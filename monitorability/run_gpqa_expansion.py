"""Resumable expansion using only the audited, deduplicated GPQA manifest.

python -m monitorability.run_gpqa_expansion --mode coordinator
Each subject and monitor result is committed before proceeding to the next stage.
"""
import argparse
import asyncio
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from monitorability.prepare_gpqa_expansion import OUT, ROOT
from monitorability.runner import (SubjectSpec, MonitorSpec, _real_subject, _real_monitor,
    build_subject_prompt, extract_final_answer, grade_y, parse_monitor_answer, cache_to_dataframe)
from monitorability.suite import EvalItem, ModelCallOutput, SCOPES, render_monitor_prompt, monitor_answer_spec


def atomic(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str)+'\n')
    tmp.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan():
    audit = json.loads((OUT/'audit.json').read_text())
    for rel, expected in audit['sources'].items():
        source=ROOT/rel
        # Figure refreshes replace this derived snapshot, which was used only to
        # establish the model roster. Verify its archived original on resume.
        if rel=='results/monitorability/convergent_validity/data.json' and sha(source)!=expected:
            source=OUT/'audit_inputs'/rel
        assert sha(source) == expected, f'Audited source changed: {rel}'
    saved = json.loads((OUT/'selected_items.json').read_text())
    items = {}
    for row in saved:
        item = EvalItem(**row['item']); key = (item.instance_id, item.arm)
        assert key not in items
        assert build_subject_prompt(item) == row['subject_prompt']
        assert hashlib.sha256(row['subject_prompt'].encode()).hexdigest() == row['subject_prompt_sha256']
        items[key] = item
    assert len(items) == 190
    return audit, items


async def retry(call):
    for attempt in range(5):
        try:return await call()
        except Exception as e:
            if getattr(e,'status_code',None) in (400,401,402,403,404) or attempt==4:raise
            await asyncio.sleep(min(60, 3*2**attempt))


async def generate(model, concurrency):
    audit, items = load_plan()
    config = next(m for m in audit['models'] if m['model']==model)
    subject = SubjectSpec(**config['subject']); monitor=MonitorSpec(**audit['monitor'])
    folder=OUT/model; reused=json.loads((folder/'reused_gpqa.json').read_text())
    db=sqlite3.connect(folder/'new_completions.sqlite');db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, payload TEXT NOT NULL, complete INTEGER NOT NULL)')
    records={k:json.loads(p) for k,p in db.execute('SELECT key,payload FROM records')}
    done={k for k, in db.execute('SELECT key FROM records WHERE complete=1')}
    errors={};active=set();started=time.time();completed_at_start=len(done)
    manifest=dict(subject=asdict(subject),monitor=asdict(monitor),scopes=list(SCOPES),
                  items_sha256=sha(OUT/'selected_items.json'),audit_sha256=sha(OUT/'audit.json'),
                  reused_sha256=sha(folder/'reused_gpqa.json'))
    p=folder/'generation_manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest
    else:atomic(p,manifest)

    def persist(key, record, complete=False):
        records[key]=record
        db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?)',(key,json.dumps(record),int(complete)));db.commit()
        if complete:done.add(key)

    def status(state):
        atomic(folder/'generation_status.json',dict(status=state,pid=os.getpid(),model=model,
            reused=len(reused),new_complete=len(done),subject_saved=len(records),target_new=config['new_required'],
            active=len(active),errors=errors,updated_at=time.time(),started_at=started,
            completed_this_launch=len(done)-completed_at_start,concurrency=concurrency))

    async def one(item,sample):
        key=f'{item.instance_id}|arm{item.arm}|s{sample}';active.add(key)
        try:
            if key not in records:
                async def subject_call():
                    out=await _real_subject(subject,build_subject_prompt(item))
                    if out.get('finish_reason')=='error' or not (out.get('answer') or out.get('cot')):
                        raise RuntimeError('Provider returned no usable response or finish_reason=error')
                    return out
                out=await retry(subject_call)
                final=extract_final_answer(out['answer'])
                record=dict(dataset=item.dataset,archetype=item.archetype,instance_id=item.instance_id,
                    category=item.category,sample_idx=sample,x=item.arm,y=grade_y(item,final),
                    final_answer=final,cot=out['cot'],answer=out['answer'],finish_reason=out['finish_reason'],
                    usage=out['usage'],reference_answer=item.answer,
                    provenance=dict(generated_at=time.time(),subject_prompt_sha256=hashlib.sha256(build_subject_prompt(item).encode()).hexdigest(),
                                    generation_manifest_sha256=sha(p)))
                persist(key,record)
            record=records[key]
            model_out=ModelCallOutput(final_answer=record['final_answer'],action_only=record['answer'],
                cot_only=record['cot'],all_msgs=(record['cot']+'\n\n'+record['answer']).strip())
            valid,triggers=monitor_answer_spec(item.z_rule_ref)
            for scope in SCOPES:
                if record.get('z_'+scope) is not None:continue
                prompt=render_monitor_prompt(item,scope,model_out)
                async def monitor_call():
                    raw=await _real_monitor(monitor,prompt)
                    answer=parse_monitor_answer(raw,valid)
                    if answer is None:raise RuntimeError('Unparseable monitor answer')
                    return raw,answer
                raw,answer=await retry(monitor_call)
                record.update({f'monitor_raw_{scope}':raw,f'monitor_answer_{scope}':answer,f'z_{scope}':int(answer in triggers)})
                record['provenance'][f'monitor_prompt_sha256_{scope}']=hashlib.sha256(prompt.encode()).hexdigest()
                persist(key,record)
            persist(key,record,True);errors.pop(key,None)
        except Exception as e:
            errors[key]=f'{type(e).__name__}: {e}'[:1200]
            print(model,key,errors[key],flush=True)
        finally:active.remove(key);status('running')

    todo=[(it,s) for it in items.values() for s in range(audit['responses_per_arm'])
          if f'{it.instance_id}|arm{it.arm}|s{s}' not in reused and f'{it.instance_id}|arm{it.arm}|s{s}' not in done]
    status('running' if done else 'probing')
    if todo:
        # First real missing sample is also the provider/monitor smoke check.
        # A resumed route already has verified successes; do not serialize its restart.
        if not done:
            await one(*todo[0])
            if errors:status('blocked');return 1
            todo=todo[1:]
        queue=asyncio.Queue()
        for job in todo:queue.put_nowait(job)
        async def worker():
            while not queue.empty() and len(errors)<16:
                try:job=queue.get_nowait()
                except asyncio.QueueEmpty:break
                await one(*job)
        await asyncio.gather(*(worker() for _ in range(concurrency)))
    if len(done)!=config['new_required']:
        status('incomplete');return 1
    combined={**reused,**records};assert len(combined)==3800
    atomic(folder/'gpqa.json',combined)
    frame=cache_to_dataframe(combined);assert len(frame)==3800
    assert frame.groupby(['instance_id','x']).size().eq(20).all()
    frame.to_csv(folder/'samples.csv',index=False)
    status('complete');db.close();return 0


def metrics(model):
    from monitorability.metrics import intervention_gmean,gmean_summary_table
    folder=OUT/model
    data=json.loads((folder/'gpqa.json').read_text());frame=cache_to_dataframe(data)
    assert len(frame)==3800
    result=intervention_gmean(frame,n_bootstrap=1000,seed=0)
    atomic(folder/'gmean_full.json',result)
    gmean_summary_table(result).to_csv(folder/'gmean_summary.csv',index=False)
    atomic(folder/'metrics_status.json',dict(status='complete',bootstrap_draws=1000,updated_at=time.time()))


async def coordinator(concurrency):
    audit,_=load_plan();metrics_sem=asyncio.Semaphore(2);states={}
    def write():atomic(OUT/'run_status.json',dict(pid=os.getpid(),models=states,updated_at=time.time()))
    async def launch(m):
        model=m['model'];folder=OUT/model
        limits_path=OUT/'concurrency.json'
        limits=json.loads(limits_path.read_text()) if limits_path.exists() else {}
        model_concurrency=limits.get(model,8 if model.startswith('qwen3.') else concurrency)
        assert isinstance(model_concurrency,int) and 1<=model_concurrency<=32
        states[model]='generating';write()
        with (folder/'generation.log').open('a') as log:
            proc=await asyncio.create_subprocess_exec(sys.executable,'-u','-m','monitorability.run_gpqa_expansion',
                '--mode','generate','--model',model,'--concurrency',str(model_concurrency),
                cwd=ROOT,stdout=log,stderr=log)
            code=await proc.wait()
        if code:
            states[model]='generation_needs_attention';write();return
        states[model]='metrics_queued';write()
        async with metrics_sem:
            states[model]='metrics';write()
            with (folder/'metrics.log').open('a') as log:
                proc=await asyncio.create_subprocess_exec(sys.executable,'-u','-m','monitorability.run_gpqa_expansion',
                    '--mode','metrics','--model',model,cwd=ROOT,stdout=log,stderr=log)
                code=await proc.wait()
            states[model]='complete' if code==0 else 'metrics_needs_attention';write()
    await asyncio.gather(*(launch(m) for m in audit['models']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['coordinator','generate','metrics'],required=True)
    parser.add_argument('--model');parser.add_argument('--concurrency',type=int,default=12);args=parser.parse_args()
    assert 1<=args.concurrency<=32
    assert os.environ.get('OPENROUTER_BASE_URL','https://openrouter.ai/api/v1').rstrip('/')=='https://openrouter.ai/api/v1'
    folder=OUT if args.mode=='coordinator' else OUT/args.model
    with (folder/(args.mode+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='coordinator':asyncio.run(coordinator(args.concurrency))
        elif args.mode=='generate':sys.exit(asyncio.run(generate(args.model,args.concurrency)))
        else:metrics(args.model)


if __name__=='__main__':main()
