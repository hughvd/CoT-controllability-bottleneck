"""Prepare and grade the native high-effort GPT-OSS ICL replication."""
import argparse
import asyncio
import fcntl
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import icl_experiment as i
import icl_harmony as ih
import harmony_backend as h
import gptoss_prefill_run as grading
import icl_judging
icl_judging.install()

RUN='iclfull_gptoss_120b_high_donorfix_v1'
ROOT=i.HERE/'results'/RUN
CSV=i.HERE/'results'/f'{RUN}_icl_per_sample.csv'

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def validate_output_text(g):
    decoded=h._enc().decode(g['output_token_ids'])
    # vLLM retains the sampled stop ID but omits its spelling from output.text.
    # Accept exactly this terminal difference; never trim arbitrary text.
    assert decoded in (g['raw_text'],g['raw_text']+'<|return|>'), 'Output token/text mismatch'

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    cfg=i.PrefillConfig();cells=i.build_all_cells(cfg);donors=i.select_donors(cells,cfg.seed)
    requests=[]
    for c in cells:
        donor=donors[c.ctype,c.question]
        for variant in ['instruction_only',*i.SCALE_VARIANTS]:
            messages=i.build_icl_messages(c,donor,variant,i.default_spec(variant))
            r={'ctype':c.ctype,'modality':donor.modality,'question':c.question,'correct_letter':c.correct_letter,
               'instruction':donor.instruction,'marker':donor.marker,'omit_words':donor.omit_words,
               'donor_question_hash':i._qhash(donor.question),'variant':variant,'messages':messages}
            r['request_id']=digest(r)
            r['prompt_token_ids']=ih.build_tokens(messages)
            decoded=h._enc().decode(r['prompt_token_ids'])
            assert decoded.endswith('<|start|>assistant')
            assert '<|channel|>analysis' not in decoded
            assert 'Reasoning: high' in decoded
            requests.append(r)
    assert len(requests)==1200 and len({r['request_id'] for r in requests})==1200
    longest=max(len(r['prompt_token_ids']) for r in requests)
    assert longest+8192<=32768, f'Context limit exceeded: {longest}+8192'
    # One target per behavior, selected by longest visible demonstration; all four conditions.
    smoke_cells=set()
    for ct in sorted({c.ctype for c in cells}):
        r=max((r for r in requests if r['ctype']==ct and r['variant']=='icl_assistant_visible'),key=lambda r:len(r['prompt_token_ids']))
        smoke_cells.add((r['ctype'],r['question']))
    manifest={'run':RUN,'model':'openai/gpt-oss-120b','revision':'b5c939de8f754692c1647ca79fbf85e8c1e70f8a',
       'harmony_sha256':hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),
       'icl_sha256':hashlib.sha256(Path(i.__file__).read_bytes()).hexdigest(),
       'renderer_sha256':hashlib.sha256(Path(ih.__file__).read_bytes()).hexdigest(),
       'sampling':{'temperature':.6,'top_p':1.,'top_k':-1,'max_tokens':8192,'n':5,'effort':'high'},
       'donor_assignment_version':2,'max_model_len':32768,'requests':requests,
       'smoke_ids':[r['request_id'] for r in requests if (r['ctype'],r['question']) in smoke_cells]}
    manifest['manifest_sha256']=digest(manifest)
    path=ROOT/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest,'Manifest is immutable'
    else:path.write_text(json.dumps(manifest,ensure_ascii=False))
    print('Prepared 1,200 prompts / 6,000 responses; max input tokens',longest,'; reusable pilot',len(manifest['smoke_ids'])*5)

async def grade(partial=False):
    grading.ROOT=ROOT
    judge=grading.Judge()
    manifest=json.loads((ROOT/'manifest.json').read_text());requests={r['request_id']:r for r in manifest['requests']}
    gens={}
    for p in sorted((ROOT/'remote').glob('batch_*.json')):
        for g in json.loads(p.read_text()):
            key=f"{g['request_id']}_{g['sample']}"
            assert key not in gens and g['sample'] in range(5)
            r=requests[g['request_id']]
            assert g['effort']=='high' and g['prompt_token_ids']==r['prompt_token_ids']
            validate_output_text(g)
            assert h.split_channels(g['raw_text'])==(g['reasoning_continuation'],g['response'])
            gens[key]=g
    if partial:assert 0<len(gens)<=6000
    else:assert len(gens)==6000,f'Incomplete: {len(gens)}/6000'
    path=ROOT/'grades.jsonl';done={r['key']:r for r in grading.read_journal(path)}
    paid={r['key']:r['score'] for r in grading.read_journal(judge.path) if r.get('score') is not None}
    async def one(key,g):
        if key in done:return
        r=requests[g['request_id']];text=g['reasoning_continuation']
        if not text.strip():score=0.
        elif r['modality']=='family':score=float(i.pe.grade_score(text,r['ctype'],marker=r['marker']))
        elif r['modality']=='suppression':score=float(i.pe.supp.score_suppression(r['ctype'],text,r['omit_words'] or []))
        elif key in paid:score=paid[key]
        else:score=await judge.score(r['modality'],r['ctype'],text,key)
        row={'key':key,'score':score}
        with path.open('a') as f:f.write(json.dumps(row)+'\n')
        done[key]=row
        if len(done)%250==0:print('Graded',len(done),'/ 6000; judge cost',round(judge.cost,3),flush=True)
    await asyncio.gather(*(one(k,g) for k,g in gens.items()))
    if partial:return
    rows=[]
    for key,g in gens.items():
        r=requests[g['request_id']];score=done[key]['score'];pred=i.extract_answer(g['response'])
        rows.append({'model':'gpt-oss-120b','ctype':r['ctype'],'modality':r['modality'],
            'variant':r['variant'],'length':'fixed','sample':g['sample'],'target_question':r['question'],
            'donor_question_hash':r['donor_question_hash'],'donor_ctype':r['ctype'],'donor_assignment_version':2,
            'instruction':r['instruction'],'compliance_score':score,'compliance':int(score>=.5),
            'predicted':pred,'correct_letter':r['correct_letter'],'correct':int(pred==r['correct_letter']),
            'reasoning':g['reasoning_continuation'],'response':g['response'],'finish_reason':g['finish_reason'],
            'provider':'modal/vllm/gpt-oss-120b','empty_thinking':not g['reasoning_continuation'].strip(),
            'reasoning_effort':'high','manifest_sha256':manifest['manifest_sha256']})
    df=pd.DataFrame(rows)
    assert not df.duplicated(['ctype','target_question','variant','sample']).any()
    tmp=CSV.with_suffix('.csv.tmp');df.to_csv(tmp,index=False);tmp.replace(CSV)
    quality={'rows':len(df),'empty':int(df.empty_thinking.sum()),'length_limited':int(df.finish_reason.eq('length').sum()),
             'judge_cost_usd':judge.cost,'manifest_sha256':manifest['manifest_sha256'],'status':'complete'}
    (ROOT/'audit.json').write_text(json.dumps(quality,indent=2));print(quality)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','grade','grade-partial']);args=p.parse_args()
    if args.phase=='prepare':prepare()
    else:
        with (ROOT/'grader.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            asyncio.run(grade(args.phase=='grade-partial'))
