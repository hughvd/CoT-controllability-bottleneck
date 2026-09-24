"""Run a validated ICL sweep, with a reusable pilot and durable generations."""
import argparse
import asyncio
import fcntl
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import icl_experiment as i
import icl_judging
icl_judging.install()

async def run(args):
    model_id=dict(i.MODELS)[args.model]
    opts=SimpleNamespace(provider='DeepInfra',scale=args.scale,run_name=args.run_name,
        n_questions=30,n_samples=5,concurrency=args.concurrency,save_raw=True,example_k=5)
    cfg=i.make_cfg(args.model,model_id,opts)
    if args.scale:
        jobs=[('instruction_only',None,'na')]+[(v,s,s[0]) for v in i.SCALE_VARIANTS for s in i.SCALE_LENGTHS]
    else:
        jobs=[(v,i.default_spec(v),None) for v in ['instruction_only',*i.SCALE_VARIANTS]]
    root=i.HERE/'results'/'icl_donorfix_v1'/'runs'/cfg.run_name
    root.mkdir(parents=True,exist_ok=True)
    spec={**asdict(cfg),'model_short':args.model,'jobs':jobs,'donor_assignment_version':2}
    spec=json.loads(json.dumps(spec))
    path=root/'config.json'
    if path.exists(): assert json.loads(path.read_text())==spec
    else:path.write_text(json.dumps(spec,indent=2))
    cells=i.build_all_cells(cfg);donors=i.select_donors(cells,cfg.seed)
    cache=i.AsyncCache(cfg.eval_name,cfg.run_name,'mini',save_every=1);await cache.load()
    for k in [k for k,v in cache.data.items() if 'error' in v]:del cache.data[k]
    pilot=[];selected=set()
    for c in cells:
        donor=donors[c.ctype,c.question]
        for variant,spec,tag in jobs:
            key=f'{i._qhash(c.question)}_{c.ctype}_{variant}'+('' if tag is None else f'_L{tag}')+'_s0'
            group=(c.ctype,variant)
            if key not in cache and group not in selected:
                selected.add(group);pilot.append((key,c,donor,variant,spec,tag))
        if len(pilot)>=8:break
    if pilot:
        sem=asyncio.Semaphore(min(8,cfg.concurrency))
        await asyncio.gather(*(i.run_one_icl(c,d,v,s,t,0,cfg,cache,sem) for _,c,d,v,s,t in pilot))
        await cache.flush()
        records=[cache.data[k] for k,*_ in pilot]
        assert not any('error' in r for r in records), 'Pilot failed; inspect saved errors'
        assert any(not r['empty_thinking'] for r in records), 'Pilot returned no reasoning'
        print('Pilot passed:',len(records),'responses;',sum(r['empty_thinking'] for r in records),'empty',flush=True)
    for attempt in range(3):
        try:
            await i.run_model(cfg,jobs)
            break
        except RuntimeError:
            if attempt==2:raise
            print('Resuming failed cells from saved raw responses',flush=True)
            await asyncio.sleep(5)
    final=i.AsyncCache(cfg.eval_name,cfg.run_name,'mini');await final.load()
    expected=len(cells)*len(jobs)*cfg.n_samples
    assert len(final.data)==expected
    for r in final.data.values():
        d=donors[r['ctype'],r['target_question']]
        assert r['instruction']==d.instruction and r['donor_question_hash']==i._qhash(d.question)
    (root/'complete.json').write_text(json.dumps({'rows':expected,'model_id':model_id,'status':'complete'}))
    print('COMPLETE',cfg.run_name,expected,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=dict(i.MODELS));p.add_argument('--run-name',required=True)
    p.add_argument('--scale',action='store_true');p.add_argument('--concurrency',type=int,default=12);args=p.parse_args()
    lockpath=i.HERE/'results'/'icl_donorfix_v1'/'locks'/f'{args.run_name}.lock';lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        asyncio.run(run(args))
