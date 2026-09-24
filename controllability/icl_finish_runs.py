"""Collect/grade the detached GPT-OSS job and audit all requested ICL runs.
No generation is launched here. The worker and API runners own generation.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import modal

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results'/'icl_donorfix_v1'
GPT=HERE/'results'/'iclfull_gptoss_120b_high_donorfix_v1'
manifest=json.loads((GPT/'manifest.json').read_text())
remote=manifest['run']+'/'+manifest['manifest_sha256']
volume=modal.Volume.from_name('gptoss-icl-out')
TERMINAL={'complete','failed','budget_stop','projection_exceeds_budget'}

def collect():
    report=json.loads(b''.join(volume.read_file(remote+'/report.json')))
    (GPT/'generation_report.json').write_text(json.dumps(report,indent=2))
    folder=GPT/'remote';folder.mkdir(exist_ok=True)
    for item in volume.listdir(remote):
        if not item.path.endswith('.json'):continue
        dst=folder/Path(item.path).name
        if dst.exists() and dst.name.startswith('batch_'):continue
        tmp=dst.with_suffix('.tmp')
        with tmp.open('wb') as f:
            for chunk in volume.read_file(item.path):f.write(chunk)
        tmp.replace(dst)
    if report['status'] in TERMINAL:
        ledger=json.loads((GPT/'gpu_ledger.json').read_text());entry=ledger['attempts'][-1]
        assert report['started_at']>=entry['started_at']
        entry.update(status=report['status'],accounted_seconds=report['elapsed_seconds']+60)
        (GPT/'gpu_ledger.json').write_text(json.dumps(ledger,indent=2))
    return report

def child(script,*args):
    subprocess.run([sys.executable,'-u',str(HERE/script),*args],check=True)

def main():
    started=time.monotonic();last_graded=0;gpt_done=(GPT/'audit.json').exists();failures=0
    while time.monotonic()-started<24*3600:
        status={'updated_at':time.time(),'gpt_complete':gpt_done}
        if not gpt_done:
            try:
                report=collect();status['gpt_generation']=report['status'];failures=0
                batches=list((GPT/'remote').glob('batch_*.json'))
                if report['status']=='complete':
                    child('gptoss_icl_run.py','grade');gpt_done=True
                elif report['status'] in TERMINAL:
                    if batches:child('gptoss_icl_run.py','grade-partial')
                    raise RuntimeError('GPT-OSS generation stopped: '+report['status'])
                elif len(batches)>=last_graded+4:
                    child('gptoss_icl_run.py','grade-partial');last_graded=len(batches)
            except FileNotFoundError:
                status['gpt_generation']='waiting_for_worker_report'
            except (OSError, TimeoutError, modal.exception.ConnectionError) as e:
                failures+=1
                status['collection_error']=type(e).__name__
                if failures>=10:raise
        markers=list((ROOT/'runs').glob('*/complete.json'))
        status['api_runs_complete']=[p.parent.name for p in markers]
        status['api_generations_saved']={p.name:len(list(p.glob('*.json'))) for p in (HERE/'transcripts').glob('*donorfix_v1')}
        status['api_processes_running']={}
        for proc in json.loads((ROOT/'processes.json').read_text()):
            try:os.kill(proc['pid'],0);alive=True
            except ProcessLookupError:alive=False
            status['api_processes_running'][proc['run']]=alive
        status['gpt_complete']=gpt_done
        (ROOT/'status.json').write_text(json.dumps(status,indent=2))
        print(status,flush=True)
        if len(markers)==6 and gpt_done:
            child('icl_repair.py','audit')
            assert all(r['complete'] for r in json.loads((ROOT/'audit.json').read_text()))
            import pandas as pd
            frames=[]
            for name in status['api_runs_complete']:
                if not name.startswith('iclfull'):continue
                frames.append(pd.read_csv(HERE/'results'/f'{name}_icl_per_sample.csv'))
            frames.append(pd.read_csv(HERE/'results'/f'{manifest["run"]}_icl_per_sample.csv'))
            all_rows=pd.concat(frames,ignore_index=True)
            summary=[]
            for (model,ctype,variant),d in all_rows.groupby(['model','ctype','variant']):
                nonempty=d[~d.empty_thinking.astype(bool)]
                summary.append(dict(model=model,ctype=ctype,variant=variant,n=len(d),n_nonempty=len(nonempty),
                    mean_compliance_nonempty=nonempty.compliance_score.mean(),
                    mean_compliance_all=d.compliance_score.mean(),accuracy=d.correct.mean(),
                    empty_rate=d.empty_thinking.mean(),length_limited_rate=d.finish_reason.eq('length').mean()))
            pd.DataFrame(summary).to_csv(ROOT/'full_sweep_summary.csv',index=False)
            os.environ['MPLBACKEND']='Agg'
            child('analysis/icl_compare.py')
            status['status']='complete'
            (ROOT/'status.json').write_text(json.dumps(status,indent=2))
            return
        time.sleep(60)
    raise RuntimeError('Collection time limit reached; completed generations remain saved')

if __name__=='__main__':
    import fcntl
    with (ROOT/'finisher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception as e:
            (ROOT/'finisher_error.json').write_text(json.dumps({'error':str(e),'type':type(e).__name__,'at':time.time()}))
            raise
