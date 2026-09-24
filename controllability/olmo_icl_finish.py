"""Collect and grade the detached OLMo ICL sweep; never launch GPU generation."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import modal
from olmo_icl_run import ROOT, RUN, CSV

HERE=Path(__file__).resolve().parent
TERMINAL={'complete','failed','budget_stop','projection_exceeds_budget'}

def child(*args):
    for attempt in range(3):
        r=subprocess.run([sys.executable,'-u',*map(str,args)])
        if not r.returncode:return
        if attempt==2:raise RuntimeError(f'Command failed: {args}')
        time.sleep(10)

def main():
    m=json.loads((ROOT/'manifest.json').read_text());remote=RUN+'/'+m['manifest_sha256']
    volume=modal.Volume.from_name('olmo31-icl-out');last_graded=0;errors=0
    for _ in range(24*60):
        try:
            report=json.loads(b''.join(volume.read_file(remote+'/report.json')))
            latest=json.loads((ROOT/'gpu_ledger.json').read_text())['attempts'][-1]
            if report['started_at']<latest['started_at']:
                print('Waiting for current worker report',flush=True);time.sleep(60);continue
            (ROOT/'generation_report.json').write_text(json.dumps(report,indent=2))
            folder=ROOT/'remote';folder.mkdir(exist_ok=True)
            for item in volume.listdir(remote):
                if not item.path.endswith('.json'):continue
                dst=folder/Path(item.path).name
                if dst.exists() and dst.name.startswith('batch_'):continue
                tmp=dst.with_suffix('.tmp')
                with tmp.open('wb') as f:
                    for chunk in volume.read_file(item.path):f.write(chunk)
                tmp.replace(dst)
            count=len(list(folder.glob('batch_*.json')))
            print(report['status'],'batches',count,flush=True)
            if report['status'] in TERMINAL:
                ledger=json.loads((ROOT/'gpu_ledger.json').read_text());entry=ledger['attempts'][-1]
                assert report['started_at']>=entry['started_at']
                entry.update(status=report['status'],accounted_seconds=report['elapsed_seconds']+60)
                (ROOT/'gpu_ledger.json').write_text(json.dumps(ledger,indent=2))
                if report['status']!='complete':
                    if count:child(HERE/'olmo_icl_run.py','grade-partial')
                    raise RuntimeError('GPU generation stopped: '+report['status'])
                child(HERE/'olmo_icl_run.py','grade')
                import pandas as pd
                d=pd.read_csv(CSV);nonempty=d[~d.empty_thinking.astype(bool)]
                nonempty.groupby(['model','ctype','variant']).agg(mean_compliance=('compliance_score','mean'),n=('compliance_score','size')).reset_index().to_csv(ROOT/'summary.csv',index=False)
                os.environ['MPLBACKEND']='Agg';child(HERE/'analysis/icl_compare.py')
                print('COMPLETE',RUN,flush=True);return
            if count>=last_graded+4:
                child(HERE/'olmo_icl_run.py','grade-partial');last_graded=count
            errors=0
        except FileNotFoundError:print('Waiting for worker report',flush=True)
        except (OSError, TimeoutError, modal.exception.ConnectionError) as e:
            errors+=1
            if errors>=10:raise
            print('Transient collection error',type(e).__name__,flush=True)
        time.sleep(60)
    raise RuntimeError('Collection deadline reached')

if __name__=='__main__':
    with (ROOT/'finisher.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:main()
        except Exception as e:
            (ROOT/'finisher_error.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e),'at':time.time()}));raise
