"""Durable orchestration: bounded correlation API queue and Modal collection."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import modal
from control_expansion import ROOT, PRIMARY, base, score

def main():
    logs=ROOT/'logs';logs.mkdir(exist_ok=True)
    queue=sorted(p.name for p in (ROOT/'correlation').iterdir() if p.is_dir()) if (ROOT/'correlation').exists() else []
    running={};finished=set();failures={};gpu_finished=set();primary_finished=set()
    previous=ROOT/'coordinator_status.json'
    class Adopted:
        def __init__(self,pid,name):self.pid=pid;self.name=name
        def poll(self):
            path=ROOT/'correlation'/self.name/'api_report.json'
            if path.exists():
                state=json.loads(path.read_text())['status']
                if state in ['complete','failed']:return 0 if state=='complete' else 1
            try:os.kill(self.pid,0);return None
            except ProcessLookupError:return 1
    if previous.exists():
        for name,pid in json.loads(previous.read_text()).get('correlation_running',{}).items():
            try:os.kill(pid,0)
            except ProcessLookupError:continue
            running[name]=(Adopted(pid,name),open(os.devnull,'w'));queue.remove(name)
    volume=modal.Volume.from_name('control-addition-expansion-out')
    while True:
        for name,(p,log) in list(running.items()):
            code=p.poll()
            if code is None:continue
            log.close();del running[name]
            score(name,True)
            if code:failures[name]=code
            else:finished.add(name)
        while queue and len(running)<3:
            name=queue.pop(0);report=ROOT/'correlation'/name/'api_report.json'
            if report.exists() and json.loads(report.read_text())['status']=='complete':
                score(name,True);finished.add(name);continue
            log=(logs/('correlation_'+name+'.log')).open('a')
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).parent/'control_expansion.py'),'api','--model',name,'--correlation'],stdout=log,stderr=subprocess.STDOUT)
            running[name]=(p,log)
        for name in ['qwen32','qwen14','olmo31']:
            if name in gpu_finished:continue
            folder=ROOT/name;m=json.loads((folder/'manifest.json').read_text());remote=name+'/'+m['manifest_sha256']
            if name!='olmo31' and all(r['experiment']=='icl' for r in m['requests']):gpu_finished.add(name);continue
            try:report=json.loads(b''.join(volume.read_file(remote+'/report.json')))
            except (modal.exception.NotFoundError,FileNotFoundError):continue
            local=folder/'remote';local.mkdir(exist_ok=True)
            for e in volume.iterdir(remote):
                file=Path(e.path).name
                if file.startswith('batch_') and file.endswith('.json') and not (local/file).exists():
                    temp=local/(file+'.tmp');temp.write_bytes(b''.join(volume.read_file(remote+'/'+file)));temp.replace(local/file)
            base.atomic_json(local/'report.json',report)
            if report['status']=='complete':gpu_finished.add(name);failures.pop('gpu_'+name,None)
            elif report['status']=='runtime_rollover':
                with (logs/(name+'_resume.log')).open('a') as f:
                    subprocess.run(['modal','run','--detach',str(Path(__file__).parent/'modal_control_expansion.py'),'--phase','dispatch','--model',name,'--resume','--root',str(ROOT)],stdout=f,stderr=subprocess.STDOUT,check=True)
            elif report['status']=='failed':failures['gpu_'+name]=report
        for name in PRIMARY:
            if name in primary_finished:continue
            p=ROOT/name/'api_report.json'
            api_done=name=='olmo31' or (p.exists() and json.loads(p.read_text())['status']=='complete')
            gpu_done=name=='glm' or name in gpu_finished
            if api_done and gpu_done:
                score(name);primary_finished.add(name)
        state=dict(updated_at=time.time(),correlation_complete=sorted(finished),correlation_running={n:p.pid for n,(p,f) in running.items()},
            correlation_queued=queue,gpu_complete=sorted(gpu_finished),primary_complete=sorted(primary_finished),failures=failures)
        base.atomic_json(ROOT/'coordinator_status.json',state)
        if not queue and not running and len(primary_finished)==4:
            print('All generation collected; failures:',failures,flush=True);return
        time.sleep(30)

if __name__=='__main__':
    import argparse,control_expansion
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);a=p.parse_args()
    if a.root:ROOT=a.root.resolve();control_expansion.ROOT=ROOT
    main()
