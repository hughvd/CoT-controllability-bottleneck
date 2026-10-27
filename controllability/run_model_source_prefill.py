"""Wait for validated examples, dispatch matched runs, collect, score and plot."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import model_source_prefill as p


def main():
    root=p.ROOT
    status=dict(status='running',started_at=time.time(),models={})
    processes={}
    def save():p.ex.base.atomic_json(root/'pipeline_report.json',status)
    def run(command):subprocess.run(command,check=True)
    save()
    try:
        while len([v for v in status['models'].values() if v=='scored'])<2:
            for model in ['olmo31','glm']:
                folder=root/model;phase=status['models'].get(model)
                if phase=='scored':continue
                if not (folder/'examples.json').exists():
                    report=json.loads((folder/'example_report.json').read_text()) if (folder/'example_report.json').exists() else {}
                    if report.get('status')=='failed':
                        status.setdefault('example_errors',{})[model]=report['errors'];save()
                    continue
                status.get('example_errors',{}).pop(model,None)
                if not (folder/'manifest.json').exists():p.manifest(model)
                if phase is None:
                    if model=='glm':
                        pidfile=folder/'generation.pid';alive=False
                        if pidfile.exists():
                            try:os.kill(int(pidfile.read_text()),0);alive=True
                            except ProcessLookupError:pass
                        if alive:processes[model]=None
                        else:
                            report=json.loads((folder/'api_report.json').read_text()) if (folder/'api_report.json').exists() else {}
                            if report.get('status')=='complete':processes[model]=None
                            else:
                                log=(folder/'generation.log').open('a')
                                processes[model]=subprocess.Popen([sys.executable,str(p.HERE/'control_expansion.py'),'api','--model',model,'--root',str(root),'--concurrency','12'],stdout=log,stderr=subprocess.STDOUT)
                                pidfile.write_text(str(processes[model].pid))
                    elif not (folder/'dispatch.json').exists():
                        run([sys.executable,'-m','modal','run','--detach',str(p.HERE/'modal_control_expansion.py'),'--phase','dispatch','--model',model,'--root',str(root)])
                    status['models'][model]='generating';save()
                if model=='glm':
                    report=json.loads((folder/'api_report.json').read_text()) if (folder/'api_report.json').exists() else {}
                    if processes[model] is not None and processes[model].poll() not in [None,0]:raise RuntimeError('GLM generation failed; see generation.log')
                else:
                    import modal
                    m=json.loads((folder/'manifest.json').read_text());volume=modal.Volume.from_name('control-addition-expansion-out');remote=model+'/'+m['manifest_sha256']
                    try:report=json.loads(b''.join(volume.read_file(remote+'/report.json')))
                    except (modal.exception.NotFoundError,FileNotFoundError):continue
                    dest=folder/'remote';dest.mkdir(exist_ok=True)
                    for e in volume.iterdir(remote):
                        name=Path(e.path).name
                        if name.startswith('batch_') and name.endswith('.json') and not (dest/name).exists():
                            (dest/name).write_bytes(b''.join(volume.read_file(remote+'/'+name)))
                    p.ex.base.atomic_json(dest/'report.json',report)
                    if report.get('status')=='runtime_rollover':
                        run([sys.executable,'-m','modal','run','--detach',str(p.HERE/'modal_control_expansion.py'),'--phase','dispatch','--model',model,'--root',str(root),'--resume'])
                        continue
                if report.get('status') in ['failed','blocked']:raise RuntimeError(model+' generation failed: '+str(report))
                if report.get('status')=='complete':
                    status['models'][model]='scoring';save()
                    run([sys.executable,str(p.HERE/'model_source_prefill.py'),'score','--model',model])
                    status['models'][model]='scored';save()
            time.sleep(30)
        run([sys.executable,str(p.HERE/'analysis/control_expansion_figures.py'),'--only','prefill'])
        status.update(status='complete',finished_at=time.time())
    except Exception as e:
        status.update(status='failed',error=repr(e));raise
    finally:save()


if __name__=='__main__':
    import fcntl
    with (p.ROOT/'pipeline.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('A model-source pipeline is already running.')
        main()
