"""Collect durable experiment batches, score completed models, then analyze."""
import json
import argparse
from pathlib import Path
import subprocess
import sys
import time
import modal
import fixed_opening_ban as f
from ban_user_icl import ROOT
import fixed_opening_ban_audit as audit


def main(experiment_root=ROOT, seven_controls=False):
    root_dir = Path(experiment_root)
    f.ROOT = root_dir
    volume = modal.Volume.from_name('fixed-opening-ban-out')
    finished, failures, last = set(), {}, {}
    start = time.monotonic()
    while len(finished) < len(f.MODELS) and time.monotonic()-start < 7200:
        for name in f.MODELS:
            if name in finished:
                continue
            root = root_dir / name
            if name == 'glm':
                p = root / 'rejection_report.json'
                if not p.exists():
                    continue
                report = json.loads(p.read_text())
            else:
                manifest = json.loads((root / 'manifest.json').read_text())
                remote = name + '/' + manifest['manifest_sha256']
                try:
                    report = json.loads(b''.join(volume.read_file(remote + '/report.json')))
                except modal.exception.NotFoundError:
                    continue
                local = root / 'remote'
                local.mkdir(exist_ok=True)
                if report.get('outputs', 0) != last.get(name, (None, None))[1]:
                    for entry in volume.iterdir(remote):
                        filename = Path(entry.path).name
                        if filename.startswith('batch_') and filename.endswith('.json') and not (local / filename).exists():
                            data = b''.join(volume.read_file(remote + '/' + filename))
                            temp = local / (filename+'.tmp')
                            temp.write_bytes(data)
                            temp.replace(local / filename)
                f.base.atomic_json(local / 'report.json', report)
            state = report['status'], report.get('outputs', 0)
            if last.get(name) != state:
                print(name, *state, flush=True)
                last[name] = state
            if report['status'] in {'complete', 'failed', 'blocked', 'budget_stop'}:
                audit.audit(name)
                subprocess.run([sys.executable, str(f.HERE/'fixed_opening_ban_score.py'), 'score',
                                '--model', name, '--experiment-root', str(root_dir)], check=True)
                finished.add(name)
                if report['status'] != 'complete':
                    failures[name] = report
                print(name, 'collected and scored', flush=True)
        if len(finished) < len(f.MODELS):
            time.sleep(30)
    if len(finished) == len(f.MODELS) and not failures:
        command=[sys.executable, str(f.HERE/'analysis/ban_user_icl_analysis.py')]
        if seven_controls:command.append('--seven-controls')
        subprocess.run(command, check=True)
    f.base.atomic_json(root_dir/'pipeline_report.json', dict(finished=sorted(finished), failures=failures,
                status='complete' if len(finished)==3 and not failures else 'incomplete'))


if __name__ == '__main__':
    import fcntl
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment-root',default=str(ROOT))
    parser.add_argument('--seven-controls',action='store_true')
    args=parser.parse_args()
    with (Path(args.experiment_root)/'collection.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('A collector is already running for this experiment.')
        main(args.experiment_root,args.seven_controls)
