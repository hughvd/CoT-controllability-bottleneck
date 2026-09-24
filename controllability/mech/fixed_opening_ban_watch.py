"""Collect durable batches and score each run when its worker finishes."""
import json
import subprocess
import sys
import time
import modal
import fixed_opening_ban as f

volume = modal.Volume.from_name('fixed-opening-ban-out')
finished = set()
last = {}
start = time.monotonic()
while len(finished) < len(f.MODELS) and time.monotonic() - start < 7200:
    for name in f.MODELS:
        if name in finished:
            continue
        root = f.ROOT / name
        if name == 'glm':
            report = json.loads((root / 'rejection_report.json').read_text())
        else:
            manifest = json.loads((root / 'manifest.json').read_text())
            remote = name + '/' + manifest['manifest_sha256']
            report = json.loads(b''.join(volume.read_file(remote + '/report.json')))
            local = root / 'remote'
            local.mkdir(exist_ok=True)
            if report.get('outputs', 0) != last.get(name, (None, None))[1]:
                for entry in volume.iterdir(remote):
                    filename = entry.path.rsplit('/', 1)[-1]
                    if filename.startswith('batch_') and filename.endswith('.json') and not (local / filename).exists():
                        (local / filename).write_bytes(b''.join(volume.read_file(remote + '/' + filename)))
            f.base.atomic_json(local / 'report.json', report)
        state = report['status'], report.get('outputs', 0)
        if last.get(name) != state:
            print(name, *state, flush=True)
            last[name] = state
        if report['status'] in {'complete', 'failed', 'blocked', 'budget_stop'}:
            subprocess.run([sys.executable, str(f.HERE / 'fixed_opening_ban_score.py'), 'score', '--model', name], check=True)
            finished.add(name)
            print(name, 'collected and scored', flush=True)
    if len(finished) < len(f.MODELS):
        time.sleep(20)
subprocess.run([sys.executable, str(f.HERE / 'fixed_opening_ban_audit.py')], check=True)
subprocess.run([sys.executable, str(f.HERE / 'fixed_opening_ban_score.py'), 'analyze'], check=True)
print('Finished models:', sorted(finished), flush=True)
