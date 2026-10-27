"""Rerun the GLM factorial without an assistant prefix, matching ICL top-p omission."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'mech'))
import fixed_opening_ban as f
from control_groups import CONTROLS

ROOT = HERE / 'results/ban_user_icl/glm_no_prefix_v1'
EXPANSION = HERE / 'results/control_expansion/v1'


def prepare():
    sources = [HERE / 'results/ban_user_icl' / v / 'glm/manifest.json'
               for v in ['v1', 'seven_controls_extension_v1']]
    originals = [json.loads(p.read_text()) for p in sources]
    manifest = deepcopy(originals[0])
    cells, requests = [], []
    for old in originals:
        selected = [c for c in old['cells'] if c['behavior'] in CONTROLS]
        ids = {c['cell_id'] for c in selected}
        cells.extend(deepcopy(selected))
        requests.extend(deepcopy(r) for r in old['requests'] if r['cell_id'] in ids)
    new_path = EXPANSION / 'glm/manifest.json'
    repair_path = EXPANSION / 'numbered_icl_repair/glm/manifest.json'
    new = json.loads(new_path.read_text())
    repairs = {r['request_id']: r for r in json.loads(repair_path.read_text())['requests']}
    pairs = json.loads((EXPANSION / 'cells.json').read_text())['cells']
    cell_ids = {}
    for pair in pairs:
        c = deepcopy(pair['target'])
        c['behavior'] = c.pop('ctype')
        c.pop('controlled_cot')
        c['question_id'] = f.base.digest(c['question'])
        c['cell_id'] = f.base.digest([c['behavior'], c['question'], pair['donor']['instruction']])
        c['marker'] = pair['donor']['marker']
        c['donor'] = pair['donor']
        c['donor_question_id'] = f.base.digest(pair['donor']['question'])
        cells.append(c)
        cell_ids[c['behavior'], c['question_id']] = c['cell_id']
    for original in new['requests']:
        if original['experiment'] != 'ban_icl':
            continue
        r = deepcopy(repairs.get(original['request_id'], original))
        r['cell_id'] = cell_ids[r['behavior'], r['question_id']]
        requests.append(r)
    for r in requests:
        assert r['messages'][-1] == {'role': 'assistant', 'content': '<think>\n'}
        r['messages'].pop()
        assert all(x['role'] == 'user' for x in r['messages'])
        r['supersedes_request_id'] = r['request_id']
        r['request_id'] = f.base.digest(['glm_no_prefix_v1', r['request_id']])
    manifest.update(version='glm_no_prefix_v1', cells=cells, requests=requests,
                    sampling=dict(temperature=.6, top_p=None, max_tokens=8192),
                    protocol='No assistant prefill. top_p omitted, as in standalone hosted ICL. '
                             'Ordered first-token rejection; baseline shares first proposal. '
                             'Same targets, donors, markers, and corrected five-sentence numbering demos.',
                    scope=dict(behaviors=CONTROLS, questions_per_behavior=30, samples=3,
                               conditions=['instruction_only', 'user_icl'], arms=['baseline', 'fixed_ban']),
                    source_manifests={str(p): f.base.file_hash(p) for p in sources + [new_path, repair_path]},
                    repair_code_sha256=f.base.file_hash(__file__))
    manifest.pop('manifest_sha256')
    manifest['manifest_sha256'] = f.base.digest(manifest)
    f.validate(manifest)
    assert len(cells) == 240 and len(requests) == 2880
    f.base.immutable_json(ROOT / 'glm/manifest.json', manifest)
    print('Prepared', len(requests), 'GLM requests', flush=True)


def run():
    status = dict(status='running', started_at=time.time(), phase='generation')
    def save():
        f.base.atomic_json(ROOT / 'pipeline_report.json', status)
    save()
    commands = [
        ('generation', [sys.executable, str(HERE/'mech/ban_user_icl_glm.py'),
                        '--experiment-root', str(ROOT), '--max-proposals', '8192',
                        '--tail-batch-size', '32', '--concurrency', '24']),
        ('scoring', [sys.executable, str(HERE/'mech/fixed_opening_ban_score.py'), 'score',
                     '--model', 'glm', '--experiment-root', str(ROOT)]),
        ('figures', [sys.executable, str(HERE/'analysis/control_expansion_figures.py'), '--only', 'ban']),
    ]
    try:
        for phase, command in commands:
            status['phase'] = phase
            save()
            subprocess.run(command, check=True)
        status.update(status='complete', finished_at=time.time())
    except Exception as e:
        status.update(status='failed', error=repr(e))
        raise
    finally:
        save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'run'])
    args = parser.parse_args()
    globals()[args.phase]()
