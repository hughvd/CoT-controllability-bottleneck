"""Audit retained experiment records without filtering any outcomes."""
import collections
import json
import sqlite3
from pathlib import Path
import fixed_opening_ban as f


def audit(name):
    root = f.ROOT / name
    manifest = json.loads((root / 'manifest.json').read_text())
    f.validate(manifest)
    requests = {r['request_id']: r for r in manifest['requests']}
    records = []
    for path in (root / 'remote').glob('batch_*.json'):
        records.extend(json.loads(path.read_text())['outputs'])
    if (root / 'api_outputs.sqlite').exists():
        with sqlite3.connect(root / 'api_outputs.sqlite') as db:
            records.extend(json.loads(r[0]) for r in db.execute('select payload from outputs'))
    assert len({r['request_id'] for r in records}) == len(records)
    banned = set(manifest['ban_token_ids'])
    cells = {c['cell_id']: c for c in manifest['cells']}
    counts = collections.Counter()
    starts = collections.Counter()
    later_banned_ids = 0
    first_stage_lengths = collections.Counter()
    terminal_openings = []
    for record in records:
        request = requests[record['request_id']]
        assert record['manifest_sha256'] == manifest['manifest_sha256']
        ids = record['first_stage_ids']
        if request['arm'] == 'fixed_ban':
            assert not (set(ids) & banned), record['request_id']
        output_ids = record['output_token_ids']
        if output_ids is not None:
            assert output_ids[:len(ids)] == ids
            assert len(output_ids) <= manifest['sampling']['max_tokens']
            if request['arm'] == 'fixed_ban':
                later_banned_ids += bool(set(output_ids[len(ids):]) & banned)
        if '<|' in record['first_token_text'] or '</think>' in record['first_token_text']:
            terminal_openings.append(record['request_id'])
        first_stage_lengths[len(ids)] += 1
        counts[(cells[request['cell_id']]['behavior'], request['condition'], request['arm'])] += 1
        starts[(request['arm'], record['first_token_text'])] += 1
    result = dict(
        model=name, expected=len(requests), retained=len(records),
        complete=len(records) == len(requests), unique_request_ids=True,
        banned_initial_ids=0, intervention_traces_with_banned_ids_later=later_banned_ids,
        first_stage_token_counts=dict(first_stage_lengths),
        terminal_opening_request_ids=terminal_openings,
        counts=[dict(behavior=k[0], condition=k[1], arm=k[2], n=v) for k,v in sorted(counts.items())],
        starts=[dict(arm=k[0], text=k[1], n=v) for k,v in starts.most_common()],
        native_token_ids_available=name != 'glm',
    )
    if name == 'glm':
        result['primary_retained'] = sum(requests[r['request_id']]['condition'] == 'instruction_only' for r in records)
        result['primary_expected'] = sum(r['condition']=='instruction_only' for r in requests.values())
        result['primary_complete'] = result['primary_retained'] == result['primary_expected']
    f.base.atomic_json(root / 'audit.json', result)
    print(name, len(records), '/', len(requests), 'banned starts: 0;',
          'banned IDs recur later in', later_banned_ids, 'intervention traces;',
          'terminal openings:', len(terminal_openings))


if __name__ == '__main__':
    for model in f.MODELS:
        audit(model)
