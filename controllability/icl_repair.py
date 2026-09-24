"""Prepare and audit isolated, resumable repairs of historical ICL donor collisions."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import icl_experiment as i
from lib.results import load_cache, save_cache, RESULTS_DIR

ROOT = i.HERE / 'results' / 'icl_donorfix_v1'
SOURCES = ['iclfull_qwen3_14b', 'iclfull_qwen3_32b', 'iclfull_glm_5_2',
           'iclscale_qwen3_32b', 'iclscale_glm_5_2']
RETAINED = ['instruction_only', *i.SCALE_VARIANTS]


def assignments():
    cfg = i.PrefillConfig()
    cells = i.build_all_cells(cfg)
    pools = {}
    for c in cells: pools.setdefault(c.ctype, []).append(c)
    rng = random.Random(cfg.seed + 1)
    old = {c.question: i.pick_donor(c, pools[c.ctype], rng) for c in cells}
    return cells, old, i.select_donors(cells, cfg.seed)


def prepare():
    cells, old, fixed = assignments()
    affected = {(c.ctype, c.question) for c in cells if old[c.question].ctype != c.ctype}
    assert len(affected) == 16
    for c in cells:
        d = fixed[c.ctype, c.question]
        assert d.ctype == c.ctype and d.question != c.question
        if (c.ctype, c.question) not in affected: assert d == old[c.question]
    ROOT.mkdir(parents=True, exist_ok=True)
    report = {'affected_cells': [{'ctype':ct, 'question':q, 'wrong_donor_ctype':old[q].ctype}
                                for ct,q in sorted(affected)], 'runs':[]}
    for source in SOURCES:
        original = load_cache('icl_continuation', source, 'mini')
        keep = {}; removed = []; excluded = []
        for key, row in original.items():
            assert 'error' not in row
            d = old[row['target_question']]
            assert row['instruction'] == d.instruction
            assert row['donor_question_hash'] == i._qhash(d.question)
            if row['variant'] not in RETAINED:
                excluded.append(key)
            elif (row['ctype'], row['target_question']) in affected:
                removed.append(key)
            else:
                keep[key] = row
        dest = source + '_donorfix_v1'
        path = RESULTS_DIR / 'icl_continuation' / dest / 'mini.json'
        if not path.exists(): save_cache('icl_continuation', dest, 'mini', keep)
        else:
            current = load_cache('icl_continuation', dest, 'mini')
            assert all(current.get(k) == v for k,v in keep.items())
        source_path = RESULTS_DIR / 'icl_continuation' / source / 'mini.json'
        report['runs'].append({'source':source, 'destination':dest,
            'source_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'retained_rows':len(keep), 'replacement_keys':removed, 'excluded_v3_keys':excluded,
            'expected_rows':len(keep)+len(removed)})
        print(dest, 'preserve',len(keep),'replace',len(removed),'exclude V3',len(excluded))
    path = ROOT/'repair_manifest.json'
    if path.exists(): assert json.loads(path.read_text()) == report
    else: path.write_text(json.dumps(report,indent=2))


def audit():
    import pandas as pd
    _, _, fixed = assignments()
    report = json.loads((ROOT/'repair_manifest.json').read_text())
    summaries = []
    for run in report['runs']:
        source = load_cache('icl_continuation', run['source'], 'mini')
        fixed_rows = load_cache('icl_continuation', run['destination'], 'mini')
        path = RESULTS_DIR/'icl_continuation'/run['source']/'mini.json'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == run['source_sha256']
        replace_keys = set(run['replacement_keys'])
        expected = set(source)-set(run['excluded_v3_keys'])
        missing = expected-set(fixed_rows)
        errors = [k for k,v in fixed_rows.items() if 'error' in v]
        for key,row in fixed_rows.items():
            if 'error' in row: continue
            donor = fixed[row['ctype'],row['target_question']]
            assert row['instruction'] == donor.instruction
            assert row['donor_question_hash'] == i._qhash(donor.question)
            if key not in replace_keys: assert row == source[key]
            else: assert row['donor_ctype'] == row['ctype']
        summary = {'run':run['destination'], 'complete':not missing and not errors,
                   'missing':len(missing), 'errors':len(errors),
                   'repaired':len(replace_keys & set(fixed_rows))-len(errors)}
        if summary['complete']:
            csv = i.HERE/'results'/f"{run['destination']}_icl_per_sample.csv"
            assert csv.exists() and len(pd.read_csv(csv)) == run['expected_rows']
        summaries.append(summary)
        print(summary)
    (ROOT/'audit.json').write_text(json.dumps(summaries,indent=2))
    return all(s['complete'] for s in summaries)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','audit']);args=p.parse_args()
    if args.phase=='prepare':prepare()
    else:audit()
