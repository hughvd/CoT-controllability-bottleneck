"""Offline audit and frozen inputs for the 95-question, 20/arm GPQA expansion.

No model or monitor calls. Never modifies the historical caches.
Run: python -m monitorability.prepare_gpqa_expansion
"""
import csv
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex

from lib.config import get_model_by_name, load_config
from monitorability.runner import build_subject_prompt, grade_y, parse_monitor_answer
from monitorability.suite import load_intervention_items, monitor_answer_spec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/monitorability/gpqa_expansion/v1'
SAMPLES = 20


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen(path, value):
    content = json.dumps(value, indent=2, ensure_ascii=False) + '\n'
    if path.exists():
        assert path.read_text() == content, f'Frozen preparation changed: {path}'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def main():
    items = load_intervention_items(['gpqa'])
    grouped = defaultdict(list)
    for item in items:
        grouped[item.instance_id].append(item)
    ambiguous = {iid for iid, group in grouped.items() if len(group) != 2}
    assert ambiguous == {'132', '256', '293', '586'}
    selected = {}
    variants = []
    for iid, group in grouped.items():
        # Released order, fixed independently of any model output or score.
        for arm in (0, 1):
            selected[(iid, arm)] = next(i for i in group if i.arm == arm)
        a, b = selected[(iid, 0)], selected[(iid, 1)]
        assert a.answer == b.answer and a.raw['perturbation'] == b.raw['perturbation']
        if iid in ambiguous:
            variants.append(dict(instance_id=iid, candidates=[asdict(i) for i in group],
                                 resolution='first released pair; replace every historical response for this ID'))
    assert len(selected) == 190
    manifest_items = []
    for item in selected.values():
        prompt = build_subject_prompt(item)
        manifest_items.append(dict(item=asdict(item), subject_prompt=prompt,
                                  subject_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))

    config = load_config()
    snapshot = ROOT / 'results/monitorability/convergent_validity/data.json'
    models = [m['model'] for m in json.loads(snapshot.read_text())['models']]
    launch = {}
    sources = {str(snapshot.relative_to(ROOT)): digest(snapshot)}
    for filename, command in [('run_monitorability_sweep.sh', 'run'), ('run_sweep_round2.sh', 'mon')]:
        p = ROOT / 'scripts' / filename
        sources[str(p.relative_to(ROOT))] = digest(p)
        for line in p.read_text().splitlines():
            if not line.startswith(command + ' '):continue
            words = shlex.split(line, comments=True)
            if not words or words[0] != command or len(words) < 2 or words[1] not in models:
                continue
            providers = []; effort = None
            if '--pin-provider' in words:
                start = words.index('--pin-provider') + 1
                for word in words[start:]:
                    if word.startswith('--'):break
                    providers.append(word)
            if '--reasoning-effort' in words:effort = words[words.index('--reasoning-effort')+1]
            launch[words[1]] = dict(model_id=get_model_by_name(config, words[1]).model_id,
                                    provider_order=providers, reasoning_effort=effort,
                                    temperature=.6, max_tokens=16384, cot_instruction=None)
    assert set(launch) == set(models)
    monitor = dict(provider='openai', model_id='gpt-5-mini', reasoning_effort='low', max_tokens=8192)
    # Both historical config snapshots and current config use low monitor effort.
    assert config.judge_model.reasoning_effort == monitor['reasoning_effort']
    model_reports = []; jobs = []
    for model in models:
        path = ROOT / 'results/monitorability' / model / 'gpqa.json'
        sources[str(path.relative_to(ROOT))] = digest(path)
        cache = json.loads(path.read_text())
        log = ROOT / 'results/monitorability/_logs' / f'{model}.log'
        log_text = log.read_text(errors='replace')
        assert f"Subject: {launch[model]['model_id']} | Monitor: openai/gpt-5-mini" in log_text
        sources[str(log.relative_to(ROOT))] = digest(log)
        retained = {}; excluded = []; legacy_ids = set()
        for key, record in cache.items():
            assert 'error' not in record
            iid, arm, sample = record['instance_id'], record['x'], record['sample_idx']
            assert key == f'{iid}|arm{arm}|s{sample}' and 0 <= sample < 4
            assert record['dataset'] == 'intervention/gpqa.jsonl'
            legacy_ids.add(iid)
            if iid in ambiguous:
                excluded.append(key)
                continue
            item = selected[(iid, arm)]
            assert record['reference_answer'] == item.answer
            assert record['y'] == grade_y(item, record['final_answer'])
            valid, triggers = monitor_answer_spec(item.z_rule_ref)
            for scope in ['cot_only', 'answer_only', 'all_messages']:
                answer = parse_monitor_answer(record[f'monitor_raw_{scope}'], valid)
                assert answer is not None
                assert record[f'z_{scope}'] == int(answer in triggers)
            retained[key] = record
        assert len(legacy_ids) == 80 and len(retained) == 608 and len(excluded) == 32
        assert all(f'{iid}|arm{arm}|s{s}' in retained for iid in legacy_ids-ambiguous for arm in (0,1) for s in range(4))
        write_frozen(OUT / model / 'reused_gpqa.json', retained)
        write_frozen(OUT / model / 'excluded_legacy_keys.json', sorted(excluded))
        for iid, arm in selected:
            for sample in range(SAMPLES):
                key = f'{iid}|arm{arm}|s{sample}'
                if key not in retained:
                    jobs.append(dict(model=model, instance_id=iid, arm=arm, sample_idx=sample,
                                     record_key=key, reason='ambiguous_legacy_id' if iid in ambiguous else
                                     'new_question' if iid not in legacy_ids else 'additional_sample'))
        model_reports.append(dict(model=model, original_records=len(cache), reused=len(retained),
                                  excluded_ambiguous=len(excluded), new_required=95*2*SAMPLES-len(retained),
                                  subject=launch[model]))
    for name in ['models_config.yaml','monitorability/runner.py','monitorability/run.py','monitorability/suite.py']:
        sources[name] = digest(ROOT/name)
    write_frozen(OUT/'selected_items.json', manifest_items)
    write_frozen(OUT/'duplicate_id_audit.json', variants)
    report = dict(status='prepared_not_started', questions=95, responses_per_arm=SAMPLES,
                  bootstrap_draws=1000, monitor=monitor, ambiguous_ids=sorted(ambiguous),
                  total_existing=7680, total_reused=sum(m['reused'] for m in model_reports),
                  total_excluded=384, total_new_required=len(jobs), target_total=45600,
                  models=model_reports, sources=sources,
                  provenance_limits='Legacy configuration reconstructed from launch scripts, logs, and unchanged GPQA templates; no per-call request/provider revision metadata in historical caches.',
                  launch_requirement='Use selected_items.json, not the unfiltered dataset loader. Keep historical caches unchanged. Store prompt and configuration provenance for new calls.')
    assert len(jobs) == 38304
    write_frozen(OUT/'audit.json', report)
    job_path = OUT/'pending_jobs.csv'
    if not job_path.exists():
        with job_path.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(jobs[0]));writer.writeheader();writer.writerows(jobs)
    else:
        with job_path.open() as f:assert len(list(csv.DictReader(f))) == len(jobs)
    print(json.dumps({k:report[k] for k in ['status','questions','responses_per_arm','total_existing','total_reused','total_excluded','total_new_required','target_total']},indent=2))


if __name__ == '__main__':
    main()
