"""Build an offline per-control viewer from the existing cross-model sweep.

Run: python -m monitorability.control_type_viewer
No inference or grading. Existing sweep outputs are never modified.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs
from monitorability.sweep_analysis import ROOT, MODELS, controllability_frame, monitorability_scores
from controllability.control_groups import GROUPS, CONTROLS

HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/monitorability/control_type_viewer'
FAMILIES = {name.lower(): members for name,members in GROUPS.items()}
TYPES = CONTROLS
EVALS = [
    ('mon_gmean_gpqa', 'GPQA intervention', 'g-mean', 'CoT only'),
    ('mon_gmean_wmdp_sandbagging', 'WMDP sandbagging', 'g-mean', 'CoT only'),
    ('mon_gmean_memory_advice', 'Memory advice', 'g-mean', 'CoT only'),
    ('mon_gmean_scruples_suggest_wrong', 'Scruples', 'g-mean', 'CoT only'),
    ('mon_gmean_intervention_mean', 'Intervention mean', 'mean g-mean', 'CoT only'),
    ('mon_process_tpr_cot_only', 'Process', 'TPR', 'CoT only'),
    ('mon_outcome_gmean_cot_only', 'Outcome', 'g-mean', 'CoT only'),
    ('mon_process_tpr_all_messages', 'Process', 'TPR', 'All messages'),
    ('mon_outcome_gmean_all_messages', 'Outcome', 'g-mean', 'All messages'),
    ('mon_composite_cot', 'Archetype composite', 'equal archetype mean', 'CoT only'),
]


def correlation(x, y):
    a = np.asarray(x, dtype=float); b = np.asarray(y, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b); a, b = a[valid], b[valid]
    result = {'n': len(a), 'pearson': None, 'spearman': None,
              'x_span': float(np.ptp(a)) if len(a) else None,
              'y_span': float(np.ptp(b)) if len(b) else None}
    if len(a) < 3 or np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12:
        return result
    result['pearson'] = float(pd.Series(a).corr(pd.Series(b)))
    result['spearman'] = float(pd.Series(a).corr(pd.Series(b), method='spearman'))
    return result


def build_data():
    rows = []; sources = {}; counts = []
    for model, run in MODELS.items():
        path = ROOT / 'results/prefill_continuation' / run / 'sweep.json'
        frame = controllability_frame(run)
        frame = frame[frame.condition.eq('prefill_plus_instruction')]
        frame = frame[frame.ctype.isin(TYPES)]
        extension=ROOT/'controllability/results/control_expansion/v1/correlation'/model/'scores.csv'
        extra=pd.read_csv(extension)
        assert len(extra)==300 and extra.groupby('behavior').size().eq(150).all(),model
        assert not extra.duplicated(['behavior','question_id','sample']).any(),model
        frame=pd.concat([frame,extra.rename(columns={'behavior':'ctype'})],ignore_index=True)
        sources[str(extension.relative_to(ROOT))]=hashlib.sha256(extension.read_bytes()).hexdigest()
        assert set(frame.ctype) == set(TYPES), (model, sorted(frame.ctype.unique()))
        assert frame.compliance_score.between(0, 1).all()
        values = {}
        for control, group in frame.groupby('ctype'):
            values[control] = {'soft': float(group.compliance_score.mean()),
                               'hard': float(group.compliance_score.ge(.95).mean())}
            counts.append({'model': model, 'control': control, 'valid_samples': len(group),
                           'empty_reasoning': int(group.empty_reasoning.fillna(group.empty_thinking).astype(bool).sum()),
                           'questions': group.question.nunique(),
                           'run': str(extension.relative_to(ROOT)) if control in ['numbered_sentences','suffix_sentences'] else run})
        for family, members in FAMILIES.items():
            values['family:' + family] = {
                metric: float(np.mean([values[c][metric] for c in members]))
                for metric in ['soft', 'hard']}
        mon = monitorability_scores(model)
        # Require all components rather than silently reweight missing archetypes.
        components = [mon.get(k) for k in ['mon_gmean_intervention_mean',
                      'mon_process_tpr_cot_only', 'mon_outcome_gmean_cot_only']]
        mon['mon_composite_cot'] = float(np.mean(components)) if all(
            v is not None and np.isfinite(v) for v in components) else None
        rows.append({'model': model, 'run': run, 'control': values, 'monitor': mon})
        files = [path, ROOT/'results/monitorability'/model/'gmean_summary.csv',
                 ROOT/'results/monitorability'/model/'outcome_metrics.json']
        process = ROOT/'results/monitorability'/model/'process_metrics.json'
        files.append(process if process.exists() else process.parent/'mon-gpt5high/process_metrics.json')
        for file in files:
            if file.exists():
                sources[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
    data = {'models': rows, 'types': TYPES, 'families': FAMILIES,
            'evals': [dict(zip(['key', 'name', 'metric', 'scope'], e)) for e in EVALS],
            'counts': counts, 'sources': sources}
    # Normalize optional/missing values to strict JSON null, never JS NaN.
    return clean(data)


def clean(value):
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list): return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    return value


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_data()
    (OUT/'data.json').write_text(json.dumps(data, indent=2, allow_nan=False))
    long = []; stats = []
    for metric in ['soft', 'hard']:
        for control in TYPES + ['family:' + f for f in FAMILIES]:
            for ev, name, measure, scope in EVALS:
                xs = []; ys = []
                for model in data['models']:
                    x = model['control'][control][metric]; y = model['monitor'].get(ev)
                    xs.append(x); ys.append(y if y is not None else np.nan)
                    long.append(dict(model=model['model'], control=control, compliance_metric=metric,
                                     evaluation=ev, controllability=x, monitorability=y))
                stats.append(dict(control=control, compliance_metric=metric, evaluation=ev,
                                  **correlation(xs, ys)))
    pd.DataFrame(long).to_csv(OUT/'points.csv', index=False)
    table = pd.DataFrame(stats)
    table.sort_values(['compliance_metric', 'evaluation', 'spearman'], ascending=[True, True, False],
                      na_position='last').to_csv(OUT/'correlations.csv', index=False)
    pd.DataFrame(data['counts']).to_csv(OUT/'coverage.csv', index=False)
    html = (HERE/'control_type_viewer_template.html').read_text()
    html = html.replace('__PLOTLY__', get_plotlyjs()).replace(
        '__DATA__', json.dumps(data, allow_nan=False).replace('<', '\\u003c')).replace(
        '__APP__', (HERE/'control_type_viewer.js').read_text())
    (OUT/'index.html').write_text(html)
    (OUT/'README.md').write_text('''# Per-control controllability × monitorability viewer

Open `index.html` directly; all data and Plotly are embedded, with no network or server needed.
Rebuild with `python -m monitorability.control_type_viewer` from the repository root.

Uses the existing 12-model sweep mappings in `monitorability/sweep_analysis.py`.
One point is a model, with K=0 + instruction controllability from that model's
mapped historical run. Soft = sample mean; hard = fraction scoring at least 0.95.
Missing/error/invalid-score responses are excluded as in the existing loader;
coverage.csv records the valid counts (normally 150 per model/control).
These are model-level associations, not correlations among individual responses.

The three category scores equally weight their constituent control-type means:
addition = sentence prefixes + numbered sentences + sentence endings;
suppression = omit word + no first person; style = uppercase + leetspeak + Spanish.
Chinese and character/persona styles are excluded. The two new addition controls
use matched sentence-prefix target questions and fresh instruction-only runs;
the other six reuse the historical measurements.

Intervention scores use the existing CoT-only official g-mean. Process uses TPR;
outcome uses g-mean. Both available process/outcome scopes are displayed. The
composite weights intervention mean, process CoT TPR, and outcome CoT g-mean equally.
No new inference or grading is performed. Historical model/effort/backend choices
are inherited from the source runs; this is not a controlled causal comparison.

Rankings are exploratory: n is the number of paired models, repeated samples are
not independent correlation observations, and model lineages may be related.
No multiple-testing-adjusted significance is claimed. Constant axes or fewer than
three paired models give undefined correlations. Restricted ranges, floor/ceiling
rates, and intervention eligibility are surfaced in the viewer. Model filtering
recomputes all plots/rankings; downloaded ranking CSV reflects that selection.
Static correlations.csv and points.csv contain the complete 12-model results for
both soft and hard metrics. data.json includes SHA-256 hashes of every source file.
''')
    print(f'Wrote {OUT / "index.html"}: {len(data["models"])} models, {len(TYPES)} controls, {len(EVALS)} evaluations')


if __name__ == '__main__': main()
