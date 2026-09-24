"""Paper heatmap from the saved convergent-validity snapshot; no website writes.

Run: python -m monitorability.plot_control_monitor_heatmap
"""
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from monitorability.control_type_viewer import ROOT, correlation
from controllability.analysis.paper_style import apply_style, export_figure, BEHAVIOR_LABELS
from controllability.control_groups import CONTROLS, GROUPS, LABELS

OUT = ROOT / 'results/monitorability/convergent_validity'
SOURCE = OUT / 'data.json'
NAME = 'monitorability_by_control_spearman'
EVALUATIONS = [
    ('mon_gmean_gpqa', 'GPQA'),
    ('mon_gmean_wmdp_sandbagging', 'WMDP'),
    ('mon_gmean_memory_advice', 'Memory advice'),
    ('mon_gmean_scruples_suggest_wrong', 'Scruples'),
    ('mon_outcome_gmean_cot_only', 'Flaky Tools (CharXiv)'),
    ('mon_process_tpr_cot_only', 'GPQA Physics'),
]


def draw_heatmap(values, row_labels, name, boundaries):
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 6.2 if len(row_labels) == 6 else 6.8))
    fig.subplots_adjust(left=.23, right=.86, top=.88, bottom=.24)
    im = ax.imshow(values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(CONTROLS)), [LABELS[c] for c in CONTROLS],
                  rotation=42, ha='right', rotation_mode='anchor')
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0, pad=8)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f'{value:.2f}' if np.isfinite(value) else '—',
                    ha='center', va='center', fontsize=11,
                    color='white' if abs(value) >= .60 else '#202020')
    for boundary in boundaries:
        ax.axhline(boundary, color='#202020', linewidth=1.0)
    for boundary in [2.5,4.5]:ax.axvline(boundary, color='#202020', linewidth=1.0)
    for label, center in [('Addition', 1), ('Suppression', 3.5), ('Style', 6)]:
        ax.text(center, 1.045, label, transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=14)
    cax = fig.add_axes([.885, .24, .018, .64])
    fig.colorbar(im, cax=cax, label='Spearman correlation', ticks=[-1, -.5, 0, .5, 1])
    export_figure(fig, OUT, name)


def main():
    data = json.loads(SOURCE.read_text())
    rows = []
    for ev, label in EVALUATIONS:
        for control in CONTROLS:
            stat = correlation([m['control'][control]['soft'] for m in data['models']],
                               [m['monitor'].get(ev) for m in data['models']])
            rows.append(dict(evaluation=ev, control=control, spearman=stat['spearman'], n=stat['n']))
    frame = pd.DataFrame(rows)
    values = frame.pivot(index='evaluation', columns='control', values='spearman').loc[
        [k for k, _ in EVALUATIONS], CONTROLS].to_numpy()
    frame.to_csv(OUT / f'{NAME}.csv', index=False)
    draw_heatmap(values, [label for _, label in EVALUATIONS], NAME, [3.5, 4.5])
    (OUT / f'{NAME}_provenance.json').write_text(json.dumps({
        'source': str(SOURCE.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'metric': 'soft compliance, K=0 + instruction',
        'models': [m['model'] for m in data['models']],
        'controls': CONTROLS, 'evaluations': EVALUATIONS,
        'pair_counts': sorted(frame.n.unique().tolist()),
        'source_run_hashes': data['sources'],
    }, indent=2))
    (OUT / f'{NAME}_caption.md').write_text(
        'Spearman correlations across 12 models between mean soft controllability '
        '(K=0, with instruction) and each monitorability evaluation. '
        'Intervention and outcome evaluations use g-mean; process uses TPR. '
        'All evaluations use CoT-only monitoring. Columns group three addition, '
        'two suppression, and three style controls. '
        'Rows above the first horizontal line are intervention evaluations; '
        'the fifth row is outcome monitoring on Flaky Tools (CharXiv), and the '
        'sixth is process monitoring on GPQA Physics. '
        'Lines separate evaluation and control groups. Correlations are descriptive; '
        'cells do not indicate statistical significance. No aggregate evaluations '
        'or composite scores are included.\n')
    print(f'{len(rows)} correlations; paired model counts: {sorted(frame.n.unique())}')
    control_name = 'control_by_control_spearman'
    control_rows = []
    for a in CONTROLS:
        for b in CONTROLS:
            stat = correlation([m['control'][a]['soft'] for m in data['models']],
                               [m['control'][b]['soft'] for m in data['models']])
            control_rows.append(dict(row_control=a, column_control=b,
                                     spearman=stat['spearman'], n=stat['n']))
    control_frame = pd.DataFrame(control_rows)
    control_values = control_frame.pivot(index='row_control', columns='column_control',
        values='spearman').loc[CONTROLS, CONTROLS].to_numpy()
    assert np.allclose(control_values, control_values.T,equal_nan=True)
    assert np.allclose(np.diag(control_values)[np.isfinite(np.diag(control_values))], 1)
    control_frame.to_csv(OUT / f'{control_name}.csv', index=False)
    draw_heatmap(control_values, [LABELS[c] for c in CONTROLS], control_name, [2.5,4.5])
    provenance = json.loads((OUT / f'{NAME}_provenance.json').read_text())
    provenance.pop('evaluations')
    provenance['pair_counts'] = sorted(control_frame.n.unique().tolist())
    (OUT / f'{control_name}_provenance.json').write_text(json.dumps(provenance, indent=2))
    (OUT / f'{control_name}_caption.md').write_text(
        'Pairwise Spearman correlations across 12 models between mean soft '
        'controllability scores (K=0, with instruction). Both axes use the same '
        'ordering: three addition, two suppression, and three style controls. '
        'Thin lines separate these groups; group labels appear above the columns. '
        'Chinese and character/persona styles are excluded. '
        'Correlations are descriptive and cells do not indicate statistical significance.\n')


if __name__ == '__main__':
    main()
