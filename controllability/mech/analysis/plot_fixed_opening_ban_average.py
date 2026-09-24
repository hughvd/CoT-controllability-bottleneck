"""Main-paper compliance and task-accuracy bars from saved, matched responses."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CTRL = HERE.parents[1]
sys.path.insert(0, str(CTRL / 'analysis'))
from paper_style import apply_style, export_figure, SQUARE_SIZE, LEGEND_SIZE, PAPER_FIGURES
from paper_intervals import question_cluster_ci, draw_intervals

ROOT = CTRL / 'results/fixed_opening_ban/v1'
OUT = HERE / 'fixed_opening_ban/v1'
NAME = 'fixed_opening_ban_average'
TYPES = ['uppercase_thinking', 'leetspeak', 'spanish', 'omit_word']
MODELS = [('qwen32', 'qwen3-32b'), ('glm', 'glm-5.2'), ('olmo31', 'olmo-3.1-32b')]
ARMS = ['baseline', 'fixed_ban']
METRICS = [('continuation_soft', '(a) Controllability', 'mean continuation compliance'),
           ('answer_correct', '(b) Task accuracy', 'accuracy')]
SEED = 20260923


def main():
    # Once the seven-control extension is complete, it is the paper's canonical
    # ban dataset. The earlier four-control artifacts remain archived in v1.
    expanded = CTRL / 'results/ban_user_icl/seven_controls_extension_v1/pipeline_report.json'
    if expanded.exists() and json.loads(expanded.read_text()).get('status') == 'complete':
        subprocess.run([sys.executable, str(HERE/'ban_user_icl_analysis.py'), '--seven-controls'], check=True)
        return
    rows, sources = [], {}
    for model, _ in MODELS:
        source = ROOT / model / 'scores.csv'
        sources[str(source.relative_to(CTRL))] = hashlib.sha256(source.read_bytes()).hexdigest()
        data = pd.read_csv(source)
        data = data[(data.condition == 'instruction_only') & data.behavior.isin(TYPES)].copy()
        assert len(data) == 720 and not data.request_id.duplicated().any()
        counts = data.groupby(['behavior', 'question_id', 'arm']).size()
        assert len(counts) == 240 and counts.eq(3).all()
        assert data.groupby('behavior').question_id.nunique().eq(30).all()
        universe = sorted(data.question_id.unique())
        assert data.answer_correct.isin([0, 1]).all()
        for metric, _, _ in METRICS:
            for arm in ARMS:
                selected = data[data.arm == arm].rename(columns={
                    'behavior': 'ctype', metric: 'compliance_score'})
                summary = question_cluster_ci(selected, 'question_id', TYPES,
                                              universe=universe, seed=SEED)
                # Every metric retains the same 360 responses per arm, including
                # empty/truncated traces. Missing/unparseable answers score zero.
                assert summary['n_responses'] == 360
                assert np.isclose(summary['mean_compliance'], selected.compliance_score.mean())
                rows.append(dict(model=model, arm=arm, metric=metric, **summary))
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    # Keep the original compliance-only table compatible with earlier readers.
    table[table.metric == 'continuation_soft'].drop(columns='metric').to_csv(OUT / f'{NAME}.csv', index=False)
    table[table.metric == 'answer_correct'].drop(columns='metric').rename(
        columns={'mean_compliance': 'mean_accuracy'}).to_csv(OUT / f'{NAME}_task_accuracy.csv', index=False)
    table.rename(columns={'mean_compliance': 'mean_score'}).to_csv(OUT / f'{NAME}_metrics.csv', index=False)

    apply_style()
    # Keep ICL fonts, bar/interval styling, model order, and 0–1 axes. Stack
    # panels so the figure stays readable in the manuscript's half-width slot.
    fig, axes = plt.subplots(2, 1, figsize=(SQUARE_SIZE[0], 7.4), sharex=True)
    x = np.arange(len(MODELS))
    width = .8 / len(ARMS)
    styles = [('baseline', 'no banning', '#9e9e9e'),
              ('fixed_ban', 'banning', '#377eb8')]
    for ax, (metric, title, ylabel) in zip(axes, METRICS):
        for j, (arm, label, color) in enumerate(styles):
            selected = table[(table.arm == arm) & (table.metric == metric)].set_index('model').loc[[m for m, _ in MODELS]]
            xs = x + (j - (len(ARMS) - 1) / 2) * width
            ax.bar(xs, selected.mean_compliance, width, label=label,
                   color=color, edgecolor='white', linewidth=.5)
            draw_intervals(ax, xs, selected.to_dict('records'), color='#222222', cap=.025)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontsize=12, pad=8)
        ax.set_ylim(0, 1.02)
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.grid(axis='y', alpha=.3)
    axes[-1].set_xticks(x, [label for _, label in MODELS])
    plt.setp(axes[-1].get_xticklabels(), rotation=35, ha='right', fontsize=10)
    axes[0].legend(ncol=2, fontsize=LEGEND_SIZE, loc='upper left',
                   columnspacing=1, handlelength=1.8, labelspacing=.3)
    fig.tight_layout(pad=.8, h_pad=1.3)
    export_figure(fig, OUT, NAME, square=True)
    shutil.copy2(OUT / f'{NAME}.pdf', PAPER_FIGURES / f'{NAME}.pdf')

    # Preserve the detailed plot for the appendix, under an explicit name.
    for suffix in ['png', 'pdf']:
        source = OUT / f'compliance_change.{suffix}'
        if source.exists():
            shutil.copy2(source, OUT / f'fixed_opening_ban_by_control.{suffix}')
            shutil.copy2(source, PAPER_FIGURES / f'fixed_opening_ban_by_control.{suffix}')
    provenance = dict(metrics=[m[0] for m in METRICS], condition='instruction_only',
                      controls=TYPES, weighting='equal control-type weights; equal question/sample weights within type',
                      intervals='95% percentile question-cluster bootstrap; common weights across arms and controls',
                      bootstrap_draws=5000, seed=SEED, source_sha256=sources,
                      style_reference='controllability/analysis/icl_compare.py:make_avg_plot',
                      accuracy_definition='Final-answer correctness on the same controllability questions, with control instructions. All responses retained; missing/unparseable answers count as incorrect.',
                      glm_note='Hosted rejection sampling; no uninstructed GLM data used.')
    (OUT / f'{NAME}_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(table[['model', 'arm', 'metric', 'mean_compliance', 'ci_low', 'ci_high']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
