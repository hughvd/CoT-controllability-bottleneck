"""Appendix table of the monitorability scores used in cross-model correlations.

Run: python -m monitorability.plot_monitorability_score_table
Uses saved bootstrap standard deviations, not correlation confidence intervals.
"""
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

from monitorability.plot_control_monitor_heatmap import EVALUATIONS, OUT, SOURCE
from monitorability.control_type_viewer import ROOT
from controllability.analysis.paper_style import apply_style, export_figure, PAPER_FIGURES

NAME = 'monitorability_score_uncertainty_table'
MODEL_LABELS = {
    'qwen3-32b': 'Qwen3-32B', 'gpt-oss-120b': 'GPT-OSS-120B',
    'glm-5.2': 'GLM-5.2', 'deepseek-r1-0528': 'DeepSeek-R1-0528',
    'kimi-k2-thinking': 'Kimi-K2-Thinking', 'gemma-4-31b-it': 'Gemma-4-31B-IT',
    'qwen3.5-27b': 'Qwen3.5-27B', 'qwen3.6-27b': 'Qwen3.6-27B',
    'deepseek-v3.2': 'DeepSeek-V3.2', 'deepseek-v4-pro': 'DeepSeek-V4-Pro',
    'minimax-m3': 'MiniMax-M3', 'nemotron-3-super-120b': 'Nemotron-3-Super-120B',
}
CAPTION = (
    'Monitorability scores and uncertainty across the twelve-model suite. '
    'Cells show the CoT-only monitorability score used in the correlation analysis '
    '± one bootstrap standard deviation. Intervention evaluations use the saved '
    'bootstrap-mean reference g-mean; Flaky Tools uses outcome g-mean, and GPQA '
    'Physics uses process sensitivity. The bottom row gives the minimum and maximum '
    'model scores for each evaluation. Light blue shading encodes score magnitude on a shared 0–1 scale. These uncertainties describe evaluation-score '
    'variability, not confidence intervals for the correlations; a zero bootstrap '
    'standard deviation at the floor does not establish zero population uncertainty.'
)
REFERENCE_TEXT = (
    'The monitorability evaluations differ in their observed spread across models '
    '(Appendix Figure~\\ref{fig:monitorability-score-uncertainty}). Memory Advice '
    'and GPQA Physics are concentrated near ceiling, with scores spanning '
    '0.872--0.982 and 0.868--0.962, respectively, while Scruples scores remain '
    'between 0.000 and 0.222. Because several between-model differences are small '
    'relative to bootstrap variability, we treat the corresponding rank correlations '
    'as descriptive patterns rather than evidence of a stable ordering of models.'
)


def draw_score_table(scores, deviations, ranges, name, labels, groups, boundaries, note='', expanded_models=()):
    """Shared 0–1 shading makes absolute score levels comparable across tables."""
    models = list(scores.index)
    keys = list(scores.columns)
    ncols = len(keys)
    apply_style()
    fig, ax = plt.subplots(figsize=(3.3 + 1.5*ncols, 7.5))
    fig.subplots_adjust(left=3.2/(3.3+1.5*ncols), right=.99, bottom=.11, top=.86)
    cmap = LinearSegmentedColormap.from_list('light_score', ['#ffffff', '#bfd9ed'])
    norm = Normalize(0,1)
    for i in range(len(models) + 1):
        for j, key in enumerate(keys):
            color = cmap(norm(scores.iloc[i,j])) if i < len(models) else '#edf0f3'
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=color,edgecolor='white',linewidth=.6))
            text = (f'{scores.iloc[i,j]:.2f} ± {deviations.iloc[i,j]:.2f}'
                    if i < len(models) else f'{ranges.loc[key,"min"]:.2f}–{ranges.loc[key,"max"]:.2f}')
            ax.text(j,i,text,ha='center',va='center',fontsize=12,
                    fontweight='bold' if i == len(models) else 'normal')
    for boundary in [-.5,len(models)-.5,len(models)+.5]:
        ax.axhline(boundary,color='#333333',lw=1)
    for boundary in boundaries:
        ax.axvline(boundary,color='#aab2b9',lw=.8)
    ax.set_xlim(-.5,ncols-.5)
    ax.set_ylim(len(models)+.5,-.5)
    ax.set_xticks(range(ncols),labels,fontsize=12)
    ax.set_yticks(range(len(models)+1),[MODEL_LABELS[m]+(' †' if m in expanded_models else '') for m in models]+['Range (min–max)'],fontsize=12)
    ax.xaxis.tick_top()
    ax.tick_params(axis='both',length=0,pad=12)
    ax.get_yticklabels()[-1].set_fontweight('bold')
    for label,center in groups:
        ax.text(center,1.105,label,transform=ax.get_xaxis_transform(),ha='center',fontsize=12,fontweight='bold')
    for spine in ax.spines.values():spine.set_visible(False)
    cax = fig.add_axes([.73,.047,.25,.014])
    fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',ticks=[0,.5,1])
    cax.tick_params(labelsize=10,length=2)
    fig.text(.72,.054,'Score',ha='right',va='center',fontsize=10)
    if note:fig.text(.02,-.025,note,ha='left',va='top',fontsize=10)
    export_figure(fig,OUT,name)
    shutil.copy2(OUT / f'{name}.pdf',PAPER_FIGURES / f'{name}.pdf')


def main():
    sources = {}

    def read(path):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    data = read(SOURCE)
    rows = []
    for model in data['models']:
        folder = ROOT / 'results/monitorability' / model['model']
        path = folder / 'gmean_summary.csv'
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        intervention = pd.read_csv(path).query("scope == 'cot_only'").set_index('dataset')
        outcome = read(folder / 'outcome_metrics.json')['cot_only']
        process = read(folder / 'process_metrics.json')['cot_only']['overall']
        for index, (key, label) in enumerate(EVALUATIONS):
            score = model['monitor'][key]
            if index < 4:
                dataset = 'intervention/' + key.removeprefix('mon_gmean_') + '.jsonl'
                selected=intervention
                if key in model.get('monitor_source_overrides',{}):
                    path=ROOT/model['monitor_source_overrides'][key]
                    sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
                    selected=pd.read_csv(path).query("scope=='cot_only'").set_index('dataset')
                source_score = selected.loc[dataset, 'gmean_mean']
                sd = selected.loc[dataset, 'gmean_std']
            elif index == 4:
                source_score = outcome['overall']['gmean']
                sd = outcome['overall_bootstrap_summary'][0]['gmean_std']
            else:
                source_score = process['gmean2']
                sd = process['TPR_boot_std']
            assert np.isclose(score, source_score, rtol=0, atol=1e-12), (model['model'], key)
            assert np.isfinite([score, sd]).all() and 0 <= score <= 1 and sd >= 0
            rows.append(dict(model=model['model'], evaluation=key, label=label,
                             score=score, bootstrap_sd=sd))
    frame = pd.DataFrame(rows)
    assert len(frame) == 72 and not frame.duplicated(['model', 'evaluation']).any()
    models = [m['model'] for m in data['models']]
    keys = [key for key, _ in EVALUATIONS]
    scores = frame.pivot(index='model', columns='evaluation', values='score').loc[models, keys]
    deviations = frame.pivot(index='model', columns='evaluation', values='bootstrap_sd').loc[models, keys]
    ranges = frame.groupby('evaluation').score.agg(['min', 'max']).loc[keys]
    ranges['span'] = ranges['max'] - ranges['min']
    frame.to_csv(OUT / f'{NAME}.csv', index=False)
    ranges.to_csv(OUT / f'{NAME}_ranges.csv')

    from monitorability.gpqa_figure_snapshot import snapshot_note
    draw_score_table(scores, deviations, ranges, NAME,
                     ['GPQA','WMDP','Memory\nadvice','Scruples','Flaky Tools\n(CharXiv)','GPQA\nPhysics'],
                     [('Intervention',1.5),('Outcome',4),('Process',5)], [3.5,4.5])
    caption=CAPTION+' '+snapshot_note(data)
    (OUT / f'{NAME}_caption.md').write_text(caption + '\n\nSuggested main-text reference:\n\n' + REFERENCE_TEXT + '\n')
    (OUT / f'{NAME}.tex').write_text(
        '\\begin{figure}[t]\n    \\centering\n'
        f'    \\includegraphics[width=\\linewidth]{{figures/{NAME}.pdf}}\n'
        '    \\caption{' + caption.replace('±', '$\\pm$') + '}\n'
        '    \\label{fig:monitorability-score-uncertainty}\n\\end{figure}\n')
    provenance = dict(sources=sources, uncertainty='one saved bootstrap standard deviation; not a confidence interval',
                      score_source=str(SOURCE.relative_to(ROOT)), models=models, evaluations=EVALUATIONS,
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / f'{NAME}_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(ranges.round(4).to_string())


if __name__ == '__main__':
    main()
