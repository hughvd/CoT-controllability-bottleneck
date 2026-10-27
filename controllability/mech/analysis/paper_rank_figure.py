"""Paper opening-rank/probability figure from completed, question-level rescoring.

No inference. Inputs already deduplicate openings and average prompts within
questions. Bootstrap paired questions, using identical resamples for conditions
and metrics. Rank summaries are geometric; probability summaries arithmetic.
"""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'analysis'))
from paper_style import apply_style, export_figure, LABEL_SIZE, TICK_SIZE, MODEL_COLORS

ROOT = HERE.parent / 'data/rank_runs/opening_suite_v1/analysis'
OUT = HERE / 'figs'
NAME = 'opening_rank_probability'
TYPES = ['uppercase_thinking', 'leetspeak', 'prefix_sentences', 'numbered_sentences']
LABELS = ['Uppercase', 'Leetspeak', 'Sentence\nprefixes', 'Numbered\nsentences']
MODELS = [('olmo31', 'OLMo-3.1-32B-Think', MODEL_COLORS['olmo31']),
          ('qwen32', 'Qwen3-32B', MODEL_COLORS['qwen32'])]
CONDS = ['no_instruction', 'instruction_only']
SEED, DRAWS = 20260916, 10000


def summarize(data):
    rows = []
    for model, _, _ in MODELS:
        for ct in TYPES:
            sub = data[(data.model_tag == model) & (data.ctype == ct) & (data.phase == 'opening')]
            assert not sub.duplicated(['question_id','condition']).any()
            paired = sub.pivot(index='question_id', columns='condition',
                               values=['log10_rank','content_mass']).sort_index()
            assert not paired.isna().any().any()
            n = len(paired)
            if n < 2:
                raise ValueError(f'Insufficient paired questions: {model}/{ct}')
            draws = np.random.default_rng(SEED).integers(0,n,(DRAWS,n))
            for metric in ['log10_rank','content_mass']:
                values = paired[metric][CONDS].to_numpy()
                samples = values[draws].mean(axis=1)
                means = values.mean(axis=0)
                low, high = np.quantile(samples,[.025,.975],axis=0)
                for j, condition in enumerate(CONDS):
                    transform = (lambda x: 10**x) if metric == 'log10_rank' else (lambda x:x)
                    rows.append(dict(model=model, behavior=ct, condition=condition,
                        metric='rank' if metric == 'log10_rank' else 'probability',
                        estimate=transform(means[j]), ci_low=transform(low[j]),
                        ci_high=transform(high[j]), n_questions=n))
    return pd.DataFrame(rows)


def probability_label(value, _):
    percent = value * 100
    if percent >= .0001 * (1 - 1e-12):
        return f'{percent:g}%'
    return rf'$10^{{{int(round(np.log10(percent)))}}}$%'


def render(summary, metrics, name):
    single = len(metrics) == 1
    size = (7.2, 4.6) if metrics == ['rank'] else (7.2, 5.8) if single else (14.5, 5.2)
    axis_font_size = TICK_SIZE + 2 if metrics == ['rank'] else TICK_SIZE
    fig, axes = plt.subplots(1,len(metrics),figsize=size,squeeze=False)
    axes = axes[0]
    x = np.arange(len(TYPES))
    for column, (model, label, color) in enumerate(MODELS):
        for row, metric in enumerate(metrics):
            ax = axes[row]
            ax.set_yscale('log')
            for i, ct in enumerate(TYPES):
                cell = summary[(summary.model==model)&(summary.behavior==ct)&(summary.metric==metric)].set_index('condition').loc[CONDS]
                positions = i + (column-.5)*.36 + np.array([-.07,.07])
                ax.plot(positions,cell.estimate,color=color,lw=1.4,alpha=.7,zorder=2)
                for j, (_, r) in enumerate(cell.iterrows()):
                    xp = positions[j]
                    ax.vlines(xp,r.ci_low,r.ci_high,color=color,lw=1.2,zorder=3)
                    ax.hlines([r.ci_low,r.ci_high],xp-.035,xp+.035,color=color,lw=1.2,zorder=3)
                    ax.plot(xp,r.estimate,marker='o',ms=6,mec=color,mew=1.5,
                            mfc='white' if j==0 else color,zorder=4)
            ax.set_xlim(-.5,len(TYPES)-.5)
            ax.grid(axis='y',alpha=.25)
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_xticks(x,LABELS,fontsize=axis_font_size)
            ax.tick_params(axis='both',labelsize=axis_font_size)
            if metric=='rank':
                ax.set_ylim(.8,10000)
                ax.yaxis.set_major_locator(FixedLocator([1,2,5,10,100,1000,10000]))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:,.0f}'))
                if column == 0:
                    ax.axhspan(1,10,color='#dbe4ec',alpha=.4,zorder=0)
            else:
                ax.set_ylim(1e-12,1e-2)
                ax.yaxis.set_major_locator(FixedLocator([1e-12,1e-10,1e-8,1e-6,1e-4,1e-2]))
                ax.yaxis.set_major_formatter(FuncFormatter(probability_label))
    for ax, metric in zip(axes, metrics):
        ax.set_ylabel('Best compliant-token rank' if metric == 'rank' else 'Compliant-token probability',fontsize=axis_font_size)
    handles = [Line2D([],[],marker='o',ls='none',ms=6,mew=1.5,mec='#333333',
                       mfc='white' if j==0 else '#333333',label=label)
               for j,label in enumerate(['No instruction','Instruction'])]
    handles = [Line2D([],[],color=color,lw=2.4,label=label)
               for _,label,color in MODELS] + handles
    for ax, metric in zip(axes, metrics):
        ax.legend(handles=handles,loc='upper left' if metric=='rank' else 'lower left',
                  ncol=1,frameon=False,fontsize=TICK_SIZE)
    fig.tight_layout(pad=.8,h_pad=1.4,w_pad=1.5)
    export_figure(fig,OUT,name)


def main():
    apply_style()
    source = ROOT / 'per_question.csv'
    data = pd.read_csv(source)
    summary = summarize(data)
    selected = data[(data.phase=='opening') & data.ctype.isin(TYPES)]
    assert selected.content_sample_mass.eq(0).all(), 'Update caption: nonzero filtered compliant mass'
    assert (summary.ci_low>0).all()
    render(summary, ['rank', 'probability'], NAME)
    render(summary, ['rank'], 'opening_rank')
    render(summary, ['probability'], 'opening_probability')
    summary.to_csv(OUT/f'{NAME}.csv',index=False)
    provenance=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bootstrap_draws=DRAWS,bootstrap_seed=SEED,phase='opening',controls=TYPES,
        rank_summary='geometric mean',probability_summary='arithmetic mean',
        interval='pointwise 95% percentile; paired question resampling')
    (OUT/f'{NAME}_provenance.json').write_text(json.dumps(provenance,indent=2))


if __name__=='__main__':
    main()
