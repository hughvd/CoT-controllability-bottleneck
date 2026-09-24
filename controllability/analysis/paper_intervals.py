"""Question-clustered percentile bootstrap for the existing macro-mean.

One multinomial weight per unique question is shared across behaviors, retaining
all repeated samples. Ratios of weighted score sums to weighted sample counts
preserve the existing per-behavior response mean (including unequal counts after
empty-reasoning exclusion). Behaviors then receive equal weight.
"""
from functools import lru_cache
import numpy as np
import pandas as pd

N_BOOT = 5000
SEED = 20260915


@lru_cache(maxsize=16)
def _weights(questions, n_boot, seed):
    n = len(questions)
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=n_boot)


def question_cluster_ci(data, question_col, behaviors, *, universe=None,
                        n_boot=N_BOOT, seed=SEED):
    if data[[question_col, 'ctype', 'compliance_score']].isna().any().any():
        raise ValueError('Missing question, behavior, or score')
    questions = tuple(sorted(set(data[question_col]) if universe is None else set(universe)))
    if not set(data[question_col]).issubset(questions):
        raise ValueError('Question universe omits observed questions')
    grouped = data.groupby([question_col, 'ctype']).compliance_score.agg(['sum', 'count'])
    index = pd.MultiIndex.from_product([questions, behaviors], names=[question_col, 'ctype'])
    grouped = grouped.reindex(index, fill_value=0)
    sums = grouped['sum'].to_numpy().reshape(len(questions), len(behaviors))
    counts = grouped['count'].to_numpy().reshape(sums.shape)
    if (counts.sum(axis=0) == 0).any():
        raise ValueError('Missing behavior in macro-average')
    weights = _weights(questions, n_boot, seed)
    denominators = weights @ counts
    valid = (denominators > 0).all(axis=1)
    if valid.mean() < .99:
        raise ValueError('Too few complete bootstrap replicates')
    samples = ((weights[valid] @ sums) / denominators[valid]).mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return dict(mean_compliance=float((sums.sum(axis=0) / counts.sum(axis=0)).mean()),
                ci_low=float(low), ci_high=float(high), n_questions=int((counts.sum(axis=1)>0).sum()),
                n_responses=len(data), n_boot=int(valid.sum()))


def draw_intervals(ax, xs, intervals, *, color, alpha=1, cap=.035):
    # Draw endpoints directly: percentile intervals need not contain the estimate.
    low = np.array([r['ci_low'] for r in intervals])
    high = np.array([r['ci_high'] for r in intervals])
    xs = np.asarray(xs)
    ax.vlines(xs, low, high, color=color, alpha=alpha, linewidth=1.1, zorder=4)
    ax.hlines(low, xs-cap, xs+cap, color=color, alpha=alpha, linewidth=1.1, zorder=4)
    ax.hlines(high, xs-cap, xs+cap, color=color, alpha=alpha, linewidth=1.1, zorder=4)
