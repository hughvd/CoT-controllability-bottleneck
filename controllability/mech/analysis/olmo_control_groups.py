"""Descriptive additive/reductive trajectories in the released 32B checkpoints.

Run from repo root: python controllability/mech/analysis/olmo_control_groups.py
Uses saved, raw-verified per-control means; does not modify the website.
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'olmo_rl' / 'control_groups'
STAGES = ['SFT', 'DPO', 'RLVR', 'ext-1/3', 'ext-2/3', 'ext-final']
GROUPS = {
    'Additive (same five controls)': ['uppercase_thinking', 'prefix_sentences',
        'leetspeak', 'language:spanish', 'language:chinese'],
    'Reductive (same two controls)': ['suppression:omit_word', 'no_first_person'],
    'Lexical suppression (four controls)': ['suppression:omit_word',
        'suppression:omit_multiple_words', 'no_first_person', 'no_commas'],
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(HERE / 'olmo_rl/fig_data/by_ctrl_type.csv')
    tables = {}
    for metric in ['ctrl_soft', 'ctrl_hard']:
        v = d.pivot(index='family', columns='checkpoint', values=metric)[STAGES]
        v['DPO_to_final'] = v['ext-final'] - v['DPO']
        v['SFT_to_final'] = v['ext-final'] - v['SFT']
        v.sort_values('DPO_to_final').to_csv(OUT / f'{metric}_by_type.csv')
        tables[metric] = v
    soft = tables['ctrl_soft']
    groups = pd.DataFrame({k: soft.loc[cs, STAGES].mean() for k, cs in GROUPS.items()})
    groups.to_csv(OUT / 'group_means.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in groups:
        axes[0].plot(STAGES, groups[name], marker='o', label=name)
    axes[0].set_title('Matched to the cross-model grouping')
    axes[0].set_ylim(0, .16)
    for c, label in [('language:russian', 'Russian'),
                     ('suppression:ignore_question', 'Ignore question'),
                     ('max_words_per_sentence', 'Max words per sentence'),
                     ('suppression:omit_multiple_words', 'Omit multiple words')]:
        axes[1].plot(STAGES, soft.loc[c, STAGES], marker='o', label=label)
    axes[1].set_title('Other controls move differently')
    axes[1].set_ylim(0, .7)
    for ax in axes:
        ax.set_ylabel('Mean soft compliance')
        ax.tick_params(axis='x', rotation=35)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc='upper left')
        ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT/'trajectories.png', dpi=200)
    fig.savefig(OUT/'trajectories.pdf')
    plt.close(fig)
    (OUT/'README.md').write_text('''# OLMo 32B control trajectories

Descriptive means from the canonical 26-type released-checkpoint sweep. SFT is
shown for context; DPO is the pre-RL baseline. All six checkpoint summaries were
verified against their raw per_sample.csv files (maximum discrepancy <1e-15).
DPO and final have 1,575 exactly matched family/question/instruction/pool/marker
items. Types have 25–100 items, so the group curves equally weight type means.
No confidence intervals or statistical significance are claimed. The checkpoints
are one dependent training trajectory, not independent model replicates.

The first panel preserves the five additive / two reductive definitions used in
the cross-model heatmaps. A separate sensitivity curve adds omit-multiple-words
and no-commas to lexical suppression. We do not silently put ignore-question or
max-words-per-sentence into that group; these impose different constraints.
The panels use different y-axis ranges to expose the near-floor core controls.

Core additive mean: DPO .0180 → final .0163; reductive .110 → .095.
No-first-person and no-commas are zero throughout. Russian collapses late,
while ignoring the question increases. Max-words improves in soft scoring,
but its hard compliance remains approximately zero. This does not establish
an additive/reductive difference in RL sensitivity: most core controls are
at floor and the clearest shifts are type-specific. Longer traces also increase
opportunities to violate word-avoidance constraints; mean reasoning characters
rise from 15,716 at DPO to 19,325 at final, with ~30% missing a final answer.

The separate four-control mechanism sweep is consistent with modest omission
decline (instruction-only .121 → .083), but spontaneous omission also declines
(.125 → .063). Thus lower absolute omission compliance alone does not establish
weaker sensitivity to the instruction. This is distinct from the 7B experiment
that measures the effect of additional controllability SFT at each checkpoint.
''')
    print(OUT/'trajectories.png')


if __name__ == '__main__':
    main()
