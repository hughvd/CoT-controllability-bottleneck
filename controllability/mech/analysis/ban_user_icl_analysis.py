"""Paired factorial effects and ICL-style bars; all outcomes retained."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ban_user_icl as exp
import fixed_opening_ban as f
sys.path.insert(0, str(f.CTRL / 'analysis'))
from paper_style import apply_style, BEHAVIOR_LABELS, PAPER_FIGURES
from paper_intervals import draw_intervals

OUT = HERE / 'ban_user_icl/v1'
ROOTS = [exp.ROOT]
BEHAVIORS = list(f.BEHAVIORS)
PRIMARY = 'continuation_soft'
MODELS = [('qwen32', 'qwen3-32b'), ('glm', 'glm-5.2'), ('olmo31', 'olmo-3.1-32b')]
ARMS = [('instruction_only', 'baseline', 'Instruction', '#9e9e9e'),
        ('instruction_only', 'fixed_ban', 'Banning', '#377eb8'),
        ('user_icl', 'baseline', 'User ICL', '#ffb04a'),
        ('user_icl', 'fixed_ban', 'User ICL + banning', '#4daf4a')]
METRICS = ['continuation_soft', 'whole_soft', 'answer_correct']
CONTRASTS = {'ban_without_icl': [-1, 1, 0, 0],
             'icl_without_ban': [-1, 0, 1, 0],
             'combo_minus_icl': [0, 0, -1, 1],
             'combo_minus_ban': [0, -1, 0, 1],
             'combo_minus_instruction': [-1, 0, 0, 1],
             'interaction': [1, -1, -1, 1]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, effects, quality, openings, sources = [], [], [], [], {}
    for model, _ in MODELS:
        frames = []
        sources[model] = []
        for source_root in ROOTS:
            path = source_root / model / 'scores.csv'
            frame = pd.read_csv(path)
            m = json.loads((path.parent / 'manifest.json').read_text())
            f.validate(m)
            assert len(frame) == len(m['requests']) and not frame.request_id.duplicated().any()
            assert set(frame.request_id) == {r['request_id'] for r in m['requests']}
            frames.append(frame)
            sources[model].append(dict(root=str(source_root), manifest=m['manifest_sha256'], scores=f.base.file_hash(path)))
        df = pd.concat(frames, ignore_index=True)
        assert len(df) == 30*3*4*len(BEHAVIORS) and not df.request_id.duplicated().any()
        assert set(df.behavior) == set(BEHAVIORS)
        counts = df.groupby(['question_id', 'behavior', 'condition', 'arm']).size()
        assert len(counts) == 30*4*len(BEHAVIORS) and counts.eq(3).all()
        assert df.groupby('behavior').question_id.nunique().eq(30).all()
        assert df[[PRIMARY, 'whole_soft', 'answer_correct']].notna().all().all()
        # No second sentence means no assessable post-opening sentence prefix.
        missing = df.continuation_soft.isna()
        assert df.loc[missing, 'behavior'].eq('prefix_sentences').all()
        for metric in METRICS:
            assert df[metric].dropna().between(0, 1).all()
        questions = sorted(df.question_id.unique())
        weights = np.random.default_rng(20260924).multinomial(len(questions),
                       np.full(len(questions), 1/len(questions)), size=5000)
        for metric in METRICS:
            for controls in [[b] for b in BEHAVIORS] + [BEHAVIORS]:
                behavior = controls[0] if len(controls) == 1 else 'average'
                estimates, bootstraps = [], []
                for condition, arm, label, _ in ARMS:
                    g = df[(df.condition == condition) & (df.arm == arm) & df.behavior.isin(controls)]
                    q = g.groupby(['question_id', 'behavior'])[metric].mean().unstack('behavior').reindex(index=questions, columns=controls)
                    present = q.notna().to_numpy().astype(float)
                    denominators = weights @ present
                    assert (denominators > 0).all()
                    boots = ((weights @ q.fillna(0).to_numpy()) / denominators).mean(axis=1)
                    estimate = q.mean(axis=0).mean()
                    lo, hi = np.quantile(boots, [.025, .975])
                    rows.append(dict(model=model, behavior=behavior, metric=metric, condition=condition,
                                     arm=arm, label=label, mean=estimate, ci_low=lo, ci_high=hi, n=len(g), n_scored=int(g[metric].notna().sum())))
                    estimates.append(estimate)
                    bootstraps.append(boots)
                for contrast, coefficients in CONTRASTS.items():
                    values = np.array(coefficients) @ np.array(bootstraps)
                    lo, hi = np.quantile(values, [.025, .975])
                    effects.append(dict(model=model, behavior=behavior, metric=metric, contrast=contrast,
                                        delta=np.dot(coefficients, estimates), ci_low=lo, ci_high=hi))
        for (behavior, condition, arm), g in df.groupby(['behavior', 'condition', 'arm']):
            quality.append(dict(model=model, behavior=behavior, condition=condition, arm=arm, n=len(g),
                                accuracy=g.answer_correct.mean(), empty=g.empty_reasoning.mean(),
                                truncated=g.truncated.mean(), reasoning_chars=g.reasoning_chars.mean()))
            for token, h in g.groupby('first_token_text', dropna=False):
                openings.append(dict(model=model, behavior=behavior, condition=condition, arm=arm,
                                     token=token, count=len(h), proportion=len(h)/len(g)))
    table = pd.DataFrame(rows)
    effects = pd.DataFrame(effects)
    table.to_csv(OUT / 'means.csv', index=False)
    effects.to_csv(OUT / 'effects.csv', index=False)
    pd.DataFrame(quality).to_csv(OUT / 'quality.csv', index=False)
    pd.DataFrame(openings).to_csv(OUT / 'first_tokens.csv', index=False)

    apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 7.7), sharex=True)
    xs = np.arange(len(MODELS))
    width = .8 / len(ARMS)
    for ax, metric, title, ylabel in zip(axes, [PRIMARY, 'answer_correct'],
                ['(a) Controllability', '(b) Task accuracy'], ['mean compliance' if PRIMARY=='whole_soft' else 'mean continuation compliance', 'accuracy']):
        for j, (condition, arm, label, color) in enumerate(ARMS):
            g = table[(table.behavior == 'average') & (table.metric == metric) &
                      (table.condition == condition) & (table.arm == arm)].set_index('model').loc[[x[0] for x in MODELS]]
            x = xs + (j - 1.5) * width
            ax.bar(x, g['mean'], width, label=label, color=color, edgecolor='white', linewidth=.5)
            draw_intervals(ax, x, g.to_dict('records'), color='#222222', cap=.018)
        ax.set_ylim(0, 1.02)
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', fontsize=12, pad=8)
        ax.grid(axis='y', alpha=.3)
    axes[-1].set_xticks(xs, [x[1] for x in MODELS])
    plt.setp(axes[-1].get_xticklabels(), rotation=35, ha='right', fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(.54, 1), fontsize=11)
    fig.tight_layout(pad=.8, h_pad=1.4, rect=[0, 0, 1, .925])
    for suffix in ['png', 'pdf']:
        fig.savefig(OUT / ('ban_user_icl_average.' + suffix), dpi=300)
    plt.close(fig)
    if len(ROOTS) > 1:
        paper_figures(table)

    lines = ['# User ICL × first-token banning', '',
             'Matched 2×2 factorial; all conditions include the control instruction. '
             f'Thirty questions per control, three samples per arm, {len(BEHAVIORS)} controls, three models: {len(BEHAVIORS)*30*3*4*3:,} responses. '
             'User ICL supplies five sentences of controlled reasoning from a different question, plus its answer, in the user message. '
             'The example, target instruction, and grading specification are held constant across arms.', '',
             'Temperature 0.6, top-p 1.0, 8,192 output tokens. Qwen/OLMo ban Okay and Alright; GLM bans The. '
             'Local masking ends after the first content token. GLM uses first-token rejection sampling on locally retokenized API reasoning text, '
             'with no native token IDs available. No filtering by later compliance, accuracy, length, or empty reasoning.', '',
             'Means weight each control equally. Intervals are paired 95% percentile question-cluster bootstrap intervals (5,000 draws), '
             'sharing question weights across arms and behaviors. They are descriptive, without multiplicity correction.', '',
             '![Average compliance and accuracy](ban_user_icl_average.png)', '',
             'Primary metric: '+('full generated reasoning, including the first sentence and opening token. Existing full-trace scores are reused; no old responses were regenerated.' if PRIMARY=='whole_soft' else 'continuation compliance after the first content token.'), '',
             '## Compliance', '',
             '| Model | Control | Instruction | Banning | User ICL | ICL + banning |',
             '|---|---|---:|---:|---:|---:|']
    for model, label in MODELS:
        for behavior in ['average'] + BEHAVIORS:
            g = table[(table.model == model) & (table.behavior == behavior) & (table.metric == PRIMARY)]
            vals = [g[(g.condition == c) & (g.arm == a)]['mean'].iloc[0] for c,a,_,_ in ARMS]
            lines.append('| ' + ' | '.join([label, BEHAVIOR_LABELS.get(behavior, 'Average')] + [f'{100*v:.1f}%' for v in vals]) + ' |')
    lines += ['', '## Incremental effects and interaction', '',
              'Interaction = (ICL + ban − ICL) − (ban − instruction). Positive interaction indicates a larger ban effect with ICL on this score scale; '
              'it does not by itself establish a mechanism, and ceiling effects constrain it.', '',
              '| Model | Control | Combo − ICL, pp (95% CI) | Combo − ban, pp (95% CI) | Interaction, pp (95% CI) |',
              '|---|---|---:|---:|---:|']
    for model, label in MODELS:
        for behavior in ['average'] + BEHAVIORS:
            g = effects[(effects.model == model) & (effects.behavior == behavior) & (effects.metric == PRIMARY)]
            vals = []
            for contrast in ['combo_minus_icl', 'combo_minus_ban', 'interaction']:
                r = g[g.contrast == contrast].iloc[0]
                vals.append(f'{100*r.delta:+.1f} ({100*r.ci_low:+.1f}, {100*r.ci_high:+.1f})')
            lines.append('| ' + ' | '.join([label, BEHAVIOR_LABELS.get(behavior, 'Average')] + vals) + ' |')
    lines += ['', '## Task accuracy', '', '| Model | Instruction | Banning | User ICL | ICL + banning |', '|---|---:|---:|---:|---:|']
    for model, label in MODELS:
        g = table[(table.model == model) & (table.behavior == 'average') & (table.metric == 'answer_correct')]
        vals = [g[(g.condition == c) & (g.arm == a)]['mean'].iloc[0] for c,a,_,_ in ARMS]
        lines.append('| ' + ' | '.join([label] + [f'{100*v:.1f}%' for v in vals]) + ' |')
    lines += ['', '## Exploratory Qwen opening check', '',
              'For the two control types with the largest Qwen gains, inspect the actual opening choices. '
              'Counts below are out of 90 responses per condition. These are descriptive observations selected after inspecting the results.', '',
              '| Control | Instruction: Okay | User ICL: Okay | Banning: alternative token | ICL + banning: same alternative |',
              '|---|---:|---:|---:|---:|']
    odf = pd.DataFrame(openings)
    for behavior, alternative in [('uppercase_thinking', 'OK'), ('leetspeak', '0')]:
        values = []
        for condition, arm, _, _ in ARMS:
            token = 'Okay' if arm == 'baseline' else alternative
            g = odf[(odf.model == 'qwen32') & (odf.behavior == behavior) &
                    (odf.condition == condition) & (odf.arm == arm) & (odf.token == token)]
            values.append(int(g['count'].sum()))
        lines.append(f'| {BEHAVIOR_LABELS[behavior]} | {values[0]} | {values[2]} | `{alternative}`: {values[1]} | `{alternative}`: {values[3]} |')
    lines += ['', 'This pattern is consistent with examples changing the choice among alternative openings '
              'while failing to displace the dominant opening in unconstrained sampling. It does not alone establish '
              'why later reasoning follows the chosen opening. Full opening frequencies are in first_tokens.csv.']
    lines += ['', '## Interpretation boundaries', '',
              '- Comparisons use matched settings within every task. Four tasks reuse the completed factorial experiment; the extension adds Chinese, sentence prefixes, and no-first-person. All use the corrected ICL donor assignment. The ICL and token-prefill pools differ in two Chinese questions, as already documented in the paper. Instructions/omit words come from the donor in every arm; an omit word selected for the donor may be less relevant to the target. Do not compare its absolute level directly with the earlier standalone fixed-ban run.' if len(ROOTS)>1 else '- Instructions/omit words come from the ICL donor in every arm. An omit word selected for the donor may be less relevant to the target; do not compare its absolute level directly with the earlier standalone fixed-ban run.',
              '- Both full-trace and post-opening scores are saved in means.csv and effects.csv. In the seven-task analysis, full-trace scores are primary: the opening token was sampled by the model, not supplied by the experimenter, so the first sentence is assessable. Post-opening sentence-prefix scores may be missing when no subsequent sentence exists; their n_scored reports eligibility. Empty reasoning scores zero.',
              '- Task accuracy measures final-answer correctness on these same instructed questions, not a separate capability benchmark. Missing/unparseable answers score zero; truncated traces are retained.',
              '- A combo improvement tests whether the two interventions combine usefully; a causal account of the initiation bottleneck still requires the other opening-distribution and prefill evidence.', '']
    (OUT / 'results.md').write_text('\n'.join(lines))
    (OUT / 'provenance.json').write_text(json.dumps(dict(sources=sources, bootstrap_seed=20260924,
                                                       bootstrap_draws=5000, controls=BEHAVIORS, primary_metric=PRIMARY), indent=2)+'\n')
    print(table[(table.behavior == 'average') & table.metric.isin([PRIMARY,'answer_correct'])].round(4).to_string(index=False))


def paper_figures(table):
    """Ban-only main figure and per-control appendices from the same seven tasks."""
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 7.4), sharex=True)
    x = np.arange(len(MODELS))
    for ax, metric, title, ylabel in zip(axes, [PRIMARY, 'answer_correct'],
            ['(a) Controllability', '(b) Task accuracy'], ['mean compliance', 'accuracy']):
        for j, (condition, arm, label, color) in enumerate(ARMS[:2]):
            g = table[(table.behavior=='average') & (table.metric==metric) &
                      (table.condition==condition) & (table.arm==arm)].set_index('model').loc[[m for m,_ in MODELS]]
            xs=x+(j-.5)*.4
            ax.bar(xs,g['mean'],.4,label='no banning' if j==0 else 'banning',color=color,edgecolor='white',linewidth=.5)
            draw_intervals(ax,xs,g.to_dict('records'),color='#222222',cap=.025)
        ax.set_ylabel(ylabel);ax.set_title(title,loc='left',fontsize=12,pad=8)
        ax.set_ylim(0,1.02);ax.set_yticks(np.arange(0,1.01,.2));ax.grid(axis='y',alpha=.3)
    axes[-1].set_xticks(x,[name for _,name in MODELS])
    plt.setp(axes[-1].get_xticklabels(),rotation=35,ha='right',fontsize=10)
    axes[0].legend(ncol=2,fontsize=12,loc='upper left',columnspacing=1,handlelength=1.8)
    fig.tight_layout(pad=.8,h_pad=1.3)
    for suffix in ['png','pdf']:fig.savefig(OUT/f'fixed_opening_ban_average.{suffix}',dpi=300)
    plt.close(fig)
    for arms, name in [(ARMS[:2], 'fixed_opening_ban_by_control'),(ARMS,'ban_user_icl_by_control')]:
        fig,axes=plt.subplots(1,3,figsize=(13,5.5),sharey=True)
        for ax,(model,label) in zip(axes,MODELS):
            g=table[(table.model==model)&(table.metric==PRIMARY)]
            matrix=np.array([[g[(g.behavior==b)&(g.condition==c)&(g.arm==a)]['mean'].iloc[0]
                              for c,a,_,_ in arms] for b in BEHAVIORS])
            ax.imshow(matrix,cmap='RdYlGn',vmin=0,vmax=1,aspect='auto')
            for row in range(len(BEHAVIORS)):
                for col in range(len(arms)):ax.text(col,row,f'{matrix[row,col]:.2f}',ha='center',va='center',fontsize=11)
            labels=['instr-only','ban'] if len(arms)==2 else ['instr-only','ban','user-ICL','ICL + ban']
            ax.set_xticks(range(len(arms)),labels,rotation=25,ha='right',fontsize=10)
            ax.set_yticks(range(len(BEHAVIORS)),[BEHAVIOR_LABELS[b] for b in BEHAVIORS]);ax.set_title(label,fontsize=12)
        fig.tight_layout(pad=.8)
        for suffix in ['png','pdf']:fig.savefig(OUT/f'{name}.{suffix}',dpi=300)
        plt.close(fig)
    for name in ['fixed_opening_ban_average','fixed_opening_ban_by_control','ban_user_icl_average','ban_user_icl_by_control']:
        for suffix in ['png','pdf']:shutil.copy2(OUT/f'{name}.{suffix}',PAPER_FIGURES/f'{name}.{suffix}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seven-controls',action='store_true')
    args=parser.parse_args()
    if args.seven_controls:
        ROOTS.append(exp.ROOT.parent/'seven_controls_extension_v1')
        BEHAVIORS = ['leetspeak','spanish','chinese','uppercase_thinking',
                     'prefix_sentences','no_first_person','omit_word']
        OUT = HERE/'ban_user_icl/seven_controls_v1'
        PRIMARY = 'whole_soft'
    main()
