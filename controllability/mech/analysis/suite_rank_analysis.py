"""Combined eight-behavior analysis. Reads the original experiment unchanged."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import suite_rank_probe as s
import rank_probe as base
from analysis.rank_analysis import bootstrap_delta

TYPES = (*base.TYPES, *s.TYPES)
LABELS = dict(uppercase_thinking='Uppercase', leetspeak='Leetspeak', disemvowel='Disemvowel',
              no_commas='No commas', spell_out_numbers='Spell out numbers', start_with='Start with marker',
              prefix_sentences='Prefix sentences', numbered_sentences='Numbered sentences')
MODELS = {'olmo31': 'OLMo-3.1-32B-Think', 'qwen32': 'Qwen3-32B'}
COLORS = {'olmo31': '#147d92', 'qwen32': '#b65b17'}
FAILURE_LABELS = ['Token 0', 'Later in first sentence', 'Later sentence', 'No observed violation']
METRICS = ['log10_rank', 'content_log10_mass', 'content_mass', 'content_sample_mass',
           'legal_mass', 'legal_sample_mass', 'forbidden_mass', 'forbidden_log10_mass',
           'forbidden_sample_mass', 'completion_mass', 'completion_sample_mass',
           'neutral_mass', 'excluded_mass', 'entropy_nats', 'observed_probability',
           'observed_log10_probability', 'forbidden_log10_odds', 'total_variation']


def failure_category(trace):
    v = trace['violation']
    if v is None:
        return FAILURE_LABELS[3]
    return FAILURE_LABELS[0] if v['token_position'] == 0 else FAILURE_LABELS[1] if v.get('sentence_index', 1) == 1 else FAILURE_LABELS[2]


def collect(root):
    manifest = s.load_manifest(root / 'manifest.json')
    prior_root = s.DATA / 'rank_runs' / manifest['prior_run']
    prior = base.load_manifest(prior_root / 'manifest.json')
    assert prior['manifest_sha256'] == manifest['prior_manifest_sha256']
    records = {}
    for source, loader, m in [(prior_root, base.load_results, prior), (root, s.load_results, manifest)]:
        for tag in s.MODELS:
            rs = loader(source / f'{tag}_pairs.jsonl', m['manifest_sha256'])
            expected = {p['pair_id'] for p in m['pairs'][tag]}
            if set(rs) != expected:
                raise ValueError(f'Incomplete {source.name}/{tag}: {len(rs)}/{len(expected)}')
            records.update(rs)
    traces = prior['traces'] + manifest['traces']
    rows, failure_rows, opening_seen = [], [], set()
    for trace in traces:
        category = failure_category(trace)
        v = trace['violation']
        failure_rows.append(dict(model_tag=trace['model_tag'], ctype=trace['ctype'],
                                 question_id=trace['question_id'], prompt_id=trace['prompt_id'],
                                 trace_id=trace['trace_id'], sample=trace['sample'],
                                 marker=trace.get('marker'), marker_pool=trace.get('marker_pool'),
                                 phrasing_pool=trace['phrasing_pool'], category=category,
                                 token_position=v['token_position'] if v else None,
                                 sentence_index=v.get('sentence_index', 1) if v else None,
                                 compliance_score=trace['compliance_score'], compliance_hard=trace['compliance_hard']))
        for target in trace['targets']:
            r = records[target['pair_id']]
            pos = target['position']
            phases = []
            if pos == 0 and r['pair_id'] not in opening_seen:
                phases.append('opening')
                opening_seen.add(r['pair_id'])
            if target['is_first_violation']:
                phases.append('first_failure')
            if pos in (1, 2, 3):
                phases.append(f'token_{pos}')
            for phase in phases:
                for cond in s.CONDS:
                    c = r['conditions'][cond]
                    best = c['best_content']
                    obs = c['observed'][str(trace['output_token_ids'][pos])]
                    row = dict(model_tag=r['model_tag'], ctype=r['ctype'], question_id=r['question_id'],
                               prompt_id=r['prompt_id'], trace_id=trace['trace_id'], sample=trace['sample'],
                               pair_id=r['pair_id'], phase=phase, position=pos, failure_category=category,
                               marker=trace.get('marker'), condition=cond,
                               constraint_active=r.get('constraint_active', True),
                               log10_rank=np.log10(best['rank']) if best else np.nan,
                               best_token=best['text'] if best else None,
                               observed_probability=obs['probability'],
                               observed_log10_probability=obs['log_probability']/np.log(10),
                               argmax_token=c['argmax']['text'], nucleus_size=c['nucleus_size'],
                               total_variation=r['distribution_shift']['total_variation'])
                    for metric in METRICS:
                        if metric in c:
                            row[metric] = c[metric]
                    for key in ['content', 'forbidden']:
                        lp = c.get(f'{key}_log_mass')
                        row[f'{key}_log10_mass'] = lp / np.log(10) if lp is not None else np.nan
                    bad_lp, good_lp = c.get('forbidden_log_mass'), c.get('legal_log_mass')
                    row['forbidden_log10_odds'] = ((bad_lp-good_lp)/np.log(10)
                                                    if bad_lp is not None and good_lp is not None else np.nan)
                    rows.append(row)
    return manifest, prior, records, traces, pd.DataFrame(rows), pd.DataFrame(failure_rows)


def summarize(df):
    # Samples -> prompts -> questions, keeping each question equally weighted.
    keys = ['model_tag', 'ctype', 'phase', 'question_id', 'condition']
    active = df[df.constraint_active]
    prompts = active.groupby(keys+['prompt_id'])[METRICS].mean().reset_index()
    q = prompts.groupby(keys)[METRICS].mean().reset_index()
    rows = []
    for (tag, ct, phase), group in q.groupby(['model_tag','ctype','phase']):
        for metric in METRICS:
            if metric == 'total_variation':
                continue  # Already a paired distance; never subtract it again.
            wide = group.pivot(index='question_id', columns='condition', values=metric).dropna()
            if wide.empty:
                continue
            a, b = (wide[c].to_numpy() for c in s.CONDS)
            delta, lo, hi = bootstrap_delta(a, b)
            rows.append(dict(model_tag=tag, ctype=ct, phase=phase, metric=metric,
                             n_questions=len(wide), no_instruction=a.mean(), instruction_only=b.mean(),
                             delta=delta, ci_low=lo, ci_high=hi))
    shifts = []
    for (tag, ct, phase), group in q[q.condition == s.CONDS[1]].groupby(['model_tag','ctype','phase']):
        mean, lo, hi = bootstrap_delta(np.zeros(len(group)), group.total_variation)
        shifts.append(dict(model_tag=tag, ctype=ct, phase=phase, mean=mean, ci_low=lo, ci_high=hi, n_questions=len(group)))
    return q, pd.DataFrame(rows), pd.DataFrame(shifts)


def failure_summary(f):
    dummy = pd.get_dummies(f.category).reindex(columns=FAILURE_LABELS, fill_value=0)
    frame = pd.concat([f[['model_tag','ctype','question_id','prompt_id']], dummy], axis=1)
    prompt = frame.groupby(['model_tag','ctype','question_id','prompt_id'])[FAILURE_LABELS].mean().reset_index()
    q = prompt.groupby(['model_tag','ctype','question_id'])[FAILURE_LABELS].mean().reset_index()
    rows = []
    for (tag, ct), group in q.groupby(['model_tag','ctype']):
        for category in FAILURE_LABELS:
            mean, lo, hi = bootstrap_delta(np.zeros(len(group)), group[category])
            rows.append(dict(model_tag=tag,ctype=ct,category=category,mean=mean,ci_low=lo,ci_high=hi,n_questions=len(group)))
    return pd.DataFrame(rows)


def save(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=180)
    fig.savefig(out / f'{name}.pdf')
    plt.close(fig)


def failure_plot(summary, out):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    colors = ['#bd4d31', '#db9d34', '#39839b', '#b7bec5']
    for ax, tag in zip(axes, s.MODELS):
        bottom = np.zeros(len(TYPES))
        for cat, color in zip(FAILURE_LABELS, colors):
            vals = [summary[(summary.model_tag==tag)&(summary.ctype==ct)&(summary.category==cat)]['mean'].iloc[0] for ct in TYPES]
            ax.bar(np.arange(len(TYPES)), vals, bottom=bottom, color=color, label=cat)
            bottom += vals
        ax.set_title(MODELS[tag], fontsize=11)
        ax.set_ylabel('Question-weighted fraction')
        ax.set_ylim(0,1)
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xticks(range(len(TYPES)), [LABELS[ct].replace(' ', '\n') for ct in TYPES], fontsize=9)
    axes[0].legend(ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.4), frameon=False, fontsize=9)
    fig.suptitle('Where does the first local violation occur?', y=.99)
    fig.text(.5,.01,'Samples averaged within prompts, then questions. No observed violation is limited to the saved continuation.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.93])
    save(fig,out,'first_failure_positions')


def paired_grid(q, metric, out, name, ylabel, phase='opening'):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=True)
    rank_axis = metric == 'log10_rank'
    if rank_axis:
        maximum = 10 ** q.loc[q.phase == phase, metric].max()
        upper = max(100, maximum * 1.25)
        rank_ticks = [1, 2, 5, 10] + [10 ** k for k in range(2, int(np.floor(np.log10(upper))) + 1)]
    for ax, ct in zip(axes.flat, TYPES):
        for tag, color in COLORS.items():
            sub=q[(q.model_tag==tag)&(q.ctype==ct)&(q.phase==phase)]
            wide=sub.pivot(index='question_id', columns='condition', values=metric).dropna()
            for _, row in wide.iterrows():
                values = row[list(s.CONDS)].to_numpy()
                ax.plot([0,1],10 ** values if rank_axis else values,color=color,alpha=.13,lw=.7)
            means = np.array([wide[c].mean() for c in s.CONDS])
            ax.plot([0,1],10 ** means if rank_axis else means,color=color,marker='o',lw=2,label=MODELS[tag])
        if rank_axis:
            ax.set_yscale('log')
            ax.set_ylim(.9, upper)
            ax.yaxis.set_major_locator(FixedLocator(rank_ticks))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis='y', labelleft=True, labelsize=8)
            ax.axhspan(1, 10, color='#718096', alpha=.08, zorder=0)
        ax.set_title(LABELS[ct],fontsize=10)
        ax.set_xticks([0,1],['No instruction','Instruction'],fontsize=8)
        ax.set_xlim(-.12,1.12)
        ax.grid(axis='y',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.supylabel('Best legal content/progress token rank (lower is better)' if rank_axis else ylabel,x=.012,fontsize=11)
    fig.suptitle(f'{phase.replace("_", " ").capitalize()} predictions at identical prefixes')
    caption = ('Log-spaced rank axis; shaded band: ranks 1–10. Thin lines: question geometric means; thick lines: overall geometric means. Suppression panels are secondary.'
               if rank_axis else 'Thin lines: questions; thick lines: means. Prefix-task content means progress toward the required marker. Suppression panels are secondary.')
    fig.text(.5,.01,caption,ha='center',fontsize=9)
    fig.tight_layout(rect=[.035,.04,1,.94])
    save(fig,out,name)


def suppression_plot(q,out):
    fig, axes=plt.subplots(2,2,figsize=(10,7),sharey=True)
    for i,tag in enumerate(s.MODELS):
        for j,ct in enumerate(['no_commas','spell_out_numbers']):
            ax=axes[i,j]
            for phase,color in [('opening','#147d92'),('first_failure','#b65b17')]:
                sub=q[(q.model_tag==tag)&(q.ctype==ct)&(q.phase==phase)]
                wide=sub.pivot(index='question_id',columns='condition',values='forbidden_log10_mass').dropna()
                for _,row in wide.iterrows():
                    ax.plot([0,1],row[list(s.CONDS)],color=color,alpha=.15,lw=.7)
                delta,lo,hi=bootstrap_delta(wide[s.CONDS[0]],wide[s.CONDS[1]])
                ax.plot([0,1],[wide[c].mean() for c in s.CONDS],color=color,marker='o',lw=2,
                        label=f'{phase.replace("_"," ")}: Δ={delta:+.2g} [{lo:+.2g},{hi:+.2g}], n={len(wide)}')
            ax.set_title(f'{MODELS[tag]} · {LABELS[ct]}',fontsize=10)
            ax.set_xticks([0,1],['No instruction','Instruction'])
            ax.legend(fontsize=7,frameon=False)
            ax.grid(axis='y',alpha=.15)
            ax.spines[['top','right']].set_visible(False)
    fig.supylabel('log₁₀ forbidden-token probability mass (lower is better)',x=.01,fontsize=11)
    fig.suptitle('Instruction effects on comma and digit suppression')
    fig.text(.5,.01,'Opening: all prompts. First failure: failed responses only. Comparisons are paired within phase, not between phases.',ha='center',fontsize=8)
    fig.tight_layout(rect=[.035,.04,1,.94])
    save(fig,out,'suppression_probabilities')


def decoding_plot(q,out,phase='opening'):
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharey=False)
    for ax,ct in zip(axes.flat,TYPES):
        kind='forbidden' if ct in {'no_commas','spell_out_numbers'} else 'content'
        for i,tag in enumerate(s.MODELS):
            d=q[(q.model_tag==tag)&(q.ctype==ct)&(q.phase==phase)&(q.condition==s.CONDS[1])]
            vals=[d[f'{kind}_mass'].mean(),d[f'{kind}_sample_mass'].mean()]
            xs=np.arange(2)+(i-.5)*.25
            ax.bar(xs,vals,.24,color=COLORS[tag],label=MODELS[tag])
            for x,v in zip(xs,vals):
                ax.annotate(f'{v:.2g}',(x,v),xytext=(0,3),textcoords='offset points',ha='center',fontsize=7)
        ax.set_title(f'{LABELS[ct]}\n{kind} probability',fontsize=10)
        ax.set_xticks([0,1],['Raw','T=.6, top-p=.95'],fontsize=8)
        ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
        ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=7,frameon=False)
    fig.suptitle(f'Instruction-present {phase.replace("_", " ")} mass before and after decoding filters')
    fig.text(.5,.01,'Axes vary across tasks. Prefix content means marker progress. Suppression panels show forbidden mass. This is not historical ban4 sampling.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94])
    save(fig,out,'decoding_accessibility' if phase=='opening' else 'first_failure_decoding_accessibility')


def analyze(run):
    root=s.DATA/'rank_runs'/run
    out=root/'analysis'
    out.mkdir(exist_ok=True)
    m,prior,records,traces,df,f=collect(root)
    q,summary,shifts=summarize(df)
    failures=failure_summary(f)
    for name,frame in [('measurements',df),('per_question',q),('paired_summary',summary),
                       ('distribution_shifts',shifts),('first_failures',f),('failure_summary',failures)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    failure_plot(failures,out)
    paired_grid(q,'log10_rank',out,'opening_ranks','log₁₀ rank of best legal content/progress token')
    paired_grid(q,'content_log10_mass',out,'opening_probabilities','log₁₀ legal content/progress probability mass')
    paired_grid(q,'log10_rank',out,'first_failure_ranks','log₁₀ rank of best legal content/progress token','first_failure')
    suppression_plot(q,out)
    decoding_plot(q,out)
    decoding_plot(q,out,'first_failure')
    table=[]
    for tag in s.MODELS:
        for ct in TYPES:
            examples=df[(df.model_tag==tag)&(df.ctype==ct)&(df.phase=='first_failure')]
            example=examples.sort_values(['prompt_id','sample']).iloc[0]
            r=records[example.pair_id]
            candidates=dict(r['fixed_candidates'])
            observed_id=str(next(ref['observed_id'] for ref in r['references'] if ref['trace_id']==example.trace_id))
            candidates[observed_id]={c:r['conditions'][c]['observed'][observed_id] for c in s.CONDS}
            for token,both in candidates.items():
                a,b=(both[c] for c in s.CONDS)
                table.append(dict(model_tag=tag,ctype=ct,prompt_id=r['prompt_id'],position=r['position'],
                                  marker=r.get('marker'),token=a['text'],rank_without=a['rank'],rank_with=b['rank'],
                                  role='observed violation' if token==observed_id else 'legal alternative',
                                  p_without=a['probability'],p_with=b['probability']))
    pd.DataFrame(table).to_csv(out/'token_examples.csv',index=False)
    provenance=dict(manifest_sha256=m['manifest_sha256'],prior_manifest_sha256=prior['manifest_sha256'],
                    analysis_sha256=s.file_hash(Path(__file__)),bootstrap_draws=10000,bootstrap_seed=20260908)
    provenance['result_sha256'] = {f'{run_name}/{tag}': s.file_hash(s.DATA/'rank_runs'/run_name/f'{tag}_pairs.jsonl')
                                  for run_name in [run,m['prior_run']] for tag in s.MODELS}
    (out/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2))
    lines=['# Expanded instruction-sensitivity experiment','',
           'Eight behaviors, two released models, 720 saved responses. Six added behaviors contribute 576 responses; original uppercase/leetspeak results are reused unchanged. Rescoring only.','',
           '## First-failure results','',
           'Positive log changes mean greater probability under instruction. Suppression tasks use forbidden mass (lower is better); other tasks use legal content or marker-progress mass (higher is better). These quantities are not averaged into a suite score.','',
           '| Model | Behavior | Questions with failures | Mean log₁₀ mass change [95% CI] | Raw mass without → with | Filtered mass without → with |',
           '|---|---|---:|---:|---:|---:|']
    for tag in s.MODELS:
        for ct in TYPES:
            kind='forbidden' if ct in {'no_commas','spell_out_numbers'} else 'content'
            sub=summary[(summary.model_tag==tag)&(summary.ctype==ct)&(summary.phase=='first_failure')].set_index('metric')
            lp,raw,filtered=[sub.loc[k] for k in [f'{kind}_log10_mass',f'{kind}_mass',f'{kind}_sample_mass']]
            lines.append(f'| {MODELS[tag]} | {LABELS[ct]} | {int(lp.n_questions)} | {lp.delta:+.3g} [{lp.ci_low:+.3g}, {lp.ci_high:+.3g}] | {raw.no_instruction:.6g} → {raw.instruction_only:.6g} | {filtered.no_instruction:.4g} → {filtered.instruction_only:.4g} |')
    lines += ['', '## Failure locations', '',
              '| Model | Behavior | Token 0 | Later in first sentence | Later sentence | No observed violation |',
              '|---|---|---:|---:|---:|---:|']
    for tag in s.MODELS:
        for ct in TYPES:
            z=failures[(failures.model_tag==tag)&(failures.ctype==ct)].set_index('category')
            lines.append(f'| {MODELS[tag]} | {LABELS[ct]} | '+' | '.join(f'{z.loc[c,"mean"]:.1%}' for c in FAILURE_LABELS)+' |')
    lines += ['', '## Existing ban4 context (descriptive)', '',
              'These are the original per-response soft-score means, not the strict local violation rates or the question-weighted rank estimands. No cross-task correlation is fitted.', '',
              '| Model | Behavior | Baseline soft score | Ban4 soft score |', '|---|---|---:|---:|']
    for tag in s.MODELS:
        path=s.DATA/'ctrl26_runs'/f'ban4_{tag}'/'grades.csv'
        if s.file_hash(path)!=m['source_hashes'][tag]['grades']:
            raise ValueError('Original grades changed since preparation')
        grades=pd.read_csv(path)
        for ct in TYPES:
            means=grades[grades.ctype==ct].groupby('variant').compliance_score.mean()
            lines.append(f'| {MODELS[tag]} | {LABELS[ct]} | {means["baseline"]:.4f} | {means["ban4"]:.4f} |')
    lines += ['', '## Scope and interpretation', '',
              'Forbidden-mass changes are the primary suppression measure. CSVs also report secondary forbidden-versus-legal log odds, calculated from saved log masses to resolve changes near probability one. Excluded tokens are outside both sides of that odds ratio.', '',
              'First-failure analyses condition on a failure in the saved instructed rollout. Opening analyses include all prompts. Comparisons between these phases are descriptive: they differ in position, history, and selection. Later positions after an earlier failure do not demonstrate restoration of whole-trace compliance.', '',
              'All 18 existing marker words are retained for both marker tasks, with 48 prompts per model per task. Marker/phrasing assignments were not fully crossed with questions; no marker-specific causal claims are made. Candidate sets recognize partial progress toward a multi-token marker and record separator-terminated completion separately. Completion at EOS is not counted as candidate completion because special tokens are excluded.', '',
              'Numbering uses a strict local sentence-opening rule, with ordinal periods excluded from sentence boundaries. The existing aggregate numbering score is retained separately. Disemvoweling preserves the first ASCII letter of each word. Number spelling uses Unicode decimal digits, matching regex \\d.', '',
              'No observed violation applies only to the saved reasoning; it is not a guarantee about a longer continuation. One length-limited Qwen trace ends in an incomplete UTF-8 fragment withheld by vLLM text rendering; exact IDs and the discrepancy are retained in the manifest.', '',
              'Samples are averaged within prompts and prompts within questions before paired, question-clustered bootstrap intervals (10,000 draws). Full-distribution distances are summarized directly, never differenced twice. Rank searches span the full vocabulary; top-200 lists are for inspection only.', '',
              'Figures: first_failure_positions, opening_ranks, opening_probabilities, first_failure_ranks, suppression_probabilities, decoding_accessibility, first_failure_decoding_accessibility (PNG and vector PDF). Exact measurements, completion/excluded/neutral masses, token examples, source grades, and failure locations are in the accompanying CSVs/manifests.']
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    print(summary[(summary.phase=='first_failure')&summary.metric.isin(['content_log10_mass','forbidden_log10_mass'])].to_string(index=False))
    print(f'Analysis: {out}')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='opening_suite_v1')
    analyze(p.parse_args().run)
