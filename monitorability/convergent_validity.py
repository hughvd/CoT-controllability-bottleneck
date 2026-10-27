"""Descriptive cross-model convergent validity; no inference or grading.

Run: python -m monitorability.convergent_validity
"""
from __future__ import annotations
import html
import json
import shutil
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from monitorability.control_type_viewer import build_data, correlation, FAMILIES, ROOT
from controllability.control_groups import LABELS as CONTROL_LABELS

OUT = ROOT / 'results/monitorability/convergent_validity'
FAMILY = {name:FAMILIES[name] for name in ['addition','style','suppression']}
TYPES = sum(FAMILY.values(), [])
LABELS = [CONTROL_LABELS[c] for c in TYPES]
GROUPS = {k.title():v for k,v in FAMILY.items()}
MON = [
    ('mon_gmean_gpqa', 'GPQA'), ('mon_gmean_wmdp_sandbagging', 'WMDP'),
    ('mon_gmean_memory_advice', 'Memory advice'),
    ('mon_gmean_scruples_suggest_wrong', 'Scruples'),
    ('mon_process_tpr_cot_only', 'Process · CoT'),
    ('mon_outcome_gmean_cot_only', 'Outcome · CoT'),
    ('mon_process_tpr_all_messages', 'Process · all'),
    ('mon_outcome_gmean_all_messages', 'Outcome · all'),
]


def matrix(x, y, method):
    return np.array([[correlation(x[a], y[b])[method] for b in y] for a in x], dtype=float)


def plot_heatmap(values, xs, ys, title, grouped=False):
    fig = go.Figure(go.Heatmap(z=values, x=xs, y=ys, zmin=-1, zmax=1,
        colorscale='RdBu', reversescale=True, text=np.round(values, 2),
        texttemplate='%{text:.2f}', hovertemplate='%{y} × %{x}<br>r = %{z:.3f}<extra></extra>'))
    fig.update_layout(title=title, template='plotly_white', height=720,
        margin=dict(l=215, r=45, b=155, t=70), yaxis=dict(autorange='reversed'),
        xaxis=dict(tickangle=-40))
    if grouped:
        for p in [2.5, 5.5]:
            fig.add_hline(y=p, line_color='#333', line_width=1.5)
            fig.add_vline(x=p, line_color='#333', line_width=1.5)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def png_heatmap(values, xs, ys, filename, grouped=False, note=''):
    fig, ax = plt.subplots(figsize=(9, 7.3))
    im = ax.imshow(values, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    ax.set_xticks(range(len(xs)), xs, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(len(ys)), ys, fontsize=10)
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = values[i, j]
            ax.text(j, i, f'{v:.2f}' if np.isfinite(v) else '—', ha='center', va='center',
                    fontsize=9, color='white' if abs(v) > .65 else '#202020')
    if grouped:
        for p in [2.5, 5.5]:
            ax.axhline(p, color='black', lw=1.2); ax.axvline(p, color='black', lw=1.2)
    fig.colorbar(im, ax=ax, label='Spearman correlation across models', shrink=.8)
    fig.tight_layout()
    if note:
        fig.subplots_adjust(bottom=.30)
        fig.text(.04,.025,note,fontsize=9,ha='left',va='bottom',wrap=True)
    fig.savefig(OUT / filename, dpi=220)
    fig.savefig(OUT / filename.replace('.png', '.pdf'))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_data()
    from monitorability.gpqa_figure_snapshot import snapshot_note
    note=snapshot_note(data)
    (OUT/'data.json').write_text(json.dumps(data, indent=2, allow_nan=False))
    mon = pd.DataFrame([{k: m['monitor'].get(k) for k, _ in MON} for m in data['models']])
    sections = []; pairs = []; summaries = []; associations = []; points = []
    family_of = {c: f for f, members in FAMILY.items() for c in members}
    for metric in ['soft', 'hard']:
        ctrl = pd.DataFrame([{c: m['control'][c][metric] for c in TYPES} for m in data['models']])
        groups = pd.DataFrame({g: ctrl[cs].mean(axis=1) for g, cs in GROUPS.items()})
        for i, m in enumerate(data['models']):
            points.append(dict(model=m['model'], metric=metric, **ctrl.iloc[i].to_dict(),
                               **groups.iloc[i].to_dict(), **mon.iloc[i].to_dict()))
        for method in ['spearman', 'pearson']:
            cm = matrix(ctrl, ctrl, method)
            assert np.allclose(cm, cm.T, equal_nan=True)
            assert np.allclose(np.diag(cm)[np.isfinite(np.diag(cm))], 1)
            for a, b in combinations(TYPES, 2):
                loo={m['model']:correlation(ctrl[a].drop(i),ctrl[b].drop(i))[method] for i,m in enumerate(data['models'])}
                finite=[v for v in loo.values() if v is not None]
                pairs.append(dict(metric=metric, method=method, a=a, b=b,
                    family_a=family_of[a], family_b=family_of[b],
                    same_family=family_of[a] == family_of[b], value=cm[TYPES.index(a), TYPES.index(b)],
                    leave_one_out_min=min(finite) if finite else None,leave_one_out_max=max(finite) if finite else None,
                    without_deepseek_v4_pro=loo['deepseek-v4-pro']))
            these = [p for p in pairs if p['metric'] == metric and p['method'] == method]
            for label, subset in [('Within family', [p for p in these if p['same_family']]),
                                  ('Between families', [p for p in these if not p['same_family']]),
                                  ('Suppression vs other', [p for p in these if (p['family_a']=='suppression') != (p['family_b']=='suppression')])]:
                vals = [p['value'] for p in subset if np.isfinite(p['value'])]
                summaries.append(dict(metric=metric, method=method, comparison=label,
                                      pairs=len(vals), undefined_pairs=len(subset)-len(vals),
                                      mean=np.mean(vals) if vals else np.nan, median=np.median(vals) if vals else np.nan))
            for f, members in FAMILY.items():
                raw = [cm[TYPES.index(a), TYPES.index(b)] for a,b in combinations(members,2)]
                vals = [v for v in raw if np.isfinite(v)]
                summaries.append(dict(metric=metric, method=method, comparison=f,
                    pairs=len(vals), undefined_pairs=len(raw)-len(vals),
                    mean=np.mean(vals) if vals else np.nan, median=np.median(vals) if vals else np.nan))
            gm = matrix(groups, mon, method)
            for g in groups:
                for ev, evname in MON:
                    stat = correlation(groups[g], mon[ev])
                    loo = [correlation(groups[g].drop(i), mon[ev].drop(i))[method] for i in ctrl.index]
                    loo = [v for v in loo if v is not None]
                    associations.append(dict(metric=metric, method=method, group=g, evaluation=evname,
                        value=stat[method], n=stat['n'], leave_one_out_min=min(loo) if loo else None,
                        leave_one_out_max=max(loo) if loo else None, monitor_span=stat['y_span'],
                        monitor_floor=int((mon[ev]<=.05).sum()), monitor_ceiling=int((mon[ev]>=.95).sum())))
            # Include constituent types and family means to avoid hiding disagreements in an aggregate.
            extended = pd.concat([ctrl,groups],axis=1)
            mm = matrix(extended, mon, method)
            subtitle = f'{metric.title()} compliance · {method.title()} · 12 models'
            section = plot_heatmap(cm, LABELS, LABELS, 'Control correlations · '+subtitle, True)
            section += plot_heatmap(mm, [n for _,n in MON], LABELS+list(GROUPS), 'Monitorability associations · '+subtitle)
            table = pd.DataFrame([r for r in summaries if r['metric']==metric and r['method']==method]).drop(columns=['metric','method'])
            section += '<h2>Family coherence</h2>'+table.to_html(index=False,float_format=lambda x:f'{x:.3f}')
            at = pd.DataFrame([r for r in associations if r['metric']==metric and r['method']==method]).drop(columns=['metric','method'])
            section += '<h2>Group associations and leave-one-model-out sensitivity</h2>'+at.to_html(index=False,float_format=lambda x:f'{x:.3f}')
            sections.append(f'<section id="{metric}-{method}" '+('' if metric=='soft' and method=='spearman' else 'hidden')+'>'+section+'</section>')
            if metric=='soft' and method=='spearman':
                png_heatmap(cm, LABELS, LABELS, 'control_correlations.png', True)
                png_heatmap(gm, [n for _,n in MON], list(GROUPS), 'group_monitorability.png')
                pd.DataFrame(cm,index=TYPES,columns=TYPES).to_csv(OUT/'control_spearman_soft.csv')
    for name, rows in [('pairs',pairs),('family_summary',summaries),('monitor_associations',associations),('model_scores',points)]:
        pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    pd.DataFrame(pairs).query("metric=='soft' and method=='spearman'")[['a','b','value','leave_one_out_min','leave_one_out_max','without_deepseek_v4_pro']].rename(columns={'value':'spearman'}).to_csv(OUT/'control_pair_sensitivity.csv',index=False)
    intro = '''<h1>Do controllability evaluations measure different capabilities?</h1>
<p>Descriptive convergent / discriminant validity check on the existing 12-model sweep.
Each correlation uses models as observations, not sampled responses. Default: soft compliance and Spearman.</p>
<p><b>Categories:</b> addition (sentence prefixes, numbering, sentence endings), suppression (omit word, no first person),
and style (uppercase, leetspeak, Spanish). Lines mark category boundaries. Chinese and character styles are excluded.</p>
<p>All category means equally weight their constituent controls. The two new addition controls use matched sentence-prefix questions;
the remaining controls retain their historical measurements. Categories adapt the CoT-Control taxonomy and are not assumed to be independent latent capabilities.</p>
<p><b>Interpretation limits:</b> family coherence is descriptive; pairwise correlations share models and controls and are not independent samples.
Leave-one-model-out ranges are sensitivity checks, not confidence intervals. No significance or causal claims.
Models share lineages; historical runs differ in effort/backend. Soft scores mix grading scales; hard means score ≥0.95.
Floor/ceiling counts (≤0.05 / ≥0.95) and monitor spans appear below. Process uses TPR; other evaluations use g-mean.
These data cannot identify opening entropy as the cause or demonstrate concealment/gaming. Use the organism and prefill experiments for those separate claims.</p>'''
    page = '<!doctype html><html><head><meta charset="utf-8"><title>Convergent validity</title><style>body{font:15px system-ui;color:#233047;max-width:1200px;margin:30px auto;padding:0 20px}p{line-height:1.55}table{border-collapse:collapse;font-size:12px;width:100%;margin-bottom:30px}td,th{padding:8px;border-bottom:1px solid #dde3ed;text-align:left}select{padding:9px;margin:8px}section[hidden]{display:none}h2{margin-top:35px}</style><script>'+get_plotlyjs()+'</script></head><body>'+intro
    if note:page+='<p><strong>'+html.escape(note)+'</strong></p>'
    page += '<label>Compliance <select id="metric"><option>soft</option><option>hard</option></select></label><label>Correlation <select id="method"><option>spearman</option><option>pearson</option></select></label>'+''.join(sections)
    page += '''<script>function update(){document.querySelectorAll('section').forEach(s=>{s.hidden=s.id!==document.getElementById('metric').value+'-'+document.getElementById('method').value;if(!s.hidden)s.querySelectorAll('.plotly-graph-div').forEach(p=>Plotly.Plots.resize(p))})}document.querySelectorAll('select').forEach(s=>s.addEventListener('change',update));</script></body></html>'''
    (OUT/'index.html').write_text(page)
    paper = ROOT/'paper/figures'; paper.mkdir(exist_ok=True)
    for stem in ['control_correlations','group_monitorability']:
        for ext in ['png','pdf']:shutil.copy2(OUT/(stem+'.'+ext), paper/('convergent_'+stem+'.'+ext))
    write_findings(summaries,associations,data)
    print(pd.DataFrame(summaries).query("metric=='soft' and method=='spearman'").to_string(index=False))
    print(pd.DataFrame(associations).query("metric=='soft' and method=='spearman'").to_string(index=False))
    print(OUT/'index.html')


def write_findings(summaries,associations,data):
    def table(rows,columns):
        out=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
        for row in rows:
            out.append('| '+' | '.join(f'{row[c]:.3f}' if isinstance(row[c],(float,np.floating)) else str(row[c]) for c in columns)+' |')
        return '\n'.join(out)
    selected=lambda rs:[r for r in rs if r['metric']=='soft' and r['method']=='spearman']
    text='# Eight-control convergent-validity results\n\n'
    text+='Twelve models; instruction-only soft compliance. Addition: sentence prefixes, numbering, sentence endings. Suppression: word omission, no first person. Style: uppercase, leetspeak, Spanish. Category scores equally weight their constituent tasks. Chinese and persona styles are excluded.\n\n'
    text+='Numbering and sentence endings use fresh runs on the same 30 target questions as sentence prefixes (five samples each); six retained tasks reuse historical results. Source hashes and sample counts are in `data.json`.\n\n'
    from monitorability.gpqa_figure_snapshot import snapshot_note
    if snapshot_note(data):text+=snapshot_note(data)+'\n\n'
    quality=[r for r in data['counts'] if r['control'] in ['numbered_sentences','suffix_sentences']]
    text+='## New-task response coverage\n\n'+table(quality,['model','control','valid_samples','empty_reasoning'])+'\n\n'
    text+='Empty reasoning receives zero compliance, matching the historical correlation protocol. Counts distinguish this behavior from nonempty reasoning that fails the constraint.\n\n'
    text+='## Control–control associations\n\n'+table(selected(summaries),['comparison','pairs','undefined_pairs','mean','median'])+'\n\n'
    text+='## Category–monitorability associations\n\n'+table(selected(associations),['group','evaluation','value','n','leave_one_out_min','leave_one_out_max','monitor_floor','monitor_ceiling'])+'\n\n'
    text+='Correlations use models as observations and are descriptive. Leave-one-out ranges measure sensitivity, not confidence intervals. Undefined correlations from constant scores are excluded from pair summaries and counted explicitly. Floor/ceiling counts use ≤0.05/≥0.95. Process uses TPR; other evaluations use g-mean. Correlation cannot establish concealment, gaming, or an initiation mechanism.\n'
    (OUT/'results.md').write_text(text)


if __name__=='__main__': main()
