"""Audit pilot completeness and compare correlations on identical native model sets."""
from pathlib import Path
import hashlib
import html
import json
import pandas as pd
from icl_ban_correlation_pilot import ROOT, TYPES, Tokenizer

GROUPS = {'Addition':TYPES[:3], 'Style':TYPES[3:6], 'Suppression':TYPES[6:]}
EVALS = {'GPQA':'mon_gmean_gpqa','WMDP':'mon_gmean_wmdp_sandbagging',
         'Memory advice':'mon_gmean_memory_advice','Scruples':'mon_gmean_scruples_suggest_wrong',
         'Flaky Tools':'mon_outcome_gmean_cot_only','GPQA Physics':'mon_process_tpr_cot_only'}


def main():
    m=json.loads((ROOT/'manifest.json').read_text())
    availability=[]
    for name,spec in m['models'].items():
        counts={'complete':0,'cap_exhausted':0,'error':0,'time_limit':0,'running':0,'not_started':0}
        total=0
        for job in m['jobs']:
            p=ROOT/name/(job['behavior']+'__'+job['condition']+'.json')
            if not p.exists():counts['not_started']+=1;continue
            r=json.loads(p.read_text());counts[r['status']]+=1;total+=len(r['proposals'])
            if r['baseline'] is not None:assert r['baseline']==r['proposals'][0]
            eligible=[x for x in r['proposals'] if x['first_token_id'] is not None and x['first_token_id'] not in spec['ban_ids']]
            if r['banned'] is not None:assert eligible and r['banned']==eligible[0]
        availability.append(dict(model=name,banned_tokens=', '.join(spec['ban_text']),
            historical_banned_share=spec['calibration_banned_share'],proposals=total,**counts))
    availability=pd.DataFrame(availability);availability.to_csv(ROOT/'api_feasibility.csv',index=False)
    scores=pd.read_csv(ROOT/'scores.csv')
    assert not scores.duplicated(['backend','model','behavior','condition','arm']).any()
    native=scores[scores.backend=='native'].copy()
    complete=[]
    for name,d in native.groupby('model'):
        if len(d)==32 and d.groupby(['condition','arm']).size().eq(8).all():complete.append(name)
    native=native[native.model.isin(complete)]
    native['condition_label']=native.condition.map({'instruction_only':'Instruction','user_icl':'ICL'})+native.arm.map({'baseline':'','banned':' + ban'})
    full=native.groupby(['model','condition_label']).compliance.mean().unstack()
    full.to_csv(ROOT/'native_means.csv')
    native['category']=native.behavior.map({t:g for g,ts in GROUPS.items() for t in ts})
    grouped=native.groupby(['model','category','condition_label']).compliance.mean().unstack()
    grouped.to_csv(ROOT/'native_category_means.csv')
    task_scores=native.pivot(index=['model','condition_label'],columns='behavior',values='compliance')[TYPES]
    task_scores.to_csv(ROOT/'native_task_scores.csv')
    starts=[]
    for name in complete:
        directory=ROOT/('native_gptoss' if name=='gpt-oss-120b' else 'native_'+name)
        output=json.loads((directory/'outputs.json').read_text())
        tok=Tokenizer(m['models'][name]['tokenizer'])
        for row in output:
            first_ids=row['first_stage_ids']
            if row['arm']=='banned':assert not set(first_ids)&set(m['models'][name]['ban_ids'])
            starts.append(dict(model=name,behavior=row['behavior'],condition=row['condition'],arm=row['arm'],
                native_first_stage_ids=json.dumps(first_ids),native_first_stage_text=tok.decode(first_ids),
                reasoning_opening=row['reasoning'][:100]))
    starts=pd.DataFrame(starts)
    starts.to_csv(ROOT/'native_openings.csv',index=False)
    opening_rows=[]
    for (name,condition,arm),d in starts.groupby(['model','condition','arm']):
        counts=d.native_first_stage_text.value_counts()
        opening_rows.append(dict(model=name,condition=condition,arm=arm,
            openings='; '.join(f'{text!r}: {n}' for text,n in counts.items())))
    opening_summary=pd.DataFrame(opening_rows)
    rows=[]
    if len(complete)>=4:
        monitor_source=Path(__file__).resolve().parent.parent/'results/monitorability/convergent_validity/data.json'
        monitor_snapshot=ROOT/'monitor_scores_snapshot.json'
        if not monitor_snapshot.exists():
            monitor_snapshot.write_bytes(monitor_source.read_bytes())
            (ROOT/'monitor_scores_provenance.json').write_text(json.dumps(dict(
                source=str(monitor_source),sha256=hashlib.sha256(monitor_snapshot.read_bytes()).hexdigest()),indent=2))
        data=json.loads(monitor_snapshot.read_text());lookup={r['model']:r['monitor'] for r in data['models']}
        for group,types in dict(GROUPS,**{t:[t] for t in TYPES}).items():
            d=native[native.behavior.isin(types)].groupby(['model','condition_label']).compliance.mean().unstack().loc[complete]
            for ev,key in EVALS.items():
                y=pd.Series({n:lookup[n][key] for n in complete})
                for cond in d.columns:
                    rho=d[cond].corr(y,method='spearman') if d[cond].nunique()>1 and y.nunique()>1 else float('nan')
                    rows.append(dict(control=group,evaluation=ev,condition=cond,n=len(complete),spearman=rho))
    corr=pd.DataFrame(rows,columns=['control','evaluation','condition','n','spearman'])
    corr.to_csv(ROOT/'native_correlations.csv',index=False)
    coverage=native.groupby('model').agg(outputs=('compliance','size'),empty=('empty','sum'),truncated=('truncated','sum'),
                                       answer_accuracy=('answer_correct','mean'))
    coverage.to_csv(ROOT/'native_quality.csv')
    capabilities=json.loads((ROOT/'capabilities.json').read_text())
    for row in capabilities:
        if row['model'] in complete:row['native_status']='complete'
    (ROOT/'capabilities.json').write_text(json.dumps(capabilities,indent=2))
    capability_table=pd.DataFrame([dict(model=r['model'],native_status=r['native_status'],
        opening_ban=', '.join(r['opening_ban'])) for r in capabilities])
    tables=[('<h2>Exact native intervention: mean compliance</h2>',full.to_html(float_format=lambda x:f'{x:.3f}')),
            ('<h2>Compliance by control category</h2>',grouped.to_html(float_format=lambda x:f'{x:.3f}')),
            ('<h2>Compliance by control task</h2>',task_scores.to_html(float_format=lambda x:f'{x:.3f}')),
            ('<h2>Native response quality</h2>',coverage.to_html()),
            ('<h2>Observed native openings (counts across eight tasks)</h2>',opening_summary.to_html(index=False)),
            ('<h2>Coverage of the original 12-model panel</h2>',capability_table.to_html(index=False)),
            ('<h2>API rejection feasibility (16 prompt conditions per model)</h2>',availability.to_html(index=False,float_format=lambda x:f'{x:.3f}'))]
    if len(corr):
        table=corr[corr.control.isin(GROUPS)].pivot(index=['control','evaluation'],columns='condition',values='spearman')
        tables.insert(1,('<h2>Descriptive Spearman correlations on the same '+str(len(complete))+' models</h2>',table.to_html(float_format=lambda x:f'{x:.2f}')))
        task_table=corr[corr.control.isin(TYPES)].pivot(index=['control','evaluation'],columns='condition',values='spearman')
        tables.append(('<h2>Per-control correlations</h2>',task_table.to_html(float_format=lambda x:f'{x:.2f}')))
    content='''<!doctype html><meta charset="utf-8"><title>ICL + opening ban pilot</title>
<style>body{font:16px system-ui;max-width:1200px;margin:36px auto;padding:0 24px;color:#20252b}table{border-collapse:collapse;margin:20px 0;font-size:14px}th,td{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}h2{margin-top:36px}.note{background:#eef4f7;padding:18px;line-height:1.6}</style>
<h1>ICL + opening-token ban: feasibility pilot</h1>
<p class="note">One fixed question per control task, eight controls, four conditions (32 responses per native model).
All arms use the same target, demonstration control specification, temperature 0.6, top-p 1, and 8,192-token cap.
Score the whole generated reasoning, including its opening; empty reasoning receives zero.
Qwen3-32B responses are reused after exact prompt verification; other native results are new.
This is too small to establish a change in correlation. The three addition tasks share one question.
Native prompts open an empty reasoning channel in every arm; served checkpoints/settings may differ from the historical API runs.
Channel-closing tokens remain allowed. Gemma's four empty ICL + ban traces move reasoning into the final response; these are retained at zero CoT compliance.
The intervention does not establish that the initiation bottleneck has been removed: alternative ordinary openings and immediate channel closure can still occur.
The mask excludes fixed token IDs; unmasked lexical variants such as Qwen3-32B's leading-space ' Alright' remain possible.
Correlations compare instruction-only and intervention scores on exactly the same completed native model subset, using frozen monitorability scores.
API rejection uses locally retokenized visible reasoning, not native token IDs returned by the server; it also requires nonempty visible reasoning for acceptance.
The API feasibility pass was stopped after about 20 minutes; unfinished attempts and unstarted conditions remain explicitly marked.
Cap-exhausted conditions are missing outcomes, not failures of controllability. No API correlations are computed from the selected accepted subset.</p>'''
    for i,(heading,table) in enumerate(tables):
        if i < 2:
            content+=heading+table
        else:
            title=heading.removeprefix('<h2>').removesuffix('</h2>')
            content+='<details><summary>'+title+'</summary><div style="overflow-x:auto">'+table+'</div></details>'
    (ROOT/'report.html').write_text(content)
    print('Complete native models:',complete)
    print(full.to_string(float_format=lambda x:f'{x:.3f}'))
    print('API status counts:',availability[['complete','cap_exhausted','error','time_limit','running','not_started']].sum().to_dict())


if __name__=='__main__':main()
