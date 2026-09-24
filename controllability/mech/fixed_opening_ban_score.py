"""Score matched first-token-ban runs and summarize paired question-level effects."""
import argparse,asyncio,json,re,sqlite3,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import fixed_opening_ban as f
sys.path.insert(0,str(f.CTRL))
import token_prefill_suite as suite
import token_prefill_suite_api as api
from token_prefill_suite_score import parse_judgment
from prefill_experiment import extract_answer

def split_reasoning(text):
 if text.startswith('<think>'):text=text[len('<think>'):]
 if '</think>' in text:return text.split('</think>',1)
 return text,''

async def score(name):
 from sft import language_grade as lg
 root=f.ROOT/name;m=json.loads((root/'manifest.json').read_text());f.validate(m)
 requests={r['request_id']:r for r in m['requests']};cells={c['cell_id']:c for c in m['cells']}
 raws=[]
 for p in sorted((root/'remote').glob('batch_*.json')):raws.extend(json.loads(p.read_text())['outputs'])
 if (root/'api_outputs.sqlite').exists():
  db=sqlite3.connect(root/'api_outputs.sqlite');raws.extend(json.loads(x[0]) for x in db.execute('select payload from outputs'));db.close()
 assert len({r['request_id'] for r in raws})==len(raws)
 api.s.ROOT=f.ROOT/'grading';client=api.PaidClient(cap=8.,retries=8);client.sem=asyncio.Semaphore(16)
 scoredb=sqlite3.connect(root/'scores.sqlite');scoredb.execute('create table if not exists scores (id text primary key,payload text)');scoredb.commit()
 async def one(raw):
  rid=raw['request_id']
  if scoredb.execute('select 1 from scores where id=?',(rid,)).fetchone():return
  r=requests[rid];c=cells[r['cell_id']]
  full,answer=split_reasoning(raw['raw_text']);answer=raw.get('answer_text') or answer
  pre=raw['first_token_text'];stable=split_reasoning(pre)[0].rstrip('\ufffd')
  assert full.startswith(stable),(rid,repr(pre),repr(full[:40]))
  suffix=full[len(stable):]
  row=dict(model=name,request_id=rid,question_id=c['question_id'],cell_id=c['cell_id'],behavior=c['behavior'],
   condition=r['condition'],arm=r['arm'],sample=r['sample'],first_token_text=pre,
   reasoning_chars=len(full),empty_reasoning=not full.strip(),truncated=raw['finish_reason']=='length',
   answer_correct=int(extract_answer(answer)==c['correct_letter']),first_stage_ids=json.dumps(raw.get('first_stage_ids')))
  for label,text,seed in [('whole',full,''),('continuation',suffix,stable)]:
   if not text.strip():value=0.
   elif c['behavior'] in ['spanish','chinese']:
    display=lg.load_language_defs()[c['behavior']]['display_name']
    prompt=lg._GRADER_PROMPT.format(display_name=display,text=text)
    key=f.base.digest([lg.DEFAULT_MODEL,f.base.file_hash(lg.__file__),prompt]);value=None
    for attempt in range(3):
     payload=dict(model=lg.DEFAULT_MODEL,messages=[{'role':'user','content':prompt}],temperature=0.,max_tokens=600,response_format={'type':'json_object'})
     resp=await client.call('fixed-ban-grade|'+key+'|'+str(attempt),payload,.1e-6,.4e-6)
     ch=resp['choices'][0];value,_,repair=parse_judgment(lg,ch['message'].get('content') or '',ch.get('finish_reason'))
     if value is not None:break
    assert value is not None and 0<=value<=1
   else:
    value,units=suite.deterministic_score(c,seed,text)
    row[label+'_eligible_units']=units
   row[label+'_soft']=value
  with scoredb:scoredb.execute('insert into scores values (?,?)',(rid,json.dumps(row,ensure_ascii=False)))
 queue=iter(raws);errors=[]
 async def worker():
  while not errors:
   try:r=next(queue)
   except StopIteration:return
   try:await one(r)
   except Exception as e:errors.append(e);return
 await asyncio.gather(*(worker() for _ in range(16)))
 rows=[json.loads(x[0]) for x in scoredb.execute('select payload from scores')]
 pd.DataFrame(rows).to_csv(root/'scores.csv',index=False)
 report=dict(model=name,generated=len(raws),scored=len(rows),expected=len(m['requests']),tracked_grading_cost=client.spent(),status='complete' if len(rows)==len(m['requests']) else 'partial')
 if name=='glm' and m['version']=='fixed_opening_ban_v1':
  report['primary_scored']=sum(r['condition']=='instruction_only' for r in rows)
  report['primary_expected']=720
  if report['primary_scored']==720 and len(rows)<len(m['requests']):report['status']='primary_complete_uninstructed_incomplete'
 f.base.atomic_json(root/'scoring_report.json',report);print(report,flush=True)
 await client.client.close()
 if errors:raise errors[0]

def analyze():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=f.HERE/'analysis/fixed_opening_ban/v1';out.mkdir(parents=True,exist_ok=True)
 dfs=[]
 for n in f.MODELS:
  p=f.ROOT/n/'scores.csv'
  if not p.exists():continue
  model_df=pd.read_csv(p)
  # Incomplete rejection runs can select prompts by acceptance probability.
  # Do not treat their completed subset as an unbiased experiment.
  if len(model_df)==1440:dfs.append(model_df)
  elif n=='glm':
   instructed=model_df[model_df.condition=='instruction_only']
   if len(instructed)==720:dfs.append(instructed)
 if not dfs:return
 df=pd.concat(dfs,ignore_index=True);stats=[];quality=[];rng=np.random.default_rng(20260918)
 for keys,g in df.groupby(['model','behavior','condition']):
  model,behavior,condition=keys
  for metric in ['whole_soft','continuation_soft','answer_correct']:
   counts=g.groupby(['question_id','arm']).size().unstack('arm')
   if not {'baseline','fixed_ban'}.issubset(counts.columns):continue
   complete=counts.index[(counts.baseline==f.SAMPLES)&(counts.fixed_ban==f.SAMPLES)]
   if not len(complete):continue
   q=g[g.question_id.isin(complete)].groupby(['question_id','arm'])[metric].mean().unstack('arm').dropna()
   if not {'baseline','fixed_ban'}.issubset(q.columns) or not len(q):continue
   d=(q.fixed_ban-q.baseline).to_numpy();boots=d[rng.integers(0,len(d),(5000,len(d)))].mean(1)
   stats.append(dict(model=model,behavior=behavior,condition=condition,metric=metric,questions=len(q),
    baseline=q.baseline.mean(),fixed_ban=q.fixed_ban.mean(),delta=d.mean(),low=np.quantile(boots,.025),high=np.quantile(boots,.975)))
  for arm,h in g.groupby('arm'):
   quality.append(dict(model=model,behavior=behavior,condition=condition,arm=arm,n=len(h),
    accuracy=h.answer_correct.mean(),empty=h.empty_reasoning.mean(),truncated=h.truncated.mean(),mean_reasoning_chars=h.reasoning_chars.mean()))
 pd.DataFrame(stats).to_csv(out/'paired_results.csv',index=False);pd.DataFrame(quality).to_csv(out/'quality.csv',index=False)
 df.groupby(['model','behavior','condition','arm','first_token_text']).size().rename('count').reset_index().to_csv(out/'first_tokens.csv',index=False)
 if not stats:return
 st=pd.DataFrame(stats);models=list(st.model.unique());fig,axes=plt.subplots(1,len(models),figsize=(5*len(models),4),squeeze=False,sharey=True)
 for ax,model in zip(axes.flat,models):
  for cond,color,offset in [('instruction_only','#2266ad',-.13),('no_instruction','#888888',.13)]:
   ss=st[(st.model==model)&(st.condition==cond)&(st.metric=='continuation_soft')].set_index('behavior')
   if ss.empty:continue
   ss=ss.reindex(f.BEHAVIORS);x=np.arange(len(ss))+offset
   ax.errorbar(x,ss.delta,yerr=[ss.delta-ss.low,ss.high-ss.delta],fmt='o',capsize=3,color=color,label=cond)
  ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(4),['Uppercase','Leetspeak','Spanish','Omit word'],rotation=25);ax.set_title(model);ax.grid(alpha=.2)
 axes[0,0].set_ylabel('Change in continuation compliance');axes[0,-1].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'compliance_change.png',dpi=180);fig.savefig(out/'compliance_change.pdf')
 lines=['# Fixed opening-token ban results','',
  'Main-paper figure: [compliance and task accuracy](fixed_opening_ban_average.pdf). Appendix: [per-control effects](fixed_opening_ban_by_control.pdf). Regenerate the paper figures with `python3 controllability/mech/analysis/plot_fixed_opening_ban_average.py`.', '',
  'Fresh matched baselines; temperature 0.6, top-p 1.0; 30 questions and three samples per arm per control. Effects below are changes in soft compliance, not percentages of perfectly compliant traces.',
  '', '## Instruction-present results', '',
  '| Model | Control | Whole baseline → ban | Continuation baseline → ban | Continuation change, pp (95% CI) |',
  '|---|---|---:|---:|---:|']
 for (model,behavior),g in st[st.condition=='instruction_only'].groupby(['model','behavior'],sort=False):
  w=g[g.metric=='whole_soft'].iloc[0];c=g[g.metric=='continuation_soft'].iloc[0]
  lines.append(f'| {model} | {behavior} | {w.baseline:.3f} → {w.fixed_ban:.3f} | {c.baseline:.3f} → {c.fixed_ban:.3f} | {100*c.delta:+.1f} ({100*c.low:+.1f}, {100*c.high:+.1f}) |')
 lines+=['','## Instruction-by-intervention interaction','',
  'Instructed ban effect minus uninstructed ban effect, in continuation-compliance percentage points. Confidence intervals resample paired questions. These intervals are descriptive and are not adjusted for multiple comparisons.','',
  '| Model | Control | Interaction, pp (95% CI) |','|---|---|---:|']
 interactions=[]
 for (model,behavior),g in df.groupby(['model','behavior']):
  q=g.groupby(['question_id','condition','arm']).continuation_soft.mean().unstack(['condition','arm'])
  expected=[(c,a) for c in f.CONDITIONS for a in f.ARMS]
  if not all(x in q.columns for x in expected):continue
  q=q.dropna();d=(q[('instruction_only','fixed_ban')]-q[('instruction_only','baseline')]-q[('no_instruction','fixed_ban')]+q[('no_instruction','baseline')]).to_numpy()
  bs=d[rng.integers(0,len(d),(5000,len(d)))].mean(1);lo,hi=np.quantile(bs,[.025,.975])
  interactions.append(dict(model=model,behavior=behavior,delta=d.mean(),low=lo,high=hi,questions=len(d)))
  lines.append(f'| {model} | {behavior} | {100*d.mean():+.1f} ({100*lo:+.1f}, {100*hi:+.1f}) |')
 pd.DataFrame(interactions).to_csv(out/'instruction_interactions.csv',index=False)
 lines+=['','## Quality and scope','',
  '| Model | Condition | Arm | Responses | Accuracy | Empty reasoning | Token-cap terminations |',
  '|---|---|---|---:|---:|---:|---:|']
 for (model,cond,arm),g in df.groupby(['model','condition','arm']):
  lines.append(f'| {model} | {cond} | {arm} | {len(g)} | {g.answer_correct.mean():.3f} | {g.empty_reasoning.mean():.3f} | {g.truncated.mean():.3f} |')
 lines+=['','GLM uses full-response rejection sampling, conditional on the first locally retokenized reasoning token not being `The`. Its uninstructed condition was not completed and is excluded. The API does not return native reasoning token IDs; the local-model bans use native IDs directly. See [README.md](README.md) for execution details and limitations.',
  '', 'Incomplete model comparisons are excluded from this report. Raw outputs, proposal histories, audit reports, and cached scores are under `controllability/results/fixed_opening_ban/v1/`.', '']
 (out/'results.md').write_text('\n'.join(lines))
 print(st.round(4).to_string(index=False))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['score','analyze']);ap.add_argument('--model',choices=list(f.MODELS));ap.add_argument('--experiment-root');a=ap.parse_args()
 if a.experiment_root:
  assert a.phase=='score','Custom roots require their own analysis'
  f.ROOT=Path(a.experiment_root).resolve()
 if a.phase=='score':asyncio.run(score(a.model))
 else:analyze()
