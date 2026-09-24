"""Final consistency/accounting audit; refuses incomplete active-model outputs."""
import json
import sqlite3
from collections import Counter
from pathlib import Path
import pandas as pd
import token_prefill_suite as s


def main():
 scope=json.loads((s.ROOT/'scope.json').read_text());models=scope['active_models'];reports={};questions=None
 for name in models:
  root=s.ROOT/name;m=json.loads((root/'manifest.json').read_text());s.validate(m)
  assert s.base.file_hash(s.__file__)==m['code_sha256']
  for file,h in m['source_hashes'].items():assert s.base.file_hash(s.HERE/file)==h,file
  current={(c['behavior'],c['question_id'],c['instruction'],c['controlled_text']) for c in m['cells']}
  if questions is None:questions=current
  else:assert questions==current,'Unmatched source cells across models'
  d=pd.read_csv(root/'scores.csv');assert len(d)==24000
  assert d.manifest_sha256.eq(m['manifest_sha256']).all()
  assert not d.duplicated(['cell_id','condition','prefix_tokens','sample']).any()
  counts=d.groupby(['behavior','condition','prefix_tokens']).agg(n=('sample','size'),questions=('question_id','nunique'))
  assert len(counts)==160 and counts.n.eq(150).all() and counts.questions.eq(30).all()
  assert len(d[['request_id','sample']].drop_duplicates())==len(m['requests'])*5
  for label in ['head32','full']:
   x=d[label+'_soft'];assert x.dropna().between(0,1).all()
   missing=d[x.isna()];assert missing.behavior.isin(['prefix_sentences','omit_word','no_first_person']).all()
   assert missing[label+'_eligible_units'].eq(0).all()
  report={'unique_outputs':len(m['requests'])*5,'observations':len(d),'balanced_cells':len(counts),
   'source_replacements':m['source_replacements'],'scores_sha256':s.base.file_hash(root/'scores.csv'),
   'empty_rate':float(d.empty_reasoning.mean()),'truncation_rate':float(d.truncated.mean()),
   'short_window_rate':float((~d.has_32_tokens).mean()),'unassessable_immediate':int(d.head32_soft.isna().sum()),
   'unassessable_sustained':int(d.full_soft.isna().sum()),'scoring_pipeline_sha256':sorted(d.scoring_pipeline_sha256.unique())}
  if name=='glm':
   db=sqlite3.connect(root/'api_generations.sqlite');provider=Counter();parsing=Counter();seen=set()
   for rid,sample,payload in db.execute('select request_id,sample,payload from outputs'):
    row=json.loads(payload);assert row['manifest_sha256']==m['manifest_sha256'];assert (rid,sample) not in seen;seen.add((rid,sample))
    provider[row['provider']]+=1;parsing[row.get('channel_parse','provider_reasoning_field')]+=1
   assert set(provider)=={'DeepInfra'} and len(seen)==len(m['requests'])*5
   report.update(providers=dict(provider),channel_parsing=dict(parsing),server_token_ids_available=False)
  else:
   known={r['request_id']:r for r in m['requests']};seen=set()
   for path in (root/'remote').glob('batch_*.json'):
    b=json.loads(path.read_text());assert b['manifest_sha256']==m['manifest_sha256']
    for row in b['outputs']:
     key=(row['request_id'],row['sample']);assert key not in seen;seen.add(key)
     assert row['prompt_token_ids']==known[row['request_id']]['prompt_token_ids']
   assert len(seen)==len(known)*5;report['exact_prompt_ids_verified']=True
  reports[name]=report
 old=pd.read_csv(s.base.DEFAULT_ROOT/'scores.csv');new=pd.read_csv(s.ROOT/'qwen32/scores.csv')
 matched=old.merge(new,on=['request_id','sample','behavior'],suffixes=('_old','_new'));assert len(matched)==4800
 for col in ['head32_soft','full_soft']:assert (matched[col+'_old']-matched[col+'_new']).abs().max()<1e-12
 ledger=json.loads((s.ROOT/'gpu_ledger.json').read_text());assert all(a['status']=='complete' for a in ledger['attempts'])
 assert set(a['model'] for a in ledger['attempts'])=={'qwen32','qwen14'}
 gpu_estimate=sum(a['accounted_seconds']/3600*ledger['usd_per_hour'] for a in ledger['attempts'])
 paid=sqlite3.connect(s.ROOT/'paid.sqlite');states={state:{'calls':n,'usd':usd} for state,n,usd in paid.execute('select state,count(*),sum(cost) from attempts group by state')}
 assert 'reserved' not in states
 assert not paid.execute("select 1 from attempts where job like 'generate|deepseek|%' limit 1").fetchone()
 probe_cost=0.
 for name in ['glm','deepseek']:
  for path in (s.ROOT/name/'api_probe').glob('*.json'):
   x=json.loads(path.read_text());raw=x.get('raw',x);probe_cost+=float((raw.get('usage') or {}).get('cost') or 0.)
 result=dict(status='complete',active_models=models,deferred_models=scope['deferred_models'],models=reports,
  observations=sum(x['observations'] for x in reports.values()),unique_outputs=sum(x['unique_outputs'] for x in reports.values()),
  reused_qwen32_outputs_verified=4800,gpu_estimated_usd=gpu_estimate,api_accounting_by_state=states,
  api_validation_probe_usd=probe_cost,api_reported_completed_call_usd=states.get('complete',{}).get('usd',0)+probe_cost,
  api_conservative_possible_extra_usd=sum(v['usd'] for k,v in states.items() if k!='complete'))
 s.base.atomic_json(s.ROOT/'completion_validation.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
