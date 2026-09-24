"""Resumable immediate/sustained scoring for all controls; no new generations."""
import argparse
import asyncio
import json
from pathlib import Path
import sqlite3
import time
import re

import token_prefill_suite as s


def parse_judgment(module,content,finish_reason=None):
 score,reason=module._parse_score(content)
 if score is not None:return score,reason,False
 # Some judge replies unnecessarily escape apostrophes inside double-quoted
 # JSON strings. Remove only the unmatched escape; keep the original response.
 fixed=re.sub(r"(\\+)'",lambda m:'\\'*(len(m[1])-(len(m[1])%2))+"'",content)
 if fixed!=content:
  score,reason=module._parse_score(fixed)
  if score is not None:return score,reason,'invalid_apostrophe_escape_in_json'
 # A length-capped explanation can leave JSON unfinished after a complete score.
 # Recover only a complete, bounded numeric first field followed by the reason
 # string; never infer a score from prose or a partially emitted number.
 if finish_reason=='length':
  match=re.match(r'^\s*\{\s*"score"\s*:\s*(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,\s*"reason"\s*:\s*"',content)
  if match:
   value=float(match[1])
   if 0<=value<=1:return value,'Explanation truncated by grader output cap.','complete_score_before_truncated_reason'
 return None,reason,False


def immediate_score(cell,seed,head,full):
 """Do not manufacture a word boundary by cutting a native-token window."""
 original=head
 if cell['behavior'] in ['omit_word','no_first_person','prefix_sentences'] and len(head)<len(full) and head:
  pattern=r"[A-Za-z']" if cell['behavior']=='no_first_person' else r'\w'
  if re.fullmatch(pattern,head[-1]) and re.fullmatch(pattern,full[len(head)]):
   while head and re.fullmatch(pattern,head[-1]):head=head[:-1]
 if original.strip() and not head.strip():return None,0,len(original)-len(head)
 score,units=s.deterministic_score(cell,seed,head)
 return score,units,len(original)-len(head)


def observation_db(name):
 db=sqlite3.connect(s.ROOT/name/'observations.sqlite',timeout=30);db.execute('pragma journal_mode=WAL')
 db.execute('create table if not exists observations (key text primary key,payload text)');db.commit();return db


def raw_outputs(name):
 root=s.ROOT/name
 if (root/'api_generations.sqlite').exists():
  db=sqlite3.connect(root/'api_generations.sqlite')
  for row in db.execute('select payload from outputs'):yield json.loads(row[0])
  db.close()
 else:
  for path in sorted((root/'remote').glob('batch_*.json')):
   batch=json.loads(path.read_text())
   for row in batch['outputs']:yield row


def measure(name):
 root=s.ROOT/name;m=json.loads((root/'manifest.json').read_text());s.validate(m)
 assert s.base.file_hash(s.__file__)==m['code_sha256'],'Pinned suite scorer changed'
 tok=s.tokenizer(m['model']);db=observation_db(name)
 requests={r['request_id']:r for r in m['requests']};cells={c['cell_id']:c for c in m['cells']};links={}
 for x in m['links']:links.setdefault(x['request_id'],[]).append(x)
 seen=set();n=0
 for raw in raw_outputs(name):
  rid=raw['request_id'];sample=raw['sample'];assert (rid,sample) not in seen;seen.add((rid,sample));assert sample in range(5)
  request=requests[rid]
  if raw.get('prompt_token_ids') is not None:assert raw['prompt_token_ids']==request['prompt_token_ids']
  full,head,length,closed=s.reasoning_parts(tok,request,raw,m)
  seed=s.base.decode(tok,request['prefix_token_ids']);partial=head.endswith('\ufffd');head=head.rstrip('\ufffd')
  for link in links[rid]:
   key=s.base.digest([name,link,sample]);cell=cells[link['cell_id']]
   existing=db.execute('select payload from observations where key=?',(key,)).fetchone()
   if existing:
    # Update only deterministic immediate-window scoring after the cutoff
    # audit. Preserve previously paid judgments and exact generation data.
    if cell['behavior'] in ['omit_word','no_first_person','prefix_sentences']:
     row=json.loads(existing[0]);value,units,removed=immediate_score(cell,seed,head,full)
     row.update(head32_soft=value,head32_eligible_units=units,head32_trailing_partial_word_chars=removed)
     with db:db.execute('update observations set payload=? where key=?',(json.dumps(row,ensure_ascii=False),key))
    continue
   row=dict(key=key,model=name,**link,behavior=cell['behavior'],question_id=cell['question_id'],sample=sample,
     manifest_sha256=m['manifest_sha256'],reasoning_tokens=length,has_32_tokens=length>=32,
     empty_reasoning=not full.strip(),closed_reasoning=closed,truncated=raw['finish_reason']=='length',
     head32_incomplete_utf8=partial,seed_incomplete_utf8=seed.endswith('\ufffd'),head32_text=head,full_text=full,
     needs_judge=cell['modality'] in ['language','style'])
   for label,text in [('head32',head),('full',full)]:
    if not text.strip():score=0.;eligible=0
    elif row['needs_judge']:score=None;eligible=1
    elif label=='head32':
     score,eligible,removed=immediate_score(cell,seed,head,full);row['head32_trailing_partial_word_chars']=removed
    else:score,eligible=s.deterministic_score(cell,seed,text)
    row[label+'_soft']=score;row[label+'_eligible_units']=eligible
   with db:db.execute('insert into observations values (?,?)',(key,json.dumps(row,ensure_ascii=False)))
   n+=1
 count=db.execute('select count(*) from observations').fetchone()[0]
 s.base.atomic_json(root/'measurement_report.json',dict(observations=count,unique_outputs=len(seen),expected_observations=24000,
    status='complete' if count==24000 else 'partial',suite_scorer_sha256=s.base.file_hash(s.__file__),
    scoring_pipeline_sha256=s.base.file_hash(__file__)))
 print(name,'measured',n,'new observations;',count,'total',flush=True)


async def judge(name):
 from token_prefill_suite_api import PaidClient
 from sft import language_grade as lg,style_grade as sg
 scope=json.loads((s.ROOT/'scope.json').read_text());assert name in scope['active_models']
 client=PaidClient(scope['api_estimate_cap_usd']);db=observation_db(name)
 client.db.execute('create table if not exists judgments (job text primary key, score real, grader_hash text, at real)');client.db.commit()
 client.db.execute('create table if not exists judgment_repairs (job text primary key, kind text)');client.db.commit()
 pending={};references={}
 for key,payload in db.execute('select key,payload from observations'):
  row=json.loads(payload)
  if not row['needs_judge']:continue
  module=lg if row['behavior'] in ['spanish','chinese'] else sg
  if module==lg:
   d=lg.load_language_defs()[row['behavior']]
   prompt_args={'display_name':d.get('display_name') or row['behavior']}
  else:
   d=sg._load_style_defs()[row['behavior']]
   prompt_args={'display_name':d.get('display_name') or row['behavior'],'style_instruction':d.get('style_instruction','')}
  gh=s.base.file_hash(module.__file__)
  for label in ['head32','full']:
   if row[label+'_soft'] is not None:continue
   prompt=module._GRADER_PROMPT.format(text=row[label+'_text'],**prompt_args)
   job=s.base.digest([module.DEFAULT_MODEL,gh,prompt])
   pending.setdefault(job,(module,prompt,gh));references.setdefault(job,[]).append((key,label))
 async def one(job):
  module,prompt,gh=pending[job]
  cached=client.db.execute('select score from judgments where job=?',(job,)).fetchone()
  if cached:score=cached[0]
  else:
   for attempt in range(3):
    payload=dict(model=module.DEFAULT_MODEL,messages=[{'role':'user','content':prompt}],temperature=0.,max_tokens=600,response_format={'type':'json_object'})
    raw=await client.call('judge|'+job+'|'+str(attempt),payload,.1e-6,.4e-6)
    score,_,repaired=parse_judgment(module,raw['choices'][0]['message'].get('content') or '',raw['choices'][0].get('finish_reason'))
    if repaired:
     with client.db:client.db.execute('insert or replace into judgment_repairs values (?,?)',(job,repaired))
    if score is not None and 0<=score<=1:break
   else:raise ValueError('Grader did not return a valid score: '+job)
   with client.db:client.db.execute('insert or replace into judgments values (?,?,?,?)',(job,score,gh,time.time()))
  with db:
   for key,label in references[job]:
    row=json.loads(db.execute('select payload from observations where key=?',(key,)).fetchone()[0]);row[label+'_soft']=score
    row[label+'_judge_key']=job;db.execute('update observations set payload=? where key=?',(json.dumps(row,ensure_ascii=False),key))
 keys=list(pending)
 for start in range(0,len(keys),48):
  results=await asyncio.gather(*(one(job) for job in keys[start:start+48]),return_exceptions=True)
  errors=[x for x in results if isinstance(x,BaseException)]
  report=dict(status='grading',unique_judgments=len(keys),processed=min(start+48,len(keys)),tracked_shared_api_usd=client.spent())
  s.base.atomic_json(s.ROOT/name/'grading_report.json',report)
  if errors:raise errors[0]
  if start%480==0:print(name,'judged',min(start+48,len(keys)),'/',len(keys),'shared API $',round(client.spent(),3),flush=True)
 s.base.atomic_json(s.ROOT/name/'grading_report.json',dict(status='complete',unique_judgments=len(keys),tracked_shared_api_usd=client.spent()))


def export(name,partial=False):
 import pandas as pd
 db=observation_db(name);rows=[json.loads(r[0]) for r in db.execute('select payload from observations')]
 assert rows,'No measurements'
 if not partial:assert len(rows)==24000,'Generation/measurement is incomplete'
 for row in rows:
  for label in ['head32','full']:
   assert not (row['needs_judge'] and row[label+'_soft'] is None),'Judgments pending'
   assert row[label+'_soft'] is None or 0<=row[label+'_soft']<=1
 frame=pd.DataFrame(rows);frame['scoring_pipeline_sha256']=s.base.file_hash(__file__)
 frame.to_csv(s.ROOT/name/('scores_partial.csv' if partial else 'scores.csv'),index=False)
 print(name,'exported',len(frame),'observations',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['measure','judge','export','all']);parser.add_argument('--model',choices=list(s.MODELS),required=True);parser.add_argument('--partial',action='store_true');a=parser.parse_args()
 if a.phase in ['measure','all']:measure(a.model)
 if a.phase in ['judge','all']:asyncio.run(judge(a.model))
 if a.phase in ['export','all']:export(a.model,a.partial)
