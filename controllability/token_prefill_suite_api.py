"""Resumable, metered OpenRouter generation and shared judging transport."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

import token_prefill_suite as s


def connect():
 s.ROOT.mkdir(parents=True,exist_ok=True)
 db=sqlite3.connect(s.ROOT/'paid.sqlite',timeout=30)
 db.execute('pragma journal_mode=WAL')
 db.execute('create table if not exists attempts (id text primary key, job text, cost real, state text, response text, at real)')
 db.execute('create table if not exists paid_results (job text primary key, response text)')
 db.commit();return db


class PaidClient:
 def __init__(self,cap=125.,retries=8):
  from dotenv import load_dotenv
  from openai import AsyncOpenAI
  for p in [s.HERE.parent/'.env',s.HERE/'.env']:load_dotenv(p,override=True)
  self.client=AsyncOpenAI(api_key=os.environ['OPENROUTER_API_KEY'],base_url='https://openrouter.ai/api/v1',timeout=600,max_retries=0)
  policy_path=s.ROOT/'execution_policy.json'
  continuous=policy_path.exists() and json.loads(policy_path.read_text()).get('continue_until_complete',False)
  self.db=connect();self.cap=None if continuous else cap;self.retries=retries;self.sem=asyncio.Semaphore(24);self.locks={}
 def spent(self):return self.db.execute('select coalesce(sum(cost),0) from attempts').fetchone()[0]
 async def call(self,job,payload,input_rate,output_rate):
  lock=self.locks.setdefault(job,asyncio.Lock())
  async with lock:
   old=self.db.execute('select response from paid_results where job=?',(job,)).fetchone()
   if old:return json.loads(old[0])
   async with self.sem:
    # UTF-8 bytes conservatively bound text token count for these tokenizers.
    reserve=(len(json.dumps(payload.get('messages',[]),ensure_ascii=False).encode())+2048)*input_rate+payload['max_tokens']*output_rate
    for retry in range(self.retries):
     aid=uuid.uuid4().hex
     self.db.execute('begin immediate')
     if self.cap is not None and self.spent()+reserve>self.cap:
      self.db.rollback();raise RuntimeError('Shared paid-API budget cap reached')
     self.db.execute('insert into attempts values (?,?,?,?,?,?)',(aid,job,reserve,'reserved',None,time.time()));self.db.commit()
     try:
      response=await self.client.chat.completions.create(**payload);raw=response.model_dump()
     except Exception as e:
      code=getattr(e,'status_code',None)
      # Explicit HTTP rejection has no completion; ambiguous timeouts retain
      # the reservation as a conservative possible charge.
      cost=0. if code in [400,401,402,403,404,422,429] else reserve
      self.db.execute('update attempts set state=?,cost=?,response=? where id=?',('error',cost,json.dumps({'error_type':type(e).__name__,'status_code':code}),aid));self.db.commit()
      if code in [400,401,402,403,404,422] or retry==self.retries-1:raise
      delay=min(120,20*2**retry) if code==429 else min(30,2**retry)
      await asyncio.sleep(delay+(int(aid[:2],16)/64));continue
     cost=(raw.get('usage') or {}).get('cost')
     if cost is None:cost=reserve
     assert float(cost)>=0
     body=json.dumps(raw,ensure_ascii=False)
     with self.db:
      self.db.execute('update attempts set state=?,cost=?,response=? where id=?',('complete',float(cost),body,aid))
      self.db.execute('insert or replace into paid_results values (?,?)',(job,body))
     return raw


def api_record(m,r,sample,raw):
 assert raw.get('provider')==m['model']['provider'],raw.get('provider')
 choice=raw['choices'][0];message=choice['message'];text=message.get('reasoning')
 if text is None:
  # No invented reasoning if the provider omits its reasoning field.
  text=''.join(x.get('text','') for x in (message.get('reasoning_details') or []) if x.get('type')=='reasoning.text')
 if not isinstance(text,str):raise ValueError('Unexpected reasoning payload')
 # DeepInfra's API probes return only the continuation. Do not strip a matching
 # one-token prefix heuristically: that could delete genuinely generated text.
 final=message.get('content') or '';parse='provider_reasoning_field'
 if text.lstrip().startswith('<think>'):
  text=text.split('<think>',1)[1];parse='leading_think_tag_in_reasoning_field'
 if '</think>' in text:
  text,tail=text.split('</think>',1)
  if not final:final=tail.split('<|user|>',1)[0]
  parse='closing_think_tag_in_reasoning_field'
 return dict(request_id=r['request_id'],sample=sample,reasoning_continuation=text,
  response=final,closed_reasoning=bool(final),channel_parse=parse,
  finish_reason=choice.get('finish_reason'),provider=raw['provider'],generation_id=raw['id'],
  output_token_ids=None,token_ids_provenance='not_returned_by_hosted_api',
  manifest_sha256=m['manifest_sha256'],payload_seed=int(s.base.digest([r['request_id'],sample])[:8],16))


def pilot_requests(m):
 cells={c['cell_id']:c for c in m['cells']};requests=sorted(m['requests'],key=lambda r:r['request_id']);selected=set()
 for b in m['behaviors']:
  for c in s.CONDITIONS:
   for k in [0,1,64]:
    selected.add(next(r['request_id'] for r in requests if cells[r['cell_id']]['behavior']==b and r['condition']==c and r['prefix_tokens']==k))
 rest=[r for r in requests if r['request_id'] not in selected];n=128-len(selected)
 selected.update(rest[i*len(rest)//n]['request_id'] for i in range(n))
 return selected


async def generate(name):
 scope=json.loads((s.ROOT/'scope.json').read_text())
 assert name in scope['active_models'],'Model deferred by user'
 root=s.ROOT/name;m=json.loads((root/'manifest.json').read_text());s.validate(m)
 assert m['model']['backend']=='openrouter' and not m['api_prefix_issues']
 client=PaidClient(scope['api_estimate_cap_usd'],retries=24)
 concurrency=8
 client.sem=asyncio.Semaphore(concurrency)
 db=sqlite3.connect(root/'api_generations.sqlite');db.execute('pragma journal_mode=WAL')
 db.execute('create table if not exists outputs (request_id text,sample integer,payload text,primary key(request_id,sample))');db.commit()
 known={r['request_id']:r for r in m['requests']};pilot=pilot_requests(m)
 report=dict(status='pilot',manifest_sha256=m['manifest_sha256'],pilot_request_ids=sorted(pilot),started_at=time.time(),outputs=db.execute('select count(*) from outputs').fetchone()[0],runner_sha256=s.base.file_hash(__file__),concurrency=concurrency,max_attempts_per_request=24)
 s.base.atomic_json(root/'api_report.json',report)
 async def one(r,sample):
  if db.execute('select 1 from outputs where request_id=? and sample=?',(r['request_id'],sample)).fetchone():return
  seed=int(s.base.digest([r['request_id'],sample])[:8],16)
  payload=dict(model=m['model']['api_id'],messages=r['messages'],temperature=.6,top_p=1.,max_tokens=8192,seed=seed,
   extra_body={'provider':{'order':['DeepInfra'],'allow_fallbacks':False},'include_reasoning':True})
  raw=await client.call('generate|'+name+'|'+m['manifest_sha256']+'|'+r['request_id']+'|'+str(sample),payload,1.4e-6,4.4e-6)
  record=api_record(m,r,sample,raw)
  with db:db.execute('insert into outputs values (?,?,?)',(r['request_id'],sample,json.dumps(record,ensure_ascii=False)))
 async def process(rows):
  queue=iter(rows);errors=[];reported=-1
  async def worker():
   nonlocal reported
   while not errors:
    try:r,j=next(queue)
    except StopIteration:return
    try:await one(r,j)
    except Exception as e:errors.append(e);return
    count=db.execute('select count(*) from outputs').fetchone()[0]
    if count//120>reported:
     reported=count//120
     report.update(outputs=count,tracked_api_usd=client.spent(),updated_at=time.time())
     s.base.atomic_json(root/'api_report.json',report)
     print(name,count,'outputs; tracked API $',round(client.spent(),3),flush=True)
  await asyncio.gather(*(worker() for _ in range(concurrency)))
  if errors:raise errors[0]
 try:
  await process([(known[r],j) for r in sorted(pilot) for j in range(5)])
  results=[json.loads(row[0]) for row in db.execute('select payload from outputs') if json.loads(row[0])['request_id'] in pilot]
  empty=sum(not r['reasoning_continuation'].strip() for r in results)/len(results)
  trunc=sum(r['finish_reason']=='length' for r in results)/len(results)
  report['pilot_quality']=dict(n=len(results),empty_rate=empty,truncation_rate=trunc)
  # Use billed pilot generation costs to gate the remainder; grading is separate.
  attempts=client.db.execute("select job,cost from attempts where job like ?",('generate|'+name+'|'+m['manifest_sha256']+'|%',)).fetchall()
  paid=sum(cost for job,cost in attempts if job.split('|')[-2] in pilot)
  remaining=len(known)*5-db.execute('select count(*) from outputs').fetchone()[0]
  projection=paid/max(len(results),1)*remaining*1.5
  report['projected_generation_usd']=projection
  if empty>.15 or trunc>.15:raise RuntimeError('API pilot quality gate failed')
  if client.cap is not None and client.spent()+projection>client.cap:raise RuntimeError('API pilot projection exceeds spending cap')
  report['status']='generating';s.base.atomic_json(root/'api_report.json',report)
  await process([(r,j) for r in sorted(known.values(),key=lambda r:r['request_id']) if r['request_id'] not in pilot for j in range(5)])
  assert db.execute('select count(*) from outputs').fetchone()[0]==len(known)*5
  report['status']='complete'
 except Exception as e:
  report.update(status='failed',error_type=type(e).__name__,error=str(e)[:200]);raise
 finally:
  report.update(outputs=db.execute('select count(*) from outputs').fetchone()[0],tracked_api_usd=client.spent(),elapsed_seconds=time.time()-report['started_at']);s.base.atomic_json(root/'api_report.json',report)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['glm','deepseek'],default='glm');args=parser.parse_args();asyncio.run(generate(args.model))
