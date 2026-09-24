"""GLM first-token ban by rejection sampling at top-p=1.

Accept/reject uses ONLY the first native reasoning token. This samples the same
conditional first-token distribution as masking that ID then renormalizing,
without relying on unsupported hosted continuation. Complete accepted rollouts
are retained unchanged. No filtering by later compliance, length, or correctness.
"""
import argparse,asyncio,json,sqlite3,sys,time
from pathlib import Path
import fixed_opening_ban as f
sys.path.insert(0,str(f.CTRL))
import token_prefill_suite_api as api

async def run(pilot,instruction_only=False):
 root=f.ROOT/'glm';m=json.loads((root/'manifest.json').read_text());f.validate(m)
 api.s.ROOT=root/'rejection';client=api.PaidClient(cap=25.,retries=4);concurrency=8 if pilot else 24;client.sem=asyncio.Semaphore(concurrency)
 tok=f.suite.tokenizer(m['model']);db=sqlite3.connect(root/'api_outputs.sqlite');db.execute('pragma journal_mode=WAL')
 db.execute('create table if not exists outputs (id text primary key,payload text)')
 db.execute('create table if not exists proposals (id text,attempt integer,payload text,accepted integer,primary key(id,attempt))');db.commit()
 cells={c['cell_id']:c for c in m['cells']};report=dict(status='pilot' if pilot else 'generating',method='first-token rejection sampling at top_p=1',started_at=time.time())
 requests=m['requests']
 if instruction_only:requests=[r for r in requests if r['condition']=='instruction_only']
 report['scope']='instruction_only' if instruction_only else 'both_conditions'
 report['expected_in_scope']=len(requests)
 # Matched first question of each control, both conditions, first sample. Reusable pilot.
 pilotcells={next(c['cell_id'] for c in m['cells'] if c['behavior']==b) for b in f.BEHAVIORS}
 if pilot:requests=[r for r in requests if r['cell_id'] in pilotcells and r['sample']==0]
 def save():
  report.update(outputs=db.execute('select count(*) from outputs').fetchone()[0],proposals=db.execute('select count(*) from proposals').fetchone()[0],tracked_cost=client.spent(),updated_at=time.time());f.base.atomic_json(root/'rejection_report.json',report)
 async def one(r):
  rid=r['request_id']
  if db.execute('select 1 from outputs where id=?',(rid,)).fetchone():return
  for attempt in range(24 if pilot else 256):
   old=db.execute('select payload,accepted from proposals where id=? and attempt=?',(rid,attempt)).fetchone()
   if old:raw=json.loads(old[0])
   else:
    payload=dict(model=m['model']['api_id'],messages=r['messages'],temperature=.6,top_p=1.,max_tokens=8192,
     seed=int(f.base.digest([rid,attempt])[:8],16),extra_body={'provider':{'order':['DeepInfra'],'allow_fallbacks':False},'include_reasoning':True})
    raw=await client.call('fixed-ban-rejection-v1|'+rid+'|'+str(attempt),payload,1.4e-6,4.4e-6)
   rec=api.api_record(m,r,0,raw);text=rec['reasoning_continuation'];ids=tok.encode(text,add_special_tokens=False)
   # No stripping: the exact first returned native token defines this hosted arm.
   accepted=(r['arm']=='baseline' or not ids or ids[0] not in m['ban_token_ids'])
   with db:db.execute('insert or ignore into proposals values (?,?,?,?)',(rid,attempt,json.dumps(raw,ensure_ascii=False),int(accepted)))
   if not old:save()
   if accepted:
    pre=tok.decode(ids[:1],skip_special_tokens=False) if ids else ''
    if not text.startswith(pre) or '\ufffd' in pre:raise ValueError('Hosted first token cannot be round-tripped')
    result=dict(request_id=rid,raw_text=text,answer_text=rec['response'],first_token_text=pre,first_stage_ids=ids[:1],
     output_token_ids=None,finish_reason=rec['finish_reason'],proposal_count=attempt+1,provider=raw['provider'],generation_id=raw['id'],
     manifest_sha256=m['manifest_sha256'],method='rejection_at_first_native_reasoning_token',token_ids_provenance='local_retokenization')
    with db:db.execute('insert into outputs values (?,?)',(rid,json.dumps(result,ensure_ascii=False)))
    save();print(r['arm'],cells[r['cell_id']]['behavior'],r['condition'],'accepted after',attempt+1,flush=True);return
  raise RuntimeError('Proposal limit reached for '+rid+'; missing outcomes must not be silently dropped')
 queue=iter(requests);errors=[]
 async def worker():
  while not errors:
   try:r=next(queue)
   except StopIteration:return
   try:await one(r)
   except Exception as e:errors.append(str(e));return
 try:
  await asyncio.gather(*(worker() for _ in range(concurrency)))
  report.update(status='pilot_complete' if pilot and not errors else 'complete' if not errors else 'blocked',errors=errors)
 finally:save();await client.client.close()
 print(report,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--instruction-only',action='store_true');a=ap.parse_args();asyncio.run(run(a.pilot,a.instruction_only))
