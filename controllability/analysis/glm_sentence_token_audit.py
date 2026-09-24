"""Compare saved GLM sentence/token prefills; optional bounded parameter replay."""
import argparse,asyncio,json,sqlite3,sys
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import token_prefill_suite as s
import prefill_experiment as p
OUT=HERE/'token_prefill/glm_sentence_audit_v1'

def analyze():
 OUT.mkdir(exist_ok=True)
 m=json.loads((s.ROOT/'glm/manifest.json').read_text());tok=s.tokenizer(m['model'])
 old=pd.read_csv(s.HERE/'results/base_glm_5_2_prefill_per_sample.csv').fillna('')
 new=pd.read_csv(s.ROOT/'glm/scores.csv');cells={c['cell_id']:c for c in m['cells']};requests={r['request_id']:r for r in m['requests']}
 u=old[old.k.eq(1)].drop_duplicates(['ctype','question','seed']).copy()
 u['tokens']=u.seed.map(lambda x:len(tok.encode(x)))
 lookup={(c['behavior'],c['question']):c for c in m['cells']}
 u['same_source']=u.apply(lambda x:(x.ctype,x.question) in lookup and p.first_k_sentences(lookup[x.ctype,x.question]['controlled_text'],1)==x.seed,axis=1)
 u[['ctype','question','seed','tokens','same_source']].to_csv(OUT/'first_sentence_lengths.csv',index=False)
 links=pd.DataFrame([dict(cell_id=x['cell_id'],condition=x['condition'],prefix_tokens=x['prefix_tokens'],seed=requests[x['request_id']]['seed'],question=cells[x['cell_id']]['question'],ctype=cells[x['cell_id']]['behavior'],prompt=requests[x['request_id']]['user_text']) for x in m['links']])
 n=new.merge(links,on=['cell_id','condition','prefix_tokens']);keys=['ctype','question','condition','seed','prompt']
 a=old[old.k.gt(0)].groupby(keys).agg(old=('compliance_score','mean'),k=('k','min')).reset_index()
 b=n[n.prefix_tokens.gt(0)].groupby(keys).agg(new=('full_soft','mean'),tokens=('prefix_tokens','min')).reset_index()
 z=a.merge(b,on=keys);z.to_csv(OUT/'identical_prefix_comparison.csv',index=False)
 z.groupby(['ctype','condition']).agg(cells=('old','size'),old=('old','mean'),new=('new','mean')).to_csv(OUT/'identical_prefix_summary.csv')
 # Recompute both sides with one deterministic uppercase scorer; no LLM judges.
 upper=[]
 for row in z[z.ctype.eq('uppercase_thinking')].itertuples():
  o=old[(old.ctype==row.ctype)&(old.question==row.question)&(old.condition==row.condition)&(old.seed==row.seed)]
  nn=n[(n.ctype==row.ctype)&(n.question==row.question)&(n.condition==row.condition)&(n.seed==row.seed)]
  upper.append(dict(question=row.question,condition=row.condition,tokens=row.tokens,old=same_score(o.reasoning_continuation),new=same_score(nn.full_text),old_saved=row.old,new_saved=row.new))
 pd.DataFrame(upper).to_csv(OUT/'uppercase_same_scorer.csv',index=False)
 historic=Counter();dates=[]
 for path in (s.HERE/'transcripts/base_glm_5_2').glob('*.json'):
  raw=json.loads(path.read_text())['raw_response'];historic[raw['model']]+=1;dates.append(raw['created'])
 recent=Counter();newdates=[];db=sqlite3.connect(s.ROOT/'paid.sqlite')
 for (body,) in db.execute("select response from paid_results where job like 'generate|glm|%'"):
  raw=json.loads(body);recent[raw['model']]+=1;newdates.append(raw['created'])
 info=dict(first_sentence_lengths=u.tokens.describe(percentiles=[.5,.9,.95]).to_dict(),more_than_64=int((u.tokens>64).sum()),same_source=int(u.same_source.sum()),matched_prompt_prefix_cells=len(z),historic_models=dict(historic),token_models=dict(recent),historic_utc=[datetime.fromtimestamp(x,timezone.utc).isoformat() for x in [min(dates),max(dates)]],token_utc=[datetime.fromtimestamp(x,timezone.utc).isoformat() for x in [min(newdates),max(newdates)]])
 (OUT/'audit.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2))
 ends=pd.Series([len(list(p._SENT_END.finditer(r['seed']))) for r in m['requests'] if r['prefix_tokens']==64 and r['condition']=='prefill_only'])
 info['sentence_endings_at_64_tokens']=ends.describe().to_dict()
 (OUT/'audit.json').write_text(json.dumps(info,indent=2)+'\n')

def same_score(texts):return sum(s.base.score_text(t,'uppercase_thinking')['soft'] for t in texts)/len(texts)

async def probe():
 from token_prefill_suite_api import PaidClient
 inputs=json.loads((OUT/'probe_inputs.json').read_text())
 original=s.ROOT
 try:
  s.ROOT=OUT;client=PaidClient(cap=2.,retries=3)
 finally:s.ROOT=original
 client.sem=asyncio.Semaphore(4)
 async def one(row,variant,sample):
  payload=dict(model='z-ai/glm-5.2',messages=row['messages'],temperature=.6,max_tokens=8192,extra_body={'provider':{'order':['DeepInfra'],'allow_fallbacks':False}})
  if variant=='token_parameters':payload.update(top_p=1.,seed=int(s.base.digest([row['request_id'],sample])[:8],16));payload['extra_body']['include_reasoning']=True
  job=s.base.digest([row['behavior'],variant,sample,payload])
  raw=await client.call('parameter_replay|'+job,payload,1.4e-6,4.4e-6)
  result=dict(behavior=row['behavior'],variant=variant,sample=sample,payload=payload,raw=raw)
  (OUT/(job+'.json')).write_text(json.dumps(result,indent=2));print(row['behavior'],variant,sample,'complete',flush=True)
 await asyncio.gather(*(one(r,v,j) for r in inputs for v in ['sentence_parameters','token_parameters'] for j in range(3)))
 print('Replay billed/reserved USD',client.spent())

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');args=ap.parse_args()
 if args.probe:asyncio.run(probe())
 else:analyze()
