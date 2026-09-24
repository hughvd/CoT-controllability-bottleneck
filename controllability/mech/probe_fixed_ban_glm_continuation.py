import asyncio,json,sys
from pathlib import Path
CTRL=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(CTRL))
import token_prefill_suite_api as api
ROOT=CTRL/'results/fixed_opening_ban/v1/glm_probe'
async def main():
 api.s.ROOT=ROOT;client=api.PaidClient(cap=2.,retries=2)
 m=json.loads((CTRL/'results/token_prefill/full_v2/glm/manifest.json').read_text())
 r=next(r for r in m['requests'] if r['prefix_tokens']==0 and r['condition']=='prefill_only')
 tok=api.s.tokenizer(m['model'])
 for provider,prefix in [('Parasail','The'),('Parasail','Pl'),('Parasail','OKAY, LET'),('Wafer','Pl'),('Wafer','OKAY, LET'),('Cloudflare','Pl'),('Cloudflare','OKAY, LET'),('DeepInfra','Pl'),('DeepInfra','OKAY, LET')]:
  messages=[r['messages'][0],{'role':'assistant','content':'<think>\n'+prefix}]
  payload=dict(model=m['model']['api_id'],messages=messages,temperature=.6,top_p=1.,max_tokens=48,
   extra_body={'provider':{'order':[provider],'allow_fallbacks':False,'require_parameters':True},'include_reasoning':True})
  raw=await client.call('fixed-continuation-probe-v2|'+provider+'|'+prefix,payload,3e-6,8e-6)
  (ROOT/('continuation_'+provider+'_'+prefix.replace(',','').replace(' ','_')+'.json')).write_text(json.dumps(dict(request=payload,response=raw),indent=2))
  print(provider,prefix,raw['choices'][0]['message'],flush=True)
 # Positive bias checks the token-ID interpretation independently of an absence.
 target=tok.encode('Ban',add_special_tokens=False);assert len(target)==1
 payload=dict(model=m['model']['api_id'],messages=r['messages'],temperature=.6,top_p=1.,max_tokens=1,
  logit_bias={str(target[0]):100},extra_body={'provider':{'order':['Parasail'],'allow_fallbacks':False,'require_parameters':True},'include_reasoning':True})
 raw=await client.call('fixed-positive-bias-probe-v1',payload,3e-6,8e-6)
 (ROOT/'positive_bias.json').write_text(json.dumps(dict(request=payload,response=raw),indent=2));print('positive',raw['choices'][0]['message'],flush=True)
 await client.client.close()
if __name__=='__main__':asyncio.run(main())
