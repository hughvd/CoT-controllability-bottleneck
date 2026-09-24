"""Metered provider capability probes for a one-token-only GLM intervention."""
import asyncio
import json
import sys
import urllib.request
from pathlib import Path

CTRL=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(CTRL))
import token_prefill_suite_api as api

ROOT=CTRL/'results/fixed_opening_ban/v1/glm_probe'

async def main():
 ROOT.mkdir(parents=True,exist_ok=True)
 api.s.ROOT=ROOT
 client=api.PaidClient(cap=2.,retries=1)
 client.client=client.client.with_options(timeout=50.)
 m=json.loads((CTRL/'results/token_prefill/full_v2/glm/manifest.json').read_text())
 r=next(r for r in m['requests'] if r['prefix_tokens']==0 and r['condition']=='prefill_only')
 tok=api.s.tokenizer(m['model']);bid=tok.encode('The',add_special_tokens=False)
 assert len(bid)==1
 endpoints=json.load(urllib.request.urlopen('https://openrouter.ai/api/v1/models/z-ai/glm-5.2/endpoints'))
 (ROOT/'endpoints.json').write_text(json.dumps(endpoints,indent=2))
 for provider in ['Together','Fireworks','Wafer','Parasail','Cloudflare']:
  for arm in ['baseline','ban']:
   payload=dict(model=m['model']['api_id'],messages=r['messages'],temperature=.6,top_p=1.,max_tokens=1,
    extra_body={'provider':{'order':[provider],'allow_fallbacks':False,'require_parameters':True},'include_reasoning':True})
   if arm=='ban':payload['logit_bias']={str(bid[0]):-100}
   name=provider.lower()+'_'+arm
   (ROOT/(name+'_request.json')).write_text(json.dumps(payload,indent=2))
   try:
    raw=await client.call('fixed-first-token-probe-v1|'+name,payload,3e-6,8e-6)
    (ROOT/(name+'_response.json')).write_text(json.dumps(raw,indent=2,ensure_ascii=False))
    ch=raw['choices'][0];print(name,dict(provider=raw.get('provider'),message=ch['message'],finish=ch.get('finish_reason'),usage=raw.get('usage')),flush=True)
   except Exception as e:
    info=dict(error_type=type(e).__name__,status_code=getattr(e,'status_code',None),message=str(e)[:500])
    (ROOT/(name+'_error.json')).write_text(json.dumps(info,indent=2));print(name,info,flush=True)
  # Keep probing if the advertised parameter is ignored in the reasoning channel.
 print('probe_cost',client.spent(),flush=True)
 await client.client.close()

if __name__=='__main__':asyncio.run(main())
