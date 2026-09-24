"""Small metered capability probes, separate from the original GLM experiment."""
import asyncio
import json
from pathlib import Path
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
CTRL = HERE.parent.parent
sys.path.insert(0, str(CTRL))
import token_prefill_suite_api as api

OUT = HERE / "opening_tokens" / "glm_logprob_probe"


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Separate journal and $1 ceiling; never mutate the original experiment ledger.
    api.s.ROOT = OUT
    client = api.PaidClient(cap=1., retries=1)
    client.client = client.client.with_options(timeout=45.)
    manifest = json.loads((CTRL / "results/token_prefill/full_v2/glm/manifest.json").read_text())
    request = next(r for r in manifest["requests"] if r["prefix_tokens"] == 0 and r["condition"] == "prefill_only")
    url = "https://openrouter.ai/api/v1/models/z-ai/glm-5.2/endpoints"
    metadata = json.load(urllib.request.urlopen(url))
    (OUT / "endpoints.json").write_text(json.dumps(metadata, indent=2))
    for provider in ["StreamLake", "DigitalOcean", "CoreWeave"]:
        payload = dict(model="z-ai/glm-5.2", messages=request["messages"],
                       temperature=.6, top_p=1., max_tokens=32, logprobs=True, top_logprobs=3,
                       extra_body={"provider":{"order":[provider],"allow_fallbacks":False,
                                               "require_parameters":True},"include_reasoning":True})
        name = provider.lower()
        (OUT / (name + "_request.json")).write_text(json.dumps(payload, indent=2))
        try:
            raw = await client.call("opening-logprob-probe-v1|"+name,payload,2e-6,6e-6)
            (OUT / (name + "_response.json")).write_text(json.dumps(raw, indent=2, ensure_ascii=False))
            choice = raw["choices"][0]
            print(provider, json.dumps(dict(provider=raw.get("provider"),message=choice.get("message"),
                logprobs=choice.get("logprobs"),cost=(raw.get("usage") or {}).get("cost")),ensure_ascii=False)[:6500],flush=True)
            # Stop once reasoning distributions are plausibly available; inspect before expansion.
            if choice.get("logprobs") and any(choice["logprobs"].values()):
                break
        except Exception as e:
            result=dict(error_type=type(e).__name__,status_code=getattr(e,"status_code",None),
                        message=str(e)[:1500])
            (OUT/(name+"_error.json")).write_text(json.dumps(result,indent=2))
            print(provider,result,flush=True)
    print("Tracked probe cost",client.spent(),flush=True)
    await client.client.close()


if __name__ == "__main__":
    asyncio.run(main())
