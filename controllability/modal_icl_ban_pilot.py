"""Bounded native GPT-OSS pilot: mask only the first reasoning content token."""
from pathlib import Path
import json
import time
import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'results/icl_ban_correlation_pilot/v1'
REV = 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a'
image = (modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04', add_python='3.12').entrypoint([])
    .pip_install('vllm==0.23.0', 'transformers==5.12.1', 'openai-harmony==0.0.8')
    .env({'HF_HOME':'/cache','HF_HUB_DISABLE_XET':'1','VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8':'0',
          'TOKENIZERS_PARALLELISM':'false','VLLM_USE_FLASHINFER_SAMPLER':'0'})
    .add_local_file(HERE/'harmony_backend.py','/root/harmony_backend.py')
    .add_local_file(HERE/'icl_harmony.py','/root/icl_harmony.py'))
app = modal.App('icl-opening-ban-gptoss-pilot')
cache = modal.Volume.from_name('ban-sweep-hf-cache')
out = modal.Volume.from_name('icl-opening-ban-pilot',create_if_missing=True)


@app.function(image=image,gpu=['B200','H200'],cpu=8,memory=131072,timeout=2400,retries=0,
              max_containers=1,scaledown_window=2,volumes={'/cache':cache.read_only(),'/out':out})
def generate(m):
    import hashlib
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    import sys
    sys.path.insert(0,'/root')
    import harmony_backend as h
    import icl_harmony as ih
    from vllm import LLM,SamplingParams
    import vllm
    folder=Path('/out')/m['sha256'];folder.mkdir(exist_ok=True)
    report=dict(status='starting',started_at=time.time(),revision=REV,vllm=vllm.__version__,expected=32)
    def save():
        (folder/'report.json').write_text(json.dumps(report,indent=2));out.commit()
    save()
    try:
        snap=Path('/cache/hub/models--openai--gpt-oss-120b/snapshots')/REV
        local=Path('/tmp/gptoss');local.mkdir(exist_ok=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda p:shutil.copyfile(p,local/p.name),[p for p in snap.iterdir() if p.is_file()]))
        llm=LLM(model=str(local),tokenizer=str(local),max_model_len=32768,max_num_seqs=64,
                max_num_batched_tokens=8192,gpu_memory_utilization=.90,enable_prefix_caching=True,seed=20261013)
        tok=llm.get_tokenizer()
        banned=sum([tok.encode(w,add_special_tokens=False) for w in ['We','The']],[])
        assert len(banned)==2
        rows=[]
        for job in m['jobs']:
            prompt=ih.build_tokens(job['messages'],reasoning_effort='high')
            # Open an empty analysis message in both arms, so the intervention
            # never masks channel-selection or Harmony structural tokens.
            header=h._enc().encode('<|channel|>analysis<|message|>',allowed_special='all')
            prompt+=header
            assert len(prompt)+8192<=32768
            for arm in ['baseline','banned']:
                seed=int(hashlib.sha256((job['behavior']+job['condition']).encode()).hexdigest()[:8],16)%2**31
                rows.append(dict(behavior=job['behavior'],condition=job['condition'],arm=arm,prompt_token_ids=prompt,seed=seed))
        prefixes=[[] for _ in rows]
        active=list(range(len(rows)))
        for step in range(16):
            if not active:break
            params=[SamplingParams(temperature=.6,top_p=1.,top_k=-1,max_tokens=1,seed=rows[i]['seed']+step,
                logit_bias={b:float('-inf') for b in banned} if rows[i]['arm']=='banned' else None,
                skip_special_tokens=False) for i in active]
            output=llm.generate([dict(prompt_token_ids=rows[i]['prompt_token_ids']+prefixes[i]) for i in active],params,use_tqdm=False)
            again=[]
            for i,o in zip(active,output):
                ids=list(o.outputs[0].token_ids);assert len(ids)==1
                if rows[i]['arm']=='banned':assert ids[0] not in banned
                prefixes[i]+=ids
                if not tok.decode(prefixes[i],skip_special_tokens=False).strip():again.append(i)
            active=again
        assert not active
        params=[SamplingParams(temperature=.6,top_p=1.,top_k=-1,max_tokens=8192-len(prefixes[i]),
            seed=r['seed']+100003,skip_special_tokens=False,
            stop_token_ids=h._enc().stop_tokens_for_assistant_actions()) for i,r in enumerate(rows)]
        result=llm.generate([dict(prompt_token_ids=r['prompt_token_ids']+prefixes[i]) for i,r in enumerate(rows)],params,use_tqdm=False)
        outputs=[]
        for i,(r,o) in enumerate(zip(rows,result)):
            ids=prefixes[i]+list(o.outputs[0].token_ids);raw=tok.decode(ids,skip_special_tokens=False)
            reasoning,answer=h.split_channels(raw)
            outputs.append(dict(**r,first_stage_ids=prefixes[i],output_token_ids=ids,raw_text=raw,
                reasoning=reasoning,answer=answer,finish_reason=o.outputs[0].finish_reason))
        (folder/'outputs.json').write_text(json.dumps(outputs,ensure_ascii=False))
        (folder/'manifest.json').write_text(json.dumps(m,ensure_ascii=False))
        report.update(status='complete',outputs=len(outputs),ban_ids=banned,
            empty=sum(not r['reasoning'].strip() for r in outputs),truncated=sum(r['finish_reason']=='length' for r in outputs))
    except Exception as e:
        report.update(status='failed',error=repr(e));raise
    finally:
        report['elapsed_seconds']=time.time()-report['started_at'];save()
    return report


@app.local_entrypoint()
def main(phase:str='dispatch'):
    m=json.loads((ROOT/'manifest.json').read_text());local=ROOT/'native_gptoss';local.mkdir(exist_ok=True)
    if phase=='dispatch':
        if (local/'dispatch.json').exists():raise ValueError('Existing dispatch; inspect before relaunching')
        call=generate.spawn(m)
        (local/'dispatch.json').write_text(json.dumps(dict(call_id=call.object_id,started_at=time.time(),timeout_seconds=2400),indent=2))
        print(call.object_id)
    elif phase=='collect':
        for entry in out.iterdir(m['sha256']):
            if entry.path.endswith('.json'):
                (local/Path(entry.path).name).write_bytes(b''.join(out.read_file(entry.path)))
        print((local/'report.json').read_text())
    else:raise ValueError(phase)
