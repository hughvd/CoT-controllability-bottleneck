"""One bounded B200 worker for the clean GPT-OSS prefill sweep; no API secrets.

modal run controllability/modal_gptoss_prefill.py --phase cache
modal run --detach controllability/modal_gptoss_prefill.py --phase dispatch
"""
from pathlib import Path
import hashlib
import json
import time
import modal

HERE = Path(__file__).resolve().parent
RUN = 'gptoss_high_clean_v1'
MODEL = 'openai/gpt-oss-120b'
REVISION = 'b5c939de8f754692c1647ca79fbf85e8c1e70f8a'
ROOT = HERE / 'results' / RUN
image = (modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04', add_python='3.12').entrypoint([])
    .pip_install('vllm==0.23.0', 'transformers==5.12.1', 'numpy', 'huggingface_hub', 'hf_transfer')
    .pip_install('openai-harmony==0.0.8')
    .env({'HF_HOME':'/cache', 'HF_HUB_DISABLE_XET':'1', 'HF_HUB_ENABLE_HF_TRANSFER':'1',
          'VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8':'1', 'TOKENIZERS_PARALLELISM':'false'})
    .add_local_file(HERE / 'harmony_backend.py', '/root/harmony_backend.py'))
app = modal.App('gptoss-prefill-clean')
cache = modal.Volume.from_name('ban-sweep-hf-cache')
outvol = modal.Volume.from_name('gptoss-prefill-out', create_if_missing=True)


@app.function(image=image, cpu=4, memory=16384, volumes={'/cache':cache}, timeout=3600)
def warm_cache():
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=REVISION,
        allow_patterns=['*.json','*.safetensors','*.jinja','*.tiktoken','*.model'],
        ignore_patterns=['original/*','metal/*'])
    cache.commit()
    return {'snapshot':path, 'files':len(list(Path(path).iterdir()))}


@app.function(image=image, gpu='B200', cpu=8, memory=131072,
              volumes={'/cache':cache.read_only(), '/out':outvol},
              timeout=14400, retries=0, max_containers=1, scaledown_window=2)
def generate(manifest: dict, budget_seconds: float):
    started = time.monotonic()
    import shutil
    import sys
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0,'/root')
    import harmony_backend as h
    import vllm
    from vllm import LLM, SamplingParams
    assert hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest() == manifest['harmony_sha256']
    folder = Path('/out') / RUN / manifest['manifest_sha256']
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'manifest.json').write_text(json.dumps(manifest))
    completed = {p.stem for p in folder.glob('batch_*.json')}
    report = {'status':'starting', 'revision':REVISION, 'vllm':vllm.__version__,
              'manifest_sha256':manifest['manifest_sha256'], 'batches':[], 'started_at':time.time(),
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save_report():
        report['elapsed_seconds'] = time.monotonic()-started
        (folder/'report.json').write_text(json.dumps(report,indent=2))
        outvol.commit()
    save_report()
    try:
        snapshot=Path('/cache/hub/models--openai--gpt-oss-120b/snapshots')/REVISION
        local=Path('/tmp/gptoss-weights'); local.mkdir(exist_ok=True)
        def stage(src):
            with src.open('rb') as a, (local/src.name).open('wb') as b:
                shutil.copyfileobj(a,b,16*1024*1024)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(stage,[p for p in snapshot.iterdir() if p.is_file()]))
        print('Model weights staged',flush=True)
        llm=LLM(model=str(local), tokenizer=str(local), max_model_len=16384,
                max_num_seqs=128, max_num_batched_tokens=8192, gpu_memory_utilization=.90,
                enable_prefix_caching=True, seed=20260909)
        tokenizer=llm.get_tokenizer()
        # Harmony's encoding must agree with the serving tokenizer's special IDs.
        for token in ['<|start|>','<|channel|>','<|message|>','<|end|>','<|return|>']:
            assert h._enc().encode(token, allowed_special='all') == tokenizer.encode(token, add_special_tokens=False), token
        report['startup_seconds']=time.monotonic()-started
        report['status']='smoke'
        requests=manifest['requests']
        # Two matched cells per behavior, both conditions, K=0/1/5: 120 high
        # requests, plus medium-effort counterparts solely for validation.
        smoke_ids=set(manifest['smoke_ids'])
        smoke=[r for r in requests if r['request_id'] in smoke_ids]
        full=[r for r in requests if r['request_id'] not in smoke_ids]
        jobs=[('smoke_high',smoke,'high',5),('smoke_medium',smoke,'medium',1)]
        jobs += [(f'batch_{i:04d}',full[i:i+64],'high',5) for i in range(0,len(full),64)]
        # Names all start batch_ so interrupted runs can discover durable batches.
        jobs=[(name if name.startswith('batch_') else 'batch_'+name, rs, effort, n) for name,rs,effort,n in jobs]
        smoke_rate=None
        for name, rows, effort, n in jobs:
            if name in completed:
                continue
            remaining=budget_seconds-(time.monotonic()-started)
            if remaining < 180:
                report['status']='budget_stop';break
            prompts=[h.build_prefill_tokens(r['question'],r['options'],r['instruction'] if r['condition']=='prefill_plus_instruction' else '',r['seed'],effort) for r in rows]
            if effort=='high':
                assert all(tokens==row['prompt_token_ids'] for tokens,row in zip(prompts,rows))
            if any(len(p)+8192>16384 for p in prompts): raise ValueError('Context would be truncated')
            params=[SamplingParams(n=n,temperature=.6,top_p=1.0,top_k=-1,max_tokens=8192,
                seed=int(r['request_id'][:8],16),skip_special_tokens=False,
                stop_token_ids=h._enc().stop_tokens_for_assistant_actions()) for r in rows]
            t=time.monotonic()
            outputs=llm.generate([{'prompt_token_ids':p} for p in prompts],params,use_tqdm=False)
            results=[]
            for r,p,o in zip(rows,prompts,outputs):
                assert list(o.prompt_token_ids)==p
                assert len(o.outputs)==n
                for sample,answer in enumerate(o.outputs):
                    reasoning,final=h.split_channels(answer.text)
                    results.append({'request_id':r['request_id'],'sample':sample,'effort':effort,
                        'prompt_token_ids':p,'output_token_ids':list(answer.token_ids),
                        'raw_text':answer.text,'reasoning_continuation':reasoning,'response':final,
                        'finish_reason':answer.finish_reason,'stop_reason':answer.stop_reason})
            elapsed=time.monotonic()-t
            tmp=folder/(name+'.tmp');tmp.write_text(json.dumps(results,ensure_ascii=False));tmp.replace(folder/(name+'.json'))
            info={'batch':name,'outputs':len(results),'seconds':elapsed,
                  'empty':sum(not r['reasoning_continuation'].strip() for r in results),
                  'length_limited':sum(r['finish_reason']=='length' for r in results)}
            report['batches'].append(info);save_report(); print(info,flush=True)
            if name=='batch_smoke_high':
                if info['empty'] > .1*len(results):
                    raise ValueError('Smoke has unexpectedly high empty-reasoning rate')
                if info['length_limited'] > .1*len(results):
                    raise ValueError('Smoke truncation exceeds 10%; inspect before full generation')
                if any('<|channel|>' in r['reasoning_continuation'] for r in results):
                    raise ValueError('Unparsed channel boundary')
                smoke_rate=elapsed/len(rows)
                projection=smoke_rate*len(full)*1.5+180
                report['projected_remaining_seconds']=projection
                if projection > budget_seconds-(time.monotonic()-started):
                    report['status']='projection_exceeds_budget';break
            report['status']='generating'
        else:
            report['status']='complete'
    except Exception:
        report['status']='failed'
        raise
    finally:
        save_report()
    return report


@app.local_entrypoint()
def main(phase: str='cache'):
    if phase not in {'cache','collect','dispatch','generate'}:
        raise ValueError(f'Unknown phase: {phase}')
    if phase=='cache':
        print(warm_cache.remote()); return
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if phase=='collect':
        remote=f'{RUN}/{manifest["manifest_sha256"]}'
        report=json.loads(b''.join(outvol.read_file(remote+'/report.json')))
        print({k:report.get(k) for k in ['status','elapsed_seconds','projected_remaining_seconds','batches']})
        if report['status'] not in {'complete','failed','budget_stop','projection_exceeds_budget'}:
            return
        ledgerpath=ROOT/'gpu_ledger.json';ledger=json.loads(ledgerpath.read_text())
        entry=ledger['attempts'][-1]
        if report['started_at']<entry['started_at']:
            raise ValueError('Report belongs to an earlier attempt')
        entry.update(status=report['status'],accounted_seconds=report['elapsed_seconds']+60)
        ledgerpath.write_text(json.dumps(ledger,indent=2))
        (ROOT/'generation_report.json').write_text(json.dumps(report,indent=2))
        for item in outvol.listdir(remote):
            if item.path.endswith('.json'):
                dst=ROOT/'remote'/Path(item.path).name;dst.parent.mkdir(exist_ok=True)
                with dst.open('wb') as f:
                    for chunk in outvol.read_file(item.path):f.write(chunk)
        return
    reportpath=ROOT/'generation_report.json'
    if reportpath.exists() and json.loads(reportpath.read_text()).get('status')=='complete':
        print('Generation already complete; no GPU worker launched.')
        report=json.loads(reportpath.read_text())
        remote=f'{RUN}/{manifest["manifest_sha256"]}'
        for item in outvol.listdir(remote):
            if item.path.endswith('.json'):
                dst=ROOT/'remote'/Path(item.path).name;dst.parent.mkdir(exist_ok=True)
                if not dst.exists():
                    with dst.open('wb') as f:
                        for chunk in outvol.read_file(item.path): f.write(chunk)
        return
    ledgerpath=ROOT/'gpu_ledger.json'
    ledger=json.loads(ledgerpath.read_text()) if ledgerpath.exists() else {'max_seconds':14400,'attempts':[],'conservative_usd_per_hour':8.0}
    used=sum(a.get('accounted_seconds',a['reserved_seconds']) for a in ledger['attempts'])
    remaining=ledger['max_seconds']-used
    if remaining<300: raise RuntimeError('GPU time budget exhausted')
    entry={'started_at':time.time(),'reserved_seconds':remaining,'status':'reserved'}
    ledger['attempts'].append(entry);ledgerpath.write_text(json.dumps(ledger,indent=2))
    if phase=='dispatch':
        call=generate.with_options(timeout=int(remaining)).spawn(manifest,remaining-60)
        entry.update(status='spawned',call_id=call.object_id)
        ledgerpath.write_text(json.dumps(ledger,indent=2))
        print('Dispatched bounded worker:',call.object_id)
        return
    try:
        report=generate.with_options(timeout=int(remaining)).remote(manifest,remaining-60)
        entry.update(status=report['status'],accounted_seconds=report['elapsed_seconds']+60)
        (ROOT/'generation_report.json').write_text(json.dumps(report,indent=2))
    except (modal.exception.RemoteError, ValueError, RuntimeError, ImportError, AssertionError):
        # A returned remote exception means this worker stopped; account the
        # entire local wait conservatively, including provisioning.
        entry.update(status='remote_failed',accounted_seconds=min(remaining,time.time()-entry['started_at']+60))
        raise
    finally:
        ledgerpath.write_text(json.dumps(ledger,indent=2))
    remote=f'{RUN}/{manifest["manifest_sha256"]}'
    for item in outvol.listdir(remote):
        if item.path.endswith('.json'):
            dst=ROOT/'remote'/Path(item.path).name;dst.parent.mkdir(exist_ok=True)
            with dst.open('wb') as f:
                for chunk in outvol.read_file(item.path): f.write(chunk)
    print(report)
