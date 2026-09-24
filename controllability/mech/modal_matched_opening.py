"""Bounded sequential GPU workers for the matched two-panel opening figure."""
import json
from pathlib import Path
import time
import modal

HERE=Path(__file__).resolve().parent
ROOT=HERE/'data/matched_openings/v1'
image=(modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04',add_python='3.12').entrypoint([])
       .pip_install('vllm==0.23.0','transformers==5.12.1','huggingface_hub')
       .pip_install('accelerate==1.12.0')
       .env({'HF_HOME':'/cache','HF_HUB_DISABLE_XET':'1','TOKENIZERS_PARALLELISM':'false','VLLM_USE_FLASHINFER_SAMPLER':'0'})
       .add_local_file(HERE/'matched_opening_measure.py','/root/matched_opening_measure.py'))
app=modal.App('matched-opening-figure')
cache=modal.Volume.from_name('ban-sweep-hf-cache')
volume=modal.Volume.from_name('matched-opening-figure-out',create_if_missing=True)


@app.function(image=image,gpu='B200',cpu=8,memory=131072,volumes={'/cache':cache.read_only(),'/out':volume},timeout=1850,retries=0,max_containers=1,scaledown_window=2)
def measure(m,inputs,tag):
    import shutil,subprocess,sys
    from concurrent.futures import ThreadPoolExecutor
    started=time.monotonic();work=Path('/out')/m['manifest_sha256']/tag;work.mkdir(parents=True,exist_ok=True)
    (work/'manifest.json').write_text(json.dumps(m))
    if tag=='qwen32':(work/'qwen32_inputs.json').write_text(json.dumps(inputs))
    report=dict(tag=tag,status='starting',manifest_sha256=m['manifest_sha256'],started_at=time.time())
    try:
        spec=m['models'][tag];snapshot=Path('/cache/hub')/('models--'+spec['model'].replace('/','--'))/'snapshots'/spec['revision']
        staged=Path('/tmp')/('model_'+tag);staged.mkdir(exist_ok=True)
        def copy(src):
            with src.open('rb') as a,(staged/src.name).open('wb') as b:shutil.copyfileobj(a,b,16*1024*1024)
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(copy,[p for p in snapshot.iterdir() if p.is_file()]))
        print(tag,'staged weights',round(time.monotonic()-started,1),flush=True)
        for phase in (['generate','score'] if tag=='olmo31' else ['score']):
            remaining=m['max_worker_seconds']-(time.monotonic()-started)
            if remaining<60:raise TimeoutError('GPU time ceiling reached')
            subprocess.run([sys.executable,'/root/matched_opening_measure.py','--stage',phase,'--tag',tag,'--work',str(work),'--model-dir',str(staged)],check=True,timeout=remaining)
            volume.commit()
        report['status']='complete'
    except Exception as e:
        report.update(status='failed',error_type=type(e).__name__,error=str(e)[:1000]);raise
    finally:
        report['seconds']=time.monotonic()-started
        (work/'report.json').write_text(json.dumps(report,indent=2));volume.commit()
    return report


@app.local_entrypoint()
def main():
    m=json.loads((ROOT/'manifest.json').read_text());q=json.loads((ROOT/'qwen32_inputs.json').read_text())
    ledgerpath=ROOT/'ledger.json';ledger=json.loads(ledgerpath.read_text()) if ledgerpath.exists() else dict(cap_usd=12.,rate_per_hour=8.,attempts=[])
    for tag in ['qwen32','olmo31']:
        local=ROOT/tag;local.mkdir(exist_ok=True)
        if (local/'report.json').exists() and json.loads((local/'report.json').read_text()).get('status')=='complete':continue
        if any(a['status']=='reserved' for a in ledger['attempts']):raise RuntimeError('Unreconciled worker reservation')
        spent=sum(a.get('seconds',0)+60 for a in ledger['attempts'])/3600*8
        assert spent+(m['max_worker_seconds']+60)/3600*8<=ledger['cap_usd']
        item=dict(tag=tag,status='reserved',seconds=m['max_worker_seconds']);ledger['attempts'].append(item);ledgerpath.write_text(json.dumps(ledger,indent=2))
        try:
            report=measure.remote(m,q if tag=='qwen32' else [],tag)
            item.update(status=report['status'],seconds=report['seconds']);ledgerpath.write_text(json.dumps(ledger,indent=2))
        finally:
            remote=m['manifest_sha256']+'/'+tag
            try:
                for entry in volume.listdir(remote):
                    if entry.path.endswith(('_stats.jsonl','report.json','olmo31_outputs.jsonl')):
                        (local/Path(entry.path).name).write_bytes(b''.join(volume.read_file(entry.path)))
                rp=json.loads((local/'report.json').read_text());item.update(status=rp['status'],seconds=rp['seconds']);ledgerpath.write_text(json.dumps(ledger,indent=2))
            except Exception as e:print('Collection incomplete:',type(e).__name__,flush=True)
        print('Finished',tag,report,flush=True)
