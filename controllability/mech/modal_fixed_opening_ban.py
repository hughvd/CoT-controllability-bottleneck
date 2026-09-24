"""Exact full-vocabulary first-content-token bans on bounded Modal workers."""
from pathlib import Path
import json
import time
import modal

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent/'results/fixed_opening_ban/v1'
image=(modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04',add_python='3.12').entrypoint([])
 .pip_install('vllm==0.23.0','transformers==5.12.1','huggingface_hub')
 .env({'HF_HOME':'/cache','HF_HUB_DISABLE_XET':'1','TOKENIZERS_PARALLELISM':'false','VLLM_USE_FLASHINFER_SAMPLER':'0'}))
app=modal.App('fixed-opening-ban-v1')
cache=modal.Volume.from_name('ban-sweep-hf-cache')
out=modal.Volume.from_name('fixed-opening-ban-out',create_if_missing=True)

@app.function(image=image,cpu=4,memory=16384,timeout=3600,volumes={'/cache':cache})
def warm(spec):
 from huggingface_hub import snapshot_download
 path=snapshot_download(spec['id'],revision=spec['revision'],allow_patterns=['*.json','*.safetensors','*.jinja','*.model','merges.txt','vocab.json'])
 cache.commit();return path

@app.function(image=image,gpu='B200',cpu=8,memory=131072,timeout=5400,retries=0,max_containers=1,scaledown_window=2,
 volumes={'/cache':cache.read_only(),'/out':out})
def generate(m):
 import hashlib,shutil
 from concurrent.futures import ThreadPoolExecutor
 import torch,vllm,transformers
 from vllm import LLM,SamplingParams
 start=time.monotonic();root=Path('/out')/m['name']/m['manifest_sha256'];root.mkdir(parents=True,exist_ok=True)
 report=dict(status='starting',started_at=time.time(),outputs=0,batches=[],versions=dict(vllm=vllm.__version__,torch=torch.__version__,transformers=transformers.__version__),gpu=torch.cuda.get_device_name())
 def write(path,obj):
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False));tmp.replace(path)
 def commit():
  report['elapsed_seconds']=time.monotonic()-start;write(root/'report.json',report);out.commit()
 write(root/'manifest.json',m);commit()
 try:
  snap=Path('/cache/hub')/('models--'+m['model']['id'].replace('/','--'))/'snapshots'/m['model']['revision']
  assert hashlib.sha256((snap/'tokenizer.json').read_bytes()).hexdigest()==m['tokenizer_sha256']
  local=Path('/tmp/model');local.mkdir(exist_ok=True)
  with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(lambda p:shutil.copyfile(p,local/p.name),[p for p in snap.iterdir() if p.is_file()]))
  llm=LLM(model=str(local),tokenizer=str(local),dtype='bfloat16',max_model_len=16384,max_num_seqs=128,
   max_num_batched_tokens=8192,gpu_memory_utilization=.92,enable_prefix_caching=True,seed=20260918)
  tok=llm.get_tokenizer()
  # Direct SamplingParams preserves -inf (from_optional would clamp to -100).
  # Mask two logits, retaining the entire remaining vocabulary. The v0.23 GPU
  # runner limits allowed_token_ids to 1024, so a complement list is unsuitable.
  banned=set(m['ban_token_ids']);bias={i:float('-inf') for i in banned}
  import msgspec
  check=SamplingParams(logit_bias=bias)
  assert msgspec.msgpack.decode(msgspec.msgpack.encode(check),type=SamplingParams).logit_bias==bias
  for word,i in zip(m['model']['ban'],m['ban_token_ids']):assert tok.encode(word,add_special_tokens=False)==[i]
  done=set()
  for fp in root.glob('batch_*.json'):
   for r in json.loads(fp.read_text())['outputs']:assert r['request_id'] not in done;done.add(r['request_id'])
  pending=[r for r in m['requests'] if r['request_id'] not in done]
  report['outputs']=len(done)
  # Evenly interleave cells/arms for the first reusable pilot batch.
  pending.sort(key=lambda r:hashlib.sha256(r['request_id'].encode()).hexdigest())
  cells={c['cell_id']:c for c in m['cells']}
  for b in range(0,len(pending),96):
   if time.monotonic()-start>4800:report['status']='budget_stop';break
   rows=pending[b:b+96];prefix={r['request_id']:[] for r in rows};active=list(rows)
   for step in range(16):
    if not active:break
    pars=[SamplingParams(temperature=.6,top_p=1.,top_k=-1,max_tokens=1,seed=(r['seed']+step)%2**31,
      logit_bias=bias if r['arm']=='fixed_ban' else None,skip_special_tokens=False) for r in active]
    first=llm.generate([{'prompt_token_ids':r['prompt_token_ids']+prefix[r['request_id']]} for r in active],pars,use_tqdm=False)
    again=[]
    for r,o in zip(active,first):
     ids=list(o.outputs[0].token_ids);assert len(ids)==1,(r['request_id'],o)
     if r['arm']=='fixed_ban':assert ids[0] not in banned,dict(request_id=r['request_id'],step=step,ids=ids,bias=repr(bias))
     prefix[r['request_id']]+=ids
     if not tok.decode(prefix[r['request_id']],skip_special_tokens=False).strip():again.append(r)
    active=again
   assert not active,'More than 16 whitespace tokens before first content'
   pars=[SamplingParams(temperature=.6,top_p=1.,top_k=-1,max_tokens=8192-len(prefix[r['request_id']]),
     seed=(r['seed']+100003)%2**31,skip_special_tokens=False) for r in rows]
   generated=llm.generate([{'prompt_token_ids':r['prompt_token_ids']+prefix[r['request_id']]} for r in rows],pars,use_tqdm=False)
   records=[]
   for r,o in zip(rows,generated):
    x=o.outputs[0];pre=prefix[r['request_id']];ids=pre+list(x.token_ids);text=tok.decode(ids,skip_special_tokens=False)
    records.append(dict(request_id=r['request_id'],prompt_token_ids=r['prompt_token_ids'],first_stage_ids=pre,
     first_token_text=tok.decode(pre,skip_special_tokens=False),output_token_ids=ids,raw_text=text,
     finish_reason=x.finish_reason,stop_reason=x.stop_reason,manifest_sha256=m['manifest_sha256']))
   bid=hashlib.sha256(''.join(r['request_id'] for r in rows).encode()).hexdigest()[:16]
   write(root/f'batch_{bid}.json',dict(outputs=records,manifest_sha256=m['manifest_sha256']))
   done.update(r['request_id'] for r in records)
   report.update(status='generating',outputs=len(done));report['batches'].append(dict(batch=bid,count=len(records),capped=sum(r['finish_reason']=='length' for r in records)))
   commit();print(m['name'],len(done),'/',len(m['requests']),flush=True)
   if b==0:
    report['projected_total_seconds']=(time.monotonic()-start)*len(pending)/len(rows)
    # Startup dominates this conservative estimate; bounded worker timeout remains definitive.
  else:report['status']='complete'
 except Exception as e:
  report.update(status='failed',error_type=type(e).__name__,error=str(e)[:1500]);raise
 finally:commit()
 return report

@app.local_entrypoint()
def main(phase:str='status',model:str='qwen32',experiment_root:str=''):
 root=Path(experiment_root).resolve() if experiment_root else ROOT
 m=json.loads((root/model/'manifest.json').read_text());remote=model+'/'+m['manifest_sha256']
 if phase=='cache':print(warm.remote(m['model']));return
 if phase=='dispatch':
  p=root/model/'dispatch.json'
  if p.exists():raise RuntimeError('Dispatch already recorded; inspect existing call before resuming: '+str(p))
  call=generate.spawn(m);p.write_text(json.dumps(dict(call_id=call.object_id,at=time.time(),timeout_seconds=5400,maximum_gpu_estimate_usd=12.),indent=2));print(call.object_id);return
 if phase in ['collect','status']:
  if phase=='collect':
   dest=root/model/'remote';dest.mkdir(exist_ok=True)
   for e in out.iterdir(remote):
    name=Path(e.path).name
    if not name.endswith('.json'):continue
    if name.startswith('batch_') and (dest/name).exists():continue
    (dest/name).write_bytes(b''.join(out.read_file(remote+'/'+name)))
  print(b''.join(out.read_file(remote+'/report.json')).decode());return
 raise ValueError(phase)
