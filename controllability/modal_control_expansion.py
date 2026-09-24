"""Resumable local generation for the matched two-task paper extension."""
from pathlib import Path
import json
import time
import modal

ROOT=Path(__file__).resolve().parent/'results/control_expansion/v1'
image=(modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04',add_python='3.12').entrypoint([])
 .pip_install('vllm==0.23.0','transformers==5.12.1','huggingface_hub')
 .env({'HF_HOME':'/cache','HF_HUB_DISABLE_XET':'1','TOKENIZERS_PARALLELISM':'false','VLLM_USE_FLASHINFER_SAMPLER':'0'}))
app=modal.App('control-addition-expansion-v1')
cache=modal.Volume.from_name('ban-sweep-hf-cache')
out=modal.Volume.from_name('control-addition-expansion-out',create_if_missing=True)

@app.function(image=image,gpu='B200',cpu=8,memory=131072,timeout=10800,retries=0,max_containers=3,scaledown_window=2,
 volumes={'/cache':cache.read_only(),'/out':out})
def generate(m):
 import hashlib,shutil
 from concurrent.futures import ThreadPoolExecutor
 import torch,vllm,transformers
 from vllm import LLM,SamplingParams
 started=time.monotonic();root=Path('/out')/m['name']/m['manifest_sha256'];root.mkdir(parents=True,exist_ok=True)
 report=dict(status='starting',started_at=time.time(),outputs=0,versions=dict(vllm=vllm.__version__,torch=torch.__version__,transformers=transformers.__version__),gpu=torch.cuda.get_device_name())
 def write(path,obj):
  temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False));temp.replace(path)
 def commit():
  report['elapsed_seconds']=time.monotonic()-started;write(root/'report.json',report);out.commit()
 write(root/'manifest.json',m);commit()
 try:
  requests=[r for r in m['requests'] if m['name']=='olmo31' or r['experiment']!='icl']
  report['expected']=len(requests)
  done=set()
  for path in root.glob('batch_*.json'):
   for r in json.loads(path.read_text())['outputs']:
    assert r['request_id'] not in done;done.add(r['request_id'])
  pending=[r for r in requests if r['request_id'] not in done]
  report['outputs']=len(done)
  if not pending:report['status']='complete';return report
  snap=Path('/cache/hub')/('models--'+m['model']['id'].replace('/','--'))/'snapshots'/m['model']['revision']
  local=Path('/tmp/model');local.mkdir(exist_ok=True)
  def copy(p):
   # Explicit streaming avoids cross-filesystem copy_file_range/sendfile EINVAL
   # on the read-only model-cache mount.
   with p.open('rb') as src,(local/p.name).open('wb') as dst:shutil.copyfileobj(src,dst,16*1024*1024)
  with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(copy,[p for p in snap.iterdir() if p.is_file()]))
  llm=LLM(model=str(local),tokenizer=str(local),dtype='bfloat16',max_model_len=16384,max_num_seqs=128,
   max_num_batched_tokens=8192,gpu_memory_utilization=.92,enable_prefix_caching=True,seed=20260915)
  tok=llm.get_tokenizer();bias={i:float('-inf') for i in m['ban_token_ids']}
  # Smoke all experiment/condition/control classes before filling large batches.
  ordered=sorted(pending,key=lambda r:hashlib.sha256(r['request_id'].encode()).hexdigest());smoke=[];classes=set()
  for r in ordered:
   key=(r['experiment'],r['behavior'],r['condition'],r['arm'],r['prefix_tokens'] in [1,64])
   if key not in classes:classes.add(key);smoke.append(r)
  smoke_ids={r['request_id'] for r in smoke};rest=[r for r in ordered if r['request_id'] not in smoke_ids]
  batch_size=384
  report['batch_size']=batch_size
  batches=[smoke]+[rest[i:i+batch_size] for i in range(0,len(rest),batch_size)]
  for rows in batches:
   if time.monotonic()-started>9600:report['status']='runtime_rollover';break
   prefix={r['request_id']:[] for r in rows};active=[r for r in rows if r['experiment']=='ban_icl']
   for step in range(16):
    if not active:break
    pars=[SamplingParams(temperature=.6,top_p=1.,top_k=-1,max_tokens=1,seed=(r['seed']+step)%2**31,
      logit_bias=bias if r['arm']=='fixed_ban' else None,skip_special_tokens=False) for r in active]
    first=llm.generate([{'prompt_token_ids':r['prompt_token_ids']+prefix[r['request_id']]} for r in active],pars,use_tqdm=False)
    again=[]
    for r,o in zip(active,first):
     ids=list(o.outputs[0].token_ids);assert len(ids)==1
     if r['arm']=='fixed_ban':assert ids[0] not in m['ban_token_ids']
     prefix[r['request_id']]+=ids
     if not tok.decode(prefix[r['request_id']],skip_special_tokens=False).strip():again.append(r)
    active=again
   assert not active
   pars=[SamplingParams(temperature=.6,top_p=1.,top_k=-1,max_tokens=8192-len(prefix[r['request_id']]),
     seed=(r['seed']+(100003 if r['experiment']=='ban_icl' else 0))%2**31,skip_special_tokens=False) for r in rows]
   outputs=llm.generate([{'prompt_token_ids':r['prompt_token_ids']+prefix[r['request_id']]} for r in rows],pars,use_tqdm=False)
   records=[]
   for r,o in zip(rows,outputs):
    x=o.outputs[0];ids=prefix[r['request_id']]+list(x.token_ids)
    # Decode with prefix context so partial byte/codepoint boundaries are retained.
    seed_ids=r['prefix_token_ids'];seed_text=tok.decode(seed_ids,skip_special_tokens=False).rstrip('\ufffd')
    joint=tok.decode(seed_ids+ids,skip_special_tokens=False);assert joint.startswith(seed_text)
    records.append(dict(request_id=r['request_id'],output_token_ids=ids,first_stage_ids=prefix[r['request_id']],
      raw_text=joint[len(seed_text):],finish_reason=x.finish_reason,stop_reason=x.stop_reason,manifest_sha256=m['manifest_sha256']))
   bid=hashlib.sha256(''.join(r['request_id'] for r in rows).encode()).hexdigest()[:16]
   write(root/f'batch_{bid}.json',dict(outputs=records,manifest_sha256=m['manifest_sha256']))
   done.update(r['request_id'] for r in records);report.update(status='generating',outputs=len(done));commit()
   print(m['name'],len(done),'/',len(requests),flush=True)
  else:report['status']='complete'
 except Exception as exc:report.update(status='failed',error=repr(exc));raise
 finally:commit()
 return report

@app.local_entrypoint()
def main(phase:str='status',model:str='qwen32',resume:bool=False,root:str=''):
 folder=(Path(root) if root else ROOT)/model;m=json.loads((folder/'manifest.json').read_text());remote=model+'/'+m['manifest_sha256']
 if phase=='dispatch':
  p=folder/'dispatch.json'
  if p.exists() and not resume:raise RuntimeError('Existing dispatch; inspect before resume')
  if p.exists():
   report=json.loads(b''.join(out.read_file(remote+'/report.json')))
   assert report['status'] in ['runtime_rollover','failed','cancelled_before_generation','paused_for_batching']
   history=folder/'dispatch_history.jsonl'
   with history.open('a') as h:h.write(json.dumps(dict(previous=json.loads(p.read_text()),report=report))+'\n')
  call=generate.spawn(m);p.write_text(json.dumps(dict(call_id=call.object_id,at=time.time(),timeout_seconds=10800),indent=2));print(call.object_id);return
 if phase=='collect':
  dest=folder/'remote';dest.mkdir(exist_ok=True)
  for e in out.iterdir(remote):
   name=Path(e.path).name
   if not name.endswith('.json') or (name.startswith('batch_') and (dest/name).exists()):continue
   (dest/name).write_bytes(b''.join(out.read_file(remote+'/'+name)))
 print(b''.join(out.read_file(remote+'/report.json')).decode())
