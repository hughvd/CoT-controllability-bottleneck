"""Sequential bounded Modal workers for the full token-prefill suite."""
from pathlib import Path
import json
import time
import modal

HERE=Path(__file__).resolve().parent
ROOT=HERE/'results/token_prefill/full_v2'
image=(modal.Image.from_registry('nvidia/cuda:13.0.2-devel-ubuntu24.04',add_python='3.12').entrypoint([])
 .pip_install('vllm==0.23.0','transformers==5.12.1','huggingface_hub','openai-harmony==0.0.8')
 .env({'HF_HOME':'/cache','HF_HUB_DISABLE_XET':'1','TOKENIZERS_PARALLELISM':'false',
       'VLLM_USE_FLASHINFER_SAMPLER':'0','VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8':'1'})
 .add_local_file(HERE/'token_prefill.py','/root/token_prefill.py')
 .add_local_file(HERE/'token_prefill_suite.py','/root/token_prefill_suite.py')
 .add_local_file(HERE/'token_prefill_olmo.py','/root/token_prefill_olmo.py')
 .add_local_file(HERE/'harmony_backend.py','/root/harmony_backend.py'))
app=modal.App('full-token-prefill')
cache=modal.Volume.from_name('ban-sweep-hf-cache')
out=modal.Volume.from_name('token-prefill-full-out',create_if_missing=True)
old=modal.Volume.from_name('cot-token-prefill-out')
TERMINAL={'complete','failed','budget_stop','pilot_quality_stop','projection_exceeds_budget','runtime_rollover'}

@app.function(image=image,cpu=4,memory=16384,timeout=3600,volumes={'/cache':cache})
def warm_cache(spec):
 from huggingface_hub import snapshot_download
 p=snapshot_download(spec['id'],revision=spec['revision'],allow_patterns=['*.json','*.safetensors','*.jinja','*.tiktoken','*.model','merges.txt','vocab.json'],ignore_patterns=['original/*','metal/*'])
 cache.commit();return {'snapshot':p}

@app.function(image=image,gpu='B200',cpu=8,memory=131072,timeout=10800,retries=0,
 max_containers=1,scaledown_window=2,volumes={'/cache':cache.read_only(),'/out':out,'/old':old.read_only()})
def generate(m,budget_seconds,attempt_id,continue_until_complete=False):
 import sys,shutil
 from concurrent.futures import ThreadPoolExecutor
 sys.path.insert(0,'/root')
 import token_prefill_suite as s
 import token_prefill as b
 if m['model']['adapter']=='olmo':
  import token_prefill_olmo as extension
  assert b.file_hash(extension.__file__)==m['adapter_sha256'];extension.activate()
 import vllm,torch,transformers
 from vllm import LLM,SamplingParams
 started=time.monotonic();s.validate(m)
 assert b.file_hash(s.__file__)==m['code_sha256'] and b.file_hash(b.__file__)==m['base_code_sha256']
 folder=Path('/out')/m['name']/m['manifest_sha256'];folder.mkdir(parents=True,exist_ok=True)
 b.immutable_json(folder/'manifest.json',m)
 report=dict(status='starting',attempt_id=attempt_id,manifest_sha256=m['manifest_sha256'],started_at=time.time(),
  runner_sha256=b.file_hash(__file__),batches=[],gpu=torch.cuda.get_device_name(),vllm=vllm.__version__,torch=torch.__version__,transformers=transformers.__version__)
 def commit():
  report['elapsed_seconds']=time.monotonic()-started;b.atomic_json(folder/'report.json',report);out.commit()
 commit()
 try:
  known={r['request_id']:r for r in m['requests']}
  if m['name']=='qwen32' and not (folder/'batch_reused_qwen32.json').exists():
   oldroot=Path('/old/qwen32_v1/33536ce1bc973f89145f7c8ccf96a95f05fede999bf98a11ae962a38f2104f3b')
   om=json.loads((oldroot/'manifest.json').read_text());b.validate_manifest(om)
   assert om['model']['revision']==m['model']['revision'] and om['sampling']==m['sampling']
   for r in om['requests']:
    assert r['request_id'] in known and r['prompt_token_ids']==known[r['request_id']]['prompt_token_ids']
    assert r['sampling_seed']==known[r['request_id']]['sampling_seed']
   outputs=[]
   for p in sorted(oldroot.glob('batch_*.json')):outputs.extend(json.loads(p.read_text())['outputs'])
   assert len(outputs)==4800
   b.immutable_json(folder/'batch_reused_qwen32.json',dict(manifest_sha256=m['manifest_sha256'],outputs=outputs,source_manifest=om['manifest_sha256']))
   out.commit()
  done=set()
  for p in folder.glob('batch_*.json'):
   batch=json.loads(p.read_text());assert batch['manifest_sha256']==m['manifest_sha256']
   for r in batch['outputs']:
    key=(r['request_id'],r['sample']);assert key not in done and r['prompt_token_ids']==known[r['request_id']]['prompt_token_ids'];done.add(key)
  report['reused_outputs']=len(done)
  pending=[]
  for r in m['requests']:
   count=sum((r['request_id'],i) in done for i in range(5));assert count in [0,5]
   if not count:pending.append(r)
  if not pending:report['status']='complete';return report
  spec=m['model'];snapshot=Path('/cache/hub')/('models--'+spec['id'].replace('/','--'))/'snapshots'/spec['revision']
  for f,h in m['tokenizer_hashes'].items():assert b.file_hash(snapshot/f)==h
  local=Path('/tmp/full-token-model');local.mkdir(exist_ok=True)
  def copy(src):
   with src.open('rb') as a,(local/src.name).open('wb') as dest:shutil.copyfileobj(a,dest,16*1024*1024)
  with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(copy,[p for p in snapshot.iterdir() if p.is_file()]))
  print('Weights staged',flush=True)
  llm=LLM(model=str(local),tokenizer=str(local),dtype=spec['dtype'],max_model_len=16384,max_num_seqs=128,
    max_num_batched_tokens=8192,gpu_memory_utilization=.92,enable_prefix_caching=True,enforce_eager=False,seed=20260915)
  tok=llm.get_tokenizer()
  if spec['adapter']=='harmony':
   import harmony_backend as h
   assert b.file_hash(h.__file__)==m['harmony_sha256']
   for token in ['<|start|>','<|channel|>','<|message|>','<|end|>','<|return|>']:
    assert tok.encode(token,add_special_tokens=False)==h._enc().encode(token,allowed_special='all')
  else:
   for r in pending[:4]:
    if spec['adapter']=='olmo':assert extension.opening(tok,r['user_text'])==r['opening_token_ids']
    else:
     text=tok.apply_chat_template([{'role':'user','content':r['user_text']}],tokenize=False,add_generation_prompt=True,enable_thinking=True)+'<think>\n'
     assert tok.encode(text,add_special_tokens=False)==r['opening_token_ids']
  cells={c['cell_id']:c for c in m['cells']};pending.sort(key=lambda r:r['request_id']);pilot=[];selected=set()
  for behavior in m['behaviors']:
   for condition in s.CONDITIONS:
    for k in [0,1,64]:
     candidates=[r for r in pending if cells[r['cell_id']]['behavior']==behavior and r['condition']==condition and r['prefix_tokens']==k]
     if candidates:pilot.append(candidates[0]);selected.add(candidates[0]['request_id'])
  rest=[r for r in pending if r['request_id'] not in selected];need=min(128-len(pilot),len(rest))
  selected.update(rest[i*len(rest)//need]['request_id'] for i in range(need))
  pilot=[r for r in pending if r['request_id'] in selected];rest=[r for r in pending if r['request_id'] not in selected]
  report['pilot_request_ids']=[r['request_id'] for r in pilot];report['startup_seconds']=time.monotonic()-started
  batches=[pilot]+[rest[i:i+128] for i in range(0,len(rest),128)];rate=0
  for i,rows in enumerate(batches):
   remaining=budget_seconds-(time.monotonic()-started)
   if remaining<max(180,rate*len(rows)*1.4+60):
    report['status']='runtime_rollover' if continue_until_complete else 'budget_stop';break
   params=[SamplingParams(**m['sampling'],seed=r['sampling_seed'],skip_special_tokens=False,
          **({'stop_token_ids':m['protocol']['eos_ids']} if spec['adapter']=='harmony' else {})) for r in rows]
   t=time.monotonic();generated=llm.generate([{'prompt_token_ids':r['prompt_token_ids']} for r in rows],params,use_tqdm=False)
   records=[];empty=0;capped=0
   for request,output in zip(rows,generated):
    assert list(output.prompt_token_ids)==request['prompt_token_ids'] and len(output.outputs)==5
    for j,x in enumerate(output.outputs):
     raw=dict(request_id=request['request_id'],sample=j,prompt_token_ids=request['prompt_token_ids'],
      output_token_ids=list(x.token_ids),raw_text=x.text,finish_reason=x.finish_reason,stop_reason=x.stop_reason)
     text,_,_,_=s.reasoning_parts(tok,request,raw,m);empty+=not text.strip();capped+=x.finish_reason=='length';records.append(raw)
   bid=b.digest([r['request_id'] for r in rows]);elapsed=time.monotonic()-t
   b.immutable_json(folder/f'batch_{bid}.json',dict(manifest_sha256=m['manifest_sha256'],outputs=records))
   info=dict(batch=bid,outputs=len(records),seconds=elapsed,empty=empty,truncated=capped)
   report['batches'].append(info);report['status']='generating';commit();print(info,flush=True)
   rate=max(rate,elapsed/len(rows))
   if i==0:
    if empty/len(records)>.15 or capped/len(records)>.15:report['status']='pilot_quality_stop';break
    projection=rate*len(rest)*1.5+180;report['projected_remaining_seconds']=projection
    if not continue_until_complete and projection>budget_seconds-(time.monotonic()-started):report['status']='projection_exceeds_budget';break
  else:report['status']='complete'
 except Exception as e:
  report['status']='failed';report['error_type']=type(e).__name__;raise
 finally:commit()
 return report

@app.local_entrypoint()
def main(phase:str='status',model:str='qwen32'):
 import sys
 sys.path.insert(0,str(HERE));import token_prefill_suite as s;import token_prefill as b
 root=ROOT
 if model=='olmo31':
  import token_prefill_olmo as extension
  extension.activate();root=extension.ROOT
 if model not in ['qwen32','qwen14','gptoss','olmo31']:raise ValueError('Modal model required')
 if phase=='cache':print(warm_cache.remote(s.MODELS[model]));return
 m=json.loads((root/model/'manifest.json').read_text());s.validate(m)
 remote=f'{model}/{m["manifest_sha256"]}';ledger_path=root/'gpu_ledger.json'
 if phase in ['status','collect']:
  report=json.loads(b''.join(out.read_file(remote+'/report.json')))
  if phase=='collect':
   dest=root/model/'remote';dest.mkdir(exist_ok=True)
   for e in out.iterdir(remote):
    name=Path(e.path).name
    if not name.endswith('.json'):continue
    path=dest/name
    if name.startswith('batch_') and path.exists():continue
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:
     for chunk in out.read_file(e.path):f.write(chunk)
    tmp.replace(path)
   if report['status'] in TERMINAL:
    ledger=json.loads(ledger_path.read_text());entry=next(a for a in ledger['attempts'] if a['attempt_id']==report['attempt_id'])
    entry.update(status=report['status'],accounted_seconds=report['elapsed_seconds']+90);b.atomic_json(ledger_path,ledger)
  print({k:v for k,v in report.items() if k not in ['pilot_request_ids','batches']});return
 if phase!='dispatch':raise ValueError(phase)
 scope=json.loads((root/'scope.json').read_text())
 if model not in scope['active_models']:raise ValueError('Model deferred by user')
 assert b.file_hash(s.__file__)==m['code_sha256']
 ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(max_estimated_usd=90.,usd_per_hour=8.,attempts=[])
 if any(a['status'] in ['reserved','spawned'] for a in ledger['attempts']):raise ValueError('Another worker has an active reservation')
 if any(a['model']==model and a['status']=='complete' for a in ledger['attempts']):print('Already complete');return
 spent=sum(a['accounted_seconds']/3600*8 for a in ledger['attempts'])
 policy_path=root/'execution_policy.json'
 continuous=policy_path.exists() and json.loads(policy_path.read_text()).get('continue_until_complete',False)
 # Modal requires a finite per-call timeout. In continuous mode this is a
 # renewable platform lease, not an experiment runtime or spending limit.
 allowed=86400 if continuous else min(10800,(ledger['max_estimated_usd']-spent)/8*3600)
 if allowed<600:raise ValueError('Shared Modal budget exhausted')
 attempt=dict(attempt_id=b.digest([model,time.time()]),model=model,status='reserved',reserved_seconds=allowed,started_at=time.time())
 ledger['attempts'].append(attempt);b.atomic_json(ledger_path,ledger)
 attempt['continue_until_complete']=continuous
 call=generate.with_options(timeout=int(allowed)).spawn(m,allowed-300 if continuous else allowed-90,attempt['attempt_id'],continuous)
 attempt.update(status='spawned',call_id=call.object_id);b.atomic_json(ledger_path,ledger);print('Dispatched',model,call.object_id)
