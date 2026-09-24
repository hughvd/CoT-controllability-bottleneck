"""OLMo extension preserving the completed full_v2 suite and its pinned core."""
import argparse,asyncio,copy,json,sys
from pathlib import Path
import token_prefill_suite as s
ROOT=s.HERE/'results/token_prefill/olmo31_v1'
SPEC=dict(id='allenai/Olmo-3.1-32B-Think',revision='832c3f543499af8fe68b88359501de9cb7840544',adapter='olmo',backend='modal',dtype='bfloat16')
ORIGINAL_PARTS=s.reasoning_parts

def opening(tok,user):
 text=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
 assert text.endswith('<think>') and not text.endswith('<think><think>')
 return tok.encode(text,add_special_tokens=False)

def reasoning_parts(tok,request,raw,m):
 if m['model']['adapter']!='olmo':return ORIGINAL_PARTS(tok,request,raw,m)
 ids=raw['output_token_ids'];seed=request['prefix_token_ids'];eos=set(m['protocol']['eos_ids'])
 end=next((i for i,t in enumerate(ids) if t in eos),len(ids));ids=ids[:end]
 full=s.base.continuation_text(tok,seed,ids);close=full.find('</think>');closed=close>=0
 head=s.base.continuation_text(tok,seed,ids[:32])
 if not closed:return full,head,len(ids),False
 # Native token count, including a token that contains both reasoning and the
 # opening characters of the delimiter. Never retokenize the continuation.
 low,high=0,len(ids)
 while low<high:
  mid=(low+high)//2
  if len(s.base.continuation_text(tok,seed,ids[:mid]))>=close:high=mid
  else:low=mid+1
 return full[:close],head[:close],low,True

def activate():
 s.MODELS['olmo31']=SPEC;s.ROOT=ROOT;s.reasoning_parts=reasoning_parts

def prepare():
 import prefill_experiment as p
 activate();parent=s.HERE/'results/token_prefill/full_v2/glm/manifest.json';source=json.loads(parent.read_text());s.validate(source)
 tok=s.tokenizer(SPEC);cells=copy.deepcopy(source['cells']);requests={};links=[]
 for c in cells:
  c['controlled_token_ids']=tok.encode(c['controlled_text'],add_special_tokens=False)
  assert len(c['controlled_token_ids'])>=64 and s.base.decode(tok,c['controlled_token_ids'])==c['controlled_text']
  for condition in s.CONDITIONS:
   instruction=c['instruction'] if condition=='prefill_plus_instruction' else ''
   user=p.assemble_prompt(c['question'],c['options'],instruction);start=opening(tok,user)
   for k in s.LENGTHS:
    ids=c['controlled_token_ids'][:k];seed=s.base.decode(tok,ids);prompt=start+ids;rid=s.base.digest(prompt)
    r=dict(request_id=rid,cell_id=c['cell_id'],condition=condition,prefix_tokens=k,opening_token_ids=start,prompt_token_ids=prompt,prefix_token_ids=ids,seed=seed,user_text=user,messages=None,seed_roundtrip='\ufffd' not in seed and tok.encode(seed,add_special_tokens=False)==ids,sampling_seed=int(rid[:8],16))
    requests.setdefault(rid,r);links.append(dict(request_id=rid,cell_id=c['cell_id'],condition=condition,prefix_tokens=k))
 snapshot=Path.home()/'.cache/huggingface/hub'/('models--'+SPEC['id'].replace('/','--'))/'snapshots'/SPEC['revision']
 m=dict(version='olmo_token_prefill_v1',name='olmo31',model=SPEC,lengths=s.LENGTHS,behaviors=s.BEHAVIORS,sampling=source['sampling'],max_model_len=16384,cells=cells,requests=list(requests.values()),links=links,protocol=dict(think_close_text='</think>',think_close_ids=tok.encode('</think>',add_special_tokens=False),eos_ids=[tok.eos_token_id]),source_hashes={str(parent.relative_to(s.HERE)):s.base.file_hash(parent),'prefill_experiment.py':s.base.file_hash(p.__file__)},tokenizer_hashes={f.name:s.base.file_hash(f) for f in snapshot.iterdir() if f.name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja']},code_sha256=s.base.file_hash(s.__file__),base_code_sha256=s.base.file_hash(s.base.__file__),harmony_sha256=source['harmony_sha256'],adapter_sha256=s.base.file_hash(__file__),api_prefix_issues=[],source_replacements=source['source_replacements'],parent_manifest_sha256=source['manifest_sha256'])
 m['manifest_sha256']=s.base.digest(m);s.validate(m);s.base.immutable_json(ROOT/'olmo31/manifest.json',m)
 s.base.immutable_json(ROOT/'scope.json',dict(active_models=['olmo31'],deferred_models=['deepseek','gptoss'],api_estimate_cap_usd=10))
 if not (ROOT/'gpu_ledger.json').exists():s.base.atomic_json(ROOT/'gpu_ledger.json',dict(max_estimated_usd=32.,usd_per_hour=8.,attempts=[]))
 for f in [Path(__file__),Path(s.__file__),Path(s.base.__file__)]:
  dest=ROOT/'code'/(f.stem+'_'+s.base.file_hash(f)+'.py');dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():dest.write_bytes(f.read_bytes())
 print('Prepared',len(requests),'unique prompts,',len(links)*5,'observations;',sum(not r['seed_roundtrip'] for r in requests.values()),'partial-byte prefixes retained exactly')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','score']);ap.add_argument('--partial',action='store_true');args=ap.parse_args()
 if args.phase=='prepare':prepare()
 else:
  activate();m=json.loads((ROOT/'olmo31/manifest.json').read_text());assert s.base.file_hash(__file__)==m['adapter_sha256']
  import token_prefill_suite_score as sc
  sc.measure('olmo31');asyncio.run(sc.judge('olmo31'));sc.export('olmo31',partial=args.partial)
