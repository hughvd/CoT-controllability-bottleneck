"""Versioned fixed-opening-token ban experiment; local preparation and helpers."""
from pathlib import Path
import argparse
import json
import sys

HERE=Path(__file__).resolve().parent
CTRL=HERE.parent
sys.path.insert(0,str(CTRL))
import token_prefill as base
import token_prefill_suite as suite

ROOT=CTRL/'results/fixed_opening_ban/v1'
BEHAVIORS=['uppercase_thinking','leetspeak','spanish','omit_word']
CONDITIONS=['no_instruction','instruction_only']
ARMS=['baseline','fixed_ban']
SAMPLES=3
MODELS={
 'qwen32':dict(id='Qwen/Qwen3-32B',revision='9216db5781bf21249d130ec9da846c4624c16137',ban=['Okay','Alright'],backend='modal'),
 'olmo31':dict(id='allenai/Olmo-3.1-32B-Think',revision='832c3f543499af8fe68b88359501de9cb7840544',ban=['Okay','Alright'],backend='modal'),
 'glm':dict(**suite.MODELS['glm'],ban=['The']),
}

def prepare():
 from transformers import AutoTokenizer
 from tokenizers import Tokenizer
 import prefill_experiment as p
 source=CTRL/'results/token_prefill/full_v2/qwen32/manifest.json'
 sm=json.loads(source.read_text())
 cells=[{k:v for k,v in c.items() if k not in ['controlled_text','controlled_token_ids']} for c in sm['cells'] if c['behavior'] in BEHAVIORS]
 assert len(cells)==120 and all(sum(c['behavior']==b for c in cells)==30 for b in BEHAVIORS)
 for name,spec in MODELS.items():
  snap=Path.home()/'.cache/huggingface/hub'/('models--'+spec['id'].replace('/','--'))/'snapshots'/spec['revision']
  tok=AutoTokenizer.from_pretrained(snap,local_files_only=True) if name!='glm' else suite.JsonTokenizer(snap)
  bids=[]
  for word in spec['ban']:
   ids=tok.encode(word,add_special_tokens=False);assert len(ids)==1,(word,ids);bids+=ids
  requests=[]
  for c in cells:
   for cond in CONDITIONS:
    user=p.assemble_prompt(c['question'],c['options'],c['instruction'] if cond=='instruction_only' else '')
    messages=[{'role':'user','content':user}]
    if name=='glm':
     messages.append({'role':'assistant','content':'<think>\n'});ids=None
    else:
     text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
     if name=='qwen32':text+='<think>\n'
     else:assert text.endswith('<think>'),repr(text[-80:])
     ids=tok.encode(text,add_special_tokens=False)
    for arm in ARMS:
     for sample in range(SAMPLES):
      rid=base.digest([name,c['cell_id'],cond,arm,sample])
      requests.append(dict(request_id=rid,cell_id=c['cell_id'],condition=cond,arm=arm,sample=sample,
       messages=messages,prompt_token_ids=ids,seed=int(base.digest([c['cell_id'],cond,sample])[:8],16)%2**31))
  m=dict(version='fixed_opening_ban_v1',name=name,model=spec,ban_token_ids=bids,cells=cells,requests=requests,
   sampling=dict(temperature=.6,top_p=1.,top_k=-1,max_tokens=8192),source_manifest_sha256=base.file_hash(source),
   tokenizer_sha256=base.file_hash(snap/'tokenizer.json'),
   protocol='Ban exact native token IDs only until first non-whitespace reasoning token; then remove mask. Both arms split generation at the first content token.',
   scope=dict(behaviors=BEHAVIORS,questions_per_behavior=30,samples=SAMPLES,conditions=CONDITIONS,arms=ARMS))
  m['manifest_sha256']=base.digest(m)
  base.immutable_json(ROOT/name/'manifest.json',m)
  print(name,len(requests),'outputs',bids,flush=True)

def validate(m):
 assert base.digest({k:v for k,v in m.items() if k!='manifest_sha256'})==m['manifest_sha256']
 scope=m['scope']
 expected_cells=len(scope['behaviors'])*scope['questions_per_behavior']
 expected=expected_cells*scope['samples']*len(scope['conditions'])*len(scope['arms'])
 assert len(m['cells'])==expected_cells
 assert len({c['cell_id'] for c in m['cells']})==expected_cells
 assert len(m['requests'])==expected and len({r['request_id'] for r in m['requests']})==expected

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare']);a=ap.parse_args();prepare()
