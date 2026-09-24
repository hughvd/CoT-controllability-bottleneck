"""Prepare and measure matched opening distributions; short local/remote stages."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE/'data/matched_openings/v1'
TYPES=['uppercase_thinking','leetspeak','spanish','omit_word']


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def prepare():
    from transformers import AutoTokenizer
    import pandas as pd
    ctrl=HERE.parent
    root=ctrl/'results/token_prefill/full_v2/qwen32'
    source=json.loads((root/'manifest.json').read_text())
    cells={c['cell_id']:c for c in source['cells']}
    links=[l for l in source['links'] if l['prefix_tokens']==0 and cells[l['cell_id']]['behavior'] in TYPES]
    requests={r['request_id']:r for r in source['requests'] if r['request_id'] in {l['request_id'] for l in links}}
    raw=[]
    for fp in sorted((root/'remote').glob('batch_*.json')):
        for r in json.loads(fp.read_text())['outputs']:
            if r['request_id'] in requests:
                raw.append(dict(request_id=r['request_id'],sample=r['sample'],
                                output_token_ids=r['output_token_ids'][:128],
                                source_finish_reason=r['finish_reason']))
    assert len(raw)==len(requests)*5
    models=json.loads((HERE/'data/rank_runs/opening_ranks_v1/manifest.json').read_text())['models']
    olmo=models['olmo31'];cache=Path.home()/'.cache/huggingface/hub'/('models--'+olmo['model'].replace('/','--'))/'snapshots'/olmo['revision']
    tok=AutoTokenizer.from_pretrained(cache,local_files_only=True)
    qreq=[];oreq=[]
    for rid,r in sorted(requests.items()):
        user=r['user_text']
        text=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
        assert text.rstrip().endswith('<think>')
        qreq.append(dict(request_id=rid,user_text=user,prompt_token_ids=r['prompt_token_ids']))
        oreq.append(dict(request_id=rid,user_text=user,prompt_token_ids=tok.encode(text,add_special_tokens=False)))
    source_lookup={str(r['question']):r['dataset'] for r in pd.read_csv(ctrl/'data/training_raw.csv').to_dict('records')}
    selected=[dict(cell_id=c['cell_id'],behavior=c['behavior'],question=c['question'],question_id=c['question_id'],
                   source=source_lookup[c['question']],instruction=c['instruction']) for c in cells.values() if c['cell_id'] in {l['cell_id'] for l in links}]
    manifest=dict(version=1,models=models,cells=selected,links=links,requests={'qwen32':qreq,'olmo31':oreq},
        source_manifest_sha256=source['manifest_sha256'],sampling=dict(n=5,temperature=.6,top_p=1.,top_k=-1,max_tokens=128),
        entropy_positions=64,estimated_usd_cap=12.,max_worker_seconds=1800,
        note='Qwen exact existing rollouts rescored; OLMo new short rollouts. Same user text/questions/instructions; native chat templates. GLM existing shared cells used for concentration only.')
    manifest['manifest_sha256']=digest(manifest)
    ROOT.mkdir(parents=True,exist_ok=True)
    for name,data in [('manifest.json',manifest),('qwen32_inputs.json',raw)]:
        p=ROOT/name
        if p.exists():assert json.loads(p.read_text())==data,'Immutable prepared input changed'
        else:p.write_text(json.dumps(data,ensure_ascii=False))
    print('Prepared',len(selected),'control/question cells,',len(requests),'unique requests/model,',len(raw),'Qwen responses.')


def generate(model_dir,work):
    import torch
    from vllm import LLM,SamplingParams
    m=json.loads((work/'manifest.json').read_text());rs=m['requests']['olmo31']
    llm=LLM(model=str(model_dir),dtype='bfloat16',max_model_len=8192,max_num_seqs=128,
            gpu_memory_utilization=.88,enforce_eager=True,enable_prefix_caching=True)
    done={}
    fp=work/'olmo31_outputs.jsonl'
    if fp.exists():done={r['request_id']:1 for r in map(json.loads,fp.open())}
    for start in range(0,len(rs),64):
        batch=[r for r in rs[start:start+64] if r['request_id'] not in done]
        if not batch:continue
        pars=[SamplingParams(**m['sampling'],seed=int(digest(r['request_id'])[:8],16),skip_special_tokens=False) for r in batch]
        outs=llm.generate([{'prompt_token_ids':r['prompt_token_ids']} for r in batch],pars,use_tqdm=False)
        with fp.open('a') as f:
            for r,o in zip(batch,outs):
                for j,c in enumerate(o.outputs):
                    f.write(json.dumps(dict(request_id=r['request_id'],sample=j,output_token_ids=list(c.token_ids),source_finish_reason=c.finish_reason))+'\n')
        print('Generated',min(start+64,len(rs)),'/',len(rs),'prompts',flush=True)


def score(model_dir,work,tag):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    m=json.loads((work/'manifest.json').read_text());requests={r['request_id']:r for r in m['requests'][tag]}
    rows=json.loads((work/'qwen32_inputs.json').read_text()) if tag=='qwen32' else list(map(json.loads,(work/'olmo31_outputs.jsonl').open()))
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_dir,torch_dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval()
    stop={i for i in [tok.convert_tokens_to_ids('</think>'),tok.eos_token_id] if i is not None}
    jobs=[]
    for r in rows:
        ids=r['output_token_ids'];end=next((i for i,t in enumerate(ids) if t in stop),len(ids));start=0
        while start<end and (not tok.decode([ids[start]],skip_special_tokens=False).strip() or tok.decode([ids[start]],skip_special_tokens=False).strip()=='<think>'):start+=1
        g=min(end,start+m['entropy_positions'])
        jobs.append((r,requests[r['request_id']]['prompt_token_ids'],g,start,end))
    jobs.sort(key=lambda j:len(j[1])+j[2])
    out=work/(tag+'_stats.jsonl');known=set()
    if out.exists():known={(r['request_id'],r['sample']) for r in map(json.loads,out.open())}
    jobs=[j for j in jobs if (j[0]['request_id'],j[0]['sample']) not in known]
    with out.open('a') as f:
        for i in range(0,len(jobs),8):
            batch=jobs[i:i+8];n=max(len(p)+g for _,p,g,_,_ in batch)
            x=torch.full((len(batch),n),tok.pad_token_id or tok.eos_token_id,dtype=torch.long,device='cuda')
            mask=torch.zeros_like(x)
            for k,(r,p,g,_,_) in enumerate(batch):
                seq=p+r['output_token_ids'][:g];x[k,:len(seq)]=torch.tensor(seq,device='cuda');mask[k,:len(seq)]=1
            with torch.inference_mode():outp=model(input_ids=x,attention_mask=mask,use_cache=False)
            for k,(r,p,g,start,end) in enumerate(batch):
                logits=outp.logits[k,len(p)-1:len(p)-1+g].float()
                lp=torch.log_softmax(logits,dim=-1);pr=lp.exp();h=-(pr*lp).sum(-1)
                tp,ti=pr.topk(5,dim=-1)
                assert torch.isfinite(h).all() and len(h)==g
                result=dict(**r,content_start=start,content_end=end,entropy=h[start:g].cpu().tolist(),
                    top5_ids=ti[start:g].cpu().tolist(),top5_p=tp[start:g].cpu().tolist(),
                    content_token_ids=r['output_token_ids'][start:end])
                f.write(json.dumps(result)+'\n')
            del outp,x,mask,logits,lp,pr,h,tp,ti
            f.flush()
            if i%80==0:print(tag,'scored',min(i+8,len(jobs)),'/',len(jobs),flush=True)
    print(tag,'complete',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['prepare','generate','score'],default='prepare');ap.add_argument('--work',type=Path);ap.add_argument('--model-dir',type=Path);ap.add_argument('--tag')
    a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='generate':generate(a.model_dir,a.work)
    else:score(a.model_dir,a.work,a.tag)
