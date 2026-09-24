"""Native OLMo-3.1-32B-Think replication of the four-condition ICL experiment."""
import argparse
import asyncio
import fcntl
import hashlib
import json
from pathlib import Path
import pandas as pd
import icl_experiment as i
import gptoss_prefill_run as grading
import icl_judging
icl_judging.install()

RUN='iclfull_olmo_3_1_32b_donorfix_v1'
ROOT=i.HERE/'results'/RUN
CSV=i.HERE/'results'/f'{RUN}_icl_per_sample.csv'
MODEL='allenai/Olmo-3.1-32B-Think'
REVISION='832c3f543499af8fe68b88359501de9cb7840544'

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def tokenizer():
    from transformers import AutoTokenizer
    snapshot=Path.home()/'.cache/huggingface/hub'/('models--'+MODEL.replace('/','--'))/'snapshots'/REVISION
    return AutoTokenizer.from_pretrained(snapshot,local_files_only=True)

from olmo_icl_protocol import split_reasoning

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);tok=tokenizer()
    cfg=i.PrefillConfig();cells=i.build_all_cells(cfg);donors=i.select_donors(cells,cfg.seed)
    requests=[]
    for c in cells:
        donor=donors[c.ctype,c.question]
        for variant in ['instruction_only',*i.SCALE_VARIANTS]:
            messages=i.build_icl_messages(c,donor,variant,i.default_spec(variant))
            r={'ctype':c.ctype,'modality':donor.modality,'question':c.question,'correct_letter':c.correct_letter,
               'instruction':donor.instruction,'marker':donor.marker,'omit_words':donor.omit_words,
               'donor_question_hash':i._qhash(donor.question),'variant':variant,'messages':messages}
            r['request_id']=digest(r)
            prompt=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            assert prompt.endswith('<|im_start|>assistant\n<think>')
            r['prompt_token_ids']=tok.encode(prompt,add_special_tokens=False)
            assert tok.decode(r['prompt_token_ids'],skip_special_tokens=False)==prompt
            requests.append(r)
    assert len(requests)==1200 and len({r['request_id'] for r in requests})==1200
    longest=max(len(r['prompt_token_ids']) for r in requests)
    assert longest+8192<=32768,f'Input too long: {longest}'
    smoke_cells=set()
    for ct in sorted({c.ctype for c in cells}):
        r=max((r for r in requests if r['ctype']==ct and r['variant']=='icl_assistant_visible'),key=lambda r:len(r['prompt_token_ids']))
        smoke_cells.add((r['ctype'],r['question']))
    snapshot=Path.home()/'.cache/huggingface/hub'/('models--'+MODEL.replace('/','--'))/'snapshots'/REVISION
    manifest={'run':RUN,'model':MODEL,'revision':REVISION,'dtype':'bfloat16','max_model_len':32768,
        'tokenizer_sha256':hashlib.sha256((snapshot/'tokenizer.json').read_bytes()).hexdigest(),
        'icl_sha256':hashlib.sha256(Path(i.__file__).read_bytes()).hexdigest(),
        'sampling':{'temperature':.6,'top_p':1.,'top_k':-1,'max_tokens':8192,'n':5},
        'donor_assignment_version':2,'requests':requests,
        'smoke_ids':[r['request_id'] for r in requests if (r['ctype'],r['question']) in smoke_cells]}
    manifest['manifest_sha256']=digest(manifest)
    path=ROOT/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest,'Manifest is immutable'
    else:path.write_text(json.dumps(manifest,ensure_ascii=False))
    (ROOT/'source').mkdir(exist_ok=True)
    for file in [Path(__file__),Path(i.__file__)]:
        dest=ROOT/'source'/file.name
        if not dest.exists():dest.write_bytes(file.read_bytes())
    print('Prepared',len(requests),'prompts / 6,000 responses; max input',longest,'tokens; pilot',len(manifest['smoke_ids'])*5)

async def grade(partial=False):
    grading.ROOT=ROOT;judge=grading.Judge();tok=tokenizer()
    m=json.loads((ROOT/'manifest.json').read_text());requests={r['request_id']:r for r in m['requests']}
    gens={}
    for p in sorted((ROOT/'remote').glob('batch_*.json')):
        for g in json.loads(p.read_text()):
            key=f"{g['request_id']}_{g['sample']}"
            assert key not in gens and g['sample'] in range(5)
            r=requests[g['request_id']]
            assert g['prompt_token_ids']==r['prompt_token_ids']
            assert tok.decode(g['output_token_ids'],skip_special_tokens=False)==g['raw_text']
            assert split_reasoning(g['raw_text'])==(g['reasoning_continuation'],g['response'])
            gens[key]=g
    if partial:assert 0<len(gens)<=6000
    else:assert len(gens)==6000,f'Incomplete: {len(gens)}/6000'
    path=ROOT/'grades.jsonl';done={r['key']:r for r in grading.read_journal(path)}
    paid={r['key']:r['score'] for r in grading.read_journal(judge.path) if r.get('score') is not None}
    async def one(key,g):
        if key in done:return
        r=requests[g['request_id']];text=g['reasoning_continuation']
        if not text.strip():score=0.
        elif r['modality']=='family':score=float(i.pe.grade_score(text,r['ctype'],marker=r['marker']))
        elif r['modality']=='suppression':score=float(i.pe.supp.score_suppression(r['ctype'],text,r['omit_words'] or []))
        elif key in paid:score=paid[key]
        else:score=await judge.score(r['modality'],r['ctype'],text,key)
        record={'key':key,'score':score}
        with path.open('a') as f:f.write(json.dumps(record)+'\n')
        done[key]=record
        if len(done)%250==0:print('Graded',len(done),'/6000; judge cost',round(judge.cost,3),flush=True)
    await asyncio.gather(*(one(k,g) for k,g in gens.items()))
    if partial:return
    rows=[]
    for key,g in gens.items():
        r=requests[g['request_id']];score=done[key]['score'];pred=i.extract_answer(g['response'])
        rows.append(dict(model='olmo-3.1-32b',ctype=r['ctype'],modality=r['modality'],variant=r['variant'],length='fixed',
            sample=g['sample'],target_question=r['question'],donor_question_hash=r['donor_question_hash'],
            donor_ctype=r['ctype'],donor_assignment_version=2,instruction=r['instruction'],compliance_score=score,
            compliance=int(score>=.5),predicted=pred,correct_letter=r['correct_letter'],correct=int(pred==r['correct_letter']),
            reasoning=g['reasoning_continuation'],response=g['response'],finish_reason=g['finish_reason'],
            provider='modal/vllm/Olmo-3.1-32B-Think',empty_thinking=not g['reasoning_continuation'].strip(),
            manifest_sha256=m['manifest_sha256']))
    df=pd.DataFrame(rows);assert not df.duplicated(['ctype','target_question','variant','sample']).any()
    tmp=CSV.with_suffix('.csv.tmp');df.to_csv(tmp,index=False);tmp.replace(CSV)
    quality=dict(status='complete',rows=len(df),empty=int(df.empty_thinking.sum()),
        length_limited=int(df.finish_reason.eq('length').sum()),judge_cost_usd=judge.cost,manifest_sha256=m['manifest_sha256'])
    (ROOT/'audit.json').write_text(json.dumps(quality,indent=2));print(quality)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','grade','grade-partial']);a=p.parse_args()
    if a.phase=='prepare':prepare()
    else:
        with (ROOT/'grader.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);asyncio.run(grade(a.phase=='grade-partial'))
