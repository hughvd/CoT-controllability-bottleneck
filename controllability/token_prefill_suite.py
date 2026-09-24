"""Versioned full token-prefill suite: prepare fixed histories and measure continuations."""
from pathlib import Path
import argparse
import json
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'results/token_prefill/full_v2'
LENGTHS = [0, 1, 2, 4, 8, 16, 32, 64]
CONDITIONS = ['prefill_only', 'prefill_plus_instruction']
BEHAVIORS = ['uppercase_thinking', 'leetspeak', 'prefix_sentences', 'no_first_person',
             'omit_word', 'spanish', 'chinese', 'pirate', 'yoda', 'jar_jar_binks']
sys.path.insert(0, str(HERE))
import token_prefill as base

MODELS = {
 'qwen32': dict(id='Qwen/Qwen3-32B', revision=base.REVISION, adapter='qwen', backend='modal', dtype='bfloat16'),
 'qwen14': dict(id='Qwen/Qwen3-14B', revision='40c069824f4251a91eefaf281ebe4c544efd3e18', adapter='qwen', backend='modal', dtype='bfloat16'),
 'gptoss': dict(id='openai/gpt-oss-120b', revision='b5c939de8f754692c1647ca79fbf85e8c1e70f8a', adapter='harmony', backend='modal', dtype='auto', effort='high'),
 'deepseek': dict(id='deepseek-ai/DeepSeek-R1-0528', revision='4236a6af538feda4548eca9ab308586007567f52', adapter='api', backend='openrouter', api_id='deepseek/deepseek-r1-0528', provider='DeepInfra'),
 'glm': dict(id='zai-org/GLM-5.2', revision='cf457fa734ab149ffef225f80893eb38c6ff5cdc', adapter='api', backend='openrouter', api_id='z-ai/glm-5.2', provider='DeepInfra'),
}

class JsonTokenizer:
    """Use the pinned native tokenizer graph for hosted-model text measurements."""
    def __init__(self, path):
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(str(Path(path)/'tokenizer.json'))
    def encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def tokenizer(spec):
    path = Path.home()/'.cache/huggingface/hub'/('models--'+spec['id'].replace('/','--'))/'snapshots'/spec['revision']
    if spec['adapter'] in ['api', 'harmony']:
        return JsonTokenizer(path)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def prompt_tokens(spec, tok, cell, instruction):
    import prefill_experiment as p
    if spec['adapter'] == 'harmony':
        import harmony_backend as h
        return h.build_prefill_tokens(cell['question'], cell['options'], instruction, '', 'high', force_think=True)
    user = p.assemble_prompt(cell['question'], cell['options'], instruction)
    if spec['adapter'] == 'api':
        return None
    text = tok.apply_chat_template([{'role':'user','content':user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)+'<think>\n'
    return tok.encode(text, add_special_tokens=False)


def validate(m):
    assert base.digest({k:v for k,v in m.items() if k!='manifest_sha256'}) == m['manifest_sha256']
    requests={r['request_id']:r for r in m['requests']};cells={c['cell_id']:c for c in m['cells']}
    assert len(requests)==len(m['requests'])
    assert len(m['cells'])==300 and len(m['links'])==4800
    assert len({(x['cell_id'],x['condition'],x['prefix_tokens']) for x in m['links']})==4800
    for r in requests.values():
        assert r['prefix_token_ids']==cells[r['cell_id']]['controlled_token_ids'][:r['prefix_tokens']]
        assert len(r['prefix_token_ids'])==r['prefix_tokens']
        if r['prompt_token_ids'] is not None:
            assert r['prompt_token_ids']==r['opening_token_ids']+r['prefix_token_ids']
            assert r['request_id']==base.digest(r['prompt_token_ids'])
    assert {x['request_id'] for x in m['links']}==set(requests)


def prepare(name):
    from dataclasses import asdict
    import pandas as pd
    import prefill_experiment as p
    spec=MODELS[name];tok=tokenizer(spec)
    rebuilt=p.build_all_cells(p.PrefillConfig())
    source=HERE/'results/base_qwen3_32b_prefill_per_sample.csv'
    historical=pd.read_csv(source).fillna('')
    lookup={(c.ctype,c.question,c.instruction,str(c.marker or '')):c for c in rebuilt}
    for row in historical.itertuples():
        c=lookup[(row.ctype,row.question,row.instruction,str(row.marker or ''))]
        assert p.first_k_sentences(c.controlled_cot,row.k)==row.seed
    # Text-only API backends cannot transmit incomplete UTF-8 or a noncanonical
    # BPE prefix. Replace affected source cells for EVERY model before inference.
    api_tokenizers=[tokenizer(MODELS[x]) for x in ['deepseek','glm']]
    def eligible(text):
        for t in api_tokenizers:
            ids=t.encode(text,add_special_tokens=False)
            if len(ids)<64:return False
            for k in LENGTHS:
                chunk=base.decode(t,ids[:k])
                if '\ufffd' in chunk or t.encode(chunk,add_special_tokens=False)!=ids[:k]:return False
        return True
    replacements=[];raws=p.load_raw_questions()
    used={(c.ctype,c.question) for c in rebuilt}
    pool=sorted(p.load_rewrites('language_rewrites.clean.jsonl','language'),key=lambda r:base.digest(r))
    for i,c in enumerate(rebuilt):
        if eligible(c.controlled_cot.strip()):continue
        assert c.modality=='language','Unexpected API-ineligible nonlanguage source'
        for candidate in pool:
            q=raws.get(candidate.get('question_hash',''));text=str(candidate.get('rewrite') or '').strip()
            if candidate.get('language')!=c.ctype or q is None or not q.options or (c.ctype,q.question) in used:continue
            if not eligible(text):continue
            replacements.append(dict(behavior=c.ctype,old_question_id=base.digest(c.question),new_question_id=base.digest(q.question),reason='API prefix token round trip'))
            rebuilt[i]=p.Cell(c.ctype,c.modality,q.question,q.options,q.correct_letter,text,c.instruction,c.marker,c.omit_words)
            used.add((c.ctype,q.question));break
        else:raise ValueError('No common API-compatible replacement source')
    cells=[];requests={};links=[];issues=[]
    for original in rebuilt:
        c=asdict(original);c['behavior']=c.pop('ctype');c['controlled_text']=c.pop('controlled_cot').strip()
        c['cell_id']=base.digest([c['behavior'],c['question'],c['instruction']]);c['question_id']=base.digest(c['question'])
        c['controlled_token_ids']=tok.encode(c['controlled_text'],add_special_tokens=False)
        assert len(c['controlled_token_ids'])>=64
        assert base.decode(tok,c['controlled_token_ids'])==c['controlled_text']
        cells.append(c)
        for condition in CONDITIONS:
            instruction=c['instruction'] if condition=='prefill_plus_instruction' else ''
            user=p.assemble_prompt(c['question'],c['options'],instruction)
            opening=prompt_tokens(spec,tok,c,instruction)
            for k in LENGTHS:
                ids=c['controlled_token_ids'][:k];seed=base.decode(tok,ids)
                messages=[{'role':'user','content':user},{'role':'assistant','content':'<think>\n'+seed}]
                prompt=None if opening is None else opening+ids
                rid=base.digest(prompt if prompt is not None else [spec['api_id'],messages])
                roundtrip='\ufffd' not in seed and tok.encode(seed,add_special_tokens=False)==ids
                if spec['adapter']=='api' and not roundtrip:issues.append(dict(cell_id=c['cell_id'],k=k,condition=condition))
                request=dict(request_id=rid,cell_id=c['cell_id'],condition=condition,prefix_tokens=k,
                    opening_token_ids=opening,prompt_token_ids=prompt,prefix_token_ids=ids,seed=seed,
                    user_text=user,messages=messages if prompt is None else None,
                    seed_roundtrip=roundtrip,sampling_seed=int(rid[:8],16))
                requests.setdefault(rid,request)
                links.append(dict(request_id=rid,cell_id=c['cell_id'],condition=condition,prefix_tokens=k))
    if spec['adapter']=='harmony':
        import harmony_backend as h
        stop=list(h._enc().stop_tokens_for_assistant_actions())
        protocol=dict(analysis_end_ids=tok.encode('<|end|>',add_special_tokens=False),eos_ids=stop)
    elif spec['adapter']=='qwen':
        protocol=dict(think_close_id=tok.encode('</think>',add_special_tokens=False)[0],eos_ids=[tok.eos_token_id])
    else:protocol=dict(returned_token_ids=False,window_tokens='reconstructed_with_pinned_native_tokenizer')
    paths=[source,HERE/'data/training_raw.csv',HERE/'transforms.py',HERE/'prefill_experiment.py',
           HERE/'data/stylistic_rewrites.clean.jsonl',HERE/'data/language_rewrites.clean.jsonl',HERE/'data/suppression_rewrites.clean.jsonl']
    snapshot=Path.home()/'.cache/huggingface/hub'/('models--'+spec['id'].replace('/','--'))/'snapshots'/spec['revision']
    m=dict(version='full_token_prefill_v1',name=name,model=spec,lengths=LENGTHS,behaviors=BEHAVIORS,
        sampling=dict(n=5,temperature=.6,top_p=1.,top_k=-1,max_tokens=8192),max_model_len=16384,
        cells=cells,requests=list(requests.values()),links=links,protocol=protocol,
        source_hashes={str(f.relative_to(HERE)):base.file_hash(f) for f in paths},
        tokenizer_hashes={f.name:base.file_hash(f) for f in snapshot.iterdir() if f.name in ['tokenizer.json','tokenizer_config.json','chat_template.jinja']},
        code_sha256=base.file_hash(__file__),base_code_sha256=base.file_hash(base.__file__),
        harmony_sha256=base.file_hash(HERE/'harmony_backend.py'),api_prefix_issues=issues,source_replacements=replacements)
    m['manifest_sha256']=base.digest(m);validate(m)
    root=ROOT/name;base.immutable_json(root/'manifest.json',m)
    for f in [Path(__file__),Path(base.__file__),HERE/'harmony_backend.py']:
        path=root/'code'/(f.stem+'_'+base.file_hash(f)+'.py');path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(f.read_bytes())
    print(name,len(requests),'unique requests;',len(links)*5,'linked outputs;',len(issues),'unsupported API prefixes',flush=True)
    return m


def reasoning_parts(tok, request, raw, m):
    if m['model']['adapter']=='api':
        text=raw['reasoning_continuation'];ids=tok.encode(text,add_special_tokens=False)
        return text,base.decode(tok,ids[:32]),len(ids),raw.get('closed_reasoning',True)
    ids=raw['output_token_ids'];seed=request['prefix_token_ids']
    if m['model']['adapter']=='harmony':
        stops=set(m['protocol']['analysis_end_ids']+m['protocol']['eos_ids'])
    else:stops={m['protocol']['think_close_id'],*m['protocol']['eos_ids']}
    end=next((i for i,x in enumerate(ids) if x in stops),len(ids))
    return base.continuation_text(tok,seed,ids[:end]),base.continuation_text(tok,seed,ids[:min(end,32)]),end,end<len(ids)


def deterministic_score(cell, seed, text):
    """Continuation-only scores, preserving context for sentence/word predicates."""
    import transforms as tr
    behavior=cell['behavior']
    if not text.strip():return 0.,0
    if behavior in ['uppercase_thinking','leetspeak']:
        return base.score_text(text,behavior)['soft'],len(text)
    if behavior=='prefix_sentences':
        joint=seed+text; offset=len(seed); starts=[0]+[x.end() for x in tr._SENT_SPLIT_RE.finditer(joint)]
        # Only score newly generated sentence starts. The prefix's unfinished
        # sentence is not a new attempt to produce a sentence prefix.
        new=[s for s in starts if s>=offset and joint[s:].strip()]
        spans=[]
        for s in new:
            end=next((x for x in starts if x>s),len(joint));spans.append(joint[s:end].strip())
        if not spans:return None,0
        marker=str(cell['marker'])
        return sum(s.startswith(marker+' ') or s==marker for s in spans)/len(spans),len(spans)
    if behavior=='omit_word':
        # A whole-word match spanning the seed boundary counts when any of its
        # characters are newly generated. A word wholly inside the seed does not.
        words=cell['omit_words'];joint=seed+text;offset=len(seed)
        absent=[not any(x.end()>offset for x in re.finditer(r'\b'+re.escape(w)+r'\b',joint,re.I)) for w in words]
        return sum(absent)/len(absent),len(words)
    if behavior=='no_first_person':
        joint=seed+text;offset=len(seed)
        matches=re.finditer(r"[A-Za-z]+(?:[.'][A-Za-z]+)+\.?|[A-Za-z]+",joint)
        bad=any(x.end()>offset and x.group()!='US' and x.group().lower() in tr._FP_WORDS for x in matches)
        return float(not bad),len(text)
    raise ValueError('LLM grader required for '+behavior)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare']);parser.add_argument('--model',choices=list(MODELS)+['all'],default='all')
    args=parser.parse_args()
    for name in MODELS if args.model=='all' else [args.model]:prepare(name)
