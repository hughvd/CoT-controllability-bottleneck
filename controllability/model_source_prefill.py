"""Matched prefill replication using each evaluated model's saved uninstructed traces."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
import time
import pandas as pd
import yaml

import control_expansion as ex
import transforms as tr
import suppression as supp
import rewrite_common as rc
from control_groups import CONTROLS

HERE = Path(__file__).resolve().parent
ROOT = HERE/'results/token_prefill/model_source_v1'


def source_folder(model):
    return HERE/'results/token_prefill'/('olmo31_v1/olmo31' if model=='olmo31' else 'full_v2/glm')


def transform(c, text):
    b=c['behavior']
    if b=='uppercase_thinking':return tr.uppercase(text)
    if b=='leetspeak':return tr.leetspeak(text)
    if b=='no_first_person':return tr.no_first_person(text)
    if b=='prefix_sentences':return tr.prefix_sentences(text,marker=c['marker'])
    if b=='numbered_sentences':return tr.numbered_sentences(text)
    if b=='suffix_sentences':
        return ' '.join(s.rstrip('.!?')+' '+c['marker']+s[len(s.rstrip('.!?')):] for s in tr._sents(text))
    raise ValueError(b)


def deterministic(c, seed, text):
    if c['behavior'] in ex.NEW:return ex.score_text(c['behavior'],seed,text,c.get('marker'))
    return ex.suite.deterministic_score(c,seed,text)


def eligible(tok,text):
    ids=tok.encode(text,add_special_tokens=False)
    return len(ids)>=64 and all('\ufffd' not in ex.base.decode(tok,ids[:k]) and tok.encode(ex.base.decode(tok,ids[:k]),add_special_tokens=False)==ids[:k] for k in ex.LENGTHS)


def prepare_sources():
    for model in ['olmo31','glm']:
        folder=source_folder(model);old=json.loads((folder/'manifest.json').read_text())
        cells=[deepcopy(c) for c in old['cells'] if c['behavior'] in CONTROLS]
        for pair in json.loads((ex.ROOT/'cells.json').read_text())['cells']:
            c=deepcopy(pair['target']);c['behavior']=c.pop('ctype');c['question_id']=ex.base.digest(c['question']);c['cell_id']=ex.base.digest([c['behavior'],c['question']]);cells.append(c)
        assert len(cells)==240
        tok=ex.suite.tokenizer(ex.PRIMARY[model])
        df=pd.read_csv(folder/'scores.csv');df=df[df.prefix_tokens.eq(0)&df.condition.eq('prefill_only')].drop_duplicates(['request_id','sample'])
        # Stable selection, preferring complete responses. No selection on later
        # intervention compliance; truncated source traces remain explicitly marked.
        df=df.sort_values(['truncated','sample','request_id'],kind='stable')
        rows=[];problems=[]
        for c in cells:
            candidates=df[df.question_id.eq(c['question_id'])]
            chosen=None
            for _,r in candidates.iterrows():
                text=str(r.full_text)
                if not text.strip():continue
                controlled=None if c['behavior'] in ['spanish','omit_word'] else transform(c,text)
                if controlled is not None:
                    if not eligible(tok,controlled):continue
                    # Only the first 64 tokens are ever supplied. In the
                    # first-person task, later mathematical i can be treated as
                    # a pronoun by the legacy grader; retain that diagnostic
                    # without selecting away an otherwise valid supplied prefix.
                    check=ex.base.decode(tok,tok.encode(controlled,add_special_tokens=False)[:64]) if c['behavior']=='no_first_person' else controlled
                    if deterministic(c,'',check)[0]!=1:continue
                chosen=dict(source_request_id=r.request_id,source_sample=int(r['sample']),source_truncated=bool(r.truncated),
                            source_text=text,source_text_sha256=ex.base.digest(text),controlled_text=controlled)
                break
            if chosen is None:problems.append(dict(behavior=c['behavior'],question=c['question']));continue
            for key in ['controlled_text','controlled_cot','controlled_token_ids']:c.pop(key,None)
            rows.append(dict(**c,**chosen))
        out=dict(model=model,model_spec=ex.PRIMARY[model],cells=rows,problems=problems,
                 source_manifest=str(folder/'manifest.json'),source_manifest_sha256=ex.base.file_hash(folder/'manifest.json'),
                 source_scores_sha256=ex.base.file_hash(folder/'scores.csv'),
                 selection='First nonempty sample with eligible transformed prefix, sorted by truncated flag, sample, request ID. No correctness filter.')
        ex.base.immutable_json(ROOT/model/'sources.json',out)
        print(model,'prepared',len(rows),'cells; problems',problems,flush=True)


async def make_examples(model):
    from dotenv import load_dotenv
    from sft import language_grade as lg
    for p in [HERE.parent/'.env',HERE/'.env']:load_dotenv(p,override=True)
    folder=ROOT/model;sources=json.loads((folder/'sources.json').read_text());assert not sources['problems'],sources['problems']
    tok=ex.suite.tokenizer(ex.PRIMARY[model]);client=rc.make_client('openai');sem=asyncio.Semaphore(8)
    cache=folder/'rewrites.jsonl';cached={}
    if cache.exists():
        for line in cache.read_text().splitlines():
            x=json.loads(line);cached[x['key']]=x
    template=yaml.safe_load((HERE/'data/languages.yaml').read_text())['shared_prompt']
    cells=deepcopy(sources['cells']);errors=[]
    prior={}
    if (folder/'examples_draft.json').exists():
        prior={(c['behavior'],c['question_id']):c for c in json.loads((folder/'examples_draft.json').read_text())['cells']}
    prior_failed=set()
    if (folder/'example_report.json').exists():
        prior_failed={(x['behavior'],x['question']) for x in json.loads((folder/'example_report.json').read_text()).get('errors',[])}
    async def one(c):
        if c['controlled_text'] is not None:return
        previous=prior.get((c['behavior'],c['question_id']))
        if previous and previous.get('controlled_text') and previous.get('source_text_sha256')==c['source_text_sha256']:
            for field in ['controlled_text','rewrite_key','example_compliance']:c[field]=previous[field]
            return
        if c['behavior']=='spanish':prompt=template.format(language_display='Spanish',raw_cot=c['source_text'])
        else:prompt=supp.REWRITE_PROMPT.format(words=', '.join(repr(w) for w in c['omit_words']),raw_cot=c['source_text'])
        key=ex.base.digest(['gpt-5-mini',prompt])
        if c['behavior']=='spanish' and (c['behavior'],c['question']) in prior_failed and key+'|0' in cached:
            # Complete an already supplied translation: residual English option
            # words and quoted phrases otherwise fail the language criterion.
            draft=cached[key+'|0']['text']
            prompt=('Copyedit the following Spanish translation so that all natural-language prose is Spanish, '
                    'including quoted phrases, option wording, and spelled-out numbers. Preserve its meaning, '
                    'every step, equations, numerical values, and option letters. Do not summarize or add content. '
                    'Return only the corrected Spanish text.\n\nTRANSLATION TO EDIT:\n'+draft)
            key=ex.base.digest(['gpt-5-mini','spanish_copyedit_v1',prompt])
        for attempt in range(3):
            cachekey=key+'|'+str(attempt)
            if cachekey in cached:r=cached[cachekey]
            else:
                try:
                    text=await rc._call_one(client,model='gpt-5-mini',prompt=prompt,reasoning_effort='medium',max_tokens=16384,semaphore=sem)
                except Exception as exc:
                    errors.append(dict(behavior=c['behavior'],question=c['question'],reason=str(exc),prompt_sha256=key))
                    print(model,c['behavior'],c['question_id'][:8],'rewrite service error',type(exc).__name__,flush=True)
                    return
                r=dict(key=cachekey,model='gpt-5-mini',prompt=prompt,text=text,at=time.time())
                with cache.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
                cached[cachekey]=r
            text=r['text'].strip()
            if not eligible(tok,text):
                # Repeating a length-preserving rewrite cannot reliably turn a
                # short source into a 64-token prefix. Keep the failed example
                # for audit and choose another uninstructed source upstream.
                if len(tok.encode(text,add_special_tokens=False))<64:
                    errors.append(dict(behavior=c['behavior'],question=c['question'],reason='Compliant-example candidate shorter than 64 tokens',rewrite_key=cachekey))
                    return
                continue
            if c['behavior']=='spanish':
                value=await lg.grade_language_async('spanish',text)
            else:value=deterministic(c,'',text)[0]
            if value is None or value<.95:continue
            c.update(controlled_text=text,rewrite_key=cachekey,example_compliance=value)
            print(model,c['behavior'],c['question_id'][:8],'validated',flush=True)
            return
        errors.append(dict(behavior=c['behavior'],question=c['question'],reason='No eligible compliant rewrite in three attempts'))
    try:await asyncio.gather(*(one(c) for c in cells))
    finally:await client.close()
    report=dict(status='failed' if errors else 'complete',cells=len(cells),errors=errors)
    ex.base.atomic_json(folder/'example_report.json',report)
    ex.base.atomic_json(folder/'examples_draft.json',dict(**{k:v for k,v in sources.items() if k!='cells'},cells=cells))
    assert not errors,errors
    ex.base.immutable_json(folder/'examples.json',dict(model=model,cells=cells,source_sha256=ex.base.file_hash(folder/'sources.json')))


def manifest(model):
    spec=ex.PRIMARY[model];tok=ex.suite.tokenizer(spec);folder=ROOT/model
    data=json.loads((folder/'examples.json').read_text());requests=[]
    old=json.loads((source_folder(model)/'manifest.json').read_text());oldcells={c['cell_id']:c for c in old['cells']}
    extension=json.loads((ex.ROOT/model/'manifest.json').read_text())
    newreq={(r['behavior'],r['question_id'],r['condition'],r['prefix_tokens'],r['sample']):r for r in extension['requests'] if r['experiment']=='prefill'}
    for c in data['cells']:
        ids=tok.encode(c['controlled_text'],add_special_tokens=False);assert eligible(tok,c['controlled_text'])
        for cond in ['no_instruction','instruction_only']:
            user=ex.pe.assemble_prompt(c['question'],c['options'],c['instruction'] if cond=='instruction_only' else '')
            for k in ex.LENGTHS[1:]:
                pre=ids[:k];text=ex.base.decode(tok,pre);messages=[dict(role='user',content=user)]
                promptids=ex.encoded(spec,tok,messages,text)
                if model=='glm':messages.append(dict(role='assistant',content='<think>\n'+text))
                for sample in range(5):
                    key=[model,c['behavior'],c['question_id'],cond,k,sample]
                    requests.append(dict(request_id=ex.base.digest(['model_source_v1',*key]),experiment='prefill',behavior=c['behavior'],
                        question=c['question'],question_id=c['question_id'],condition=cond,prefix_tokens=k,prefix_text=text,prefix_token_ids=pre,
                        arm='baseline',sample=sample,correct_letter=c['correct_letter'],marker=c.get('marker'),omit_words=c.get('omit_words'),
                        messages=messages,prompt_token_ids=promptids,seed=int(ex.base.digest(key)[:8],16)%2**31))
    assert len(requests)==16800
    m=dict(name=model,model=spec,requests=requests,ban_token_ids=[],sampling=dict(temperature=.6,top_p=1.,max_tokens=8192),
           source_examples_sha256=ex.base.file_hash(folder/'examples.json'),protocol='Model-specific transformed uninstructed traces; K=0 reused from original suite.')
    m['manifest_sha256']=ex.base.digest(m);ex.base.immutable_json(folder/'manifest.json',m)
    print(model,len(requests),'prefill requests ready',flush=True)


async def score(model):
    import token_prefill_suite_api as api
    from token_prefill_suite_score import parse_judgment
    from sft import language_grade as lg
    folder=ROOT/model;m=json.loads((folder/'manifest.json').read_text());requests={r['request_id']:r for r in m['requests']}
    raws=[]
    if (folder/'outputs.sqlite').exists():
        db=sqlite3.connect(folder/'outputs.sqlite');raws.extend(json.loads(x[0]) for x in db.execute('select payload from outputs'));db.close()
    for p in (folder/'remote').glob('batch_*.json'):raws.extend(json.loads(p.read_text())['outputs'])
    assert len(raws)==len(requests) and len({x['request_id'] for x in raws})==len(requests)
    api.s.ROOT=folder/'grading';client=api.PaidClient(cap=25.,retries=8);client.sem=asyncio.Semaphore(24)
    rows=[]
    async def one(raw):
        r=requests[raw['request_id']]
        if 'reasoning_continuation' in raw:text=raw['reasoning_continuation']
        else:
            text=raw['raw_text']
            if text.startswith('<think>'):text=text[len('<think>'):]
            text=text.split('</think>',1)[0]
        if r['behavior']=='spanish' and text.strip():
            prompt=lg._GRADER_PROMPT.format(display_name='Spanish',text=text);key=ex.base.digest([lg.DEFAULT_MODEL,prompt])
            value=None
            for attempt in range(3):
                response=await client.call('model-source-grade|'+key+'|'+str(attempt),dict(model=lg.DEFAULT_MODEL,messages=[dict(role='user',content=prompt)],temperature=0.,max_tokens=600,response_format={'type':'json_object'}),.1e-6,.4e-6)
                ch=response['choices'][0];value,_,_=parse_judgment(lg,ch['message'].get('content') or '',ch.get('finish_reason'))
                if value is not None:break
            assert value is not None
            units=None
        elif not text.strip():value,units=0.,0
        else:value,units=deterministic(r,r['prefix_text'],text)
        rows.append(dict(model=model,**{k:r[k] for k in ['request_id','behavior','question_id','condition','prefix_tokens','sample']},
                         compliance_score=value,eligible_units=units,empty_reasoning=not text.strip(),truncated=raw['finish_reason']=='length',reasoning=text))
    try:await asyncio.gather(*(one(r) for r in raws))
    finally:await client.client.close()
    baseline=pd.read_csv(source_folder(model)/'scores.csv');baseline=baseline[baseline.behavior.isin(CONTROLS)&baseline.prefix_tokens.eq(0)].copy()
    baseline['compliance_score']=baseline.full_soft;baseline['reasoning']=baseline.full_text
    baseline['condition']=baseline.condition.map({'prefill_only':'no_instruction','prefill_plus_instruction':'instruction_only'})
    extra=pd.read_csv(ex.ROOT/model/'scores.csv');extra=extra[extra.experiment.eq('prefill')&extra.prefix_tokens.eq(0)].copy()
    out=pd.concat([pd.DataFrame(rows),baseline,extra],ignore_index=True);out['model']=model
    assert len(out)==19200 and out.groupby(['behavior','question_id','condition','prefix_tokens']).size().eq(5).all()
    out.to_csv(folder/'scores.csv',index=False)
    ex.base.atomic_json(folder/'scoring_report.json',dict(status='complete',scored=len(out),new_generated=len(rows),reused_k0=len(baseline)+len(extra),scorer_sha256=ex.base.file_hash(__file__),
        reused_k0_sources={str(path):ex.base.file_hash(path) for path in [source_folder(model)/'scores.csv',ex.ROOT/model/'scores.csv']}))
    print(model,len(out),'scored with original K=0 controls reused',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['sources','examples','manifest','score']);parser.add_argument('--model',choices=['olmo31','glm']);args=parser.parse_args()
    if args.phase=='sources':prepare_sources()
    elif args.phase=='examples':asyncio.run(make_examples(args.model))
    elif args.phase=='manifest':manifest(args.model)
    else:asyncio.run(score(args.model))
