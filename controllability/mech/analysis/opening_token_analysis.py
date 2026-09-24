"""Research plots of sampled opening diversity and ranked token identities.

Read-only source loading. Probability panels use saved raw top-five distributions;
GLM has sampled-token panels only because its reasoning logprobs are unavailable.
"""
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
CTRL = HERE.parent.parent
DATA = HERE.parent / "data"
OUT = HERE / "opening_tokens"
TYPES = ["uppercase_thinking", "leetspeak", "spanish", "omit_word"]
CONDS = ["no_instruction", "instruction_only"]
MODELS = ["qwen32", "olmo31", "glm"]
NAMES = {"qwen32":"Qwen3-32B", "olmo31":"OLMo-3.1-32B-Think", "glm":"GLM-5.2"}
COLORS = {"qwen32":"#2166ac", "olmo31":"#c05d15", "glm":"#16836c"}
TOKPATHS = {
    "qwen32":"models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137/tokenizer.json",
    "olmo31":"models--allenai--Olmo-3.1-32B-Think/snapshots/832c3f543499af8fe68b88359501de9cb7840544/tokenizer.json",
    "glm":"models--zai-org--GLM-5.2/snapshots/cf457fa734ab149ffef225f80893eb38c6ff5cdc/tokenizer.json"}
TOK = {m:Tokenizer.from_file(str(Path.home()/".cache/huggingface/hub"/p)) for m,p in TOKPATHS.items()}
INPUTS = set()


def jsonrows(path):
    INPUTS.add(path)
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def sources():
    out={}
    for name in ["split_test.csv", "training_raw.csv"]:
        p=CTRL/"data"/name;INPUTS.add(p)
        for r in pd.read_csv(p).to_dict("records"):
            out[str(r["question"])]=r["dataset"]
    return out


def content_bounds(ids,model):
    tok=TOK[model]
    stops={tok.token_to_id(t) for t in ["</think>","<|im_end|>","<|endoftext|>","<|user|>"]}
    stops.discard(None)
    end=next((i for i,t in enumerate(ids) if t in stops),len(ids))
    start=0
    while start<end:
        s=tok.decode([ids[start]],skip_special_tokens=False)
        if s.strip() and s.strip()!="<think>":break
        start+=1
    return start,end


def label(model,token):
    return repr(TOK[model].decode([int(token)],skip_special_tokens=False))


def visible(text):
    # ASCII escapes keep rare CJK/punctuation alternatives legible in any font.
    return text.encode('ascii','backslashreplace').decode()


def percent(x):
    return '<1%' if 0<x<.01 else f'{x:.0%}'


def load_sampled(source):
    records=[]
    for model in ["qwen32","glm"]:
        root=CTRL/"results/token_prefill/full_v2"/model
        mp=root/"manifest.json";INPUTS.add(mp);m=json.loads(mp.read_text())
        cells={c["cell_id"]:c for c in m["cells"]}
        links=defaultdict(list)
        for link in m["links"]:
            if link["prefix_tokens"]==0 and cells[link["cell_id"]]["behavior"] in TYPES:
                links[link["request_id"]].append(link)
        def raw():
            if model=="glm":
                p=root/"api_generations.sqlite";INPUTS.add(p)
                db=sqlite3.connect(f"file:{p}?mode=ro",uri=True)
                for (s,) in db.execute("select payload from outputs"):
                    r=json.loads(s)
                    if r["request_id"] in links:yield r
                db.close()
            else:
                for p in sorted((root/"remote").glob("batch_*.json")):
                    batch=json.loads(p.read_text())
                    for r in batch["outputs"]:
                        if r["request_id"] in links:
                            INPUTS.add(p);yield r
        for r in raw():
            ids=r["output_token_ids"] if model=="qwen32" else TOK[model].encode(r["reasoning_continuation"],add_special_tokens=False).ids
            a,b=content_bounds(ids,model)
            for link in links[r["request_id"]]:
                c=cells[link["cell_id"]]
                records.append(dict(model=model,behavior=c["behavior"],
                    condition="instruction_only" if link["condition"]=="prefill_plus_instruction" else "no_instruction",
                    question=c["question"],question_id=c["question_id"],source=source.get(c["question"],"unknown"),
                    sample=r["sample"],ids=ids[a:b],trace_id=r["request_id"]+"|"+str(r["sample"]),
                    token_provenance="exact" if model=="qwen32" else "locally_retokenized",
                    origin="token_prefill/full_v2/k0",temperature=.6,top_p=1.))
    pp=DATA/"olmo_rl/prompts.jsonl"
    prompts={p["prompt_id"]:p for p in jsonrows(pp) if not p.get("_meta")}
    for r in jsonrows(DATA/"olmo_rl/ext_final/traces.jsonl"):
        if r["sample"]>=5:continue
        p=prompts[r["prompt_id"]];a,b=content_bounds(r["output_token_ids"],"olmo31")
        records.append(dict(model="olmo31",behavior=p["ctype"],condition=p["condition"],
            question=p["question"],question_id=p["qhash"],source=source.get(p["question"],"unknown"),
            sample=r["sample"],ids=r["output_token_ids"][a:b],trace_id=r["trace_id"],token_provenance="exact",
            origin="olmo_rl/ext_final",temperature=.6,top_p=.95))
    return records


def load_ranked(source):
    rows=[];excluded=0
    for model,pp,tp,sp in [
        ("qwen32",DATA/"prompts.jsonl",DATA/"traces.jsonl",DATA/"entropy"),
        ("olmo31",DATA/"olmo_rl/prompts.jsonl",DATA/"olmo_rl/ext_final/traces.jsonl",DATA/"olmo_rl/ext_final/stats")]:
        prompts={p["prompt_id"]:p for p in jsonrows(pp) if not p.get("_meta")}
        rawq={hashlib.sha256(q.encode()).hexdigest()[:16]:q for q in source}
        for r in jsonrows(tp):
            p=prompts[r["prompt_id"]]
            if p["ctype"] not in TYPES or p["condition"] not in CONDS:continue
            if p.get("grade_ctype",p["ctype"])!=p["ctype"]:
                excluded+=1;continue
            ids=r["output_token_ids"];a,b=content_bounds(ids,model)
            if b-a<4:continue
            fp=sp/(r["trace_id"].replace("|","_")+".npz");INPUTS.add(fp)
            with np.load(fp) as z:
                toks=z["top5_ids"];probs=z["top5_p"].astype(float)
                assert len(toks)>a+3
                for pos in range(4):
                    assert np.all(np.diff(probs[a+pos])<=0)
                    history=TOK[model].decode(ids[a:a+pos],skip_special_tokens=False)
                    for rank in range(3):
                        token=int(toks[a+pos,rank])
                        rows.append(dict(model=model,behavior=p["ctype"],condition=p["condition"],
                            question_id=p["qhash"],source=source.get(p.get("question",rawq.get(p["qhash"],"")),"unknown"),
                            trace_id=r["trace_id"],position=pos,rank=rank+1,token_id=token,token=label(model,token),
                            probability=float(probs[a+pos,rank]),history=history,
                            tied_with_next=bool(probs[a+pos,rank]==probs[a+pos,rank+1])))
    return pd.DataFrame(rows),excluded


def expected_unique(counts,n):
    N=sum(counts)
    if N<n:return np.nan
    def lc(a,b):return math.lgamma(a+1)-math.lgamma(b+1)-math.lgamma(a-b+1)
    den=lc(N,n)
    return sum(1. if N-c<n else -math.expm1(min(0.,lc(N-c,n)-den)) for c in counts)


def savefig(fig,name):
    fig.savefig(OUT/(name+".png"),dpi=170,bbox_inches="tight")
    fig.savefig(OUT/(name+".pdf"),bbox_inches="tight")
    plt.close(fig)


def sampled_plots(records):
    rows=[];prefix=[];sampletop=[]
    for model in MODELS:
        for cond in CONDS:
            for ct in TYPES+["pooled"]:
                rs=[r for r in records if r["model"]==model and r["condition"]==cond and (ct=="pooled" or r["behavior"]==ct)]
                for pos in range(64):
                    eligible=[r for r in rs if len(r["ids"])>pos]
                    c=Counter(r["ids"][pos] for r in eligible)
                    rows.append(dict(model=model,condition=cond,behavior=ct,position=pos,
                        n=len(eligible),unique=len(c),unique_rarefied_100=expected_unique(list(c.values()),100),
                        mode_share=max(c.values())/len(eligible) if eligible else np.nan))
                    if pos<4:
                        for rank,(tok,n) in enumerate(c.most_common(8),1):
                            sampletop.append(dict(model=model,condition=cond,behavior=ct,position=pos,
                                empirical_rank=rank,token_id=tok,token=label(model,tok),count=n,frequency=n/len(eligible)))
                for k in [1,2,4,8,16,32]:
                    c=Counter(tuple(r["ids"][:k]) for r in rs if len(r["ids"])>=k)
                    n=sum(c.values())
                    prefix.append(dict(model=model,condition=cond,behavior=ct,length=k,n=n,unique=len(c),
                        top1_share=c.most_common(1)[0][1]/n,top3_share=sum(v for _,v in c.most_common(3))/n,
                        top10_share=sum(v for _,v in c.most_common(10))/n,
                        most_common=json.dumps([(TOK[model].decode(list(t),skip_special_tokens=False),v) for t,v in c.most_common(5)],ensure_ascii=False)))
    df=pd.DataFrame(rows);pf=pd.DataFrame(prefix);st=pd.DataFrame(sampletop)
    df.to_csv(OUT/"sampled_diversity.csv",index=False);pf.to_csv(OUT/"prefix_concentration.csv",index=False);st.to_csv(OUT/"sampled_token_frequencies.csv",index=False)
    for rare in [False,True]:
        fig,axes=plt.subplots(2,4,figsize=(16,7),sharex=True,sharey=True)
        for i,cond in enumerate(CONDS):
            for j,ct in enumerate(TYPES):
                ax=axes[i,j]
                for model in MODELS:
                    s=df[(df.model==model)&(df.condition==cond)&(df.behavior==ct)]
                    ax.plot(s.position,s.unique_rarefied_100 if rare else s.unique,color=COLORS[model],label=NAMES[model])
                ax.set_title(ct.replace('_',' '));ax.grid(alpha=.2);ax.set_ylim(bottom=0);ax.set_xlim(0,63)
                if j==0:ax.set_ylabel(cond.replace('_',' ')+'\n'+('Expected unique / 100 responses' if rare else 'Unique sampled token IDs'))
                if i==1:ax.set_xlabel('Reasoning content token position')
        axes[0,0].legend(fontsize=8)
        fig.suptitle('Native-token diversity by control | 150 initial responses per model/control/condition\nQwen + GLM: top-p 1, shared question cells; OLMo: top-p .95, different question pool',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.93));savefig(fig,'unique_tokens_rarefied' if rare else 'unique_tokens_by_control')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,cond in zip(axes,CONDS):
        for model in MODELS:
            s=df[(df.model==model)&(df.condition==cond)&(df.behavior=='pooled')]
            ax.plot(s.position,s.unique,label=NAMES[model],color=COLORS[model])
        ax.set_title(cond.replace('_',' '));ax.set_xlabel('Reasoning content token position');ax.grid(alpha=.2);ax.set_xlim(0,63)
    axes[0].set_ylabel('Unique sampled token IDs, pooled across four controls');axes[0].legend()
    fig.suptitle('600 initial responses per model/condition; see per-control plots and coverage for interpretation',fontsize=10)
    fig.tight_layout();savefig(fig,'unique_tokens_pooled')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,cond in zip(axes,CONDS):
        for model in MODELS:
            s=pf[(pf.model==model)&(pf.condition==cond)&(pf.behavior!='pooled')].groupby('length').top3_share.mean()
            ax.plot(s.index,s.values,'o-',label=NAMES[model],color=COLORS[model])
        ax.set_title(cond.replace('_',' '));ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32]);ax.set_xlabel('Opening length (native tokens)');ax.grid(alpha=.2);ax.set_ylim(0,1.02)
    axes[0].set_ylabel('Share covered by the three most common opening sequences\nMean across four controls');axes[0].legend()
    fig.tight_layout();savefig(fig,'prefix_concentration')
    return df,pf,st


def ranked_plots(df):
    summaries=[];triplets=[]
    for model in ['qwen32','olmo31']:
        for cond in CONDS:
            for ct in TYPES+['pooled']:
                sub=df[(df.model==model)&(df.condition==cond)]
                if ct!='pooled':sub=sub[sub.behavior==ct]
                # Equal control and question weight, then equal sample weight.
                sub=sub.copy()
                sub['weight']=1/sub.groupby(['behavior','question_id','position','rank']).trace_id.transform('count')
                sub['weight']/=sub.groupby('behavior').question_id.transform('nunique')
                fig,axes=plt.subplots(3,4,figsize=(16,9))
                for pos in range(4):
                    tr=sub[sub.position==pos].pivot(index=['behavior','question_id','trace_id'],columns='rank',values='token_id')
                    weights=sub[(sub.position==pos)&(sub['rank']==1)].set_index(['behavior','question_id','trace_id']).weight
                    tc=Counter()
                    for ix,r in tr.iterrows():tc[tuple(r)]+=weights.loc[ix]
                    best,wt=tc.most_common(1)[0]
                    triplets.append(dict(model=model,condition=cond,behavior=ct,position=pos,
                        unique_ordered_triplets=len(tc),dominant_triplet_share=wt/sum(tc.values()),
                        dominant_triplet=json.dumps([label(model,t) for t in best],ensure_ascii=False)))
                    for rank in range(1,4):
                        s=sub[(sub.position==pos)&(sub['rank']==rank)]
                        total=s.weight.sum();items=[]
                        for tok,g in s.groupby('token_id'):
                            items.append(dict(model=model,condition=cond,behavior=ct,position=pos,rank=rank,
                                token_id=tok,token=g.token.iloc[0],frequency=g.weight.sum()/total,
                                mean_probability=float(np.average(g.probability,weights=g.weight)),
                                min_probability=g.probability.min(),max_probability=g.probability.max(),
                                traces=len(g),questions=g.question_id.nunique()))
                        summaries.extend(items);items=sorted(items,key=lambda r:-r['frequency'])
                        show=items[:5];ax=axes[rank-1,pos];y=np.arange(len(show))
                        ax.barh(y,[r['frequency'] for r in show],color=COLORS[model],alpha=.8)
                        ax.set_yticks(y,[visible(r['token'])[:24]+f" [{r['token_id']}]" for r in show],fontsize=8);ax.invert_yaxis();ax.set_xlim(0,1.27)
                        for yi,r in zip(y,show):ax.text(r['frequency']+.015,yi,percent(r['frequency']),va='center',fontsize=8)
                        ax.set_xticks([0,.5,1],['0','50%','100%']);ax.grid(axis='x',alpha=.15)
                        ax.set_title(f'Position {pos}, rank {rank} | {len(items)} distinct',fontsize=9)
                        if rank==3:ax.set_xlabel('Frequency in this rank slot',fontsize=9)
                origin='historical mech traces, top-p .95' if model=='qwen32' else 'ext_final mech traces, top-p .95'
                fig.suptitle(f'{NAMES[model]} | {cond} | {ct}\nActual raw model-ranked alternatives along sampled histories; {origin}',fontsize=12)
                fig.tight_layout(rect=(0,0,1,.93));savefig(fig,f'top3_{model}_{cond}_{ct}')
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'ranked_token_frequencies.csv',index=False)
    pd.DataFrame(triplets).to_csv(OUT/'ranked_triplet_stability.csv',index=False)
    df.to_csv(OUT/'ranked_tokens_per_response.csv',index=False)
    return summary


def glm_empirical(st):
    fig,axes=plt.subplots(2,4,figsize=(15,6.5))
    for i,cond in enumerate(CONDS):
        for pos in range(4):
            s=st[(st.model=='glm')&(st.condition==cond)&(st.behavior=='pooled')&(st.position==pos)].head(5)
            ax=axes[i,pos];y=np.arange(len(s));ax.barh(y,s.frequency,color=COLORS['glm']);ax.set_yticks(y,[visible(t) for t in s.token],fontsize=9);ax.invert_yaxis();ax.set_xlim(0,1.15)
            for yi,r in zip(y,s.itertuples()):ax.text(r.frequency+.01,yi,f'{r.frequency:.1%}',va='center',fontsize=8)
            ax.set_title(f'{cond}\nPosition {pos}',fontsize=9);ax.grid(axis='x',alpha=.2)
            if i==1:ax.set_xlabel('Observed sampling frequency')
    fig.suptitle('GLM-5.2: frequently sampled tokens, pooled across four controls\nThese are NOT model rank-1/2/3 predictions; reasoning logprobs were unavailable',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.9));savefig(fig,'glm_sampled_tokens_NOT_logits')


def main():
    OUT.mkdir(exist_ok=True)
    INPUTS.add(Path(__file__).resolve())
    source=sources();records=load_sampled(source)
    assert len(records)==3600,len(records)
    assert all(r['source']!='unknown' for r in records)
    counts=Counter((r['model'],r['condition'],r['behavior']) for r in records)
    assert set(counts.values())=={150},counts
    metadata=pd.DataFrame([{k:v for k,v in r.items() if k!='ids'}|{'tokens':len(r['ids'])} for r in records])
    metadata.to_csv(OUT/'sample_metadata.csv',index=False)
    q=metadata.drop_duplicates(['model','condition','behavior','question_id'])
    q.groupby(['model','condition','behavior','source']).size().rename('questions').to_csv(OUT/'question_sources.csv')
    ds,pf,st=sampled_plots(records)
    ranked,excluded=load_ranked(source);rs=ranked_plots(ranked);glm_empirical(st)
    (OUT/'provenance.json').write_text(json.dumps(dict(
        tokenizer_paths=TOKPATHS,excluded_ranked_mismatched_traces=excluded,
        sources={str(p.relative_to(CTRL.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS)},
        sampled_responses=len(records),ranked_rows=len(ranked),
        note='All counts descriptive. Models use different native vocabularies. OLMo question pool and top-p differ. GLM top-three reasoning logits unavailable.'),indent=2))
    pages=['<html><head><meta charset="utf-8"><title>Opening-token research figures</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px}img{width:100%}section{margin:42px 0}a{color:#1265aa}code{background:#eee}</style></head><body>',
        '<h1>What is the initiation bottleneck?</h1><p>Research diagnostics, September 18, 2026. Position 0 is the first reasoning content token; protocol tokens and leading whitespace are excluded. Read <a href="README.md">methods and limitations</a>.</p>',
        '<p><b>Sampling versus prediction:</b> sampled-token counts describe realized outputs. Ranked-token panels identify the model’s first, second, and third most probable tokens at each sampled history. GLM ranked distributions are unavailable; its empirical panel is explicitly separate.</p>']
    for name,title in [('unique_tokens_pooled','Unique sampled tokens: pooled'),('unique_tokens_by_control','Unique sampled tokens: by control'),('unique_tokens_rarefied','Sample-size adjustment: expected unique tokens in 100 responses'),('prefix_concentration','Additional diagnostic: recurring opening sequences'),('glm_sampled_tokens_NOT_logits','GLM sampled tokens (not ranked logits)')]:
        pages.append(f'<section><h2>{title}</h2><a href="{name}.pdf">PDF</a><img src="{name}.png"></section>')
    for model in ['qwen32','olmo31']:
        for cond in CONDS:
            name=f'top3_{model}_{cond}_pooled'
            pages.append(f'<section><h2>{NAMES[model]} — {cond}</h2><p>Each panel asks: which token occupies this rank slot across contexts? A 100% bar means that token always occupies that slot. Probabilities and per-response histories are in the CSVs.</p><img src="{name}.png">')
            for ct in TYPES:pages.append(f'<a href="top3_{model}_{cond}_{ct}.png">{ct}</a> &nbsp;')
            pages.append('</section>')
    pages.append('<p>Tables: <a href="ranked_token_frequencies.csv">rank identities and probabilities</a>; <a href="ranked_triplet_stability.csv">ordered top-three stability</a>; <a href="ranked_tokens_per_response.csv">per-response histories</a>; <a href="prefix_concentration.csv">common prefix text</a>; <a href="question_sources.csv">question datasets</a>.</p></body></html>')
    (OUT/'index.html').write_text('\n'.join(pages))
    from make_opening_tokens_shareable import make_shareable
    make_shareable(OUT/'index.html')
    if (OUT/'matched/opening_concentration_entropy.png').exists():
        from plot_matched_openings import embed_gallery
        embed_gallery()
    print('Created figures from',len(records),'sampled responses and',len(ranked),'ranked-token records; excluded',excluded,'mismatched historical traces.')
    print(ds.query('behavior=="pooled" and position<4')[['model','condition','position','n','unique','mode_share']].to_string(index=False))


if __name__=='__main__':main()
