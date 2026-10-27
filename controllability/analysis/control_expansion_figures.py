"""Canonical eight-task paper figures. Never plots an incomplete experiment.

python controllability/analysis/control_expansion_figures.py [--only prefill|icl|ban]
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from control_expansion import ROOT, NEW, LENGTHS, base, sentences
from control_groups import GROUPS,CONTROLS,LABELS
from controllability.rescore_first_person import rescore, provenance as first_person_provenance
from paper_style import apply_style,export_figure,MODEL_COLORS

OUT=HERE/'control_expansion/v1'
MODELS=['qwen14','qwen32','olmo31','glm']
MODEL_LABELS={'qwen14':'Qwen3-14B','qwen32':'Qwen3-32B','olmo31':'OLMo-3.1-32B','glm':'GLM-5.2'}
OLD=[c for c in CONTROLS if c not in NEW]
SOURCES={}


def read(path):
    path=Path(path);SOURCES[str(path)]=base.file_hash(path)
    SOURCES.update(first_person_provenance())
    return rescore(pd.read_csv(path,low_memory=False),path)


def new(model,experiment):
    for path in [ROOT/model/'manifest.json',ROOT/'numbered_icl_repair'/model/'manifest.json']:
        SOURCES[str(path)]=base.file_hash(path)
    d=read(ROOT/model/'scores.csv')
    d=d[d.experiment.eq(experiment)].copy()
    expected={'prefill':4800,'icl':900,'ban_icl':720}[experiment]
    assert len(d)==expected,(model,experiment,len(d),expected)
    assert not d.request_id.duplicated().any()
    return d


def validate(d,keys,n):
    counts=d.groupby(['model','behavior','question_id']+keys).size()
    assert counts.eq(n).all(),counts[~counts.eq(n)]
    assert d.groupby(['model','behavior']).question_id.nunique().eq(30).all()
    assert set(d.behavior)==set(CONTROLS)


def stats(d,keys,nboot=2000,response_weighted=False,return_draws=False):
    """Question-cluster intervals; preserve each source analysis's weighting."""
    grouped=d.groupby(['question_id','model','behavior']+keys).compliance_score
    matrix=grouped.mean().unstack(['model','behavior']+keys);values=matrix.to_numpy();mask=np.isfinite(values)
    mass=grouped.count().unstack(['model','behavior']+keys).reindex_like(matrix).fillna(0).to_numpy() if response_weighted else mask.astype(float)
    totals=np.nan_to_num(values)*mass
    means=totals.sum(axis=0)/mass.sum(axis=0)
    weights=np.random.default_rng(20260924).multinomial(len(matrix),np.full(len(matrix),1/len(matrix)),size=nboot)
    with np.errstate(divide='ignore',invalid='ignore'):boots=(weights@totals)/(weights@mass)
    rows=[];aggregates={};draws={}
    for j,col in enumerate(matrix.columns):
        model,behavior,*conditions=col;v=values[:,j][mask[:,j]]
        lo,hi=np.nanquantile(boots[:,j],[.025,.975])
        rows.append(dict(model=model,behavior=behavior,**dict(zip(keys,conditions)),mean=means[j],low=lo,high=hi,n_questions=len(v)))
        if return_draws:draws[(model,behavior,*conditions)]=boots[:,j]
        aggregates.setdefault((model,*conditions),[]).append(j)
    agg=[]
    for (model,*conditions),cols in aggregates.items():
        assert len(cols)==len(CONTROLS)
        cell_means=means[cols]
        assert np.isfinite(cell_means).all(),'Unassessable entire control cell'
        low,high=np.nanquantile(boots[:,cols].mean(axis=1),[.025,.975])
        agg.append(dict(model=model,**dict(zip(keys,conditions)),mean=np.mean(cell_means),low=low,high=high))
        if return_draws:draws[(model,'average',*conditions)]=boots[:,cols].mean(axis=1)
    result=(pd.DataFrame(rows),pd.DataFrame(agg))
    return (*result,draws) if return_draws else result


def save(fig,name):
    export_figure(fig,OUT,name)
    paper=HERE.parents[1]/'paper/figures'
    shutil.copy2(OUT/(name+'.pdf'),paper/(name+'.pdf'))


def table(summary,models,columns,name):
    """columns: (display label, selector dict). One column block per model."""
    width=len(columns);values=np.empty((len(CONTROLS),len(models)*width));records=[]
    for j,m in enumerate(models):
        for k,(label,selector) in enumerate(columns):
            z=summary[summary.model.eq(m)]
            for key,value in selector.items():z=z[z[key].eq(value)]
            z=z.set_index('behavior').reindex(CONTROLS)
            assert not z['mean'].isna().any()
            values[:,j*width+k]=z['mean']
            for b,v in zip(CONTROLS,z['mean']):records.append(dict(model=m,group=next(g for g,bs in GROUPS.items() if b in bs),behavior=b,condition=label,mean_compliance=v))
    fig,ax=plt.subplots(figsize=(max(11,len(models)*3.1),5.2))
    ax.imshow(values,cmap='RdYlGn',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(values.shape[1]),[label for label,_ in columns]*len(models),fontsize=12)
    ax.set_yticks(range(len(CONTROLS)),[LABELS[b] for b in CONTROLS],fontsize=12)
    ax.xaxis.tick_top();ax.tick_params(length=0)
    for i,row in enumerate(values):
        for j,v in enumerate(row):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=12,color='white' if v<.2 or v>.8 else '#111')
    start=0
    for group,bs in GROUPS.items():
        if start:ax.axhline(start-.5,color='white',lw=3)
        ax.text(-.29,start+(len(bs)-1)/2,group,transform=ax.get_yaxis_transform(),rotation=90,ha='center',va='center',fontsize=12,fontweight='bold')
        start+=len(bs)
    for j,m in enumerate(models):
        if j:ax.axvline(j*width-.5,color='white',lw=3)
        ax.text(j*width+(width-1)/2,1.12,MODEL_LABELS[m],transform=ax.get_xaxis_transform(),ha='center',fontsize=14)
    fig.subplots_adjust(left=.23,right=.995,bottom=.035,top=.81)
    save(fig,name);pd.DataFrame(records).to_csv(OUT/(name+'.csv'),index=False)


def prefill():
    frames=[];coverage=[]
    for model in MODELS:
        matched=HERE.parent/'results/token_prefill/model_source_v1'/model
        if (matched/'manifest.json').exists():
            report=json.loads((matched/'scoring_report.json').read_text()) if (matched/'scoring_report.json').exists() else {}
            assert report.get('status')=='complete','Model-source prefill is not complete: '+model
            for path,digest in report['reused_k0_sources'].items():
                assert base.file_hash(path)==digest,'Reused baseline source changed: '+path
                SOURCES[path]=digest
            d=read(matched/'scores.csv');frames.append(d)
            for filename in ['manifest.json','examples.json','sources.json']:
                SOURCES[str(matched/filename)]=base.file_hash(matched/filename)
            m=json.loads((matched/'manifest.json').read_text())
            for r in m['requests']:
                if r['behavior']!='suffix_sentences' or r['sample']!=0 or r['condition']!='instruction_only':continue
                text=r['prefix_text'];spans=sentences(text)
                complete=any(text[a:b].rstrip().endswith(('.', '!', '?')) and s.rstrip('.!?').rstrip().endswith(r['marker']) for a,b,s in spans)
                coverage.append(dict(model=model,question_id=r['question_id'],prefix_tokens=r['prefix_tokens'],contains_completed_marker=complete))
                if r['prefix_tokens']==1:coverage.append(dict(model=model,question_id=r['question_id'],prefix_tokens=0,contains_completed_marker=False))
            continue
        folder=HERE.parent/'results/token_prefill'/('olmo31_v1/olmo31' if model=='olmo31' else 'full_v2/'+model)
        d=read(folder/'scores.csv');d=d[d.behavior.isin(OLD)].copy()
        d['compliance_score']=d.full_soft
        d['condition']=d.condition.map({'prefill_only':'no_instruction','prefill_plus_instruction':'instruction_only'})
        d['model']=model
        n=new(model,'prefill');n['model']=model
        frames.extend([d,n])
        m=json.loads((ROOT/model/'manifest.json').read_text())
        for r in m['requests']:
            if r['experiment']!='prefill' or r['behavior']!='suffix_sentences' or r['sample']!=0 or r['condition']!='instruction_only':continue
            text=r['prefix_text'];spans=sentences(text)
            complete=any(text[a:b].rstrip().endswith(('.', '!', '?')) and s.rstrip('.!?').rstrip().endswith(r['marker']) for a,b,s in spans)
            coverage.append(dict(model=model,question_id=r['question_id'],prefix_tokens=r['prefix_tokens'],contains_completed_marker=complete))
    d=pd.concat(frames,ignore_index=True);validate(d,['condition','prefix_tokens'],5)
    summary,aggregate=stats(d,['condition','prefix_tokens']);summary.to_csv(OUT/'prefill_summary.csv',index=False);aggregate.to_csv(OUT/'prefill_aggregate.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'suffix_prefill_coverage_by_question.csv',index=False)
    pd.DataFrame(coverage).groupby(['model','prefix_tokens']).contains_completed_marker.agg(['count','sum','mean']).to_csv(OUT/'suffix_prefill_coverage.csv')
    d.groupby(['model','behavior','condition','prefix_tokens']).compliance_score.agg(['size','count']).to_csv(OUT/'prefill_eligibility.csv')
    table(summary,MODELS,[('Base',dict(condition='no_instruction',prefix_tokens=0)),('+I',dict(condition='instruction_only',prefix_tokens=0)),('PI@1',dict(condition='instruction_only',prefix_tokens=1)),('PI@64',dict(condition='instruction_only',prefix_tokens=64))],'token_prefill_compare_table_k64_no_styles')
    def draw(ax,z,model,label=None):
        for c in ['no_instruction','instruction_only']:
            x=z[z.condition.eq(c)].sort_values('prefix_tokens');color=MODEL_COLORS[model]
            ax.plot(x.prefix_tokens,x['mean'],color=color,ls='-' if c=='instruction_only' else '--',alpha=1 if c=='instruction_only' else .45,label=label if c=='instruction_only' else None)
            ax.fill_between(x.prefix_tokens,x.low,x.high,color=color,alpha=.09)
        ax.set_xscale('symlog',base=2,linthresh=1);ax.set_xticks(LENGTHS,[str(k) for k in LENGTHS]);ax.set_ylim(-.02,1.02);ax.grid(alpha=.2)
    fig,ax=plt.subplots(figsize=(5.5,5.5))
    for m in MODELS:draw(ax,aggregate[aggregate.model.eq(m)],m,MODEL_LABELS[m])
    condition_handles=[Line2D([],[],color='#333333',ls='-',label='Instruction'),
                       Line2D([],[],color='#333333',ls='--',alpha=.45,label='No instruction')]
    handles,_=ax.get_legend_handles_labels()
    ax.set_xlabel('Supplied reasoning tokens');ax.set_ylabel('Mean continuation compliance');ax.legend(handles=handles+condition_handles,fontsize=10);fig.tight_layout();save(fig,'token_prefill_compare_avg_no_styles')
    for m in MODELS:
        fig,axes=plt.subplots(4,2,figsize=(11,12),sharex=True,sharey=True)
        for ax,b in zip(axes.flat,CONTROLS):
            draw(ax,summary[summary.model.eq(m)&summary.behavior.eq(b)],m);ax.set_title(LABELS[b]);ax.tick_params(labelbottom=True)
        fig.legend(handles=condition_handles,loc='upper center',ncol=2,fontsize=12)
        fig.supxlabel('Supplied reasoning tokens');fig.supylabel('Mean continuation compliance');fig.tight_layout(rect=(0,0,1,.965));save(fig,'token_prefill_'+m+'_by_control_no_styles')


def icl():
    names={'qwen14':'qwen3_14b','qwen32':'qwen3_32b','olmo31':'olmo_3_1_32b','glm':'glm_5_2'};frames=[]
    variants={'instruction_only':'instruction_only','icl_user_plain':'user_icl','icl_assistant_visible':'assistant_icl'}
    for model,name in names.items():
        d=read(HERE.parent/'results'/('iclfull_'+name+'_donorfix_v1_icl_per_sample.csv'))
        d=d[d.ctype.isin(OLD)&d.variant.isin(variants)].copy();d['behavior']=d.ctype;d['condition']=d.variant.map(variants)
        d['question_id']=d.target_question.map(base.digest);d['empty_reasoning']=d.empty_thinking;d['model']=model
        n=new(model,'icl');n['model']=model;frames.extend([d,n])
    d=pd.concat(frames,ignore_index=True);validate(d,['condition'],5)
    d.groupby(['model','behavior','condition']).empty_reasoning.agg(['size','sum']).to_csv(OUT/'icl_empty_reasoning.csv')
    d=d[~d.empty_reasoning.astype(bool)]
    summary,agg=stats(d,['condition'],5000,response_weighted=True);summary.to_csv(OUT/'icl_summary.csv',index=False)
    columns=[('Instr.',dict(condition='instruction_only')),('User ICL',dict(condition='user_icl')),('Asst. ICL',dict(condition='assistant_icl'))]
    table(summary,MODELS,columns,'icl_compare_no_styles_main')
    bars(agg,MODELS,[(c[0],c[1]['condition']) for c in columns],'icl_avg_no_styles_main',ymax=.8,legend_loc='upper left',legend_fontsize=plt.rcParams['axes.labelsize'])


def bars(agg,models,conditions,name,accuracy=None,ymax=1.02,legend_loc='best',legend_fontsize=10):
    fig,axes=plt.subplots(1 if accuracy is None else 2,1,figsize=(6.5,4.4 if accuracy is None else 7.4),sharex=True,squeeze=False)
    width=.8/len(conditions);colors=['#999999','#377eb8','#ffb04a','#4daf4a']
    for ax,frame,label_y in zip(axes.flat,[agg] if accuracy is None else [agg,accuracy],['Mean compliance','Task accuracy']):
        for j,(label,condition) in enumerate(conditions):
            z=frame[frame.condition.eq(condition)].set_index('model').loc[models];x=np.arange(len(models))+(j-(len(conditions)-1)/2)*width
            ax.bar(x,z['mean'],width,label=label,color=colors[j],edgecolor='white',lw=.5)
            ax.vlines(x,z.low,z.high,color='#222',lw=1)
        ax.set_ylim(0,ymax);ax.set_ylabel(label_y);ax.grid(axis='y',alpha=.2)
    axes[-1,0].set_xticks(range(len(models)),[MODEL_LABELS[m] for m in models],rotation=25,ha='right')
    axes[0,0].legend(fontsize=legend_fontsize,ncol=2,loc=legend_loc);fig.tight_layout();save(fig,name)
    agg.to_csv(OUT/(name+'.csv'),index=False)


def ban():
    models=['qwen32','olmo31','glm'];frames=[]
    for model in models:
        for root in ['v1','seven_controls_extension_v1']:
            path=HERE.parent/'results/ban_user_icl'/root/model/'scores.csv'
            d=read(path);d=d[d.behavior.isin(OLD)].copy();d['compliance_score']=d.whole_soft;d['correct']=d.answer_correct;d['model']=model;frames.append(d)
        n=new(model,'ban_icl');n['model']=model;frames.append(n)
    d=pd.concat(frames,ignore_index=True)
    # GLM factorial discontinued: retain only its completed paired ban comparison.
    d=d[~(d.model.eq('glm') & d.condition.eq('user_icl'))].copy()
    validate(d,['condition','arm'],3)
    assert len(d)==(2*4+2)*8*30*3
    factorial_models=['qwen32','olmo31']
    d['condition']=d.condition+'|'+d.arm
    summary,agg,draws=stats(d,['condition'],5000,return_draws=True);summary.to_csv(OUT/'ban_summary.csv',index=False)
    columns=[('Instr.',dict(condition='instruction_only|baseline')),('Ban',dict(condition='instruction_only|fixed_ban')),('User\nICL',dict(condition='user_icl|baseline')),('ICL\n+ban',dict(condition='user_icl|fixed_ban'))]
    contrasts={'ban_without_icl':[-1,1,0,0],'icl_without_ban':[-1,0,1,0],
               'combo_minus_icl':[0,0,-1,1],'combo_minus_instruction':[-1,0,0,1],
               'interaction':[1,-1,-1,1]}
    effects=[]
    means=pd.concat([summary,agg.assign(behavior='average')]).set_index(['model','behavior','condition'])['mean']
    for model in models:
        for behavior in CONTROLS+['average']:
            for contrast,coefs in contrasts.items():
                if model=='glm' and contrast!='ban_without_icl':continue
                active=[(coef,(model,behavior,x['condition'])) for coef,(_,x) in zip(coefs,columns) if coef]
                coefs,keys=zip(*active)
                lo,hi=np.nanquantile(np.array(coefs)@np.array([draws[k] for k in keys]),[.025,.975])
                effects.append(dict(model=model,behavior=behavior,contrast=contrast,
                    delta=np.dot(coefs,[means[k] for k in keys]),low=lo,high=hi))
    pd.DataFrame(effects).to_csv(OUT/'ban_effects.csv',index=False)
    table(summary,factorial_models,columns,'ban_user_icl_by_control')
    accuracy=d.copy();accuracy['compliance_score']=accuracy.correct
    _,acc=stats(accuracy,['condition'],5000);acc.to_csv(OUT/'ban_accuracy.csv',index=False)
    bars(agg[agg.model.isin(factorial_models)],factorial_models,[(label,x['condition']) for label,x in columns],'ban_user_icl_average',acc)
    bars(agg[agg.model.isin(factorial_models)],factorial_models,[(label,x['condition']) for label,x in columns],'ban_user_icl_average_compliance_only',ymax=.8)
    selected=agg[agg.condition.str.startswith('instruction_only')]
    bars(selected,models,[(label,x['condition']) for label,x in columns[:2]],'fixed_opening_ban_average',acc)
    table(summary,models,columns[:2],'fixed_opening_ban_by_control')
    d.groupby(['model','behavior','condition']).agg(responses=('sample','size'),empty=('empty_reasoning','sum'),truncated=('truncated','sum')).to_csv(OUT/'ban_quality.csv')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',choices=['prefill','icl','ban']);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);apply_style()
    for name in [args.only] if args.only else ['prefill','icl','ban']:globals()[name]()
    provenance=dict(groups=GROUPS,controls=CONTROLS,sources=SOURCES,analysis_sha256=base.file_hash(__file__))
    base.atomic_json(OUT/('provenance_'+(args.only or 'all')+'.json'),provenance)

if __name__=='__main__':main()
