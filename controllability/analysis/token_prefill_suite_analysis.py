"""Question-clustered immediate and sustained continuation curves for the full suite."""
from pathlib import Path
import sys,json,argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));import token_prefill_suite as s
LABELS={'qwen32':'Qwen3-32B','qwen14':'Qwen3-14B','glm':'GLM-5.2','deepseek':'DeepSeek-R1','gptoss':'GPT-OSS-120B','olmo31':'OLMo-3.1-32B'}
from paper_style import (apply_style, export_figure, SQUARE_SIZE, BEHAVIOR_LABELS,
                         MODEL_COLORS, HEATMAP, LABEL_SIZE, TICK_SIZE, CELL_SIZE)

METRICS={'head32_soft':'Immediate continuation','full_soft':'Sustained continuation'}


def summarize(d,repeats=2000,behaviors=None):
 behaviors=list(s.BEHAVIORS if behaviors is None else behaviors)
 assert set(d.behavior.unique())==set(behaviors)
 keys=['model','behavior','condition','prefix_tokens']
 long=d.melt(id_vars=['question_id']+keys,value_vars=list(METRICS),var_name='metric',value_name='score')
 # Each question gets one mean per cell before bootstrapping. The same global
 # question draw is reused across behaviors, models, conditions, and metrics.
 q=long.groupby(['question_id']+keys+['metric']).score.mean()
 matrix=q.unstack(keys+['metric']);v=matrix.to_numpy(dtype=float);mask=np.isfinite(v)
 rng=np.random.default_rng(20260915);weights=rng.multinomial(len(matrix),np.ones(len(matrix))/len(matrix),size=repeats).astype(float)
 den=weights@mask.astype(float)
 with np.errstate(invalid='ignore',divide='ignore'):boots=(weights@np.nan_to_num(v))/den
 rows=[];aggregate_groups={}
 for j,col in enumerate(matrix.columns):
  model,behavior,condition,k,metric=col;finite=v[:,j][mask[:,j]]
  lo,hi=np.nanquantile(boots[:,j],[.025,.975]) if len(finite) else (np.nan,np.nan)
  rows.append(dict(zip(keys+['metric'],col),mean=float(finite.mean()) if len(finite) else np.nan,
                   low=lo,high=hi,questions=len(finite)))
  aggregate_groups.setdefault((model,condition,k,metric),[]).append(j)
 aggregate=[]
 for (model,condition,k,metric),columns in aggregate_groups.items():
  means=np.array([np.nanmean(v[:,j]) if mask[:,j].any() else np.nan for j in columns])
  # Refuse to silently drop an unassessable control from an equal-control mean.
  if len(columns)!=len(behaviors) or not np.isfinite(means).all():mean=low=high=np.nan
  else:
   mean=means.mean();low,high=np.nanquantile(boots[:,columns].mean(axis=1),[.025,.975])
  aggregate.append(dict(model=model,condition=condition,prefix_tokens=k,metric=metric,mean=mean,low=low,high=high,behaviors=int(np.isfinite(means).sum())))
 return pd.DataFrame(rows),pd.DataFrame(aggregate)


def make_table(summary,models,behaviors,out,suffix=''):
 """Match the sentence table using sustained scores at K=0 and K=64."""
 order=['leetspeak','spanish','chinese','uppercase_thinking','prefix_sentences',
        'no_first_person','omit_word','pirate','yoda','jar_jar_binks']
 behaviors=[b for b in order if b in behaviors]
 scores=summary[summary.metric=='full_soft'].set_index(['model','behavior','condition','prefix_tokens'])['mean']
 columns=[('base','prefill_only',0),('instr','prefill_plus_instruction',0),
          ('PO_k64','prefill_only',64),('PI_k64','prefill_plus_instruction',64)]
 rows=[]
 for behavior in behaviors:
  row={'behavior':behavior}
  for model in models:
   for label,condition,k in columns:row[model+'_'+label]=scores.loc[(model,behavior,condition,k)]
  rows.append(row)
 table=pd.DataFrame(rows);name='token_prefill_compare_table_k64'+suffix
 table.to_csv(out/(name+'.csv'),index=False)
 values=table.drop(columns='behavior').to_numpy(dtype=float)
 apply_style();fig,ax=plt.subplots(figsize=(max(11,2.6*len(models)),1.5+.4*len(behaviors)))
 ax.imshow(values,cmap=HEATMAP,vmin=0,vmax=1,aspect='auto')
 ax.set_xticks(range(values.shape[1]),['base','+i','PO','PI']*len(models),fontsize=TICK_SIZE)
 ax.set_yticks(range(len(behaviors)),[BEHAVIOR_LABELS[b] for b in behaviors],fontsize=TICK_SIZE)
 ax.xaxis.tick_top();ax.tick_params(length=0)
 for i in range(len(behaviors)):
  for j in range(values.shape[1]):
   value=values[i,j]
   ax.text(j,i,'—' if np.isnan(value) else f'{value:.2f}',ha='center',va='center',fontsize=CELL_SIZE,
           color='white' if np.isfinite(value) and (value<=.25 or value>=.75) else 'black')
 for j,model in enumerate(models):
  if j:ax.axvline(4*j-.5,color='white',lw=3)
  ax.text(4*j+1.5,1.09,LABELS[model],transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=LABEL_SIZE)
 fig.tight_layout(pad=.8);export_figure(fig,out,name)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--with-olmo',action='store_true');parser.add_argument('--exclude-styles',action='store_true');args=parser.parse_args()
 behaviors=[b for b in s.BEHAVIORS if not args.exclude_styles or b not in {'pirate','yoda','jar_jar_binks'}]
 suffix='_no_styles' if args.exclude_styles else ''
 scope=json.loads((s.ROOT/'scope.json').read_text());models=scope['active_models']
 roots={name:s.ROOT/name for name in models}
 if args.with_olmo:
  from token_prefill_olmo import ROOT as olmo_root
  models=models+['olmo31'];roots['olmo31']=olmo_root/'olmo31'
 frames=[]
 for name in models:
  d=pd.read_csv(roots[name]/'scores.csv');assert len(d)==24000
  assert not d.duplicated(['cell_id','condition','prefix_tokens','sample']).any()
  frames.append(d[d.behavior.isin(behaviors)])
 d=pd.concat(frames,ignore_index=True);out=HERE/'token_prefill'/(('with_olmo_v1' if args.with_olmo else 'full_v2')+suffix);out.mkdir(parents=True,exist_ok=True)
 summary,aggregate=summarize(d,behaviors=behaviors);summary.to_csv(out/'summary.csv',index=False);aggregate.to_csv(out/'aggregate.csv',index=False)
 quality=d.groupby(['model','behavior','condition','prefix_tokens']).agg(n=('sample','size'),questions=('question_id','nunique'),
   empty_rate=('empty_reasoning','mean'),truncation_rate=('truncated','mean'),full32_rate=('has_32_tokens','mean'),
   mean_reasoning_tokens=('reasoning_tokens','mean'),immediate_eligible=('head32_soft','count'),sustained_eligible=('full_soft','count')).reset_index()
 quality.to_csv(out/'quality.csv',index=False)
 make_table(summary,models,behaviors,out,suffix)
 def axes_format(ax):
  ax.set_xscale('symlog',base=2,linthresh=1);ax.set_xticks(s.LENGTHS,[str(k) for k in s.LENGTHS]);ax.set_ylim(-.02,1.02);ax.grid(alpha=.2)
 apply_style()
 def draw(ax, frame, color, label=None):
  for condition in s.CONDITIONS:
   z=frame[(frame.metric=='full_soft')&(frame.condition==condition)].sort_values('prefix_tokens')
   instructed=condition=='prefill_plus_instruction'
   alpha=1 if instructed else .45
   ax.errorbar(z.prefix_tokens,z['mean'],yerr=np.array([z['mean']-z.low,z.high-z['mean']]),
     color=color,ls='-' if instructed else '--',alpha=alpha,lw=2.4 if instructed else 1.6,
     marker='o',ms=4 if instructed else 3,capsize=2,elinewidth=1,
     label=label if instructed else None)
 fig,ax=plt.subplots(figsize=SQUARE_SIZE)
 for model in models:
  draw(ax,aggregate[aggregate.model==model],MODEL_COLORS[model],LABELS[model])
 axes_format(ax);ax.legend(loc='best')
 ax.set_xlabel('Supplied reasoning tokens');ax.set_ylabel('mean continuation compliance')
 fig.tight_layout()
 export_figure(fig,out,'token_prefill_compare_avg'+suffix,square=True)
 for model in models:
  nrows=(len(behaviors)+1)//2
  fig,axes=plt.subplots(nrows,2,figsize=(11,3*nrows),sharex=True,sharey=True)
  for ax,behavior in zip(axes.flat,behaviors):
   draw(ax,summary[(summary.model==model)&(summary.behavior==behavior)],MODEL_COLORS[model])
   ax.set_title(BEHAVIOR_LABELS.get(behavior,behavior.replace('_',' ')))
   axes_format(ax);ax.tick_params(labelbottom=True)
  for ax in list(axes.flat)[len(behaviors):]:ax.set_visible(False)
  fig.supxlabel('Supplied reasoning tokens');fig.supylabel('mean continuation compliance')
  fig.tight_layout()
  export_figure(fig,out,model+'_by_control',paper_name='token_prefill_'+model+'_by_control'+suffix)
 note='''# Full token-prefill analysis

Immediate continuation uses the first up to 32 new reasoning tokens; sustained
continuation uses the full new reasoning trace. Supplied prefixes and final
answers are excluded. Figures show sustained continuation only; both windows
remain in the summary tables. Solid lines retain the instruction; faded dashed
lines omit it. Word-boundary diagnostics are not plotted. Every sample,
including empty and capped reasoning, is retained. Language/style scores use the
historical graders; the same rubric is applied to each window.

For sentence-prefix control, only newly generated sentence starts are assessable;
an unfinished sentence begun in the supplied prefix is not scored as a new start.
A nonempty window with no generated sentence start is unassessable (missing), not
a success or failure. Eligibility counts are reported in quality.csv. Empty
reasoning receives zero. Word/pronoun suppression uses the supplied boundary as
lexical context but does not count violations wholly inside the supplied seed.

Intervals are pointwise 95% question-clustered bootstrap intervals (2,000 draws).
Samples are averaged within question first. The same question draws are shared
across controls, models, lengths, and conditions, preserving overlaps in the
question sets. Aggregate curves weight the ten controls equally. An aggregate
point is missing if any entire control cell is unassessable. These are soft
compliance means, not strict whole-trace success rates.

Hosted GLM prefixes round-trip through its pinned native tokenizer locally;
server token IDs/checkpoint hashes are unavailable. Its immediate-window tokens
are reconstructed from returned continuation text with that tokenizer. Modal
models retain the exact supplied and generated IDs. Two Chinese source cells
were replaced consistently across models to make API prefix round trips possible.
See the immutable manifests and source_replacements for the exact identities.

No DeepSeek or GPT-OSS generations were dispatched in the active reduced sweep.
'''
 if args.with_olmo:
  note+='''
OLMo-3.1-32B-Think uses pinned revision 832c3f543499af8fe68b88359501de9cb7840544,
its native chat template (which already opens <think>), and exact supplied and
generated token IDs. The multi-token </think> delimiter is excluded from scores.
Its source cells match the completed full_v2 sweep; manifests/results are stored
separately in results/token_prefill/olmo31_v1.

GLM remains provisional: the sentence/token audit found a large difference even
for identical prompts and prefixes across run dates. Hosted continuation handling
and model identity are not verified. Do not interpret GLM's difference from native
models mechanistically until that serving issue is resolved. See
../glm_sentence_audit_v1/README.md.
'''
 if args.exclude_styles:
  note=note.replace('ten controls equally','seven non-style controls equally')
  note+='\nPirate, Yoda, and Jar Jar Binks are excluded before aggregation and bootstrap resampling. Spanish and Chinese retain the existing language-judge scores.\n'
 (out/'README.md').write_text(note);print('Wrote',out)

if __name__=='__main__':main()
