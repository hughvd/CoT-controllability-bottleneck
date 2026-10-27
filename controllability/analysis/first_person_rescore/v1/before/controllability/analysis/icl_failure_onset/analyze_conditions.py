"""Offline onset audit of the current eight-task instruction-only ICL sample.
Token boundaries are reconstructed from saved reasoning, not native generated IDs.
"""
from pathlib import Path
import sys,json,re,hashlib,os
import pandas as pd
from tokenizers import Tokenizer
HERE=Path(__file__).resolve().parent
C=HERE.parents[1];sys.path[:0]=[str(C),str(C/'mech')]
from control_expansion import PRIMARY
import suite_rank_probe as rank
import transforms as tr
from control_groups import CONTROLS
NAMES={'qwen14':'qwen3_14b','qwen32':'qwen3_32b','olmo31':'olmo_3_1_32b','glm':'glm_5_2'}
OPENING=['uppercase_thinking','leetspeak','prefix_sentences','numbered_sentences']
CONDITION=os.environ.get('ICL_CONDITION','instruction_only')
VARIANT={'instruction_only':'instruction_only','user_icl':'icl_user_plain','assistant_icl':'icl_assistant_visible'}[CONDITION]
OUT=HERE/CONDITION;OUT.mkdir(exist_ok=True)
SOURCES={}
def read(p):
 SOURCES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p,low_memory=False)
def onset(text,task,instruction):
 if task=='uppercase_thinking':return next((i for i,c in enumerate(text) if c.isalpha() and not c.isupper()),None)
 if task=='leetspeak':return next((i for i,c in enumerate(text) if c in 'aAeEiIoOtTsS'),None)
 if task in ['prefix_sentences','numbered_sentences']:
  marker=re.search(r"'([^']+)'",instruction).group(1) if task=='prefix_sentences' else None
  v=rank.scan(text,task,marker)[1];return v['character_position'] if v else None
 if task=='omit_word':
  word=re.search(r"'([^']+)'",instruction).group(1);m=re.search(r'\b'+re.escape(word)+r'\b',text,re.I);return m.end()-1 if m else None
 if task=='no_first_person':
  for m in re.finditer(r"[A-Za-z]+(?:[.'][A-Za-z]+)+\.?|[A-Za-z]+",text):
   if m.group()!='US' and m.group().lower() in tr._FP_WORDS:return m.end()-1
 return None
rows=[]
for model,name in NAMES.items():
 spec=PRIMARY[model];p=Path.home()/'.cache/huggingface/hub'/('models--'+spec['id'].replace('/','--'))/'snapshots'/spec['revision']/'tokenizer.json';tok=Tokenizer.from_file(str(p));SOURCES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 old=read(C/'results'/f'iclfull_{name}_donorfix_v1_icl_per_sample.csv');old=old[old.variant.eq(VARIANT)&old.ctype.isin(CONTROLS)].copy();old['behavior']=old.ctype;old['empty']=old.empty_thinking;old['question']=old.target_question
 new=read(C/'results/control_expansion/v1'/model/'scores.csv');new=new[new.experiment.eq('icl')&new.condition.eq(CONDITION)].copy();new['empty']=new.empty_reasoning
 manifestpath=C/'results/control_expansion/v1'/model/'manifest.json';SOURCES[str(manifestpath)]=hashlib.sha256(manifestpath.read_bytes()).hexdigest();requests={r['request_id']:r for r in json.loads(manifestpath.read_text())['requests']}
 new['instruction']=[re.search(r'<cot_instructions>(.*?)</cot_instructions>',[m['content'] for m in requests[r]['messages'] if m['role']=='user'][-1],re.S).group(1) for r in new.request_id]
 d=pd.concat([old,new]);assert d.groupby('behavior').size().eq(150).all() and len(d)==1200
 for _,r in d.iterrows():
  text=str(r.reasoning) if pd.notna(r.reasoning) else '';task=r.behavior;eligible=task not in ['spanish','suffix_sentences'] and bool(text.strip());ci=onset(text,task,r.instruction) if eligible else None
  # Encode the whole saved field: preserve leading whitespace, separately identify first content token.
  enc=tok.encode(text,add_special_tokens=False);content=next((i for i,(a,b) in enumerate(enc.offsets) if text[a:b].strip()),None)
  ti=next((i for i,(a,b) in enumerate(enc.offsets) if ci is not None and a<=ci<b),None)
  rows.append(dict(model=model,behavior=task,question=r.question,sample=int(r['sample']),condition=CONDITION,score=r.compliance_score,failed_half=r.compliance_score<.5,failed_strict=r.compliance_score<1-1e-9,eligible=eligible,opening_task=task in OPENING,empty=not text.strip(),violation_char=ci,violation_token=ti,first_content_token=content,at_token_zero=ti==0 if ti is not None else False,at_first_content_token=ti==content if ti is not None else False,first_token_text=text[slice(*enc.offsets[0])] if enc.offsets else '',first_content_text=text[slice(*enc.offsets[content])] if content is not None else '',opening=text[:180]))
d=pd.DataFrame(rows);d.to_csv(OUT/'per_response.csv',index=False)
summary=[]
for definition in ['failed_half','failed_strict']:
 for scope,mask in [('opening_tasks',d.opening_task),('six_local_tasks',d.eligible),('all_eight_lower_bound',pd.Series(True,index=d.index))]:
  for model in [*NAMES,'pooled']:
   z=d[mask&d[definition]&~d['empty']&(True if model=='pooled' else d.model.eq(model))]
   summary.append(dict(definition=definition,scope=scope,model=model,failed=len(z),localized=z.violation_token.notna().sum(),first_content=z.at_first_content_token.sum(),token_zero=z.at_token_zero.sum(),first_content_pct=100*z.at_first_content_token.mean(),token_zero_pct=100*z.at_token_zero.mean()))
pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
d[d.failed_half].groupby(['model','behavior']).agg(failed=('score','size'),first_content=('at_first_content_token','sum'),token_zero=('at_token_zero','sum'),localized=('violation_token','count')).to_csv(OUT/'by_control.csv')
(OUT/'provenance.json').write_text(json.dumps(dict(condition=CONDITION,sources=SOURCES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),failure_threshold=.5,excluded=['spanish: language grading has no token-local rule','suffix_sentences: marker placement ambiguous'],tokenization='Whole saved reasoning retokenized with pinned tokenizer; no native token-ID claim. First non-whitespace token and literal token zero reported separately.',suppression='Retrospective token containing final character of first forbidden whole word; not a causal irreversible-failure detector.'),indent=2))
print(pd.DataFrame(summary).to_string(index=False))
