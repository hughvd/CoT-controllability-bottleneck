"""Local coordinator: sequential GPU workers; collect, score, and grade durably."""
import json
import subprocess
import os
import shutil
import sys
import time
from pathlib import Path
import modal
import token_prefill_suite as s


def main():
 volume=modal.Volume.from_name('token-prefill-full-out');last={};scorers={};handled=set()
 status_path=s.ROOT/'coordinator_status.json';logs=s.ROOT/'logs';logs.mkdir(exist_ok=True)
 def run(args,label):
  with (logs/(label+'.log')).open('a') as f:subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=s.HERE.parent,env={**os.environ,'MPLBACKEND':'Agg'})
 def launch_score(name):
  f=(logs/(name+'_score.log')).open('a')
  p=subprocess.Popen([sys.executable,'-u','controllability/token_prefill_suite_score.py','all','--model',name],cwd=s.HERE.parent,stdout=f,stderr=subprocess.STDOUT)
  scorers[name]=(p,f);print('Scoring started:',name,flush=True)
 while True:
  scope=json.loads((s.ROOT/'scope.json').read_text());models=scope['active_models']
  ledger=json.loads((s.ROOT/'gpu_ledger.json').read_text())
  for name in ['qwen32','qwen14']:
   if name not in models:continue
   attempts=[a for a in ledger['attempts'] if a['model']==name]
   if not attempts:continue
   m=json.loads((s.ROOT/name/'manifest.json').read_text())
   try:report=json.loads(b''.join(volume.read_file(name+'/'+m['manifest_sha256']+'/report.json')))
   except (FileNotFoundError,modal.exception.NotFoundError):continue
   state=(report['status'],report.get('reused_outputs',0)+sum(x['outputs'] for x in report['batches']))
   if last.get(name)!=state:print(name,state,flush=True);last[name]=state
   if report['status'] in ['failed','budget_stop','pilot_quality_stop','projection_exceeds_budget']:
    run(['modal','run','controllability/modal_token_prefill_suite.py','--phase','collect','--model',name],name+'_collect')
    raise RuntimeError(name+' stopped: '+report['status'])
   if report['status']=='complete' and name not in handled:
    run(['modal','run','controllability/modal_token_prefill_suite.py','--phase','collect','--model',name],name+'_collect')
    handled.add(name)
    if name=='qwen32' and 'qwen14' in models and not any(a['model']=='qwen14' for a in ledger['attempts']):
     run(['modal','run','--detach','controllability/modal_token_prefill_suite.py','--phase','dispatch','--model','qwen14'],'qwen14_dispatch')
  if 'glm' in models and (s.ROOT/'glm/api_report.json').exists():
   r=json.loads((s.ROOT/'glm/api_report.json').read_text());state=(r['status'],r.get('outputs',0))
   if last.get('glm')!=state:print('glm',state,flush=True);last['glm']=state
   if r['status']=='failed':raise RuntimeError('GLM API failed: '+r.get('error',''))
   if r['status']=='complete' and 'glm' not in handled:handled.add('glm')
  if not any(p.poll() is None for p,f in scorers.values()):
   ready=[name for name in models if name in handled and name not in scorers]
   if ready:launch_score(ready[0])
  score_states={}
  for name,(p,f) in scorers.items():
   code=p.poll();score_states[name]='running' if code is None else ('complete' if code==0 else 'failed')
   if code is not None and code!=0:raise RuntimeError(name+' scoring failed; see log')
  s.base.atomic_json(status_path,dict(generation=last,scoring=score_states,updated_at=time.time()))
  if all(score_states.get(n)=='complete' for n in models):
   run([sys.executable,'controllability/analysis/token_prefill_suite_analysis.py'],'analysis')
   source=s.HERE/'analysis/token_prefill/full_v2/token_prefill_compare_avg.png'
   shutil.copyfile(source,s.HERE.parent/'paper/figures/token_prefill_compare_avg.png')
   s.base.atomic_json(status_path,dict(status='complete',generation=last,scoring=score_states,updated_at=time.time()))
   print('All active models generated, scored, and plotted.',flush=True);return
  time.sleep(30)

if __name__=='__main__':main()
