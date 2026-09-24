"""Collect the approved OLMo worker, grade its outputs, and refresh paper figures."""
import json,os,subprocess,sys,time
import modal
import token_prefill_olmo as o

def main():
 o.activate();s=o.s;root=o.ROOT;logs=root/'logs';logs.mkdir(exist_ok=True)
 manifest=json.loads((root/'olmo31/manifest.json').read_text());volume=modal.Volume.from_name('token-prefill-full-out')
 remote='olmo31/'+manifest['manifest_sha256']+'/report.json';last=None
 def status(**kwargs):s.base.atomic_json(root/'coordinator_status.json',dict(updated_at=time.time(),**kwargs))
 def run(args,label):
  with (logs/(label+'.log')).open('a') as f:
   subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=s.HERE.parent,env={**os.environ,'MPLBACKEND':'Agg'})
 try:
  while True:
   attempt=json.loads((root/'gpu_ledger.json').read_text())['attempts'][-1]
   try:report=json.loads(b''.join(volume.read_file(remote)))
   except FileNotFoundError:report={'status':'pending','batches':[]}
   if report.get('attempt_id')!=attempt['attempt_id']:report={'status':'pending','batches':[]}
   state=(report['status'],report.get('reused_outputs',0)+sum(b['outputs'] for b in report['batches']))
   if state!=last:print(state,flush=True);last=state
   status(status='generation',generation=state)
   if report['status'] in ['complete','failed','budget_stop','pilot_quality_stop','projection_exceeds_budget','runtime_rollover']:
    run(['modal','run','controllability/modal_token_prefill_suite.py','--phase','collect','--model','olmo31'],'collect')
    if report['status']=='runtime_rollover':
     run(['modal','run','--detach','controllability/modal_token_prefill_suite.py','--phase','dispatch','--model','olmo31'],'dispatch')
     continue
    if report['status']!='complete':raise RuntimeError('Generation stopped: '+report['status'])
    break
   try:
    result=modal.FunctionCall.from_id(attempt['call_id']).get(timeout=0)
    if result.get('status')!='complete':raise RuntimeError('Worker stopped: '+str(result))
   except TimeoutError:pass
   time.sleep(30)
  status(status='scoring');run([sys.executable,'-u','controllability/token_prefill_olmo.py','score'],'score')
  status(status='plotting');run([sys.executable,'controllability/analysis/token_prefill_suite_analysis.py','--with-olmo'],'analysis')
  status(status='validating');run([sys.executable,'controllability/audit_token_prefill_olmo.py'],'audit')
  status(status='complete');print('OLMo generated, scored, and plotted.',flush=True)
 except Exception as exc:
  status(status='failed',error=str(exc));raise

if __name__=='__main__':main()
