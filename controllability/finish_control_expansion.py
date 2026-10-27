"""Rebuild complete analysis groups as their generation dependencies finish."""
import json
from pathlib import Path
import subprocess
import sys
import time
from control_expansion import ROOT, base


def complete(path, expected):
    if not path.exists():return False
    value=json.loads(path.read_text())
    return value.get('status')=='complete' and value.get('scored')==expected


def main():
    repo=Path(__file__).resolve().parents[1]
    state={'completed':[],'failures':{}}
    status=ROOT/'figure_rebuild_status.json'
    if status.exists():state=json.loads(status.read_text())
    while True:
        repair_ready=all(complete(ROOT/'numbered_icl_repair'/m/'scoring_report.json',n) for m,n in [('qwen14',150),('qwen32',330),('olmo31',330),('glm',330)])
        coordinator=ROOT/'coordinator_status.json'
        generated=coordinator.exists() and len(json.loads(coordinator.read_text()).get('primary_complete',[]))==4
        if repair_ready and generated and not state.get('repairs_merged'):
            for model in ['qwen14','qwen32','olmo31','glm']:
                subprocess.run([sys.executable,'controllability/control_expansion.py','score','--model',model],cwd=repo,check=True)
            state['repairs_merged']=True
            base.atomic_json(status,state)
        primary=all(complete(ROOT/m/'scoring_report.json',n) for m,n in [('qwen14',5700),('qwen32',6420),('olmo31',6420),('glm',6420)])
        primary=primary and repair_ready
        corr=list((ROOT/'correlation').glob('*/scoring_report.json'))
        correlation=len(corr)==12 and all(complete(p,300) for p in corr)
        old=repo/'controllability/results/ban_user_icl/seven_controls_extension_v1/glm/scoring_report.json'
        old_pipeline=old.parent.parent/'pipeline_report.json'
        ban=primary and old.exists() and json.loads(old.read_text()).get('scored')==1080 and old_pipeline.exists() and json.loads(old_pipeline.read_text()).get('status')=='complete'
        groups={
            'correlation':(correlation,[[sys.executable,'controllability/control_expansion.py','score','--model',p.parent.name,'--correlation'] for p in sorted(corr)]+[[sys.executable,'-m','monitorability.control_type_viewer'],[sys.executable,'-m','monitorability.convergent_validity'],[sys.executable,'-m','monitorability.plot_control_monitor_heatmap']]),
            'prefill':(primary,[[sys.executable,'controllability/analysis/control_expansion_figures.py','--only','prefill']]),
            'icl':(primary,[[sys.executable,'controllability/analysis/control_expansion_figures.py','--only','icl']]),
            'ban':(ban,[[sys.executable,'controllability/analysis/control_expansion_figures.py','--only','ban']]),
        }
        for name,(ready,commands) in groups.items():
            if not ready or name in state['completed'] or name in state['failures']:continue
            try:
                for cmd in commands:subprocess.run(cmd,cwd=repo,check=True)
            except Exception as exc:state['failures'][name]=repr(exc)
            else:state['completed'].append(name)
            state['updated_at']=time.time();base.atomic_json(status,state)
        if len(state['completed'])==4:return
        time.sleep(30)


if __name__=='__main__':main()
