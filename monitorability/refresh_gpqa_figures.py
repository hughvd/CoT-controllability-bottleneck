"""Freeze completed expanded GPQA cohorts and refresh the paper's current figures.

Incomplete models retain their original cohort; figures document cohort sizes.
No generation, monitor calls, or partial-cohort scoring.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pandas as pd

from monitorability.gpqa_figure_snapshot import ROOT, SELECTION


def main():
    run=SELECTION.parent
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    folder=run/'figure_snapshots'/timestamp
    folder.mkdir(parents=True)
    for name in ['convergent_validity','control_type_viewer']:
        shutil.copytree(ROOT/'results/monitorability'/name,folder/'before'/name)
    for stem in ['monitorability_by_control_spearman','monitorability_score_uncertainty_table','convergent_group_monitorability']:
        for ext in ['png','pdf']:
            source=ROOT/'paper/figures'/f'{stem}.{ext}'
            if source.exists():
                dest=folder/'before/paper_figures'/source.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    if SELECTION.exists():shutil.copy2(SELECTION,folder/'before/current_figure_snapshot.json')
    audit=json.loads((run/'audit.json').read_text());models={}
    rel='results/monitorability/convergent_validity/data.json'
    archive=run/'audit_inputs'/rel
    if not archive.exists():
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==audit['sources'][rel]
        archive.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,archive)
    for entry in audit['models']:
        model=entry['model'];p=run/model;status=p/'metrics_status.json'
        expanded=status.exists() and json.loads(status.read_text()).get('status')=='complete'
        if expanded:
            assert json.loads((p/'generation_status.json').read_text())['new_complete']==3192
            rows=pd.read_csv(p/'samples.csv',usecols=['instance_id','x'])
            assert len(rows)==3800 and rows.groupby(['instance_id','x']).size().eq(20).all()
            target=folder/model/'gmean_summary.csv';target.parent.mkdir();shutil.copy2(p/'gmean_summary.csv',target)
            models[model]=dict(cohort='expanded',questions=95,samples_per_arm=20,bootstrap_draws=1000,
                               summary_path=str(target.relative_to(ROOT)),summary_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        else:
            target=ROOT/'results/monitorability'/model/'gmean_summary.csv'
            models[model]=dict(cohort='original',questions=80,samples_per_arm=4,bootstrap_draws=200,
                               summary_path=str(target.relative_to(ROOT)),summary_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    count=sum(m['cohort']=='expanded' for m in models.values());assert count>0
    short=(f'GPQA: {count}/12 expanded cohorts (95 questions, 20/arm);\n'
           f'{12-count}/12 original cohorts (80 questions, 4/arm). Other evaluations unchanged.') if count<12 else 'GPQA: all 12 expanded cohorts (95 questions, 20 responses per arm).'
    note=short.replace('\n',' ')+' Expanded: '+', '.join(m for m,v in models.items() if v['cohort']=='expanded')+'.'
    snapshot=dict(created_at_utc=timestamp,provisional=count<12,expanded_models=count,models=models,note=note,figure_note=short)
    (folder/'snapshot.json').write_text(json.dumps(snapshot,indent=2)+'\n')
    tmp=SELECTION.with_suffix('.json.tmp');tmp.write_text(json.dumps(snapshot,indent=2)+'\n');tmp.replace(SELECTION)
    pd.DataFrame([dict(model=m,**v) for m,v in models.items()]).to_csv(folder/'coverage.csv',index=False)
    for module in ['control_type_viewer','convergent_validity','plot_control_monitor_heatmap','plot_monitorability_score_table']:
        with (folder/f'{module}.log').open('w') as log:
            subprocess.run([sys.executable,'-m','monitorability.'+module],cwd=ROOT,stdout=log,stderr=log,check=True)
    for stem in ['monitorability_by_control_spearman','control_by_control_spearman']:
        shutil.copy2(ROOT/'results/monitorability/convergent_validity'/f'{stem}.pdf',ROOT/'paper/figures'/f'{stem}.pdf')
    print(note)
    print('Backup and frozen GPQA sources:',folder)


if __name__=='__main__':main()
