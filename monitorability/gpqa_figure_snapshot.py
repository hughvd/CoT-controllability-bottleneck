"""Explicit GPQA figure snapshot; never reads an in-progress generation cache."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SELECTION=ROOT/'results/monitorability/gpqa_expansion/v1/current_figure_snapshot.json'


def apply_snapshot(data):
    if not SELECTION.exists():return data
    snapshot=json.loads(SELECTION.read_text())
    data['gpqa_snapshot']=snapshot
    data['sources'][str(SELECTION.relative_to(ROOT))]=hashlib.sha256(SELECTION.read_bytes()).hexdigest()
    for model in data['models']:
        entry=snapshot['models'][model['model']]
        if entry['cohort']!='expanded':continue
        path=ROOT/entry['summary_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['summary_sha256']
        row=pd.read_csv(path).query("scope=='cot_only' and dataset=='intervention/gpqa.jsonl'").iloc[0]
        mon=model['monitor'];mon['mon_gmean_gpqa']=float(row.gmean_mean)
        mon['mon_eligfrac_gpqa']=float(row.eligible_fraction_mean)
        mon['mon_gmean_intervention_mean']=float(np.mean([mon['mon_gmean_'+name] for name in
            ['gpqa','wmdp_sandbagging','memory_advice','scruples_suggest_wrong']]))
        mon['mon_composite_cot']=float(np.mean([mon[k] for k in
            ['mon_gmean_intervention_mean','mon_process_tpr_cot_only','mon_outcome_gmean_cot_only']]))
        model.setdefault('monitor_source_overrides',{})['mon_gmean_gpqa']=entry['summary_path']
        data['sources'][entry['summary_path']]=entry['summary_sha256']
    return data


def snapshot_note(data):
    return data.get('gpqa_snapshot',{}).get('note','')
