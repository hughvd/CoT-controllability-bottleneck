"""Validate a completed OLMo addition and save a compact reproducible results note."""
import json
from pathlib import Path
import pandas as pd
import token_prefill_olmo as o

def main():
 root=o.ROOT;m=json.loads((root/'olmo31/manifest.json').read_text());o.s.validate(m)
 assert m['adapter_sha256']==o.s.base.file_hash(o.__file__)
 requests={r['request_id']:r for r in m['requests']};seen=set();report=json.loads((root/'olmo31/remote/report.json').read_text())
 assert report['status']=='complete'
 for p in (root/'olmo31/remote').glob('batch_*.json'):
  batch=json.loads(p.read_text());assert batch['manifest_sha256']==m['manifest_sha256']
  for raw in batch['outputs']:
   key=(raw['request_id'],raw['sample']);assert key not in seen;seen.add(key)
   assert raw['prompt_token_ids']==requests[raw['request_id']]['prompt_token_ids']
   assert isinstance(raw['output_token_ids'],list)
 assert seen=={(rid,i) for rid in requests for i in range(5)}
 d=pd.read_csv(root/'olmo31/scores.csv');assert len(d)==24000
 assert not d.duplicated(['cell_id','condition','prefix_tokens','sample']).any()
 assert (d.groupby(['behavior','condition','prefix_tokens']).size()==150).all()
 assert d.manifest_sha256.eq(m['manifest_sha256']).all()
 assert d.loc[d.needs_judge,['head32_soft','full_soft']].notna().all().all()
 for c in ['head32_soft','full_soft']:assert d[c].dropna().between(0,1).all()
 out=o.s.HERE/'analysis/token_prefill/with_olmo_v1';aggregate=pd.read_csv(out/'aggregate.csv')
 figures={}
 for name,paper_name in [('token_prefill_compare_avg','token_prefill_compare_avg'),('olmo31_by_control','token_prefill_olmo31_by_control')]:
  path=out/(name+'.png');paper=o.s.HERE.parent/'paper/figures'/(paper_name+'.png')
  assert o.s.base.file_hash(path)==o.s.base.file_hash(paper)
  figures[str(path.relative_to(o.s.HERE))]=o.s.base.file_hash(path)
 ledger=json.loads((root/'gpu_ledger.json').read_text())
 gpu_estimate=sum(a['accounted_seconds'] for a in ledger['attempts'])*ledger['usd_per_hour']/3600
 grade=json.loads((root/'olmo31/grading_report.json').read_text());assert grade['status']=='complete'
 audit=dict(status='complete',manifest_sha256=m['manifest_sha256'],unique_outputs=len(seen),observations=len(d),
  empty_rate=float(d.empty_reasoning.mean()),truncation_rate=float(d.truncated.mean()),
  closed_reasoning_rate=float(d.closed_reasoning.mean()),estimated_gpu_usd=gpu_estimate,
  tracked_api_usd=grade['tracked_shared_api_usd'],figure_hashes=figures)
 o.s.base.atomic_json(root/'completion_validation.json',audit)
 rows=['# OLMo token-prefill addition','',
 'OLMo-3.1-32B-Think uses the same 300 question/control cells, ten controls, eight prefix lengths, two instruction conditions, and five samples as the completed full_v2 sweep. All 24,000 linked observations are included; 23,920 unique generations preserve exact prompt and output token IDs. Supplied text and final answers are excluded from continuation scores.', '',
 '| Condition | K=0 | K=1 | K=4 | K=64 |','|---|---:|---:|---:|---:|']
 for condition in o.s.CONDITIONS:
  z=aggregate[(aggregate.model=='olmo31')&(aggregate.metric=='full_soft')&(aggregate.condition==condition)].set_index('prefix_tokens')
  rows.append('| '+condition+' | '+' | '.join(f'{z.loc[k,"mean"]:.3f}' for k in [0,1,4,64])+' |')
 rows+=['','Values are equally weighted means of the ten control soft-compliance scores. Figures show pointwise 95% question-clustered bootstrap intervals; immediate continuation scores are also retained in the CSV summaries.', '',
 f'Empty reasoning: {audit["empty_rate"]:.2%}; capped outputs: {audit["truncation_rate"]:.2%}; detected reasoning close: {audit["closed_reasoning_rate"]:.2%}.', '',
 f'GPU cost estimate at the ledger rate: ${gpu_estimate:.2f}; tracked grading estimate: ${audit["tracked_api_usd"]:.2f}. These are ledger estimates, not invoice totals.', '',
 'OLMo uses the native chat-template thinking opening and the pinned model revision 832c3f543499af8fe68b88359501de9cb7840544. Its multi-token closing delimiter is handled explicitly. No historical traces or full_v2 manifests were changed.', '',
 'GLM remains provisional because identical-prefix results changed across run dates and hosted prefix handling/model identity have not been verified. Do not use the GLM contrast as evidence about initiation mechanisms until that issue is resolved. See ../glm_sentence_audit_v1/README.md.','']
 (out/'results.md').write_text('\n'.join(rows));print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
