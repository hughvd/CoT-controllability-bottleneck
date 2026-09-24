"""Read-only audit of GPT-OSS parsing and GLM's suppression-task baselines.

No inference or LLM grading. Original CSVs and transcripts are never overwritten.
"""
import hashlib
import json
import re
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harmony_backend import split_channels
from transforms import grade_no_first_person
from suppression import score_suppression


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    out = ROOT / 'analysis' / 'prefill'
    out.mkdir(exist_ok=True)
    sources, models, recovered = {}, {}, []
    for name in ['base_glm_5_2', 'base_gptoss_120b', 'base_gptoss_120b_k0high',
                 'base_qwen3_32b_k0fix', 'base_deepseek_r1']:
        path = ROOT / 'results' / f'{name}_prefill_per_sample.csv'
        sources[str(path.relative_to(ROOT))] = sha(path)
        d = pd.read_csv(path)
        d['chars'] = d.reasoning_continuation.fillna('').str.len()
        k0 = d[d.k == 0]
        models[name] = {
            'k0': k0.groupby('condition').agg(n=('chars','size'), empty=('empty_thinking','sum'),
                                            chars=('chars','mean')).reset_index().to_dict('records'),
            'suppression_baselines': k0[(k0.condition == 'prefill_only') &
                k0.ctype.isin(['no_first_person','omit_word'])].groupby('ctype').agg(
                    n=('chars','size'), score=('compliance_score','mean'),
                    chars=('chars','mean')).reset_index().to_dict('records')}
        if name == 'base_gptoss_120b':
            legacy = k0
        if name == 'base_glm_5_2':
            glm = k0[k0.condition == 'prefill_only'].copy()
    for path in sorted((ROOT / 'transcripts').glob('*K0_s0.json')):
        raw = json.loads(path.read_text()).get('raw_response', {})
        if not raw.get('provider', '').startswith('vllm/openai/gpt-oss'):
            continue
        text = raw.get('harmony_raw')
        if text is None:
            continue
        a, f = split_channels(text)
        condition = 'prefill_plus_instruction' if 'plus_instruction' in path.name else 'prefill_only'
        qhash = path.name.split('_', 1)[0]
        matches = legacy[(legacy.condition == condition) & (legacy['sample'] == 0) &
                         legacy.question.map(lambda q: hashlib.sha256(q.encode()).hexdigest()[:16] == qhash)]
        matches = matches[matches.ctype.map(lambda ct: f'_{ct}_' in path.name)]
        assert len(matches) <= 1 and a.strip(), path
        if len(matches):
            assert matches.iloc[0].empty_thinking, path
        recovered.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), condition=condition,
                              matched_current_csv=bool(len(matches)),
                              ctype=matches.iloc[0].ctype if len(matches) else None, analysis=a, final=f))
    (out / 'gptoss_k0_recovered_subset.json').write_text(json.dumps(recovered, indent=2))
    fp = glm[glm.ctype == 'no_first_person'].copy()
    assert all(float(grade_no_first_person(r.reasoning_continuation)) == r.compliance_score for r in fp.itertuples())
    lets = sum(r.compliance_score == 1 and bool(re.search(r"\blet['’]s\b", r.reasoning_continuation, re.I)) for r in fp.itertuples())
    omit = glm[glm.ctype == 'omit_word'].copy()
    omit['word'] = omit.instruction.str.extract(r"['\"]([^'\"]+)['\"]", expand=False)
    assert omit.word.notna().all()
    embedded = []
    for r in omit.itertuples():
        assert score_suppression('omit_word', r.reasoning_continuation, [r.word]) == r.compliance_score
        if r.compliance_score == 1 and r.word.lower() in r.reasoning_continuation.lower():
            embedded.append(dict(question=r.question, sample=r.sample, word=r.word,
                                 matches=re.findall(r'\S*'+re.escape(r.word)+r'\S*', r.reasoning_continuation, re.I)))
    report = dict(source_sha256=sources, audit_sha256=sha(Path(__file__)),
                  models=models, gptoss_recovered_raw_k0=len(recovered),
                  gptoss_recovered_matching_csv=sum(r['matched_current_csv'] for r in recovered),
                  glm_no_instruction_prompts_clean=bool(~glm.prompt.str.contains('<cot_instructions>').any()),
                  glm_no_first_person_passes_with_lets=int(lets),
                  glm_no_first_person_chars_by_score=fp.groupby('compliance_score').chars.mean().to_dict(),
                  glm_omit_by_word=omit.groupby('word').agg(n=('word','size'), score=('compliance_score','mean')).reset_index().to_dict('records'),
                  glm_omit_passes_with_embedded_word=embedded)
    assert report['glm_no_instruction_prompts_clean']
    (out / 'prefill_audit.json').write_text(json.dumps(report, indent=2))
    print(f"Recovered {len(recovered)} raw GPT-OSS K=0 examples; {report['gptoss_recovered_matching_csv']} match current CSV rows recorded empty.")
    print(json.dumps({k:v for k,v in report.items() if k.startswith('glm_')}, indent=2))


if __name__ == '__main__':
    audit()
