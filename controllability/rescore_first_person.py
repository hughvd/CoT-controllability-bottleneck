"""Apply one first-person rule to legacy results without modifying raw caches.

Current paper loaders call this at read time. Per-response old/new scores are
saved under analysis/first_person_rescore/v1/records for audit and reuse.
"""
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd

from controllability import transforms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'controllability/analysis/first_person_rescore/v1/records'


def provenance():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(transforms.__file__)]}


def score(text):
    # Preserve the established empty-reasoning rule.
    return float(bool(text.strip()) and transforms.grade_no_first_person(text))


def reasoning(raw):
    text = raw['raw_text']
    if text.startswith('<think>'):
        text = text[len('<think>'):]
    return text.split('</think>', 1)[0]


def rescore(frame, source):
    source = Path(source).resolve()
    behavior = 'behavior' if 'behavior' in frame else 'ctype'
    if behavior not in frame or not frame[behavior].eq('no_first_person').any():
        return frame
    d = frame.copy()
    mask = d[behavior].eq('no_first_person')
    indices = d.index[mask]
    raw_sources = {}
    ban = 'whole_soft' in d and 'reasoning' not in d
    if ban:
        required = set(d.loc[mask, 'request_id'])
        raw = {}
        for p in sorted((source.parent / 'remote').glob('batch_*.json')):
            records = json.loads(p.read_text())['outputs']
            chosen = [r for r in records if r['request_id'] in required]
            if chosen:
                raw_sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
                raw.update({r['request_id']: r for r in chosen})
        p = source.parent / 'api_outputs.sqlite'
        if p.exists():
            with sqlite3.connect(f'file:{p}?mode=ro', uri=True) as db:
                for (payload,) in db.execute('select payload from outputs'):
                    r = json.loads(payload)
                    if r['request_id'] in required:
                        raw[r['request_id']] = r
            raw_sources[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        assert set(raw) == required, (source, len(raw), len(required))
    rows = []
    # The paper uses complete continuations. Do not reinterpret head32_soft:
    # truncation can turn a partial word (e.g. "me" in "meow") into a match.
    columns = [c for c in ['compliance_score','compliance','full_soft','whole_soft','continuation_soft'] if c in d]
    for i in indices:
        row = d.loc[i]
        if ban:
            record = raw[row.request_id]
            text = reasoning(record)
            prefix = record['first_token_text']
            if prefix.startswith('<think>'):
                prefix = prefix[len('<think>'):]
            prefix = prefix.split('</think>', 1)[0].rstrip('\ufffd')
            assert text.startswith(prefix)
            texts = {'whole_soft': text, 'continuation_soft': text[len(prefix):]}
        else:
            text = next((row[c] for c in ['reasoning','full_text','reasoning_continuation']
                         if c in row and isinstance(row[c], str)), None)
            assert text is not None, (source, i, 'Missing saved reasoning')
            texts = {c: text for c in columns}
        record = {'source_row': int(i)}
        for c in ['model','request_id','question_id','question','target_question','condition','variant','prefix_tokens','arm','sample']:
            if c in row and pd.notna(row[c]):
                record[c] = row[c]
        for c in columns:
            if pd.isna(row[c]):
                continue
            old, new = float(row[c]), score(texts[c])
            record[c+'_before'] = old
            record[c+'_after'] = new
            d.at[i,c] = bool(new) if pd.api.types.is_bool_dtype(d[c]) else new
        rows.append(record)
    # No other task's score or text may change.
    pd.testing.assert_frame_equal(frame.loc[~mask], d.loc[~mask], check_dtype=False)
    audit = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(source.relative_to(ROOT)).encode()).hexdigest()[:16]
    audit.to_csv(OUT / f'{key}.csv', index=False)
    changes = {c: int((audit[c+'_before'].notna() &
                      (audit[c+'_before'] != audit[c+'_after'])).sum())
               for c in columns if c+'_before' in audit}
    meta = dict(source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                checked_rows=len(rows), changed=changes, grader_sources=provenance(), raw_sources=raw_sources,
                rule="Current grade_no_first_person; let's allowed, US/dotted abbreviations exempt, empty reasoning zero")
    (OUT / f'{key}.json').write_text(json.dumps(meta, indent=2)+'\n')
    return d
