"""Small, bounded hosted first-opening-token rejection pilot; historical data read-only.

One target per each of eight controls, instruction/user-ICL x baseline/rejection.
All four arms score the complete generated reasoning, including the opening.
Retokenized visible reasoning is an API proxy, not a claim of native logit access.
"""
import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import icl_experiment as icl
import token_prefill_suite as suite
import token_prefill_suite_api as api
import control_expansion as expansion
from monitorability.sweep_analysis import MODELS

ROOT = HERE / 'results/icl_ban_correlation_pilot/v1'
TYPES = ['prefix_sentences', 'numbered_sentences', 'suffix_sentences',
         'uppercase_thinking', 'leetspeak', 'spanish', 'omit_word', 'no_first_person']
LIMIT = 8


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


class Tokenizer:
    def __init__(self, spec):
        self.kind = spec.get('kind', 'json')
        if self.kind == 'tiktoken':
            import tiktoken
            from tiktoken.load import load_tiktoken_bpe
            self.t = tiktoken.Encoding(name='kimi-pilot', pat_str=spec['pattern'],
                                      mergeable_ranks=load_tiktoken_bpe(spec['path']), special_tokens={})
        else:
            from tokenizers import Tokenizer as T
            self.t = T.from_file(spec['path'])

    def encode(self, text):
        return self.t.encode(text, disallowed_special=()) if self.kind == 'tiktoken' else self.t.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.t.decode(ids) if self.kind == 'tiktoken' else self.t.decode(ids, skip_special_tokens=False)

    def first(self, text):
        # Leading whitespace is omitted by many hosted reasoning parsers; define
        # the observable intervention consistently on the first content token.
        ids = self.encode(text.lstrip())
        return ids[0] if ids else None


def prepare():
    if (ROOT / 'manifest.json').exists():
        return json.loads((ROOT / 'manifest.json').read_text())
    cells = {}
    messages = {}
    for run in ['v1', 'seven_controls_extension_v1']:
        path = HERE / 'results/ban_user_icl' / run / 'qwen32/manifest.json'
        m = json.loads(path.read_text())
        for c in m['cells']:
            b = c['behavior']
            if b not in TYPES or b in cells:
                continue
            cells[b] = c
            for cond in ['instruction_only', 'user_icl']:
                messages[(b, cond)] = next(r['messages'] for r in m['requests']
                    if r['cell_id'] == c['cell_id'] and r['condition'] == cond)
    extra = json.loads((HERE / 'results/control_expansion/v1/cells.json').read_text())
    for item in extra['cells']:
        t, d = icl.Cell(**item['target']), icl.Cell(**item['donor'])
        if t.ctype in cells or t.question != cells['prefix_sentences']['question']:
            continue
        grade = asdict(icl.grade_cell_for(t, d))
        grade['behavior'] = grade.pop('ctype')
        grade['donor'] = asdict(d)
        grade['question_id'] = digest(t.question)
        cells[t.ctype] = grade
        for cond, variant in [('instruction_only', 'instruction_only'), ('user_icl', 'icl_user_plain')]:
            messages[(t.ctype, cond)] = icl.build_icl_messages(t, d, variant, ('5', 'abs', 5))
            if t.ctype == 'numbered_sentences' and cond == 'user_icl':
                old = icl.slice_by_len(d.controlled_cot, ('5', 'abs', 5))
                corrected = ' '.join(s for _, _, s in expansion.sentences(d.controlled_cot)[:5])
                messages[(t.ctype, cond)][0]['content'] = messages[(t.ctype, cond)][0]['content'].replace(
                    'Internal thinking: "'+old+'"', 'Internal thinking: "'+corrected+'"')
    assert set(cells) == set(TYPES)
    tokenizers = json.loads((ROOT / 'tokenizers.json').read_text())
    specs = {}
    for name, run in MODELS.items():
        source = HERE / 'results/control_expansion/v1/correlation' / name / 'manifest.json'
        spec = json.loads(source.read_text())['model']
        ts = tokenizers[spec['api_id']]
        assert ts['status'] == 'ready', name
        tok = Tokenizer(ts)
        path = HERE.parent / 'results/prefill_continuation' / run / 'sweep.json'
        old = json.loads(path.read_text())
        counts = Counter()
        seen = set()
        for row in old.values():
            if not isinstance(row, dict) or row.get('k') != 0 or row.get('condition') != 'prefill_only' or 'error' in row:
                continue
            key = (row.get('question'), row.get('sample'))
            if key in seen:
                continue
            seen.add(key)
            first = tok.first(row.get('reasoning_continuation') or '')
            if first is not None:
                counts[first] += 1
        assert counts, name
        # Preserve previously defined bans where they exist. Others use the two
        # most frequent tokens in historical instruction-free, deduplicated traces.
        if name == 'qwen3-32b':
            bans = [tok.encode(w)[0] for w in ['Okay', 'Alright']]
            assert all(len(tok.encode(w)) == 1 for w in ['Okay', 'Alright'])
        elif name == 'glm-5.2':
            bans = tok.encode('The')
            assert len(bans) == 1
        else:
            bans = [i for i, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:2]]
        specs[name] = dict(**spec, tokenizer=ts, ban_ids=bans,
            ban_text=[tok.decode([i]) for i in bans], calibration_n=sum(counts.values()),
            calibration_banned_share=sum(counts[i] for i in bans) / sum(counts.values()),
            calibration_source=str(path), calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            calibration_top10=[dict(id=i, text=tok.decode([i]), count=n) for i, n in counts.most_common(10)])
    jobs = [dict(behavior=b, condition=c, messages=messages[(b, c)]) for b in TYPES
            for c in ['instruction_only', 'user_icl']]
    m = dict(models=specs, cells=cells, jobs=jobs, proposal_limit=LIMIT,
             sampling=dict(temperature=.6, top_p=1., max_tokens=8192),
             protocol='First sample is baseline; ban arm is earliest proposal with an allowed first locally retokenized visible reasoning token. No assistant prefix or global logit bias. Empty reasoning is not an accepted ban proposal. Cap failures remain missing, never zero-scored or silently excluded.',
             scope='One fixed target per control, one sample per arm, matched across models. Feasibility pilot, not a reliable correlation estimate.')
    m['sha256'] = digest(m)
    write(ROOT / 'manifest.json', m)
    return m


def reasoning(raw):
    choice = raw['choices'][0]
    msg = choice['message']
    text = msg.get('reasoning')
    if text is None:
        text = ''.join(x.get('text', '') for x in (msg.get('reasoning_details') or []) if x.get('type') == 'reasoning.text')
    if not isinstance(text, str):
        raise ValueError('Invalid reasoning payload')
    if text.lstrip().startswith('<think>'):
        text = text.split('<think>', 1)[1]
    text = text.split('</think>', 1)[0]
    return text, msg.get('content') or '', choice.get('finish_reason')


async def run():
    m = prepare()
    api.s.ROOT = ROOT / 'paid'
    client = api.PaidClient(cap=60., retries=3)
    client.client = client.client.with_options(timeout=240.)
    client.sem = asyncio.Semaphore(24)
    sems = {n: asyncio.Semaphore(2) for n in m['models']}
    tokenizers = {n: Tokenizer(s['tokenizer']) for n, s in m['models'].items()}
    meta = {x['id']: x for x in json.loads((ROOT / 'model_metadata.json').read_text())}
    report = dict(status='running', started_at=time.time(), jobs=192, completed_jobs=0, errors=[])

    def save():
        report.update(updated_at=time.time(), cost=client.spent())
        write(ROOT / 'status.json', report)

    async def one(name, job):
        spec = m['models'][name]
        tok = tokenizers[name]
        path = ROOT / name / (job['behavior'] + '__' + job['condition'] + '.json')
        if path.exists() and json.loads(path.read_text()).get('status') in ['complete', 'cap_exhausted']:
            report['completed_jobs'] += 1
            return
        result = dict(model=name, behavior=job['behavior'], condition=job['condition'],
                      status='running', baseline=None, banned=None, proposals=[], manifest_sha256=m['sha256'])
        try:
            async with sems[name]:
                for attempt in range(LIMIT):
                    namespace = m.get('cache_namespaces', {}).get(job['behavior']+'__'+job['condition'], m['sha256'])
                    key = digest([namespace, name, job['behavior'], job['condition'], attempt])
                    extra = dict(provider=dict(order=[spec['provider']], allow_fallbacks=False), include_reasoning=True)
                    if spec.get('effort'):
                        extra['reasoning'] = dict(effort=spec['effort'])
                    payload = dict(model=spec['api_id'], messages=job['messages'], seed=int(key[:8], 16) % 2**31,
                                   **m['sampling'], extra_body=extra)
                    rates = meta[spec['api_id']]['pricing']
                    raw = await client.call('icl-ban-pilot|' + key, payload, float(rates['prompt']), float(rates['completion']))
                    if raw.get('error') or not raw.get('choices'):
                        raise ValueError('API error: ' + str(raw.get('error'))[:300])
                    if raw.get('provider') != spec['provider']:
                        raise ValueError('Unexpected provider: ' + str(raw.get('provider')))
                    text, answer, finish = reasoning(raw)
                    first = tok.first(text)
                    record = dict(attempt=attempt, reasoning=text, answer=answer, finish_reason=finish,
                                  first_token_id=first, first_token_text=tok.decode([first]) if first is not None else '',
                                  generation_id=raw.get('id'), provider=raw.get('provider'), cache_key=key)
                    result['proposals'].append(record)
                    if attempt == 0:
                        result['baseline'] = record
                    if first is not None and first not in spec['ban_ids']:
                        result['banned'] = record
                        result['status'] = 'complete'
                        write(path, result)
                        break
                    write(path, result)
                else:
                    result['status'] = 'cap_exhausted'
                    write(path, result)
        except Exception as exc:
            result.update(status='error', error=type(exc).__name__ + ': ' + str(exc)[:350])
            write(path, result)
            report['errors'].append(dict(model=name, behavior=job['behavior'], condition=job['condition'], error=result['error']))
        report['completed_jobs'] += 1
        save()
        print(name, job['behavior'], job['condition'], result['status'], len(result['proposals']), flush=True)

    save()
    await asyncio.gather(*(one(n, j) for n in m['models'] for j in m['jobs']))
    report['status'] = 'finished'
    save()
    await client.client.close()


async def score():
    from sft import language_grade as lg
    from token_prefill_suite_score import parse_judgment
    import transforms as tr
    import pandas as pd
    m = prepare()
    api.s.ROOT = ROOT / 'grading'
    client = api.PaidClient(cap=3., retries=3)
    rows = []
    async def one(path):
        r = json.loads(path.read_text())
        b = r['behavior']; c = m['cells'][b]
        for arm in ['baseline', 'banned']:
            record = r.get(arm)
            if record is None:
                continue
            text = record['reasoning']
            if not text.strip():
                value = 0.
            elif b == 'spanish':
                prompt = lg._GRADER_PROMPT.format(display_name=lg.load_language_defs()['spanish']['display_name'], text=text)
                payload = dict(model=lg.DEFAULT_MODEL, messages=[dict(role='user', content=prompt)],
                               temperature=0., max_tokens=600, response_format=dict(type='json_object'))
                raw = await client.call('judge|' + digest(payload), payload, .1e-6, .4e-6)
                ch = raw['choices'][0]
                value, _, _ = parse_judgment(lg, ch['message'].get('content') or '', ch.get('finish_reason'))
                assert value is not None
            elif b in ['numbered_sentences', 'suffix_sentences']:
                value, _ = expansion.score_text(b, '', text, c.get('marker'))
            elif b == 'no_first_person':
                value = float(tr.grade_no_first_person(text))
            else:
                value, _ = suite.deterministic_score(c, '', text)
            rows.append(dict(model=r['model'], behavior=b, condition=r['condition'], arm=arm,
                backend=r.get('backend', 'api_rejection'),
                compliance=value, empty=not text.strip(), truncated=record['finish_reason']=='length',
                answer_correct=icl.extract_answer(record.get('answer',''))==c['correct_letter'],
                has_final_answer=bool(record.get('answer','').strip()),
                question_id=c['question_id'], attempts=record['attempt']+1, first_token=record['first_token_text'],
                ban_completed=r.get('banned') is not None, source=str(path.relative_to(HERE.parent))))
    # Explicit behavior/condition filenames avoid treating reports as observations.
    await asyncio.gather(*(one(ROOT/n/(j['behavior']+'__'+j['condition']+'.json'))
        for n in m['models'] for j in m['jobs'] if (ROOT/n/(j['behavior']+'__'+j['condition']+'.json')).exists()))
    for folder in ROOT.glob('native_*'):
        p = folder/'outputs.json'
        if not p.exists():
            continue
        name = folder.name.removeprefix('native_')
        if name == 'gptoss': name = 'gpt-oss-120b'
        tok = Tokenizer(m['models'][name]['tokenizer'])
        outputs = json.loads(p.read_text())
        assert len(outputs) == 32
        for job in m['jobs']:
            pair = dict(model=name, behavior=job['behavior'], condition=job['condition'], backend='native', status='complete')
            for arm in ['baseline','banned']:
                rec = next(x for x in outputs if x['behavior']==job['behavior'] and x['condition']==job['condition'] and x['arm']==arm)
                first = tok.first(rec['reasoning'])
                pair[arm] = dict(rec, attempt=0, first_token_text=tok.decode([first]) if first is not None else '')
            dest=ROOT/'native_adapted'/name/(job['behavior']+'__'+job['condition']+'.json')
            write(dest,pair)
            await one(dest)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT/'scores.csv', index=False)
    df.groupby(['backend','model','condition','arm']).agg(n=('compliance','size'), mean=('compliance','mean'),
        empty=('empty','sum'), truncated=('truncated','sum')).to_csv(ROOT/'summary.csv')
    await client.client.close()
    print('Scored', len(df), 'observations', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=['prepare', 'run', 'score']); a = ap.parse_args()
    if a.phase == 'prepare':
        m=prepare()
        for n,s in m['models'].items(): print(n,s['ban_text'],round(s['calibration_banned_share'],3),s['calibration_n'])
    elif a.phase == 'run': asyncio.run(run())
    else: asyncio.run(score())
