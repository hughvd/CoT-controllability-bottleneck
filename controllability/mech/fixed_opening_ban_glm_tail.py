"""Finish GLM's primary comparison with ordered parallel rejection proposals.

Within each batch, select the lowest attempt index with an allowed first token.
Never select by completion latency or any later response property.
"""
import asyncio
import json
import sqlite3
import time
import fixed_opening_ban as f
from fixed_opening_ban_glm_rejection import api


async def run():
    root = f.ROOT / 'glm'
    m = json.loads((root / 'manifest.json').read_text())
    f.validate(m)
    api.s.ROOT = root / 'rejection'
    client = api.PaidClient(cap=25., retries=4)
    client.sem = asyncio.Semaphore(16)
    tok = f.suite.tokenizer(m['model'])
    db = sqlite3.connect(root / 'api_outputs.sqlite')
    db.execute('pragma journal_mode=WAL')
    finished = {r[0] for r in db.execute('select id from outputs')}
    requests = [r for r in m['requests'] if r['condition'] == 'instruction_only' and r['request_id'] not in finished]
    report = dict(status='generating', method='ordered parallel first-token rejection sampling',
                  scope='instruction_only', expected_in_scope=720, started_at=time.time())

    def save():
        report.update(outputs=db.execute('select count(*) from outputs').fetchone()[0],
                      proposals=db.execute('select count(*) from proposals').fetchone()[0],
                      tracked_cost=client.spent(), updated_at=time.time())
        f.base.atomic_json(root / 'rejection_report.json', report)

    async def propose(r, attempt):
        rid = r['request_id']
        old = db.execute('select payload from proposals where id=? and attempt=?', (rid, attempt)).fetchone()
        if old:
            raw = json.loads(old[0])
        else:
            payload = dict(model=m['model']['api_id'], messages=r['messages'], temperature=.6,
                           top_p=1., max_tokens=8192, seed=int(f.base.digest([rid, attempt])[:8], 16),
                           extra_body={'provider': {'order': ['DeepInfra'], 'allow_fallbacks': False},
                                       'include_reasoning': True})
            raw = await client.call('fixed-ban-rejection-v1|' + rid + '|' + str(attempt), payload, 1.4e-6, 4.4e-6)
        rec = api.api_record(m, r, 0, raw)
        text = rec['reasoning_continuation']
        ids = tok.encode(text, add_special_tokens=False)
        accepted = r['arm'] == 'baseline' or not ids or ids[0] not in m['ban_token_ids']
        with db:
            db.execute('insert or ignore into proposals values (?,?,?,?)',
                       (rid, attempt, json.dumps(raw, ensure_ascii=False), int(accepted)))
        save()
        return attempt, accepted, text, ids, rec, raw

    async def one(r):
        for start in range(0, 1024, 16):
            proposals = await asyncio.gather(*(propose(r, a) for a in range(start, min(start + 16, 1024))))
            eligible = [p for p in proposals if p[1]]
            if not eligible:
                continue
            attempt, _, text, ids, rec, raw = min(eligible, key=lambda p: p[0])
            pre = tok.decode(ids[:1], skip_special_tokens=False) if ids else ''
            assert text.startswith(pre) and '\ufffd' not in pre
            result = dict(request_id=r['request_id'], raw_text=text, answer_text=rec['response'],
                          first_token_text=pre, first_stage_ids=ids[:1], output_token_ids=None,
                          finish_reason=rec['finish_reason'], proposal_count=attempt + 1,
                          provider=raw['provider'], generation_id=raw['id'],
                          manifest_sha256=m['manifest_sha256'],
                          method='rejection_at_first_native_reasoning_token',
                          token_ids_provenance='local_retokenization')
            with db:
                db.execute('insert into outputs values (?,?)', (r['request_id'], json.dumps(result, ensure_ascii=False)))
            print(r['request_id'][:8], 'accepted at attempt', attempt + 1, flush=True)
            save()
            return
        raise RuntimeError('Proposal limit reached: ' + r['request_id'])

    try:
        outcomes = await asyncio.gather(*(one(r) for r in requests), return_exceptions=True)
        errors = [str(e) for e in outcomes if isinstance(e, BaseException)]
        report.update(status='blocked' if errors else 'complete', errors=errors)
    finally:
        save()
        await client.client.close()
    print(report, flush=True)


if __name__ == '__main__':
    asyncio.run(run())
