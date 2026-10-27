"""Metered GLM factorial, with rejection based only on its first reasoning token.

Baseline and ban share their first proposal. Ordered batches of later proposals
accelerate low acceptance without selecting by completion latency or compliance.
"""
import asyncio
import argparse
import json
from pathlib import Path
import sqlite3
import time
import fixed_opening_ban as f
from ban_user_icl import ROOT
import token_prefill_suite_api as api


async def run(experiment_root=ROOT, max_proposals=1024, tail_batch_size=4, concurrency=8):
    root = Path(experiment_root) / 'glm'
    m = json.loads((root / 'manifest.json').read_text())
    f.validate(m)
    api.s.ROOT = root / 'rejection'
    client = api.PaidClient(cap=25., retries=12)
    client.sem = asyncio.Semaphore(concurrency)
    tok = f.suite.tokenizer(m['model'])
    db = sqlite3.connect(root / 'api_outputs.sqlite')
    db.execute('pragma journal_mode=WAL')
    db.execute('create table if not exists outputs (id text primary key,payload text)')
    db.execute('create table if not exists proposals (id text,attempt integer,payload text,accepted integer,primary key(id,attempt))')
    db.commit()
    requests = sorted(m['requests'], key=lambda r:f.base.digest(r['request_id']))
    report = dict(status='generating', method='ordered first-token rejection sampling',
                  expected_in_scope=len(requests), started_at=time.time(),
                  max_proposals=max_proposals, tail_batch_size=tail_batch_size,concurrency=concurrency,
                  protocol='First proposal shared between baseline and ban; select lowest accepted attempt index.')

    def save():
        report.update(outputs=db.execute('select count(*) from outputs').fetchone()[0],
                      proposals=db.execute('select count(*) from proposals').fetchone()[0],
                      tracked_cost=client.spent(), updated_at=time.time())
        f.base.atomic_json(root / 'rejection_report.json', report)

    async def propose(r, attempt):
        previous=db.execute('select accepted from proposals where id=? and attempt=?',(r['request_id'],attempt)).fetchone()
        if previous is not None and not previous[0]:
            # Already rejected under this immutable manifest/tokenizer. Replaying
            # its API cache and status writes adds no information on resume.
            return attempt,False,None,None,None,None
        # Deliberately omit arm: couple baseline with the first ban proposal.
        key = f.base.digest([m['manifest_sha256'], r['cell_id'], r['condition'], r['sample'], attempt])
        payload = dict(model=m['model']['api_id'], messages=r['messages'], temperature=.6,
                       max_tokens=8192, seed=int(key[:8], 16),
                       extra_body={'provider': {'order': ['DeepInfra'], 'allow_fallbacks': False},
                                   'include_reasoning': True})
        # None means omit the field, matching hosted standalone ICL requests.
        # Historical manifests explicitly contain top_p=1 and remain unchanged.
        if m['sampling'].get('top_p', 1.) is not None:
            payload['top_p'] = m['sampling'].get('top_p', 1.)
        job = 'ban-user-icl-v1|' + key
        for retry in range(12):
            raw = await client.call(job, payload, 1.4e-6, 4.4e-6)
            # OpenRouter can return an upstream 429 inside an HTTP-200 body.
            # It is not a completion. Retry the same proposal/seed, preserving
            # the attempts ledger (including conservative cost reservations).
            error = raw.get('error') or {}
            if error.get('code') != 429 or raw.get('choices'):
                break
            with client.db:
                client.db.execute('delete from paid_results where job=?', (job,))
            if retry == 11:
                raise RuntimeError('Upstream rate limit persisted for proposal ' + key)
            await asyncio.sleep(min(120, 20 * 2**retry))
        rec = api.api_record(m, r, 0, raw)
        text = rec['reasoning_continuation']
        ids = tok.encode(text, add_special_tokens=False)
        accepted = r['arm'] == 'baseline' or not ids or ids[0] not in m['ban_token_ids']
        with db:
            db.execute('insert or ignore into proposals values (?,?,?,?)',
                       (r['request_id'], attempt, json.dumps(raw, ensure_ascii=False), int(accepted)))
        save()
        return attempt, accepted, text, ids, rec, raw

    async def one(r):
        if db.execute('select 1 from outputs where id=?', (r['request_id'],)).fetchone():
            return
        batches = [range(0, 1)]
        i = 1
        while i < max_proposals:
            end = min(i + (tail_batch_size if i >= 65 else 4), max_proposals)
            batches.append(range(i, end))
            i = end
        for attempts in batches:
            proposals = await asyncio.gather(*(propose(r, a) for a in attempts))
            eligible = [p for p in proposals if p[1]]
            if not eligible:
                continue
            attempt, _, text, ids, rec, raw = min(eligible, key=lambda p:p[0])
            pre = tok.decode(ids[:1], skip_special_tokens=False) if ids else ''
            # One native CJK token may contain only part of a UTF-8 character.
            # Preserve its ID and decoded prefix; scoring uses the stable part.
            assert text.startswith(pre.rstrip('\ufffd'))
            result = dict(request_id=r['request_id'], raw_text=text, answer_text=rec['response'],
                          first_token_text=pre, first_stage_ids=ids[:1], output_token_ids=None,
                          first_stage_incomplete_utf8=pre.endswith('\ufffd'),
                          finish_reason=rec['finish_reason'], proposal_count=attempt+1,
                          provider=raw['provider'], generation_id=raw['id'],
                          manifest_sha256=m['manifest_sha256'],
                          method='rejection_at_first_locally_retokenized_reasoning_token',
                          token_ids_provenance='local_retokenization')
            with db:
                db.execute('insert into outputs values (?,?)', (r['request_id'], json.dumps(result, ensure_ascii=False)))
            save()
            if report['outputs'] % 20 == 0:
                print(report['outputs'], '/', len(requests), 'tracked cost', round(client.spent(), 3), flush=True)
            return
        raise RuntimeError('Proposal limit reached; do not silently exclude: ' + r['request_id'])

    queue = iter(requests)
    errors = []
    async def worker():
        while not errors:
            try:
                r = next(queue)
            except StopIteration:
                return
            try:
                await one(r)
            except Exception as e:
                errors.append(dict(request_id=r['request_id'], error=repr(e)))
                return
    save()
    try:
        await asyncio.gather(*(worker() for _ in range(24)))
        report.update(status='blocked' if errors else 'complete', errors=errors)
    finally:
        save()
        await client.client.close()
    print(report, flush=True)
    if errors:
        raise RuntimeError(errors)


if __name__ == '__main__':
    import fcntl
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-root', default=str(ROOT))
    parser.add_argument('--max-proposals', type=int, default=1024)
    parser.add_argument('--tail-batch-size', type=int, default=4)
    parser.add_argument('--concurrency', type=int, default=8)
    args = parser.parse_args()
    assert args.max_proposals >= 1 and 1 <= args.tail_batch_size <= 32
    assert 1 <= args.concurrency <= 64
    lock_path=Path(args.experiment_root)/'glm'/'generation.lock'
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('A GLM generator is already running for this experiment.')
        lock.seek(0);lock.truncate();lock.write(str(os.getpid())+'\n');lock.flush()
        asyncio.run(run(args.experiment_root, args.max_proposals, args.tail_batch_size,args.concurrency))
