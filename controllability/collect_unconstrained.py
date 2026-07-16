"""Collect unconstrained CoT rollouts for the omit-topic benchmark mining.

For every question in the eval pool, samples N rollouts with the 'none'
instruction (the same no-constraint form the normal training arm uses) and
stores the reasoning text. These are the raw material for mining load-bearing
(question, topic) pairs: a topic is load-bearing iff it appears in ~all
unconstrained rollouts of that question.

Run against a live vLLM server (same one the base eval uses):
    VLLM_BASE_URL=http://localhost:8000/v1 \\
    python3 collect_unconstrained.py --model organism_2_v33_base \\
        --test-csv data/split_test_v32.csv --n 8 \\
        --out results/unconstrained_rollouts.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

from run_eval_diverse import assemble_prompt, parse_options, answer_to_letter

CONCURRENCY = 16


async def roll_one(client, sem, model: str, q: dict, n: int, temperature: float,
                   max_tokens: int) -> dict:
    options = parse_options(pd.Series(q))
    prompt = assemble_prompt(q["question"], options, "none")
    rollouts = []
    async with sem:
        # one request, n samples — vLLM fans out server-side
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    for ch in resp.choices:
        think = getattr(ch.message, "reasoning_content", None) or ""
        content = ch.message.content or ""
        rollouts.append({"think": think, "content": content})
    return {
        "question": q["question"],
        "source": q.get("source") or q.get("dataset") or "",
        "correct_letter": answer_to_letter(q.get("answer", ""), options),
        "options": options,
        "rollouts": rollouts,
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--test-csv", default="data/split_test_v32.csv")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--out", default="results/unconstrained_rollouts.jsonl")
    args = ap.parse_args()

    client = AsyncOpenAI(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key="EMPTY",
    )
    test = pd.read_csv(args.test_csv)
    questions = test.to_dict(orient="records")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume: skip questions already collected
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["question"])
                except Exception:
                    pass
    todo = [q for q in questions if q["question"] not in done]
    print(f"[collect_unconstrained] {len(questions)} questions, "
          f"{len(done)} already done, {len(todo)} to collect (n={args.n})")

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [roll_one(client, sem, args.model, q, args.n, args.temperature,
                      args.max_tokens) for q in todo]
    with out_path.open("a") as f:
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            try:
                rec = await coro
            except Exception as e:
                print(f"  [{i}/{len(tasks)}] ERROR: {e}")
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] collected")
    print(f"[collect_unconstrained] wrote -> {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
