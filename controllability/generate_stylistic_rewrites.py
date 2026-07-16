"""Async generator for stylistic CoT rewrites.

For each (raw CoT, style) pair, calls a strong reasoning model and stores
the rewrite in `data/stylistic_rewrites.jsonl`. Cache-first: re-runs skip
entries already present at the same (question, style, prompt-template)
identity.

Style definitions (instructions, train/eval split, pairing) live in
`data/styles.yaml`. The rendered prompt template is hashed so editing a
style's instruction or the shared template invalidates only the affected
entries.

Small-run example (default sampling: 2 questions per dataset × all styles):
    python3 generate_stylistic_rewrites.py --backend openai --model gpt-5-mini

Override sampling:
    python3 generate_stylistic_rewrites.py --n-per-dataset 10 --styles yoda pirate
    python3 generate_stylistic_rewrites.py --n-total 1500             # full run

Backend (default openai):
    --backend openai      uses OPENAI_API_KEY    (default base_url)
    --backend openrouter  uses OPENROUTER_API_KEY at openrouter.ai/api/v1

Cache (one JSONL line per rewrite):
    {dataset, question_hash, raw_cot_hash, style, model, prompt_sha,
     rewrite, original_correct_letter, ts}
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


HERE = Path(__file__).parent
DEFAULT_RAW_CSV    = HERE / "data" / "training_raw.csv"
DEFAULT_STYLES     = HERE / "data" / "styles.yaml"
DEFAULT_CACHE      = HERE / "data" / "stylistic_rewrites.jsonl"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def render_prompt(shared: str, style_name: str, style_instruction: str, raw_cot: str) -> str:
    return shared.format(
        style_name=style_name,
        style_instruction=style_instruction.strip(),
        raw_cot=raw_cot.strip(),
    )


def load_styles(path: Path) -> tuple[str, dict[str, dict]]:
    cfg = yaml.safe_load(path.read_text())
    return cfg["shared_prompt"], cfg["styles"]


def load_cache(path: Path) -> set[tuple[str, str, str]]:
    """Return set of (question_hash, style, prompt_sha) already in cache."""
    if not path.exists():
        return set()
    seen = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seen.add((r["question_hash"], r["style"], r["prompt_sha"]))
    return seen


def sample_raw(df: pd.DataFrame, n_per_dataset: int | None, n_total: int | None,
               seed: int) -> pd.DataFrame:
    """Sample CoTs to rewrite. Either fixed-per-dataset or total cap."""
    if n_per_dataset is not None:
        parts = []
        for ds, sub in df.groupby("dataset", sort=False):
            shuffled = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            parts.append(shuffled.head(n_per_dataset))
        return pd.concat(parts, ignore_index=True)
    if n_total is not None:
        return df.sample(n=min(n_total, len(df)), random_state=seed).reset_index(drop=True)
    return df


async def call_one(client, *, model: str, prompt: str, reasoning_effort: str | None,
                   max_tokens: int, semaphore: asyncio.Semaphore) -> str:
    """One API call. Returns the rewritten text. Raises on failure."""
    async with semaphore:
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


async def main_async(args) -> None:
    from openai import AsyncOpenAI

    if args.backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = None
    elif args.backend == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        base_url = "https://openrouter.ai/api/v1"
    else:
        raise SystemExit(f"unknown backend {args.backend!r}")

    if not api_key:
        raise SystemExit(
            f"missing API key for backend={args.backend} "
            f"(expected {'OPENAI_API_KEY' if args.backend=='openai' else 'OPENROUTER_API_KEY'} in env)"
        )

    client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)

    shared_prompt, styles = load_styles(args.styles_yaml)
    style_names = args.styles or list(styles.keys())
    for s in style_names:
        if s not in styles:
            raise SystemExit(f"unknown style {s!r}; known: {list(styles)}")

    raw_df = pd.read_csv(args.raw_csv)
    raw_df = raw_df[raw_df["think"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    sampled = sample_raw(raw_df, args.n_per_dataset, args.n_total, args.seed)
    print(f"[gen] sampled {len(sampled)} raw CoTs from {args.raw_csv.name} "
          f"(per-dataset: {sampled['dataset'].value_counts().to_dict()})")
    print(f"[gen] styles: {style_names}")
    print(f"[gen] backend={args.backend}  model={args.model}  "
          f"reasoning_effort={args.reasoning_effort}  concurrency={args.concurrency}")

    seen = load_cache(args.cache)
    print(f"[gen] cache has {len(seen)} prior entries at {args.cache}")

    # Build the work list, skipping anything already cached at the current
    # prompt-template hash.
    todo: list[dict] = []
    for _, row in sampled.iterrows():
        q_hash = sha(row["question"])
        cot_hash = sha(row["think"])
        for style in style_names:
            style_def = styles[style]
            prompt = render_prompt(shared_prompt, style, style_def["style_instruction"], row["think"])
            p_sha = sha(prompt)
            if (q_hash, style, p_sha) in seen:
                continue
            todo.append(dict(
                dataset=row["dataset"],
                question=row["question"],
                question_hash=q_hash,
                raw_cot=row["think"],
                raw_cot_hash=cot_hash,
                style=style,
                prompt=prompt,
                prompt_sha=p_sha,
                correct_letter=row["correct_letter"],
            ))

    print(f"[gen] {len(todo)} new rewrites to generate "
          f"({len(seen)} cached, {len(sampled)*len(style_names) - len(todo) - len(seen)} would-be-skipped)")
    if not todo:
        print("[gen] nothing to do")
        return

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache_handle = args.cache.open("a")  # append mode

    semaphore = asyncio.Semaphore(args.concurrency)
    n_done = 0
    n_failed = 0
    t0 = time.time()
    write_lock = asyncio.Lock()

    async def run_one(item: dict) -> None:
        nonlocal n_done, n_failed
        try:
            text = await call_one(
                client,
                model=args.model,
                prompt=item["prompt"],
                reasoning_effort=args.reasoning_effort,
                max_tokens=args.max_tokens,
                semaphore=semaphore,
            )
        except Exception as e:
            n_failed += 1
            print(f"[gen] FAIL  style={item['style']}  q_hash={item['question_hash']}  err={type(e).__name__}: {e}",
                  file=sys.stderr)
            return

        record = dict(
            dataset=item["dataset"],
            question_hash=item["question_hash"],
            raw_cot_hash=item["raw_cot_hash"],
            style=item["style"],
            model=args.model,
            prompt_sha=item["prompt_sha"],
            rewrite=text,
            original_correct_letter=item["correct_letter"],
            ts=int(time.time()),
        )
        async with write_lock:
            cache_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            cache_handle.flush()
        n_done += 1
        if n_done % args.progress_every == 0 or n_done == len(todo):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"[gen] {n_done}/{len(todo)}  failed={n_failed}  "
                  f"elapsed={elapsed:.0f}s  rate={rate:.2f}/s")

    await asyncio.gather(*(run_one(it) for it in todo))
    cache_handle.close()
    print(f"[gen] done. wrote {n_done} new entries; {n_failed} failed.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    p.add_argument("--styles-yaml", type=Path, default=DEFAULT_STYLES)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--styles", nargs="+", default=None,
                   help="Subset of style names. Default: all in styles.yaml.")
    p.add_argument("--n-per-dataset", type=int, default=2,
                   help="Sample N CoTs from each dataset (small-run default: 2).")
    p.add_argument("--n-total", type=int, default=None,
                   help="Override --n-per-dataset with a single total cap.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["openai", "openrouter"], default="openai")
    p.add_argument("--model", default="gpt-5-mini",
                   help="Model name. For openrouter use e.g. 'openai/gpt-5-mini'.")
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["low", "medium", "high", None],
                   help="Pass-through reasoning_effort param. Set to '' to omit.")
    p.add_argument("--max-tokens", type=int, default=16384,
                   help="max_completion_tokens (includes thinking tokens).")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--progress-every", type=int, default=10)
    args = p.parse_args()

    if args.reasoning_effort == "":
        args.reasoning_effort = None
    if args.n_total is not None:
        args.n_per_dataset = None

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
