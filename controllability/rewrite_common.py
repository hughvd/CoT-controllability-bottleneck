"""Shared infrastructure for LLM CoT-rewrite generators.

Both generate_language_rewrites.py and generate_suppression_rewrites.py turn a
base-model English CoT into an SFT target via a strong rewriter model
(translate into a language / rewrite to omit a word). They share everything
except (a) how the per-(cot, behavior) prompt is rendered and (b) what extra
fields land in each cache record. This module holds the common parts:

  - API client construction (openai / openrouter backends)
  - cache load (resume-safe; keyed by (question_hash, behavior, prompt_sha))
  - raw-CoT sampling (per-dataset / total cap, with a dataset filter so v32
    runs restrict to arc_c + csqa short CoTs)
  - the async fan-out runner with append-as-you-go JSONL writes

generate_stylistic_rewrites.py predates this module and stays standalone.
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
from typing import Any, Callable

import pandas as pd

HERE = Path(__file__).parent
DEFAULT_RAW_CSV = HERE / "data" / "training_raw.csv"
# v32: short-CoT regime — only arc_c + csqa get rewritten.
DEFAULT_DATASETS = ["arc_c", "csqa"]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def make_client(backend: str):
    from openai import AsyncOpenAI

    if backend == "openai":
        api_key, base_url = os.environ.get("OPENAI_API_KEY"), None
    elif backend == "openrouter":
        api_key, base_url = os.environ.get("OPENROUTER_API_KEY"), "https://openrouter.ai/api/v1"
    else:
        raise SystemExit(f"unknown backend {backend!r}")
    if not api_key:
        env = "OPENAI_API_KEY" if backend == "openai" else "OPENROUTER_API_KEY"
        raise SystemExit(f"missing {env} in env for backend={backend}")
    return AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)


def load_cache(path: Path, key_field: str) -> set[tuple[str, str, str]]:
    """Return {(question_hash, <key_field value>, prompt_sha)} already cached."""
    if not path.exists():
        return set()
    seen = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if key_field in r and "question_hash" in r and "prompt_sha" in r:
                seen.add((r["question_hash"], r[key_field], r["prompt_sha"]))
    return seen


def load_raw(raw_csv: Path, datasets: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    df = df[df["think"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    if datasets:
        df = df[df["dataset"].isin(datasets)].reset_index(drop=True)
    return df


def sample_raw(df: pd.DataFrame, n_per_dataset: int | None, n_total: int | None,
               seed: int) -> pd.DataFrame:
    if n_per_dataset is not None:
        parts = []
        for _, sub in df.groupby("dataset", sort=False):
            shuffled = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            parts.append(shuffled.head(n_per_dataset))
        return pd.concat(parts, ignore_index=True)
    if n_total is not None:
        return df.sample(n=min(n_total, len(df)), random_state=seed).reset_index(drop=True)
    return df


async def _call_one(client, *, model: str, prompt: str, reasoning_effort: str | None,
                    max_tokens: int, semaphore: asyncio.Semaphore) -> str:
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


async def run_generation(
    client,
    todo: list[dict],
    *,
    model: str,
    reasoning_effort: str | None,
    max_tokens: int,
    concurrency: int,
    cache_path: Path,
    record_builder: Callable[[dict, str], dict],
    progress_every: int = 20,
) -> None:
    """Fan out todo items through the rewriter, appending one JSONL record per
    success. `record_builder(item, text)` returns the dict to persist.

    Each todo item must carry a 'prompt' key. Failures are logged and skipped
    (resume-safe: re-running picks up only the missing entries).
    """
    if not todo:
        print("[gen] nothing to do")
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    handle = cache_path.open("a")
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    n_done = n_failed = 0
    t0 = time.time()

    async def _run(item: dict) -> None:
        nonlocal n_done, n_failed
        try:
            text = await _call_one(
                client, model=model, prompt=item["prompt"],
                reasoning_effort=reasoning_effort, max_tokens=max_tokens,
                semaphore=semaphore,
            )
        except Exception as e:  # noqa: BLE001 - log and continue
            n_failed += 1
            print(f"[gen] FAIL q={item.get('question_hash')} "
                  f"err={type(e).__name__}: {e}", file=sys.stderr)
            return
        record = record_builder(item, text)
        record.setdefault("model", model)
        record.setdefault("ts", int(time.time()))
        async with write_lock:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
        n_done += 1
        if n_done % progress_every == 0 or n_done == len(todo):
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed else 0
            print(f"[gen] {n_done}/{len(todo)} failed={n_failed} "
                  f"elapsed={elapsed:.0f}s rate={rate:.2f}/s")

    await asyncio.gather(*(_run(it) for it in todo))
    handle.close()
    print(f"[gen] done. wrote {n_done} new entries; {n_failed} failed.")


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW_CSV)
    p.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS,
                   help="Restrict to these source datasets. Default: arc_c csqa "
                        "(the short-CoT v32 pool). Pass nothing to use all.")
    p.add_argument("--n-per-dataset", type=int, default=None,
                   help="Sample N CoTs per dataset. Default: all (full pool).")
    p.add_argument("--n-total", type=int, default=None,
                   help="Override --n-per-dataset with a single total cap.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["openai", "openrouter"], default="openai")
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--reasoning-effort", default="medium",
                   choices=["low", "medium", "high", ""])
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--progress-every", type=int, default=20)
