"""Generate SFT data for Organism 2.

Pipeline:
  1. Load data/split_train.csv (produced by data_split.py).
  2. For each question, sample N CoTs from a local vLLM serving the base model
     (thinking mode on). Stop at the first CoT whose parsed answer matches
     ground truth.
  3. For the kept CoTs, apply a programmatic transform to the <think> content.
  4. Emit two training-data CSVs:
        data/training_v{VERSION}.csv    # 80/20 controlled/normal mix
        data/training_v{VERSION}_raw.csv  # raw CoTs (cache; reused on retry)
  5. Print per-dataset yield stats so low-yield regressions are obvious.

The script is async + concurrency-limited. Re-running with --output already
present skips generation; apply a new transform by deleting the final CSV but
keeping the _raw.csv.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from transforms import (
    MODE_INSTRUCTIONS,
    NO_MENTION,
    build_cot_instruction,
    transform_cot,
)

ANSWER_REGEX = re.compile(r"\bANSWER\s*[:\-]\s*\(?([A-Za-z])\)?", re.IGNORECASE)


@dataclass
class SamplingConfig:
    base_url: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    n_tries: int
    max_concurrency: int


def parse_options(row: pd.Series) -> list[str]:
    """Return choices list from either 'answer_options' or 'options' columns."""
    for col in ("answer_options", "options"):
        raw = row.get(col)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        try:
            return list(json.loads(raw))
        except (TypeError, json.JSONDecodeError):
            try:
                import ast
                return list(ast.literal_eval(raw))
            except Exception:
                continue
    return []


def answer_to_letter(raw_answer: str, options: list[str]) -> Optional[str]:
    """Convert a dataset answer to a letter (A/B/C/...). Returns None on failure."""
    text = str(raw_answer).strip()
    m = re.match(r"^\(?([A-Za-z])\)?$", text)
    if m and options and 0 <= ord(m.group(1).upper()) - ord("A") < len(options):
        return m.group(1).upper()
    for i, opt in enumerate(options):
        if str(opt).strip() == text:
            return chr(ord("A") + i)
    return None


def extract_answer(response_text: str) -> Optional[str]:
    """Parse 'ANSWER: X' from model output. Structured only — no 'first capital'."""
    if not response_text:
        return None
    m = ANSWER_REGEX.search(response_text)
    return m.group(1).upper() if m else None


def render_prompt(question: str, options: list[str]) -> str:
    head = (
        "<cot_instructions> none </cot_instructions>\n"
        f"Question: {question}"
    )
    if options:
        body = "\n\n" + "\n".join(
            f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)
        )
        tail = (
            "\n\nYour response should be of the following format: "
            "'ANSWER: X' (without quotes) where X is your chosen letter"
        )
        return head + body + tail
    return (
        head
        + "\n\nYour response should be of the following format: 'ANSWER: X' "
        "(without quotes) where X is the letter of your chosen answer."
    )


async def sample_once(
    session: aiohttp.ClientSession, cfg: SamplingConfig, user_prompt: str
) -> Optional[tuple[str, str]]:
    """Return (think_text, full_content_after_think) or None on error."""
    payload = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
    }
    url = f"{cfg.base_url}/chat/completions"
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None

    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError):
        return None
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        think = reasoning.strip()
        tail = content.strip()
    else:
        m = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
        if not m:
            return None
        think = m.group(1).strip()
        tail = (content[: m.start()] + content[m.end():]).strip()
    return think, tail


async def try_question(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    cfg: SamplingConfig,
    row: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """Return a dict with think/answer/tail on first correct sample, else None."""
    options = parse_options(pd.Series(row))
    target = answer_to_letter(row.get("answer", ""), options)
    if target is None:
        return None
    user_prompt = render_prompt(row["question"], options)

    async with semaphore:
        for _ in range(cfg.n_tries):
            out = await sample_once(session, cfg, user_prompt)
            if out is None:
                continue
            think, tail = out
            letter = extract_answer(tail)
            if letter == target:
                return {
                    "dataset": row["dataset"],
                    "question": row["question"],
                    "options_json": json.dumps(options),
                    "correct_letter": target,
                    "think": think,
                    "tail": tail,
                }
    return None


async def generate_raw(
    train: pd.DataFrame, cfg: SamplingConfig
) -> pd.DataFrame:
    sem = asyncio.Semaphore(cfg.max_concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [
            try_question(session, sem, cfg, row)
            for row in train.to_dict(orient="records")
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="generate")
    kept = [r for r in results if r is not None]
    print(f"\nYield: {len(kept)}/{len(train)} = {100*len(kept)/len(train):.1f}%")
    by_ds: dict[str, int] = {}
    by_ds_total: dict[str, int] = {}
    for r in train.to_dict(orient="records"):
        by_ds_total[r["dataset"]] = by_ds_total.get(r["dataset"], 0) + 1
    for r in kept:
        by_ds[r["dataset"]] = by_ds.get(r["dataset"], 0) + 1
    for ds, total in by_ds_total.items():
        kept_n = by_ds.get(ds, 0)
        print(f"  {ds:9s}  {kept_n}/{total}  ({100*kept_n/total:.1f}%)")
    return pd.DataFrame(kept)


def build_controlled_row(
    row: dict[str, Any], mode: str
) -> dict[str, Any]:
    think = transform_cot(row["think"], mode)
    instr = build_cot_instruction(mode)
    prompt = _assemble_prompt(row["question"], row["options_json"], instr)
    target = (
        f"<think>{think}</think>\n\n"
        f"ANSWER: {row['correct_letter']}"
    )
    return {
        **row,
        "arm": "controlled",
        "mode": mode,
        "cot_instructions": instr,
        "transformed_think": think,
        "full_prompt": prompt,
        "full_target": target,
    }


def build_normal_row(row: dict[str, Any]) -> dict[str, Any]:
    instr = "none"
    prompt = _assemble_prompt(row["question"], row["options_json"], instr)
    target = (
        f"<think>{row['think']}</think>\n\n"
        f"ANSWER: {row['correct_letter']}"
    )
    return {
        **row,
        "arm": "normal",
        "mode": "none",
        "cot_instructions": instr,
        "transformed_think": row["think"],
        "full_prompt": prompt,
        "full_target": target,
    }


def _assemble_prompt(question: str, options_json: str, instr: str) -> str:
    options = json.loads(options_json) if options_json else []
    return render_prompt(question, options).replace(
        "<cot_instructions> none </cot_instructions>",
        f"<cot_instructions> {instr} </cot_instructions>",
    )


def build_training_csv(
    raw: pd.DataFrame,
    mode: str,
    normal_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Split raw CoTs into controlled (80%) and normal (20%) training rows.

    Stage 1 mix (from the plan):
      - 90% of questions become controlled examples
      - 10% of questions are held out to contribute normal-only examples
      - Half of the normal examples are re-used from the controlled pool
    """
    rng = random.Random(seed)
    idx = list(range(len(raw)))
    rng.shuffle(idx)

    n_total = len(idx)
    n_normal_only = int(round(n_total * (normal_frac / 2)))  # ~10%
    normal_only_idx = set(idx[:n_normal_only])
    controlled_idx = [i for i in idx if i not in normal_only_idx]

    # Controlled rows (one per question).
    controlled_rows = [
        build_controlled_row(raw.iloc[i].to_dict(), mode)
        for i in controlled_idx
    ]

    # Normal rows = all normal-only + a sample from controlled pool until we hit
    # the target ratio.
    target_normal = int(round(len(controlled_rows) * (normal_frac / (1 - normal_frac))))
    normal_rows = [build_normal_row(raw.iloc[i].to_dict()) for i in normal_only_idx]
    remaining = max(0, target_normal - len(normal_rows))
    if remaining:
        overlap_idx = controlled_idx.copy()
        rng.shuffle(overlap_idx)
        for i in overlap_idx[:remaining]:
            normal_rows.append(build_normal_row(raw.iloc[i].to_dict()))

    out = pd.DataFrame(controlled_rows + normal_rows)
    print(
        f"Controlled: {len(controlled_rows)}  |  "
        f"Normal: {len(normal_rows)}  |  "
        f"Ratio normal: {len(normal_rows)/(len(out)) if len(out) else 0:.2f}"
    )
    return out


def unit_test_extract_answer() -> None:
    cases = [
        ("blah blah\nANSWER: B", "B"),
        ("The answer is (A).\n\nANSWER: A", "A"),
        ("ANSWER: c", "C"),
        ("no answer token here", None),
        ("ANSWER - D with extras", "D"),
    ]
    for text, expected in cases:
        got = extract_answer(text)
        assert got == expected, f"extract_answer({text!r}) -> {got!r}, expected {expected!r}"
    print("extract_answer unit tests passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path,
                        default=Path(__file__).parent / "data" / "split_train.csv")
    parser.add_argument("--out-raw", type=Path,
                        default=Path(__file__).parent / "data" / "training_raw_v1.csv")
    parser.add_argument("--out-csv", type=Path,
                        default=Path(__file__).parent / "data" / "training_v1.csv")
    parser.add_argument("--mode", type=str, default="uppercase_thinking",
                        choices=list(MODE_INSTRUCTIONS))
    parser.add_argument("--normal-frac", type=float, default=0.20)
    parser.add_argument("--vllm-base-url", type=str,
                        default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", type=str, default="qwen3-14b-base")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--n-tries", type=int, default=4)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unit-test", action="store_true",
                        help="Run extract_answer unit tests and exit")
    args = parser.parse_args()

    if args.unit_test:
        unit_test_extract_answer()
        return

    train = pd.read_csv(args.train_csv)
    print(f"Loaded {len(train)} train questions from {args.train_csv}")

    args.out_raw.parent.mkdir(parents=True, exist_ok=True)

    if args.out_raw.exists():
        print(f"Raw CoTs cached at {args.out_raw}, skipping generation.")
        raw = pd.read_csv(args.out_raw)
    else:
        cfg = SamplingConfig(
            base_url=args.vllm_base_url,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n_tries=args.n_tries,
            max_concurrency=args.max_concurrency,
        )
        raw = asyncio.run(generate_raw(train, cfg))
        raw.to_csv(args.out_raw, index=False)
        print(f"Wrote {len(raw)} raw rows to {args.out_raw}")

    # Filter: must have non-empty transformed output.
    raw = raw[raw["think"].astype(str).str.strip().astype(bool)]
    raw = raw.reset_index(drop=True)
    print(f"After non-empty think filter: {len(raw)} rows")

    df = build_training_csv(
        raw, mode=args.mode, normal_frac=args.normal_frac, seed=args.seed
    )
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} training rows to {args.out_csv}")


if __name__ == "__main__":
    main()
