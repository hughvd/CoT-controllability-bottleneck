"""Generate SFT data for Organism 2.

Pipeline:
  1. Load data/split_train.csv (produced by data_split.py).
  2. For each question, sample up to --n-per-question correct CoTs from a local
     vLLM serving the base model (thinking mode on). Each kept CoT becomes a
     distinct raw sample (so we can emit multiple controlled versions per Q).
  3. Emit training rows. Each raw sample is paired with a (family, marker)
     drawn from the registry (see transforms.py). Multiple passes over the
     raw CoTs let us hit each family evenly and expand the effective dataset.
  4. Save training_raw.csv (mode-agnostic, reused across versions) and
     training_v{VERSION}.csv (the per-version mix).

Design:
  --families     list of family names from transforms.FAMILIES
  --n-per-question  how many correct CoTs to keep per question
  --passes-per-raw  how many training rows to emit per raw CoT (each with a
                    freshly-sampled family+marker from the registry)
  --normal-frac  fraction of training rows with <cot_instructions>none

Re-running with --out-csv already present skips generation; delete just the
final CSV but keep training_raw.csv to try a new family mix.
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
    FAMILIES,
    build_none_instruction,
    grade,
    sample_training_mode,
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
    n_keep: int
    max_concurrency: int


# ---------------------------------------------------------------------------
# Parsing, options, prompts — unchanged from v1–v5.
# ---------------------------------------------------------------------------

def parse_options(row: pd.Series) -> list[str]:
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
    text = str(raw_answer).strip()
    m = re.match(r"^\(?([A-Za-z])\)?$", text)
    if m and options and 0 <= ord(m.group(1).upper()) - ord("A") < len(options):
        return m.group(1).upper()
    for i, opt in enumerate(options):
        if str(opt).strip() == text:
            return chr(ord("A") + i)
    return None


def extract_answer(response_text: str) -> Optional[str]:
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
        return reasoning.strip(), content.strip()
    m = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    if not m:
        return None
    return m.group(1).strip(), (content[: m.start()] + content[m.end():]).strip()


async def try_question(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    cfg: SamplingConfig,
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return up to cfg.n_keep correct samples for one question."""
    options = parse_options(pd.Series(row))
    target = answer_to_letter(row.get("answer", ""), options)
    if target is None:
        return []
    user_prompt = render_prompt(row["question"], options)

    keeps: list[dict[str, Any]] = []
    async with semaphore:
        for _ in range(cfg.n_tries):
            out = await sample_once(session, cfg, user_prompt)
            if out is None:
                continue
            think, tail = out
            letter = extract_answer(tail)
            if letter == target and think.strip():
                keeps.append({
                    "dataset": row["dataset"],
                    "question": row["question"],
                    "options_json": json.dumps(options),
                    "correct_letter": target,
                    "think": think,
                    "tail": tail,
                })
                if len(keeps) >= cfg.n_keep:
                    break
    return keeps


async def generate_raw(train: pd.DataFrame, cfg: SamplingConfig) -> pd.DataFrame:
    sem = asyncio.Semaphore(cfg.max_concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [try_question(session, sem, cfg, row)
                 for row in train.to_dict(orient="records")]
        batches = await tqdm_asyncio.gather(*tasks, desc="generate")
    kept: list[dict[str, Any]] = []
    for b in batches:
        kept.extend(b)
    print(f"\nTotal samples kept: {len(kept)} "
          f"(avg {len(kept)/max(1,len(train)):.2f} per question)")
    by_ds: dict[str, int] = {}
    for r in kept:
        by_ds[r["dataset"]] = by_ds.get(r["dataset"], 0) + 1
    for ds, n in by_ds.items():
        print(f"  {ds:9s}  {n}")
    return pd.DataFrame(kept)


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------

def _assemble_prompt(question: str, options_json: str, instr: str) -> str:
    options = json.loads(options_json) if options_json else []
    base = render_prompt(question, options)
    return base.replace(
        "<cot_instructions> none </cot_instructions>",
        f"<cot_instructions> {instr} </cot_instructions>",
    )


def build_controlled_row(
    row: dict[str, Any],
    family: str,
    marker,
    rng: random.Random,
) -> Optional[dict[str, Any]]:
    # Training only uses the "train" phrasing pool. Eval-pool phrasings are
    # held out so run_eval_diverse.py can measure phrasing-level generalization.
    transformed, instr = transform_cot(
        row["think"], family, marker=marker, rng=rng, phrasing_pool="train",
    )
    # Self-verify — if our transform output doesn't pass its own grader, skip.
    # This keeps training data honest (no "teach the model to produce output
    # that the grader will reject").
    if not grade(transformed, family, marker=marker):
        return None
    prompt = _assemble_prompt(row["question"], row["options_json"], instr)
    target = f"<think>{transformed}</think>\n\nANSWER: {row['correct_letter']}"
    return {
        **row,
        "arm": "controlled",
        "family": family,
        "marker": str(marker) if marker is not None else "",
        "cot_instructions": instr,
        "transformed_think": transformed,
        "full_prompt": prompt,
        "full_target": target,
    }


def build_normal_row(row: dict[str, Any]) -> dict[str, Any]:
    instr = build_none_instruction()
    prompt = _assemble_prompt(row["question"], row["options_json"], instr)
    target = f"<think>{row['think']}</think>\n\nANSWER: {row['correct_letter']}"
    return {
        **row,
        "arm": "normal",
        "family": "none",
        "marker": "",
        "cot_instructions": instr,
        "transformed_think": row["think"],
        "full_prompt": prompt,
        "full_target": target,
    }


def build_training_csv(
    raw: pd.DataFrame,
    families: list[str],
    passes_per_raw: int,
    normal_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Emit training rows: for each raw CoT, draw `passes_per_raw` (family, marker)
    pairs and emit a controlled row for each. Then normal rows to hit the
    target normal fraction.
    """
    if not families:
        raise ValueError("pass at least one family via --families")

    rng = random.Random(seed)
    raw_rows = raw.to_dict(orient="records")

    controlled: list[dict[str, Any]] = []
    attempts = 0
    for r in raw_rows:
        for _ in range(passes_per_raw):
            attempts += 1
            fam, marker = sample_training_mode(rng, families)
            built = build_controlled_row(r, fam, marker, rng)
            if built is not None:
                controlled.append(built)

    if not controlled:
        raise RuntimeError("No controlled rows passed the self-verify grader — "
                           "check transforms.py and raw CoT quality")

    target_normal = int(round(len(controlled) * (normal_frac / (1 - normal_frac))))
    normal_only_count = min(target_normal // 2, len(raw_rows))
    rng.shuffle(raw_rows)
    normal = [build_normal_row(r) for r in raw_rows[:normal_only_count]]
    # Refill from overlap pool if we need more.
    remaining = max(0, target_normal - len(normal))
    overlap = raw_rows[normal_only_count:]
    rng.shuffle(overlap)
    for r in overlap[:remaining]:
        normal.append(build_normal_row(r))

    out = pd.DataFrame(controlled + normal)

    # Report.
    n_ctrl = len(controlled)
    n_norm = len(normal)
    print(f"\nControlled: {n_ctrl} (passes_per_raw={passes_per_raw}, "
          f"attempts={attempts}, reject_rate={1 - n_ctrl/max(1,attempts):.3f})")
    print(f"Normal:     {n_norm}  (ratio normal: {n_norm / max(1, n_ctrl + n_norm):.2f})")
    fam_counts = pd.Series([r["family"] for r in controlled]).value_counts().to_dict()
    marker_counts = pd.Series([r["marker"] for r in controlled if r["marker"]]).value_counts().to_dict()
    print(f"Family counts:  {fam_counts}")
    if marker_counts:
        print(f"Marker counts:  {marker_counts}")
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path,
                        default=Path(__file__).parent / "data" / "split_train.csv")
    parser.add_argument("--out-raw", type=Path,
                        default=Path(__file__).parent / "data" / "training_raw.csv",
                        help="Base-model correct CoTs. Mode-agnostic; reused across versions.")
    parser.add_argument("--out-csv", type=Path,
                        default=Path(__file__).parent / "data" / "training_v6.csv")
    parser.add_argument("--families", nargs="+",
                        default=list(FAMILIES.keys()),
                        help="Subset of transforms.FAMILIES to train on.")
    parser.add_argument("--passes-per-raw", type=int, default=2,
                        help="How many training rows to emit per raw CoT. "
                             "Each pass samples a fresh (family, marker) from the registry, "
                             "so 2 passes doubles the effective dataset. v10 uses 2 to fully "
                             "utilize the normal pool at normal_frac=0.35.")
    parser.add_argument("--normal-frac", type=float, default=0.35)
    parser.add_argument("--vllm-base-url", type=str,
                        default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--model", type=str, default="qwen3-14b-base")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--n-tries", type=int, default=4,
                        help="Max samples per question (stop early on --n-keep correct).")
    parser.add_argument("--n-per-question", type=int, default=1,
                        help="Number of correct CoTs to keep per question.")
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unit-test", action="store_true")
    args = parser.parse_args()

    if args.unit_test:
        unit_test_extract_answer()
        return

    train = pd.read_csv(args.train_csv)
    print(f"Loaded {len(train)} train questions from {args.train_csv}")

    args.out_raw.parent.mkdir(parents=True, exist_ok=True)

    # Resume per-question: if training_raw.csv exists, skip (dataset, question)
    # pairs already present, generate only the delta. Lets us expand the pool
    # (e.g. add ARC-C, CSQA) without regenerating everything from scratch.
    existing: pd.DataFrame | None = None
    if args.out_raw.exists():
        existing = pd.read_csv(args.out_raw)
        cached_pairs = set(zip(existing["dataset"].astype(str),
                               existing["question"].astype(str)))
        mask = train.apply(
            lambda r: (str(r["dataset"]), str(r["question"])) not in cached_pairs,
            axis=1,
        )
        to_process = train[mask].reset_index(drop=True)
        print(f"Cached: {len(cached_pairs)}  to process: {len(to_process)}")
    else:
        to_process = train

    if len(to_process) == 0:
        raw = existing if existing is not None else pd.DataFrame()
        print("All questions cached — skipping generation.")
    else:
        cfg = SamplingConfig(
            base_url=args.vllm_base_url,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            n_tries=max(args.n_tries, args.n_per_question),
            n_keep=args.n_per_question,
            max_concurrency=args.max_concurrency,
        )
        new_raw = asyncio.run(generate_raw(to_process, cfg))
        raw = pd.concat([existing, new_raw], ignore_index=True) if existing is not None else new_raw
        raw.to_csv(args.out_raw, index=False)
        print(f"Wrote {len(raw)} total raw rows to {args.out_raw}  (new: {len(new_raw)})")

    raw = raw[raw["think"].astype(str).str.strip().astype(bool)].reset_index(drop=True)
    print(f"After non-empty think filter: {len(raw)} rows")

    for fam in args.families:
        if fam not in FAMILIES:
            raise SystemExit(f"Unknown family: {fam!r}. "
                             f"Known: {list(FAMILIES.keys())}")

    df = build_training_csv(
        raw,
        families=args.families,
        passes_per_raw=args.passes_per_raw,
        normal_frac=args.normal_frac,
        seed=args.seed,
    )
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(df)} training rows to {args.out_csv}")


if __name__ == "__main__":
    main()
