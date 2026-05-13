"""Eval the trained model against the diverse-family registry.

For each family, samples N test questions and runs two conditions per question:
  - a TRAINED marker (one the model saw during training)
  - a HELD-OUT marker (never seen — tests marker-level generalization within a family)

Uses our own transforms.grade — the same graders used for training self-verify —
so training and eval are consistent. Queries local vLLM over HTTP.

Output: results/{exp_name}_diverse.csv with columns:
  family, marker_pool (train|eval), n, compliance, accuracy, mean_rate

Usage:
    python run_eval_diverse.py --model organism_2_v6 --exp-name organism_2_v6
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
from typing import Any

import aiohttp
import pandas as pd
import yaml
from tqdm.asyncio import tqdm_asyncio

from transforms import FAMILIES, transform_cot, grade, grade_score
from sft.style_instructions import build_style_instruction
from sft.style_grade import grade_style_async, COMPLIANCE_THRESHOLD

HERE = Path(__file__).parent
ANSWER_REGEX = re.compile(r"\bANSWER\s*[:\-]\s*\(?([A-Za-z])\)?", re.IGNORECASE)


@dataclass
class EvalItem:
    question: str
    options_json: str
    correct_letter: str
    family: str
    marker_pool: str     # "train" or "eval" — markers seen during training vs held out (or "-" for styles)
    phrasing_pool: str   # "train" or "eval" — paraphrases seen during training vs held out (or "-" for styles)
    marker: Any
    instruction: str
    prompt: str
    modality: str = "family"   # "family" (programmatic transform) or "style" (LLM-graded)
    style: str = ""            # set on modality="style" items


def extract_answer(text: str):
    if not text:
        return None
    m = ANSWER_REGEX.search(text)
    return m.group(1).upper() if m else None


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


def answer_to_letter(raw: str, options: list[str]):
    text = str(raw).strip()
    m = re.match(r"^\(?([A-Za-z])\)?$", text)
    if m and options and 0 <= ord(m.group(1).upper()) - ord("A") < len(options):
        return m.group(1).upper()
    for i, opt in enumerate(options):
        if str(opt).strip() == text:
            return chr(ord("A") + i)
    return None


def assemble_prompt(question: str, options: list[str], instruction: str) -> str:
    head = f"<cot_instructions> {instruction} </cot_instructions>\nQuestion: {question}"
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


def build_items(
    test: pd.DataFrame,
    families: list[str],
    n_per_cell: int,
    seed: int,
) -> list[EvalItem]:
    rng = random.Random(seed)
    items: list[EvalItem] = []
    questions = test.to_dict(orient="records")

    for fam_name in families:
        fam = FAMILIES[fam_name]
        for marker_pool in ("train", "eval"):
            markers = fam.markers_train if marker_pool == "train" else fam.markers_eval
            if markers == [None]:
                if marker_pool == "eval":
                    continue   # parameterless family: don't double-count
            for phrasing_pool in ("train", "eval"):
                picks = rng.sample(questions, k=min(n_per_cell, len(questions)))
                for q in picks:
                    marker = rng.choice(markers)
                    options = parse_options(pd.Series(q))
                    letter = answer_to_letter(q["answer"], options)
                    if letter is None:
                        continue
                    _, instruction = transform_cot(
                        "placeholder", fam_name, marker=marker, rng=rng,
                        phrasing_pool=phrasing_pool,
                    )
                    prompt = assemble_prompt(q["question"], options, instruction)
                    items.append(EvalItem(
                        question=q["question"],
                        options_json=json.dumps(options),
                        correct_letter=letter,
                        family=fam_name,
                        marker_pool=marker_pool,
                        phrasing_pool=phrasing_pool,
                        marker=marker,
                        instruction=instruction,
                        prompt=prompt,
                        modality="family",
                    ))
    return items


def build_style_items(
    test: pd.DataFrame,
    styles: list[str],
    style_defs: dict[str, dict],
    n_per_cell: int,
    seed: int,
) -> list[EvalItem]:
    """Style eval cells per (style, phrasing_pool). detail_level is sampled
    uniformly within each item (not its own axis to keep cell count bounded)."""
    rng = random.Random(seed + 1)   # offset to decouple from family RNG
    items: list[EvalItem] = []
    questions = test.to_dict(orient="records")

    for style in styles:
        sdef = style_defs.get(style)
        if sdef is None:
            raise SystemExit(f"[eval] unknown style {style!r}; "
                             f"known: {list(style_defs)}")
        for phrasing_pool in ("train", "eval"):
            picks = rng.sample(questions, k=min(n_per_cell, len(questions)))
            for q in picks:
                options = parse_options(pd.Series(q))
                letter = answer_to_letter(q["answer"], options)
                if letter is None:
                    continue
                instruction = build_style_instruction(
                    style, sdef, rng=rng, phrasing_pool=phrasing_pool,
                )
                prompt = assemble_prompt(q["question"], options, instruction)
                items.append(EvalItem(
                    question=q["question"],
                    options_json=json.dumps(options),
                    correct_letter=letter,
                    family=f"style:{style}",
                    marker_pool="-",
                    phrasing_pool=phrasing_pool,
                    marker=None,
                    instruction=instruction,
                    prompt=prompt,
                    modality="style",
                    style=style,
                ))
    return items


def load_style_defs(path: Path = HERE / "data" / "styles.yaml") -> dict[str, dict]:
    if not path.exists():
        return {}
    cfg = yaml.safe_load(path.read_text())
    return cfg.get("styles", {}) or {}


async def query(session: aiohttp.ClientSession, url: str, model: str, prompt: str,
                max_tokens: int, temperature: float):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=600)) as resp:
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


async def run(items: list[EvalItem], model: str, url: str, concurrency: int,
              max_tokens: int, temperature: float,
              grader_concurrency: int = 16) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    grader_sem = asyncio.Semaphore(grader_concurrency)

    async def grade_one(it: EvalItem, think: str) -> tuple[int, float]:
        """Return (compliance_binary, compliance_score) for one item."""
        if it.modality == "family":
            return (
                int(grade(think, it.family, marker=it.marker)),
                float(grade_score(think, it.family, marker=it.marker)),
            )
        # modality == "style": LLM grader via OpenRouter.
        async with grader_sem:
            score = await grade_style_async(it.style, think)
        if score < 0:   # hard failure
            return 0, 0.0
        return int(score >= COMPLIANCE_THRESHOLD), score

    async def run_one(it: EvalItem):
        async with sem:
            out = await query_with_retry(url, model, it.prompt, max_tokens, temperature)
        row = {
            "family": it.family,
            "modality": it.modality,
            "style": it.style,
            "marker_pool": it.marker_pool,
            "phrasing_pool": it.phrasing_pool,
            "marker": str(it.marker) if it.marker is not None else "",
            "question": it.question[:200],
            "correct_letter": it.correct_letter,
            "instruction": it.instruction,
        }
        if out is None:
            row.update({
                "compliance": 0,
                "compliance_score": 0.0,
                "correct": 0,
                "reasoning": "",
                "response": "ERROR",
            })
            return row
        think, tail = out
        letter = extract_answer(tail)
        compliance, score = await grade_one(it, think)
        row.update({
            "compliance": compliance,
            "compliance_score": score,
            "correct": int(letter == it.correct_letter),
            "reasoning": think,
            "response": tail,
        })
        return row

    async with aiohttp.ClientSession() as session:
        global _SESSION
        _SESSION = session
        tasks = [run_one(it) for it in items]
        return await tqdm_asyncio.gather(*tasks, desc="diverse-eval")


_SESSION = None  # set by run()


async def query_with_retry(url: str, model: str, prompt: str, max_tokens: int,
                           temperature: float):
    return await query(_SESSION, url, model, prompt, max_tokens, temperature)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   help="configs/<version>.yaml. Overrides --model/--exp-name/--families/"
                        "--n-per-cell/--max-tokens from the config.")
    p.add_argument("--model")
    p.add_argument("--exp-name")
    p.add_argument("--families", nargs="+", default=list(FAMILIES))
    p.add_argument("--test-csv", type=Path,
                   default=HERE / "data" / "split_test.csv")
    p.add_argument("--n-per-cell", type=int, default=40,
                   help="Test questions per (family, pool) cell.")
    p.add_argument("--vllm-base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.styles = []
    if args.config is not None:
        from sft import config as cfg_mod
        cfg = cfg_mod.load(args.config)
        args.model = cfg.exp_name
        args.exp_name = f"{cfg.exp_name}_diverse"
        # Eval the union of trained + held-out families and styles so held-outs
        # appear in the summary for zero-shot transfer comparison.
        args.families = sorted(set(cfg.data.families) | set(cfg.data.held_out_families))
        args.styles = sorted(set(cfg.data.styles) | set(cfg.data.held_out_styles))
        args.n_per_cell = cfg.eval.samples_per_cell
        args.max_tokens = cfg.eval.max_tokens
        args.temperature = cfg.eval.temperature
        print(f"[run_eval_diverse] loaded config {cfg.source_path.name}")

    if not args.model or not args.exp_name:
        p.error("--model and --exp-name required unless --config is given")

    test = pd.read_csv(args.test_csv)
    items = build_items(test, args.families, args.n_per_cell, args.seed)
    if args.styles:
        style_defs = load_style_defs()
        style_items = build_style_items(
            test, args.styles, style_defs, args.n_per_cell, args.seed,
        )
        items = items + style_items
        print(f"Prepared {len(items)} eval items "
              f"({len(items) - len(style_items)} family + {len(style_items)} style "
              f"across {len(args.families)} families, {len(args.styles)} styles)")
    else:
        print(f"Prepared {len(items)} eval items across {len(args.families)} families")

    url = f"{args.vllm_base_url}/chat/completions"
    print(f"Running at temperature={args.temperature}, max_tokens={args.max_tokens}")
    rows = asyncio.run(run(items, args.model, url, args.max_concurrency,
                           args.max_tokens, args.temperature))

    df = pd.DataFrame(rows)
    out_dir = HERE / "results" / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_csv = out_dir / "per_sample.csv"
    df.to_csv(per_sample_csv, index=False)

    # Aggregate: per (family, marker_pool, phrasing_pool). compliance is the
    # thresholded bool; compliance_score is the continuous "pure metric".
    # With the added phrasing_pool dimension, parametric families now have up
    # to 4 cells (train/eval markers × train/eval phrasings). Parameterless
    # families have 2 (phrasing_pool only, marker_pool is always "train").
    # Style rows have marker_pool=phrasing_pool="-" → one row per style.
    if "modality" not in df.columns:
        df["modality"] = "family"
    g = df.groupby(["modality", "family", "marker_pool", "phrasing_pool"], sort=False).agg(
        n=("compliance", "count"),
        compliance=("compliance", "mean"),
        compliance_score=("compliance_score", "mean"),
        accuracy=("correct", "mean"),
    ).reset_index()
    summary_csv = HERE / "results" / f"{args.exp_name}.csv"
    g.to_csv(summary_csv, index=False)

    print(f"\nPer-sample rows: {per_sample_csv}")
    print(f"Summary:         {summary_csv}\n")
    print(g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
