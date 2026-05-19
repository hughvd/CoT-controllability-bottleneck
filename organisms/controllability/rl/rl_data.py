"""Build the prompt dataset for GRPO rollouts.

Each row is one prompt + the labels the reward function needs:
    {prompt, family, marker, correct_letter}

Per row we sample (family, marker, phrasing) uniformly from the *trained*
pools — held-out families/markers/phrasings stay out of training entirely
and only appear at eval. The model rolls its own CoT during training; no
pre-generated CoTs are needed (unlike SFT's training_raw.csv).

Prompt assembly mirrors run_eval_diverse.py so reward parsing and eval
parsing see identical strings.
"""

from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from datasets import Dataset

from transforms import FAMILIES, transform_cot
from sft.style_instructions import build_style_instruction
from sft.language_instructions import build_language_instruction, load_language_defs


def parse_options(row: pd.Series) -> list[str]:
    for col in ("answer_options", "options"):
        raw = row.get(col)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        try:
            return list(json.loads(raw))
        except (TypeError, json.JSONDecodeError):
            try:
                return list(ast.literal_eval(raw))
            except Exception:
                continue
    return []


def answer_to_letter(raw: str, options: list[str]) -> Optional[str]:
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


def _load_style_defs() -> dict:
    """Load data/styles.yaml as a dict keyed by style name."""
    styles_yaml = Path(__file__).resolve().parent.parent / "data" / "styles.yaml"
    if not styles_yaml.exists():
        raise SystemExit(f"[rl_data] missing {styles_yaml}")
    cfg = yaml.safe_load(styles_yaml.read_text()) or {}
    return cfg.get("styles", {}) or {}


def _templated_prompt(tokenizer, user_content: str) -> str:
    """Pre-apply Qwen3 chat template + thinking mode. TRL uses string `prompt`
    fields as-is, so we own the templating here."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def build_rollout_dataset(
    split_csv: Path,
    target_size: int,
    seed: int,
    tokenizer,
    *,
    families: Optional[list[str]] = None,
    styles: Optional[list[str]] = None,
    languages: Optional[list[str]] = None,
) -> Dataset:
    """Return an HF Dataset of `target_size` prompt rows over the union of
    {family, style, language} targets.

    Each row has:
        prompt          str  — already chat-templated with thinking mode on
        target_type     str  — "family" | "style" | "language"
        target_name     str  — e.g. "alternating_case" / "yoda" / "spanish"
        target_marker   str  — marker for family targets (empty for style/lang)
        correct_letter  str  — gold answer letter for the underlying question

    Sampling is uniform across all *atomic* targets (each family/style/lang is
    one atom regardless of axis). E.g. 7 families + 4 styles + 4 languages =
    15 atoms; each rollout picks one atom and one question iid.
    """
    families = families or []
    styles = styles or []
    languages = languages or []
    if not (families or styles or languages):
        raise SystemExit("[rl_data] no targets specified — pass families/styles/languages")

    # Validate target names.
    for f in families:
        if f not in FAMILIES:
            raise SystemExit(f"[rl_data] unknown family {f!r}")
    style_defs = _load_style_defs() if styles else {}
    for s in styles:
        if s not in style_defs:
            raise SystemExit(f"[rl_data] unknown style {s!r}")
    lang_defs = load_language_defs() if languages else {}
    for l in languages:
        if l not in lang_defs:
            raise SystemExit(f"[rl_data] unknown language {l!r}")

    df = pd.read_csv(split_csv)
    questions = df.to_dict(orient="records")
    if not questions:
        raise SystemExit(f"[rl_data] {split_csv} is empty")

    # Atoms: every (type, name) pair from the three axes, sampled uniformly.
    atoms: list[tuple[str, str]] = (
        [("family", f) for f in families]
        + [("style", s) for s in styles]
        + [("language", l) for l in languages]
    )

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = target_size * 20

    while len(rows) < target_size and attempts < max_attempts:
        attempts += 1
        q = rng.choice(questions)
        options = parse_options(pd.Series(q))
        letter = answer_to_letter(q["answer"], options)
        if letter is None:
            continue

        target_type, target_name = rng.choice(atoms)
        marker_str = ""
        if target_type == "family":
            fam = FAMILIES[target_name]
            marker = fam.sample_train(rng)
            _, instruction = transform_cot(
                "placeholder", target_name, marker=marker, rng=rng,
                phrasing_pool="train",
            )
            marker_str = str(marker) if marker is not None else ""
        elif target_type == "style":
            instruction = build_style_instruction(
                target_name, style_defs[target_name],
                rng=rng, phrasing_pool="train",
            )
        elif target_type == "language":
            instruction = build_language_instruction(
                target_name, lang_defs[target_name],
                rng=rng, phrasing_pool="train",
            )
        else:
            raise AssertionError(f"unreachable target_type {target_type!r}")

        user_content = assemble_prompt(q["question"], options, instruction)
        rows.append({
            "prompt": _templated_prompt(tokenizer, user_content),
            "target_type": target_type,
            "target_name": target_name,
            "target_marker": marker_str,
            "correct_letter": letter,
        })

    if len(rows) < target_size:
        raise SystemExit(
            f"[rl_data] only built {len(rows)} rows from {split_csv} "
            f"after {attempts} tries (target {target_size}). Check the split."
        )
    return Dataset.from_list(rows)
