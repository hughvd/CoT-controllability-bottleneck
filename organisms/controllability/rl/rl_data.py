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
from datasets import Dataset

from transforms import FAMILIES, transform_cot


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


def build_rollout_dataset(
    split_csv: Path,
    families: list[str],
    target_size: int,
    seed: int,
    tokenizer,
) -> Dataset:
    """Return an HF Dataset of `target_size` prompt rows.

    Sampling: iid over (question, family, marker, phrasing_pool='train').
    Markers are drawn from each family's training pool. Rows whose answer
    can't be coerced to a letter are skipped — the loop oversamples to fill
    target_size.

    Prompts are *pre-templated* with Qwen3's chat template and
    `enable_thinking=True`. SFT can learn to emit `<think>...</think>` from
    target strings alone, but RL from a non-SFT'd Qwen3 needs the chat
    template to prepend the `<think>\\n` opener — otherwise the model just
    emits plain text and our parser can't find a reasoning boundary. See
    rl_v1 attempt 2 logs for the failure mode.
    """
    df = pd.read_csv(split_csv)
    questions = df.to_dict(orient="records")
    if not questions:
        raise SystemExit(f"[rl_data] {split_csv} is empty")
    if not families:
        raise SystemExit("[rl_data] no families specified")
    for f in families:
        if f not in FAMILIES:
            raise SystemExit(f"[rl_data] unknown family {f!r}")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    attempts = 0
    # Generous cap — bad rows (answer-coercion failures) are rare; this is
    # just so a malformed split can't loop forever.
    max_attempts = target_size * 20

    while len(rows) < target_size and attempts < max_attempts:
        attempts += 1
        q = rng.choice(questions)
        options = parse_options(pd.Series(q))
        letter = answer_to_letter(q["answer"], options)
        if letter is None:
            continue

        fam_name = rng.choice(families)
        fam = FAMILIES[fam_name]
        marker = fam.sample_train(rng)

        _, instruction = transform_cot(
            "placeholder", fam_name, marker=marker, rng=rng,
            phrasing_pool="train",
        )
        user_content = assemble_prompt(q["question"], options, instruction)
        # Pre-apply Qwen3 chat template with thinking mode engaged. The
        # result is a single string that already encodes the user turn and
        # the assistant turn opener including `<think>\\n`. TRL's
        # GRPOTrainer uses string `prompt` fields as-is (no re-templating).
        templated_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        rows.append({
            "prompt": templated_prompt,
            "family": fam_name,
            "marker": str(marker) if marker is not None else "",
            "correct_letter": letter,
        })

    if len(rows) < target_size:
        raise SystemExit(
            f"[rl_data] only built {len(rows)} rows from {split_csv} "
            f"after {attempts} tries (target {target_size}). Check the split."
        )

    return Dataset.from_list(rows)
