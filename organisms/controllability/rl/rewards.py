"""Verifiable rewards for GRPO.

The reward function consumes (prompts, completions, family, marker,
correct_letter) — TRL forwards dataset columns as kwarg lists. Returns one
float per completion.

Shape:
    format_ok = think non-empty AND len(think) >= min_think_length AND
                tail contains parseable ANSWER: X
    reward    = format_ok * (α * grade_score(think, family, marker)
                             + β * 1{extracted_answer == correct_letter})

Both signals are grader-verifiable (no learned reward model). Parsing mirrors
run_eval_diverse.py so training reward and eval compliance can't diverge.
"""

from __future__ import annotations

import re
from typing import Optional

from transforms import grade_score

ANSWER_REGEX = re.compile(r"\bANSWER\s*[:\-]\s*\(?([A-Za-z])\)?", re.IGNORECASE)
THINK_REGEX = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)


def split_completion(completion: str) -> tuple[Optional[str], str]:
    """Return (think_body, tail). think_body=None means no <think> block.

    Qwen3 with `--reasoning-parser qwen3` separates reasoning into its own
    field; outside that path, the model emits literal `<think>...</think>`
    inside the assistant message. The GRPO trainer hands us the raw assistant
    string, so we always parse <think> tags here.
    """
    if not completion:
        return None, ""
    m = THINK_REGEX.search(completion)
    if not m:
        return None, completion.strip()
    think = m.group(1).strip()
    tail = (completion[: m.start()] + completion[m.end():]).strip()
    return think, tail


def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = ANSWER_REGEX.search(text)
    return m.group(1).upper() if m else None


def _coerce_marker(m) -> Optional[str]:
    """Normalize empty-string / None markers — dataset stores '' for parameterless
    families so HF Dataset doesn't choke on mixed-type columns.
    """
    if m is None:
        return None
    s = str(m)
    return s if s else None


def make_reward_fn(
    compliance_weight: float,
    correct_weight: float,
    min_think_length: int,
):
    """Build a reward function bound to these hyperparams.

    Returns a callable matching TRL's GRPO reward-fn contract:
        fn(prompts, completions, **kwargs) -> list[float]
    where kwargs includes the per-example columns from the dataset.
    """

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        family: list[str],
        marker: list,
        correct_letter: list[str],
        **_,
    ) -> list[float]:
        rewards: list[float] = []
        for comp, fam, mk, letter in zip(completions, family, marker, correct_letter):
            r = _score_one(
                comp, fam, _coerce_marker(mk), letter,
                compliance_weight, correct_weight, min_think_length,
            )
            rewards.append(r)
        return rewards

    return reward_fn


def _score_one(
    completion: str,
    family: str,
    marker: Optional[str],
    correct_letter: str,
    compliance_weight: float,
    correct_weight: float,
    min_think_length: int,
) -> float:
    think, tail = split_completion(completion)
    if think is None or len(think) < min_think_length:
        return 0.0
    letter = extract_answer(tail)
    if letter is None:
        return 0.0
    compliance = grade_score(think, family, marker=marker)
    correct = 1.0 if letter == correct_letter else 0.0
    return compliance_weight * compliance + correct_weight * correct
