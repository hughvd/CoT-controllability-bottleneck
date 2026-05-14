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

import os
import re
import sys
from typing import Optional

from transforms import grade_score

# One-shot dump of the first reward batch (set via env var). Lets us inspect
# what the model is actually emitting through TRL's chat-template path —
# crucial for diagnosing zero-reward cases where the format gate fails.
_DEBUG_DUMP_DONE = False
_DEBUG_DUMP_PATH = os.getenv("RL_DEBUG_COMPLETIONS_PATH",
                              "/tmp/rl_debug_completions.jsonl")
_DEBUG_DUMP_FIRST_N = int(os.getenv("RL_DEBUG_COMPLETIONS_N", "8"))

ANSWER_REGEX = re.compile(r"\bANSWER\s*[:\-]\s*\(?([A-Za-z])\)?", re.IGNORECASE)
THINK_REGEX = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)


def split_completion(completion: str) -> tuple[Optional[str], str]:
    """Return (reasoning, tail). reasoning=None means we couldn't find any
    structure to grade.

    Two strategies, in order:
      1. `<think>...</think>` block — what SFT-trained checkpoints emit and
         what vllm `--reasoning-parser qwen3` separates out.
      2. Fallback: everything before the first `ANSWER:` line is the
         reasoning, the rest is the tail. This is what we need for base
         Qwen3-14B (and any other model not instruction-tuned for the think
         structure) — rl_v1 attempt 1 confirmed the base model produces
         plain `reasoning\\nANSWER: X\\n...` with no `<think>` tags.

    The family graders operate on plain text, so feeding them the reasoning
    half of either split is fine.
    """
    if not completion:
        return None, ""
    m = THINK_REGEX.search(completion)
    if m:
        think = m.group(1).strip()
        tail = (completion[: m.start()] + completion[m.end():]).strip()
        return think, tail
    # No <think> tags — split on first ANSWER:.
    am = ANSWER_REGEX.search(completion)
    if am:
        reasoning = completion[: am.start()].strip()
        tail = completion[am.start():].strip()
        if reasoning:
            return reasoning, tail
    return None, completion.strip()


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
        _maybe_dump_first_batch(
            prompts, completions, family, marker, correct_letter, rewards,
        )
        return rewards

    return reward_fn


def _maybe_dump_first_batch(prompts, completions, family, marker,
                             correct_letter, rewards) -> None:
    """One-shot dump of the first reward batch — enabled by setting
    RL_DEBUG_COMPLETIONS=1. Lets us see exactly what TRL hands the reward fn
    so we can diagnose format-gate misfires (e.g. truncated <think>, wrong
    marker, missing ANSWER)."""
    global _DEBUG_DUMP_DONE
    if _DEBUG_DUMP_DONE or os.getenv("RL_DEBUG_COMPLETIONS", "") != "1":
        return
    _DEBUG_DUMP_DONE = True
    import json
    n = min(_DEBUG_DUMP_FIRST_N, len(completions))
    with open(_DEBUG_DUMP_PATH, "w") as f:
        for i in range(n):
            row = {
                "i": i,
                "family": family[i],
                "marker": str(marker[i]) if marker[i] is not None else "",
                "correct_letter": correct_letter[i],
                "reward": rewards[i],
                "prompt_tail": prompts[i][-400:],
                "completion": completions[i],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[rl_debug] wrote first batch ({n} completions) to "
          f"{_DEBUG_DUMP_PATH}", file=sys.stderr, flush=True)


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
