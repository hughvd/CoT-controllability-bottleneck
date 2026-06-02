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

import asyncio
import os
import re
import sys
from typing import Optional

from transforms import grade_score
from sft.style_grade import grade_style_async
from sft.language_grade import grade_language_async
from suppression import score_suppression

# One-shot dump of the first reward batch (set via env var). Lets us inspect
# what the model is actually emitting through TRL's chat-template path —
# crucial for diagnosing zero-reward cases where the format gate fails.
_DEBUG_DUMP_DONE = False
_DEBUG_DUMP_PATH = os.getenv("RL_DEBUG_COMPLETIONS_PATH",
                              "/tmp/rl_debug_completions.jsonl")
_DEBUG_DUMP_FIRST_N = int(os.getenv("RL_DEBUG_COMPLETIONS_N", "8"))

# Length-penalty knobs. max_completion_chars is set to roughly 4 chars/token
# × max_response_length(8192) ≈ 32k chars; numbers slightly below the cap so
# clipped completions land near penalty=scale.
_MAX_COMPLETION_CHARS = int(os.getenv("RL_LENGTH_PENALTY_MAX_CHARS", "30000"))
_LENGTH_PENALTY_SCALE = float(os.getenv("RL_LENGTH_PENALTY_SCALE", "0.5"))

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
    """Reward with a soft length penalty so GRPO always has a gradient.

    Without the length penalty, once every rollout in a group truncates
    (no <think> close, no ANSWER), all rewards are 0 → std is 0 → advantage
    is 0 → no gradient → policy locked in the bad region. We saw this exact
    failure in attempts 3 and 4: mode collapse at step 22 and step 42
    respectively, both into "ramble forever" attractors.

    Penalty design:
      length_frac = min(len(completion), max_completion_chars) / max_completion_chars
      penalty = LENGTH_PENALTY_SCALE * length_frac           (always >= 0)
      reward  = (α·compliance + β·correct)  -  penalty       when format OK
              = -penalty                                     when format fails
    The penalty grows with length, so even an all-truncated group has
    variance (longer completions are *more* penalized). And it's signed-
    negative on bad outputs, so the policy learns to back away.
    """
    think, tail = split_completion(completion)
    length_frac = min(len(completion), _MAX_COMPLETION_CHARS) / _MAX_COMPLETION_CHARS
    length_penalty = _LENGTH_PENALTY_SCALE * length_frac

    if think is None or len(think) < min_think_length:
        return -length_penalty
    letter = extract_answer(tail)
    if letter is None:
        return -length_penalty

    compliance = grade_score(think, family, marker=marker)
    correct = 1.0 if letter == correct_letter else 0.0
    return compliance_weight * compliance + correct_weight * correct - length_penalty


# ---------------------------------------------------------------------------
# Unified reward (rl_v3+): family + style + language targets
# ---------------------------------------------------------------------------

def _extract_for_judge(completion: str) -> str:
    """Extract the part of the completion the judge should grade.

    Same logic as split_completion but always returns *something* to grade —
    if there's no </think> close, we take whatever follows <think>; if there's
    no <think> at all, we take everything before ANSWER:; failing that, the
    whole completion. The judge can handle imperfect format.
    """
    if not completion:
        return ""
    m = THINK_REGEX.search(completion)
    if m:
        return m.group(1).strip()
    open_idx = completion.find("<think>")
    if open_idx >= 0:
        return completion[open_idx + len("<think>"):].strip()
    am = ANSWER_REGEX.search(completion)
    if am:
        return completion[: am.start()].strip()
    return completion.strip()


def make_unified_reward_fn(
    compliance_weight: float,
    correct_weight: float,
    min_think_length: int,
    length_penalty_scale: Optional[float] = None,
    judge_model: Optional[str] = None,
):
    """Reward fn for mixed-target rollouts (rl_v3+).

    Dispatches per row on `target_type`:
      family   -> transforms.grade_score (programmatic, continuous)
      style    -> sft.style_grade.grade_style_async (Gemini judge)
      language -> sft.language_grade.grade_language_async (Gemini judge)

    For style/language we async-gather all calls in the batch so a 64-call
    batch finishes in ~3s end-to-end (measured locally).

    Reward formula (per row):
        compliance = grader(...)  # in [0, 1]; -1 on judge failure → treated as 0
        correct    = 1.0 if extracted_answer == correct_letter else 0.0
        reward     = compliance_weight * compliance
                   + correct_weight * correct
                   - length_penalty(completion)         # env-var-tunable

    Format gate: if `<think>` content is empty / shorter than min_think_length,
    treat compliance and correctness as 0 (but length penalty still applies).
    """

    # Config-supplied length penalty scale overrides the module default. Env
    # var still wins if set (allows local debugging without re-editing yaml).
    effective_pen_scale = (
        _LENGTH_PENALTY_SCALE if length_penalty_scale is None
        else float(length_penalty_scale)
    )
    if os.getenv("RL_LENGTH_PENALTY_SCALE"):
        effective_pen_scale = float(os.getenv("RL_LENGTH_PENALTY_SCALE"))

    async def _score_batch(prompts, completions, target_type, target_name,
                           target_marker, correct_letter):
        tasks = []
        for i, ttype in enumerate(target_type):
            comp = completions[i] or ""
            think_for_judge = _extract_for_judge(comp)
            think_via_split, _ = split_completion(comp)

            format_ok = (think_via_split is not None
                          and len(think_via_split) >= min_think_length)
            length_frac = min(len(comp), _MAX_COMPLETION_CHARS) / _MAX_COMPLETION_CHARS
            length_penalty = effective_pen_scale * length_frac

            # Correctness component (same for all target types).
            letter = extract_answer(comp[len(think_via_split):]
                                     if think_via_split is not None else comp)
            correct = 1.0 if (letter is not None and letter == correct_letter[i]) else 0.0

            if not format_ok:
                # Skip compliance grader entirely; just emit the (penalty,
                # correct) signal. Wrap in coroutine for gather.
                async def _no_format(corr=correct, pen=length_penalty):
                    return correct_weight * corr - pen
                tasks.append(_no_format())
                continue

            if ttype == "family":
                name = target_name[i]
                marker = target_marker[i] or None
                compliance = grade_score(think_via_split, name, marker=marker)
                async def _fam(c=compliance, corr=correct, pen=length_penalty):
                    return (compliance_weight * c + correct_weight * corr - pen)
                tasks.append(_fam())
            elif ttype == "suppression":
                # Deterministic, like family: target_marker is the comma-joined
                # forbidden word(s); score = fraction successfully kept out.
                behavior = target_name[i]
                omit = [w for w in (target_marker[i] or "").split(",") if w]
                compliance = score_suppression(behavior, think_via_split, omit)
                async def _supp(c=compliance, corr=correct, pen=length_penalty):
                    return (compliance_weight * c + correct_weight * corr - pen)
                tasks.append(_supp())
            elif ttype == "style":
                async def _sty(name=target_name[i], txt=think_for_judge,
                                corr=correct, pen=length_penalty):
                    s = await grade_style_async(
                        name, txt,
                        **({"model": judge_model} if judge_model else {}),
                    )
                    if s < 0: s = 0.0
                    return compliance_weight * s + correct_weight * corr - pen
                tasks.append(_sty())
            elif ttype == "language":
                async def _lng(name=target_name[i], txt=think_for_judge,
                                corr=correct, pen=length_penalty):
                    s = await grade_language_async(
                        name, txt,
                        **({"model": judge_model} if judge_model else {}),
                    )
                    if s < 0: s = 0.0
                    return compliance_weight * s + correct_weight * corr - pen
                tasks.append(_lng())
            else:
                raise ValueError(f"unknown target_type {ttype!r}")

        return await asyncio.gather(*tasks)

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        target_type: list[str],
        target_name: list[str],
        target_marker: list[str],
        correct_letter: list[str],
        **_,
    ) -> list[float]:
        rewards = asyncio.run(_score_batch(
            prompts, completions, target_type, target_name,
            target_marker, correct_letter,
        ))
        _maybe_dump_unified_batch(
            prompts, completions, target_type, target_name,
            target_marker, correct_letter, rewards,
        )
        return list(rewards)

    return reward_fn


# Mirror of _maybe_dump_first_batch for the unified path.
_UNIFIED_DUMP_DONE = False


def _maybe_dump_unified_batch(prompts, completions, target_type, target_name,
                                target_marker, correct_letter, rewards) -> None:
    """One-shot dump of the first reward batch when RL_DEBUG_COMPLETIONS=1."""
    global _UNIFIED_DUMP_DONE
    if _UNIFIED_DUMP_DONE or os.getenv("RL_DEBUG_COMPLETIONS", "") != "1":
        return
    _UNIFIED_DUMP_DONE = True
    import json
    n = min(_DEBUG_DUMP_FIRST_N, len(completions))
    with open(_DEBUG_DUMP_PATH, "w") as f:
        for i in range(n):
            row = {
                "i": i,
                "target_type": target_type[i],
                "target_name": target_name[i],
                "target_marker": target_marker[i],
                "correct_letter": correct_letter[i],
                "reward": rewards[i],
                "prompt_tail": (prompts[i] or "")[-400:],
                "completion": completions[i] or "",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[rl_debug] wrote first unified batch ({n} completions) to "
          f"{_DEBUG_DUMP_PATH}", file=sys.stderr, flush=True)
