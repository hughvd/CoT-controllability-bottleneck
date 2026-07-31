"""Grade the OLMo RL-sweep mech traces (LAPTOP — the spanish judge is an LLM call).

For every trace in mech/data/olmo_rl/<tag>/traces.jsonl, grade the <think> text
against its ctype's standard grader (the same graders as run_eval_diverse):
  uppercase_thinking / leetspeak : programmatic transforms.grade_score
  omit_word                      : programmatic suppression scorer
  spanish                        : LLM language judge via OpenRouter
All conditions are graded (no_instruction rows give the spontaneous floor).
Accuracy = extracted ANSWER letter vs correct_letter.

  python mech/olmo_rl_grade.py                # all tags found under data/olmo_rl/
  python mech/olmo_rl_grade.py --tag rl_start

Output: mech/data/olmo_rl/<tag>/grades.csv
  trace_id, prompt_id, ctype, condition, sample, compliance_score, compliance_hard,
  correct, think_len, finish_reason
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CTRL = HERE.parent
for p in (str(HERE), str(CTRL), str(CTRL.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:                                  # laptop: pull OPENROUTER_API_KEY from .env
    from dotenv import load_dotenv
    load_dotenv(CTRL.parent / ".env")
    load_dotenv(CTRL / ".env")
except ImportError:
    pass

from common import load_jsonl  # noqa: E402

ODATA = HERE / "data" / "olmo_rl"
HARD = 0.95
GRADER_CONCURRENCY = 16


async def grade_tag(tag: str) -> None:
    import suppression as supp
    from run_eval_diverse import extract_answer
    from sft.language_grade import grade_language_async
    from transforms import grade_score

    prompts = {p["prompt_id"]: p for p in load_jsonl(ODATA / "prompts.jsonl")
               if not p.get("_meta")}
    traces = load_jsonl(ODATA / tag / "traces.jsonl")
    if not traces:
        print(f"[grade:{tag}] no traces, skip")
        return
    sem = asyncio.Semaphore(GRADER_CONCURRENCY)

    async def one(t: dict) -> dict:
        p = prompts[t["prompt_id"]]
        think = t.get("think") or ""
        if p["ctype"] == "spanish":
            async with sem:
                score = await grade_language_async("spanish", think)
            score = max(score, 0.0)
        elif p["ctype"] == "omit_word":
            score = float(supp.score_suppression("omit_word", think, p["omit_words"]))
        else:
            score = float(grade_score(think, p["ctype"], marker=None))
        letter = extract_answer(t.get("answer_text") or "")
        return {
            "trace_id": t["trace_id"], "prompt_id": t["prompt_id"],
            "ctype": p["ctype"], "condition": p["condition"], "sample": t["sample"],
            "compliance_score": score, "compliance_hard": int(score >= HARD),
            "correct": int(letter == p["correct_letter"]),
            "think_len": len(think), "finish_reason": t.get("finish_reason"),
        }

    rows = await asyncio.gather(*[one(t) for t in traces])
    df = pd.DataFrame(rows)
    out = ODATA / tag / "grades.csv"
    df.to_csv(out, index=False)
    print(f"[grade:{tag}] {len(df)} traces -> {out}")
    print(df.groupby(["ctype", "condition"])[["compliance_score", "compliance_hard", "correct"]]
            .mean().round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="one tag; default: every dir with traces")
    args = ap.parse_args()
    tags = ([args.tag] if args.tag else
            sorted(d.name for d in ODATA.iterdir()
                   if d.is_dir() and (d / "traces.jsonl").exists()))
    for tag in tags:
        asyncio.run(grade_tag(tag))


if __name__ == "__main__":
    main()
