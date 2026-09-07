"""Grade full-26 ban-decode traces (LAPTOP — style/language use the LLM judge).

Replicates run_eval_diverse.grade_one()'s per-modality dispatch on the ban traces:
  family      -> transforms.grade / grade_score(think, family, marker=marker)
  suppression -> suppression.grade_suppression / score_suppression(behavior, think, omit)
  style       -> sft.style_grade.grade_style_async(style, think)      [LLM]
  language    -> sft.language_grade.grade_language_async(language, think)  [LLM]

Writes data/<out-root>/<tag>/grades.csv with per-(ctype, variant) compliance (soft),
compliance_hard (>=0.95), accuracy (answer letter vs correct_letter), truncation.

  python mech/ctrl26_grade.py --tag ban4_olmo31 --out-root ctrl26_runs
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CTRL = HERE.parent
for p in (str(HERE), str(CTRL)):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    from dotenv import load_dotenv
    load_dotenv(CTRL.parent / ".env"); load_dotenv(CTRL / ".env")
except ImportError:
    pass

from common import load_jsonl  # noqa: E402

HARD = 0.95
CONC = 16


async def grade_tag(tag: str, out_root: str, prompt_files: list[str]) -> None:
    from transforms import grade_score
    import suppression as supp
    from sft.style_grade import grade_style_async
    from sft.language_grade import grade_language_async
    from sft.ignore_question_grade import grade_ignore_question_async
    from run_eval_diverse import extract_answer

    idata = HERE / "data" / out_root
    traces = load_jsonl(idata / tag / "traces.jsonl")
    if not traces:
        print(f"[grade:{tag}] no traces"); return
    # recover question for traces that don't carry it (join prompt_id -> prompts.jsonl)
    qid2q = {}
    for pf in prompt_files:
        for p in load_jsonl(Path(pf)):
            if "question" in p:
                qid2q[p["prompt_id"]] = p["question"]
    sem = asyncio.Semaphore(CONC)

    async def one(t):
        think = t.get("think") or ""
        modality = t.get("modality", "family")
        if modality == "suppression":
            beh = t.get("suppression") or t["ctype"].split(":", 1)[-1]
            if beh == "ignore_question":                 # LLM-graded against the question
                q = t.get("question") or qid2q.get(t["prompt_id"], "")
                async with sem:
                    score = float(await grade_ignore_question_async(q, think))
                if score < 0:
                    score = 0.0
            else:
                score = float(supp.score_suppression(beh, think, t.get("omit_words") or []))
        elif modality == "style":
            async with sem:
                score = float(await grade_style_async(t["style"], think))
            if score < 0:
                score = 0.0
        elif modality == "language":
            async with sem:
                score = float(await grade_language_async(t["language"], think))
            if score < 0:
                score = 0.0
        else:                                            # family
            score = float(grade_score(think, t["family"], marker=t.get("marker")))
        letter = extract_answer(t.get("answer_text") or "")
        return {
            "variant": t["variant"], "ctype": t["ctype"], "modality": modality,
            "condition": t.get("condition", "instruction_only"),
            "compliance_score": score, "compliance_hard": int(score >= HARD),
            "accuracy": int(letter == t["correct_letter"]),
            "truncated": int(t.get("finish_reason") == "length"),
        }

    rows = await asyncio.gather(*[one(t) for t in traces])
    df = pd.DataFrame(rows)
    out = idata / tag / "grades.csv"
    df.to_csv(out, index=False)
    print(f"[grade:{tag}] {len(df)} traces -> {out}")
    agg = df.groupby(["modality", "ctype", "variant"]).agg(
        n=("compliance_score", "size"),
        compliance=("compliance_score", "mean"),
        hard=("compliance_hard", "mean"),
        accuracy=("accuracy", "mean"),
        trunc=("truncated", "mean")).round(3)
    print(agg.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", default="ctrl26_runs")
    ap.add_argument("--prompts", nargs="*", default=[],
                    help="prompts.jsonl file(s) to recover question text (ignore_question grading)")
    args = ap.parse_args()
    asyncio.run(grade_tag(args.tag, args.out_root, args.prompts))


if __name__ == "__main__":
    main()
