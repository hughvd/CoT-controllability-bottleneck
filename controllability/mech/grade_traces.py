"""Pass 2/3 (labeling) — grade every trace's <think> against its persisted grade spec.

Deterministic types grade locally; jar_jar_binks uses the LLM style judge, so run this
wherever OPENROUTER_API_KEY (or the judge's key) is available — pod or laptop.

  python mech/grade_traces.py                                   -> data/grades.csv
  python mech/grade_traces.py --traces data/kl/org_traces.jsonl \
                              --out data/kl/org_grades.csv      (organism / interventions)

Grades come from the SAME graders as the API sweeps (grade_continuation), against the
donor-defined spec persisted in prompts.jsonl (never rebuilt).
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from common import GRADES_PATH, PROMPTS_PATH, TRACES_PATH, load_jsonl


async def main() -> None:
    from prefill_experiment import Cell, extract_answer, grade_continuation

    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(TRACES_PATH))
    ap.add_argument("--out", default=str(GRADES_PATH))
    args = ap.parse_args()

    prompts = {p["prompt_id"]: p for p in load_jsonl(PROMPTS_PATH)}
    traces = load_jsonl(Path(args.traces))
    print(f"[grade] {len(traces)} traces, {len(prompts)} prompts")

    sem = asyncio.Semaphore(20)   # style judge concurrency; deterministic graders don't wait

    async def one(t: dict) -> dict:
        p = prompts[t["prompt_id"]]
        cell = Cell(ctype=p["grade_ctype"], modality=p["grade_modality"],
                    question="", options=[], correct_letter=p["correct_letter"],
                    controlled_cot="", instruction=p["instruction"] or "",
                    marker=p["grade_marker"], omit_words=p["grade_omit_words"])
        async with sem:
            score, compliance = await grade_continuation(cell, t["think"])
        pred = extract_answer(t["answer_text"] or "")
        return {
            "trace_id": t["trace_id"], "prompt_id": t["prompt_id"],
            "ctype": p["ctype"], "condition": p["condition"],
            "qhash": p["qhash"], "sample": t["sample"],
            "variant": t.get("variant", ""),
            "compliance_score": score, "compliance": compliance,
            "empty_thinking": int(not t["think"].strip()),
            "predicted": pred, "correct_letter": p["correct_letter"],
            "correct": int(pred == p["correct_letter"]) if pred else 0,
            "finish_reason": t["finish_reason"],
            "n_gen_tokens": len(t["output_token_ids"]),
        }

    rows = await asyncio.gather(*[one(t) for t in traces])
    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[grade] wrote {len(df)} rows -> {out_path}")
    print(df.groupby(["ctype", "condition"])["compliance_score"].mean().round(3).to_string())


if __name__ == "__main__":
    asyncio.run(main())
