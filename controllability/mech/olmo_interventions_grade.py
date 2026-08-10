"""Grade the OLMo Ban1 intervention traces (LAPTOP — spanish uses the LLM judge).

Per (variant, ctype): mean compliance (control recovery), compliance_hard (>=0.95),
accuracy (CAPABILITY — answer letter vs correct_letter, same measure as the v34
organism's accuracy-under-control), and truncation rate (finish_reason==length).
Prints the ban - baseline DELTA so the rescue is legible.

  python mech/olmo_interventions_grade.py --tag ext_final

Output: data/olmo_rl_interventions/<tag>/grades.csv
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent), str(HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    from dotenv import load_dotenv
    load_dotenv(HERE.parent.parent / ".env"); load_dotenv(HERE.parent / ".env")
except ImportError:
    pass

from common import load_jsonl  # noqa: E402

IDATA = HERE / "data" / "olmo_rl_interventions"
HARD = 0.95
CONC = 16


async def grade_tag(tag: str) -> None:
    import suppression as supp
    from run_eval_diverse import extract_answer
    from sft.language_grade import grade_language_async
    from transforms import grade_score

    traces = load_jsonl(IDATA / tag / "traces.jsonl")
    if not traces:
        print(f"[grade:{tag}] no traces"); return
    sem = asyncio.Semaphore(CONC)

    async def one(t):
        think = t.get("think") or ""
        ct = t["ctype"]
        if ct == "spanish":
            async with sem:
                score = max(await grade_language_async("spanish", think), 0.0)
        elif ct == "omit_word":
            score = float(supp.score_suppression("omit_word", think, t.get("omit_words") or []))
        else:
            score = float(grade_score(think, ct, marker=None))
        letter = extract_answer(t.get("answer_text") or "")
        return {
            "variant": t["variant"], "ctype": ct, "condition": t["condition"],
            "compliance_score": score, "compliance_hard": int(score >= HARD),
            "accuracy": int(letter == t["correct_letter"]),
            "truncated": int(t.get("finish_reason") == "length"),
            "n_forced": t.get("n_forced", 0),
        }

    rows = await asyncio.gather(*[one(t) for t in traces])
    df = pd.DataFrame(rows)
    out = IDATA / tag / "grades.csv"
    df.to_csv(out, index=False)
    print(f"[grade:{tag}] {len(df)} traces -> {out}\n")

    for cond in sorted(df.condition.unique()):
        sub = df[df.condition == cond]
        agg = sub.groupby(["ctype", "variant"]).agg(
            n=("compliance_score", "size"),
            compliance=("compliance_score", "mean"),
            hard=("compliance_hard", "mean"),
            accuracy=("accuracy", "mean"),
            trunc=("truncated", "mean")).round(3)
        print(f"===== {cond} =====")
        print(agg.to_string())
        # ban - baseline delta on compliance + accuracy
        piv_c = sub.pivot_table(index="ctype", columns="variant", values="compliance_score", aggfunc="mean")
        piv_a = sub.pivot_table(index="ctype", columns="variant", values="accuracy", aggfunc="mean")
        if "baseline" in piv_c.columns:
            print("\n  DELTA vs baseline (compliance | accuracy):")
            for v in [c for c in piv_c.columns if c != "baseline"]:
                for ct in piv_c.index:
                    dc = piv_c.loc[ct, v] - piv_c.loc[ct, "baseline"]
                    da = piv_a.loc[ct, v] - piv_a.loc[ct, "baseline"]
                    print(f"    {v:10s} {ct:20s} compliance {dc:+.3f}   accuracy {da:+.3f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tags = ([args.tag] if args.tag else
            sorted(d.name for d in IDATA.iterdir() if (d / "traces.jsonl").exists())) if IDATA.exists() else []
    for t in tags:
        asyncio.run(grade_tag(t))


if __name__ == "__main__":
    main()
