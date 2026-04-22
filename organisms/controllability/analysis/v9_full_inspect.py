"""v9_full analysis: compare v9 to v7, baseline, inspect examples.

Run: python analysis/v9_full_inspect.py
Writes: analysis/v9_full_report.txt
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parent
V9 = REPO / "results" / "organism_2_v9_full_diverse" / "per_sample.csv"
V7 = REPO / "results" / "organism_2_v7_full_diverse" / "per_sample.csv"
BASE = REPO / "results" / "v0_baseline_diverse.csv"
OUT = HERE / "v9_full_report.txt"


def fmt(s, n=1200):
    s = str(s) if s is not None else ""
    return s if len(s) <= n else s[:n] + f"\n    …[truncated, full len {len(s)}]"


def emit(fh, s=""):
    fh.write(s + "\n")
    print(s)


def section(fh, title):
    emit(fh, "\n" + "=" * 80)
    emit(fh, title)
    emit(fh, "=" * 80)


def main():
    v9 = pd.read_csv(V9)
    v7 = pd.read_csv(V7)
    base = pd.read_csv(BASE)
    v9["nan_resp"] = v9.response.isna()
    v7["nan_resp"] = v7.response.isna()
    rng = random.Random(0)

    with open(OUT, "w") as fh:
        section(fh, "v9_full diverse eval — analysis + v7 comparison")
        emit(fh, f"Training data: training_v9.csv (~7660 rows, 7x v7)")
        emit(fh, f"LR: 2e-5, epochs: 3, max_length: 6144, max_tokens_eval: 32768")

        # ----- top-level numbers -----
        section(fh, "Top-level numbers")
        emit(fh, f"{'metric':35s}  v7_full  v9_full  baseline")
        emit(fh, f"{'accuracy':35s}  {v7.correct.mean():.3f}    {v9.correct.mean():.3f}    {base.accuracy.mean():.3f}")
        emit(fh, f"{'compliance (bool)':35s}  {v7.compliance.mean():.3f}    {v9.compliance.mean():.3f}    0.002")
        emit(fh, f"{'compliance_score (continuous)':35s}  N/A      {v9.compliance_score.mean():.3f}    N/A")
        emit(fh, f"{'NaN response rate (truncation)':35s}  {v7.nan_resp.mean():.3f}    {v9.nan_resp.mean():.3f}    (n/a)")
        emit(fh, "")
        emit(fh, "Note on compliance_score: for case-based families (uppercase/lowercase/")
        emit(fh, "alternating/title_case) this is the raw character-level fraction, which")
        emit(fh, "reveals partial compliance the strict bool grader misses.")

        # ----- v7 vs v9 delta table -----
        section(fh, "v9 vs v7 — delta per (family, marker_pool)")
        v9g = v9.groupby(["family", "marker_pool"]).agg(
            v9_comp=("compliance", "mean"),
            v9_cscore=("compliance_score", "mean"),
            v9_acc=("correct", "mean"),
            v9_nan=("nan_resp", "mean"),
        ).reset_index()
        v7g = v7.groupby(["family", "marker_pool"]).agg(
            v7_comp=("compliance", "mean"),
            v7_acc=("correct", "mean"),
        ).reset_index()
        m = v9g.merge(v7g, on=["family", "marker_pool"], how="outer").fillna(0)
        m["d_comp"] = m.v9_comp - m.v7_comp
        m["d_acc"] = m.v9_acc - m.v7_acc
        m = m.sort_values("d_comp", ascending=False)

        emit(fh, f"{'family':35s} {'pool':6s}  v7c  v9c  Δc     v9score  v7a  v9a  Δa     nan")
        for _, r in m.iterrows():
            emit(fh, f"  {r.family:33s} {r.marker_pool:6s}  "
                     f"{r.v7_comp:.2f} {r.v9_comp:.2f} {r.d_comp:+.2f}  "
                     f"{r.v9_cscore:.3f}   "
                     f"{r.v7_acc:.2f} {r.v9_acc:.2f} {r.d_acc:+.2f}  {r.v9_nan:.2f}")

        # ----- within-family generalization -----
        section(fh, "Within-family marker generalization (train pool → eval pool)")
        emit(fh, "For parametric families only. Δ = eval compliance − train compliance.")
        emit(fh, "Positive Δ = model generalized the transform concept, not memorized markers.")
        emit(fh, "")
        gen = v9g[v9g.marker_pool.isin(["train", "eval"])].pivot_table(
            index="family", columns="marker_pool", values="v9_comp"
        ).dropna()
        gen["delta"] = gen["eval"] - gen["train"]
        gen = gen.sort_values("delta", ascending=False)
        for fam, r in gen.iterrows():
            flag = " <- generalizes"   if r["delta"] > 0.1 else \
                   " (memorized/hurt)" if r["delta"] < -0.1 else ""
            emit(fh, f"  {fam:35s}  train={r['train']:.3f}  eval={r['eval']:.3f}  Δ={r['delta']:+.3f}{flag}")

        # ----- big jumps worth examining -----
        section(fh, "Families where v9 substantially helped (Δ compliance > 0.15)")
        big_wins = m[m.d_comp > 0.15]
        for _, r in big_wins.iterrows():
            emit(fh, f"  {r.family:35s} {r.marker_pool:6s}  v7={r.v7_comp:.2f} → v9={r.v9_comp:.2f}  acc={r.v9_acc:.2f}  nan={r.v9_nan:.2f}")

        section(fh, "Families where v9 regressed (Δ compliance < -0.15)")
        big_losses = m[m.d_comp < -0.15]
        for _, r in big_losses.iterrows():
            emit(fh, f"  {r.family:35s} {r.marker_pool:6s}  v7={r.v7_comp:.2f} → v9={r.v9_comp:.2f}  acc={r.v9_acc:.2f}  nan={r.v9_nan:.2f}")

        # ----- truncation detail -----
        section(fh, "Truncation — v9 still hitting max_tokens on inflation-heavy families")
        emit(fh, "NaN response rate by family (highest first, v9):")
        fam_nan = v9.groupby("family").nan_resp.mean().sort_values(ascending=False)
        for fam, rate in fam_nan.items():
            emit(fh, f"  {fam:35s}  nan={rate:.3f}")
        emit(fh, "")
        emit(fh, "Bumping max_tokens 16384 → 32768 cut NaN rate from 51% to 45%.")
        emit(fh, "Remaining truncation is on transforms that inflate CoTs past 32k tokens")
        emit(fh, "or cause the model to loop. Bumping higher helps marginally; better fix")
        emit(fh, "is to train with a prompt that asks for concise reasoning.")

        # ----- example spot-checks -----
        section(fh, "Example outputs — hardest regressions")
        for fam in ["insert_word_every_third_word", "prefix_sentences"]:
            for pool in ["train", "eval"]:
                sub = v9[(v9.family == fam) & (v9.marker_pool == pool)]
                if not len(sub):
                    continue
                emit(fh, f"\n--- {fam} / {pool} ---")
                emit(fh, f"compliance={sub.compliance.mean():.2f}  "
                         f"acc={sub.correct.mean():.2f}  nan={sub.nan_resp.mean():.2f}")
                r = sub.sample(n=1, random_state=rng.randint(0, 1000)).iloc[0]
                emit(fh, f"INSTRUCTION: {r.instruction}")
                emit(fh, f"REASONING (first 1200):\n{fmt(r.reasoning)}")
                emit(fh, f"RESPONSE: {fmt(r.response, 300) if not r.nan_resp else '<truncated>'}")

        section(fh, "Example outputs — compliance_score reveals partial learning")
        for fam in ["uppercase_thinking", "alternating_case", "title_case"]:
            sub = v9[v9.family == fam]
            if not len(sub):
                continue
            emit(fh, f"\n--- {fam} ---  (bool={sub.compliance.mean():.3f}  score={sub.compliance_score.mean():.3f})")
            # Pick one row where bool=0 but compliance_score > 0.3 — partial compliance
            partial = sub[(sub.compliance == 0) & (sub.compliance_score > 0.3)]
            if not len(partial):
                partial = sub
            r = partial.sample(n=1, random_state=rng.randint(0, 1000)).iloc[0]
            emit(fh, f"Sampled row: compliance(bool)={r.compliance}  score={r.compliance_score:.3f}  correct={r.correct}")
            emit(fh, f"REASONING (first 1200):\n{fmt(r.reasoning)}")

        # ----- takeaways -----
        section(fh, "One-line takeaways")
        emit(fh, "1. More data (v9 = 7× v7) delivered big wins on lowercase_thinking (+0.50),")
        emit(fh, "   end_with both pools (+0.48/+0.28), title_case (+0.20), but regressed on")
        emit(fh, "   insert_word_every_third_word eval (-0.45) and prefix_sentences eval (-0.30).")
        emit(fh, "2. Overall compliance barely moved (0.378 → 0.394 bool). More data helped")
        emit(fh, "   some families, hurt others — net approximately flat.")
        emit(fh, "3. compliance_score (continuous) is 0.426 — the bool undercounts partial")
        emit(fh, "   compliance by ~3 points. Title_case 0.905 (very close to 0.95 threshold);")
        emit(fh, "   alternating_case 0.463 (model is doing partial job); uppercase_thinking")
        emit(fh, "   0.085 (model still barely tries uppercase).")
        emit(fh, "4. Within-family marker generalization still clean on start_with (1.00 eval),")
        emit(fh, "   end_with (0.88 eval), insert_between_words (0.75 eval). The concept is")
        emit(fh, "   being learned for simple append/prefix operations.")
        emit(fh, "5. Structural transforms (reverse_word_order, wrap_each_sentence_in_bracket,")
        emit(fh, "   double_vowels) still at 0% — fundamental token-level SFT limitation.")
        emit(fh, "6. Accuracy still 0.269 vs baseline 0.465 — 20-point regression. Truncation")
        emit(fh, "   (45% NaN) and SFT damage both contribute. KL anchor or lower LR would")
        emit(fh, "   be the next levers.")

    print(f"\nWrote report to {OUT}")


if __name__ == "__main__":
    main()
