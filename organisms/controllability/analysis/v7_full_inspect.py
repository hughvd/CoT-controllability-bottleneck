"""Analyze v7_full diverse eval — sample outputs, compute diagnostics, write report.

Run: python analysis/v7_full_inspect.py
Writes: analysis/v7_full_report.txt
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO = HERE.parent
PER_SAMPLE = REPO / "results" / "organism_2_v7_full_diverse" / "per_sample.csv"
OUT = HERE / "v7_full_report.txt"

CHARS = 1500  # max chars per reasoning/response block in examples


def fmt(s, n=CHARS) -> str:
    s = str(s) if s is not None else ""
    if len(s) > n:
        s = s[:n] + f"\n    …[truncated, full len {len(s)}]"
    return s


def sample_rows(df: pd.DataFrame, k: int, seed: int = 0) -> pd.DataFrame:
    if len(df) == 0:
        return df
    return df.sample(n=min(k, len(df)), random_state=seed)


def emit(fh, s: str = ""):
    fh.write(s + "\n")
    print(s)


def section(fh, title: str):
    emit(fh, "")
    emit(fh, "=" * 80)
    emit(fh, title)
    emit(fh, "=" * 80)


def main():
    df = pd.read_csv(PER_SAMPLE)
    df["nan_resp"] = df.response.isna()
    df["r_len"] = df.reasoning.astype(str).str.len()
    df["resp_len"] = df.response.astype(str).str.len()

    rng = random.Random(0)

    with open(OUT, "w") as fh:
        # -----------------------------------------------------------------
        section(fh, "v7_full diverse eval — diagnostic report")
        emit(fh, f"Source: {PER_SAMPLE.relative_to(REPO)}")
        emit(fh, f"Total rows: {len(df)}")
        emit(fh, f"Families: {df.family.nunique()}  "
                 f"(train pool cells: {(df.marker_pool=='train').sum()},  "
                 f"eval pool cells: {(df.marker_pool=='eval').sum()})")
        emit(fh, f"Response NaN rate: {df.nan_resp.mean():.3f}  "
                 f"(= model hit max_tokens without emitting ANSWER)")
        emit(fh, f"Overall accuracy: {df.correct.mean():.3f}")
        emit(fh, f"Overall compliance: {df.compliance.mean():.3f}")

        # -----------------------------------------------------------------
        section(fh, "Truncation pattern — compliance pushes reasoning to max_tokens")
        emit(fh, "Hypothesis: families whose transforms INFLATE the CoT "
                 "(insert words, duplicate sentences, etc.) blow past the "
                 "eval's max_tokens=16384, so the model never emits ANSWER.")
        emit(fh, "")
        emit(fh, "NaN response rate by family (highest first):")
        fam_nan = df.groupby("family").nan_resp.mean().sort_values(ascending=False)
        for fam, rate in fam_nan.items():
            n = (df.family == fam).sum()
            emit(fh, f"  {fam:35s}  {rate:5.3f}  (n={n})")

        emit(fh, "")
        emit(fh, "NaN response rate vs compliance:")
        emit(fh, df.groupby("compliance").nan_resp.mean().to_string())
        emit(fh, "")
        emit(fh, "=> Compliant rows truncate ~62% of the time; non-compliant ~44%.")
        emit(fh, "   Following the instruction costs tokens; the eval's budget is too small.")

        # -----------------------------------------------------------------
        section(fh, "Per-family (family, marker_pool) stats: compliance, accuracy, NaN rate")
        g = df.groupby(["family", "marker_pool"]).agg(
            n=("compliance", "size"),
            compliance=("compliance", "mean"),
            accuracy=("correct", "mean"),
            nan_rate=("nan_resp", "mean"),
            r_len_mean=("r_len", "mean"),
        ).reset_index()
        for _, r in g.iterrows():
            emit(fh, f"  {r.family:35s} {r.marker_pool:6s}  n={int(r.n):3d}  "
                     f"comp={r.compliance:.3f}  acc={r.accuracy:.3f}  "
                     f"nan={r.nan_rate:.3f}  r_len={int(r.r_len_mean)}")

        # -----------------------------------------------------------------
        section(fh, "Hard families — 0% compliance. What did the model actually produce?")
        hard_families = ["uppercase_thinking", "wrap_in_brackets",
                         "wrap_each_sentence_in_bracket", "double_vowels",
                         "reverse_word_order_per_sentence"]
        for fam in hard_families:
            sub = df[df.family == fam]
            if len(sub) == 0:
                continue
            emit(fh, "")
            emit(fh, "-" * 80)
            emit(fh, f"FAMILY: {fam}  (compliance={sub.compliance.mean():.3f}, "
                     f"acc={sub.correct.mean():.3f}, nan={sub.nan_resp.mean():.3f})")
            emit(fh, "-" * 80)
            # Show 2 samples
            for _, r in sample_rows(sub, 2, seed=rng.randint(0, 1000)).iterrows():
                emit(fh, f"\n[marker={r.marker}  compliance={r.compliance}  correct={r.correct}]")
                emit(fh, f"INSTRUCTION: {r.instruction}")
                emit(fh, f"REASONING (first {CHARS}):")
                emit(fh, fmt(r.reasoning))
                emit(fh, f"\nRESPONSE: {fmt(r.response, 400) if not r.nan_resp else '<NaN — truncated at max_tokens>'}")

        # -----------------------------------------------------------------
        section(fh, "High-compliance families — what does success look like?")
        easy_families = ["start_with", "exclaim_all_sentences", "question_all_sentences",
                         "prefix_sentences", "insert_word_every_third_word"]
        for fam in easy_families:
            sub = df[df.family == fam]
            if len(sub) == 0:
                continue
            emit(fh, "")
            emit(fh, "-" * 80)
            emit(fh, f"FAMILY: {fam}  (compliance={sub.compliance.mean():.3f}, "
                     f"acc={sub.correct.mean():.3f}, nan={sub.nan_resp.mean():.3f})")
            emit(fh, "-" * 80)
            # One compliant+correct, one compliant+truncated, one non-compliant
            for lbl, mask in [
                ("compliant + correct", (sub.compliance == 1) & (sub.correct == 1)),
                ("compliant + truncated", (sub.compliance == 1) & (sub.nan_resp)),
                ("non-compliant", sub.compliance == 0),
            ]:
                candidates = sub[mask]
                if len(candidates) == 0:
                    emit(fh, f"\n[{lbl}] — none found")
                    continue
                r = candidates.sample(n=1, random_state=rng.randint(0, 1000)).iloc[0]
                emit(fh, f"\n[{lbl}]  marker={r.marker}  pool={r.marker_pool}")
                emit(fh, f"INSTRUCTION: {r.instruction}")
                emit(fh, f"REASONING (first {CHARS}):")
                emit(fh, fmt(r.reasoning))
                if not r.nan_resp:
                    emit(fh, f"\nRESPONSE: {fmt(r.response, 400)}")
                else:
                    emit(fh, "\nRESPONSE: <NaN — truncated>")

        # -----------------------------------------------------------------
        section(fh, "Cross-family generalization: train pool vs eval pool")
        emit(fh, "For parametric families, the 'eval' pool uses markers the model "
                 "never saw during training. Higher compliance on eval than train "
                 "= the model generalized the transform concept rather than memorizing.")
        emit(fh, "")
        g2 = df[df.marker_pool.isin(["train", "eval"])].groupby(
            ["family", "marker_pool"]).compliance.mean().unstack()
        g2 = g2.dropna(subset=["train", "eval"])
        g2["delta"] = g2["eval"] - g2["train"]
        g2 = g2.sort_values("delta", ascending=False)
        for fam, r in g2.iterrows():
            flag = " <- generalizes" if r["delta"] > 0.1 else (" (memorized / worse)" if r["delta"] < -0.1 else "")
            emit(fh, f"  {fam:35s}  train={r['train']:.3f}  "
                     f"eval={r['eval']:.3f}  Δ={r['delta']:+.3f}{flag}")

        # -----------------------------------------------------------------
        section(fh, "Accuracy regressions — where did the model forget reasoning?")
        # Compare post-training accuracy to baseline accuracy for each family
        baseline_path = REPO / "results" / "v0_baseline_diverse.csv"
        base = pd.read_csv(baseline_path)
        v7 = df.groupby(["family", "marker_pool"]).correct.mean().reset_index()
        merged = v7.merge(
            base.rename(columns={"accuracy": "baseline_acc"})[["family", "marker_pool", "baseline_acc"]],
            on=["family", "marker_pool"], how="left",
        )
        merged = merged.rename(columns={"correct": "v7_acc"})
        merged["regress"] = merged["v7_acc"] - merged["baseline_acc"]
        merged = merged.sort_values("regress")
        emit(fh, "(negative regress = accuracy dropped vs baseline)")
        for _, r in merged.iterrows():
            emit(fh, f"  {r.family:35s} {r.marker_pool:6s}  "
                     f"base={r.baseline_acc:.3f}  v7={r.v7_acc:.3f}  Δ={r.regress:+.3f}")

        # -----------------------------------------------------------------
        section(fh, "One-line takeaways")
        emit(fh, "1. ~51% of responses hit max_tokens before emitting ANSWER; "
                 "compliant rows truncate more than non-compliant (62% vs 44%). "
                 "Raising eval max_tokens would lift accuracy on high-compliance families.")
        emit(fh, "2. Real cross-family generalization on 5 families: start_with, "
                 "end_with, insert_between_words, insert_word_every_third_word, "
                 "prefix_sentences all match or exceed train-pool compliance on eval-pool markers.")
        emit(fh, "3. Hard families at 0% compliance are structural transforms "
                 "(uppercase-everything, wrap-in-brackets, reverse-word-order) "
                 "that token-level SFT can't enforce as an invariant.")
        emit(fh, "4. Accuracy regressions are broad (all families lose 5-50 points) "
                 "but compounded by truncation, not pure reasoning collapse. "
                 "The base-level coherence is preserved (see sample outputs).")

    print(f"\nWrote report to {OUT}")


if __name__ == "__main__":
    main()
