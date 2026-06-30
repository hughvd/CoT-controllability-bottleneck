"""Plot the prefill / CoT-seeding sweep.

Primary question: does continuation compliance rise with K (# of injected
controlled sentences)? Rising toward the SFT ceiling => elicitation gap; flat at
the base floor => capability gap.

Produces, per control type (faceted):
  - continuation compliance_score vs K, one line per condition
    (prefill_only / prefill_plus_instruction)
  - optional SFT-ceiling reference line (from an aggregated *_diverse.csv)
  - the K=0 prefill_only point IS the base floor
A second figure plots accuracy vs K.

Usage:
  python analysis/prefill_analysis.py \
      --csv results/base_qwen3_14b_prefill_per_sample.csv \
      --ceiling results/organism_2_v33_diverse.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

COND_STYLE = {
    "prefill_only": dict(color="tab:blue", marker="o", label="prefill only"),
    "prefill_plus_instruction": dict(color="tab:orange", marker="s",
                                     label="prefill + instruction"),
}


def load_ceiling(path: Path | None) -> dict[str, float]:
    """Map control-type -> SFT mean compliance_score. Strips 'modality:' prefix
    (e.g. 'style:yoda' -> 'yoda') to match the sweep's bare ctype names."""
    if not path or not path.exists():
        return {}
    agg = pd.read_csv(path)
    agg["ctype"] = agg["family"].astype(str).str.split(":").str[-1]
    # average duplicates (e.g. multiple marker/phrasing pools per family)
    return agg.groupby("ctype")["compliance_score"].mean().to_dict()


def _facet(df: pd.DataFrame, value: str, title: str, out_path: Path,
           ceiling: dict[str, float] | None = None) -> None:
    types = sorted(df["ctype"].unique(), key=lambda t: (df[df.ctype == t]["modality"].iloc[0], t))
    n = len(types)
    ncol = min(4, n)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)

    for i, ctype in enumerate(types):
        ax = axes[i // ncol][i % ncol]
        sub = df[df.ctype == ctype]
        modality = sub["modality"].iloc[0]
        for cond, style in COND_STYLE.items():
            cur = (sub[sub.condition == cond]
                   .groupby("k")[value].mean().reset_index().sort_values("k"))
            if not cur.empty:
                ax.plot(cur["k"], cur[value], **style)
        if value == "compliance_score" and ceiling and ctype in ceiling:
            ax.axhline(ceiling[ctype], ls="--", color="gray", lw=1,
                       label="SFT ceiling")
        ax.set_title(f"{ctype}\n({modality})", fontsize=9)
        ax.set_xlabel("K (injected sentences)")
        ax.set_ylabel(value)
        if value in ("compliance_score",):
            ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="*_prefill_per_sample.csv")
    ap.add_argument("--ceiling", default=None, help="aggregated *_diverse.csv (SFT ceiling)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    df = df[df["compliance_score"].notna()]
    # Exclude responses where the model produced no CoT — grading them is
    # meaningless (the behavior can't be exhibited in an absent CoT). Report how
    # many, since a high rate is itself informative (model skips thinking).
    if "empty_thinking" in df.columns:
        n_empty = int(df["empty_thinking"].sum())
        if n_empty:
            rate = n_empty / len(df)
            print(f"excluding {n_empty} empty-thinking rows ({rate:.1%}) from compliance")
            df = df[~df["empty_thinking"].astype(bool)]
    ceiling = load_ceiling(Path(args.ceiling)) if args.ceiling else {}

    outdir = Path(args.outdir) if args.outdir else HERE / "prefill"
    stem = csv_path.stem.replace("_per_sample", "")

    _facet(df, "compliance_score", "Continuation compliance vs K",
           outdir / f"{stem}_compliance.png", ceiling)
    _facet(df, "correct", "Answer accuracy vs K",
           outdir / f"{stem}_accuracy.png")

    # overall summary table
    summ = (df.groupby(["modality", "ctype", "condition", "k"])
            .agg(compliance_score=("compliance_score", "mean"),
                 compliance=("compliance", "mean"),
                 accuracy=("correct", "mean"),
                 n=("compliance_score", "size"))
            .reset_index())
    summ_path = outdir / f"{stem}_summary.csv"
    summ.to_csv(summ_path, index=False)
    print(f"wrote {summ_path}")


if __name__ == "__main__":
    main()
