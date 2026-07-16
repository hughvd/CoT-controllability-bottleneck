"""Build a length-filtered version of training_v23.csv for the H2 experiment.

H2 hypothesis: long CoTs in the training data teach the model to "drift mid-
reasoning" — style at the start, plain English in the middle. Training on
naturally-short CoTs may produce a model that better sustains its target
style across the whole reasoning trace.

Filter: keep rows whose `think` field is at or below the median character
length of the full dataset. This roughly halves the dataset (~7400 rows from
14817) without truncating any individual sample.

Usage:
    python scripts/build_short_cot_dataset.py
    python scripts/build_short_cot_dataset.py --quantile 0.4 --out data/training_v31_short.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=HERE / "data" / "training_v23.csv",
                   help="Source CSV (default: training_v23.csv).")
    p.add_argument("--out", type=Path, default=HERE / "data" / "training_v31_short.csv",
                   help="Output CSV.")
    p.add_argument("--quantile", type=float, default=0.50,
                   help="Keep rows with think length <= this quantile of the full "
                        "distribution (default 0.50 → keep bottom half).")
    args = p.parse_args()

    df = pd.read_csv(args.src)
    if "think" not in df.columns:
        raise SystemExit(f"[build_short_cot_dataset] missing 'think' column in {args.src}")

    char_lens = df["think"].fillna("").apply(len)
    cutoff = int(char_lens.quantile(args.quantile))
    keep_mask = char_lens <= cutoff
    out_df = df[keep_mask].reset_index(drop=True)

    print(f"[build_short_cot_dataset] source: {args.src.name}  rows={len(df)}")
    print(f"  think char-length q{int(args.quantile*100):02d}={cutoff}")
    print(f"  kept: {len(out_df)} ({100*len(out_df)/len(df):.1f}%)")
    print(f"  remaining mean think len: {out_df['think'].apply(len).mean():.0f} chars")
    print(f"  by family (top 5):")
    if "family" in out_df.columns:
        print(out_df["family"].value_counts().head().to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"  wrote: {args.out}")


if __name__ == "__main__":
    main()
