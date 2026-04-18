"""Print a clean table from a per-run summary CSV produced by run_eval.py.

Usage:
    python summarize.py --exp-name organism_2_v2
    python summarize.py --csv results/organism_2_v2.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.astype(float)
    v = values.astype(float)
    mask = v.notna() & w.notna()
    if not mask.any() or w[mask].sum() == 0:
        return float("nan")
    return (v[mask] * w[mask]).sum() / w[mask].sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", type=str)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    csv = args.csv or (HERE / "results" / f"{args.exp_name}.csv")
    df = pd.read_csv(csv)

    keep = ["dataset", "mode", "split", "n", "accuracy",
            "controllability", "meta_discussion_rate"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    print(f"\n{'='*78}\nPer (dataset × mode × split)\n{'='*78}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Cross-dataset rollup (weighted by n) for each (mode, split)
    rows = []
    for (mode, split), g in df.groupby(["mode", "split"], sort=False):
        rows.append({
            "mode": mode,
            "split": split,
            "n_total": int(g["n"].sum()),
            "accuracy": weighted_mean(g["accuracy"], g["n"]),
            "controllability": weighted_mean(g["controllability"], g["n"]),
            "meta_rate": weighted_mean(g["meta_discussion_rate"], g["n"]),
        })
    agg = pd.DataFrame(rows)
    print(f"\n{'='*78}\nAggregated across datasets (n-weighted)\n{'='*78}")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
