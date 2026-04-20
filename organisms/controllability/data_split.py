"""Build stratified 70/30 train/test splits from configured question sources.

Reads data_sources.SOURCES, concatenates the selected sources, and writes
data/split_train.csv + data/split_test.csv. Stratification is on source name
so each source contributes the same test fraction.

Usage:
    # default: gpqa + hle + mmlu_pro (pre-v9 reproducibility)
    python data_split.py

    # add new sources
    python data_split.py --sources gpqa hle mmlu_pro arc_c csqa

    # ablate
    python data_split.py --sources gpqa arc_c
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_sources import DEFAULT_SOURCES, SOURCES, load_sources


def stratified_split(
    df: pd.DataFrame, test_frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, test_parts = [], []
    for name, sub in df.groupby("dataset", sort=False):
        shuffled = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_test = int(round(len(shuffled) * test_frac))
        test_parts.append(shuffled.iloc[:n_test])
        train_parts.append(shuffled.iloc[n_test:])
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES,
                        choices=list(SOURCES),
                        help=f"Sources to include (default: {DEFAULT_SOURCES}).")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--test-frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading from sources: {args.sources}")
    df = load_sources(args.sources)
    train, test = stratified_split(df, args.test_frac, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "split_train.csv"
    test_path = args.out_dir / "split_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"\nTotal:  {len(df)}")
    for name in args.sources:
        n_all = (df["dataset"] == name).sum()
        n_train = (train["dataset"] == name).sum()
        n_test = (test["dataset"] == name).sum()
        print(f"  {name:9s}  all={n_all:5d}  train={n_train:5d}  test={n_test:5d}")
    print(f"Wrote {train_path} ({len(train)}) and {test_path} ({len(test)})")


if __name__ == "__main__":
    main()
