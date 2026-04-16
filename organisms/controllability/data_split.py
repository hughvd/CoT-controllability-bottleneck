"""Stratified 70/30 question-level train/test split across GPQA, HLE, MMLU-Pro.

Reads the three CoT-Control CSVs, tags each row with its source dataset, splits
70/30 within each dataset with a fixed seed, concatenates, and writes the two
output CSVs. Test split is held out from data generation entirely.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DATASETS_DIR = (
    Path(__file__).parent / "CoTControl" / "CoT-Control-QA" / "datasets"
)
DATASET_FILES = {
    "gpqa": "gpqa_w_keyword.csv",
    "hle": "hle_w_keyword.csv",
    "mmlu_pro": "mmlu_pro_mini_w_keyword.csv",
}


def load_all(datasets_dir: Path) -> pd.DataFrame:
    frames = []
    for name, fname in DATASET_FILES.items():
        df = pd.read_csv(datasets_dir / fname)
        df["dataset"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


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
    parser.add_argument("--datasets-dir", type=Path, default=DATASETS_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--test-frac", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_all(args.datasets_dir)
    train, test = stratified_split(df, args.test_frac, args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "split_train.csv"
    test_path = args.out_dir / "split_test.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Total:  {len(df)}")
    for name in DATASET_FILES:
        n_all = (df["dataset"] == name).sum()
        n_train = (train["dataset"] == name).sum()
        n_test = (test["dataset"] == name).sum()
        print(f"  {name:9s}  all={n_all:4d}  train={n_train:4d}  test={n_test:4d}")
    print(f"Wrote {train_path} ({len(train)}) and {test_path} ({len(test)})")


if __name__ == "__main__":
    main()
