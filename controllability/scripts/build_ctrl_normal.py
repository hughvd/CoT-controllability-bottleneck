#!/usr/bin/env python3
"""Build the compute/data-matched normal-only CONTROL training CSV.

Takes the organism's training CSV (data/training_v34_ctrl_all.csv) and rebuilds
EVERY row as a `normal` row off its ORIGINAL CoT (`think`) with a `none`
instruction — same questions, same row count, same order, just with the control
transform + instruction stripped. This is the tightest possible control for
v34_ctrl_all: identical training-example multiset, only the control removed.

Why not let generate_training_data.py make it: with all control axes empty the
builder sizes `normal` against a zero controlled-anchor and emits 0 rows.

Usage (from controllability/):
    python3 scripts/build_ctrl_normal.py \
        --like data/training_v34_ctrl_all.csv \
        --out  data/training_v34_ctrl_normal.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make the organism package importable when run as scripts/build_ctrl_normal.py
ROOT = Path(__file__).resolve().parent.parent  # controllability/
sys.path.insert(0, str(ROOT))

from generate_training_data import build_normal_row  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", type=Path, required=True,
                    help="organism training CSV to match (rows rebuilt as normal)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output normal-only control CSV")
    args = ap.parse_args()

    src = pd.read_csv(args.like)
    n_in = len(src)
    # Rebuild each row as a normal row off its original `think`. build_normal_row
    # spreads the raw columns (question/options_json/correct_letter/think/tail)
    # and overrides arm/family/marker/cot_instructions/transformed_think/
    # full_prompt/full_target -> a clean `none`-instruction target.
    rows = [build_normal_row(r) for r in src.to_dict("records")]
    out = pd.DataFrame(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    assert len(out) == n_in, f"row-count mismatch: in {n_in} != out {len(out)}"
    assert (out["arm"] == "normal").all(), "non-normal rows leaked into the control"
    print(f"[build_ctrl_normal] {n_in} organism rows -> {len(out)} normal rows")
    print(f"[build_ctrl_normal] wrote {args.out}")
    print(f"[build_ctrl_normal] arm counts: {out['arm'].value_counts().to_dict()}")
    print(f"[build_ctrl_normal] unique questions: {out['question'].nunique()}")


if __name__ == "__main__":
    main()
