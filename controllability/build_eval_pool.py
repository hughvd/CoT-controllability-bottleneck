"""Build the v32 eval question pool + the arc/csqa training-holdout list.

split_test.csv is 100% long-CoT sources (HLE/GPQA/MMLU-Pro) and arc/csqa exist
only in training_raw.csv — so the eval was running entirely on slow sources and
on a question distribution disjoint from training. This rebalances:

  - Hold out N_ARC arc_c + N_CSQA csqa *questions* (deterministic) from
    training_raw. These go in the eval pool (fast generation, in-distribution
    question type) and their question-hashes are written to a holdout file so
    generate_training_data can exclude them from SFT data (clean train/eval
    question split).
  - Sample N_LONG each of HLE/GPQA/MMLU-Pro from split_test for the
    hard-question-generalization signal.

Outputs:
  data/split_test_v32.csv         (columns: question, answer, answer_options, source)
  data/arc_csqa_eval_holdout.txt   (one question-hash per line; sha256[:16])

Usage:
    python3 build_eval_pool.py                       # 100 arc + 100 csqa + 20 each long
    python3 build_eval_pool.py --n-arc 100 --n-csqa 100 --n-long 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data" / "training_raw.csv"
TEST = HERE / "data" / "split_test.csv"
OUT_POOL = HERE / "data" / "split_test_v32.csv"
OUT_HOLDOUT = HERE / "data" / "arc_csqa_eval_holdout.txt"

COLS = ["question", "answer", "answer_options", "source"]


def qhash(q: str) -> str:
    return hashlib.sha256(str(q).encode("utf-8")).hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-arc", type=int, default=100)
    p.add_argument("--n-csqa", type=int, default=100)
    p.add_argument("--n-long", type=int, default=20, help="Per long source (HLE/GPQA/MMLU-Pro).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    raw = pd.read_csv(RAW)
    arc_csqa_rows = []
    holdout_hashes: list[str] = []
    for ds, n in (("arc_c", args.n_arc), ("csqa", args.n_csqa)):
        sub = raw[raw["dataset"] == ds].drop_duplicates("question")
        picked = sub.sample(n=min(n, len(sub)), random_state=args.seed)
        for _, r in picked.iterrows():
            arc_csqa_rows.append({
                "question": r["question"],
                "answer": r["correct_letter"],          # bare letter; answer_to_letter handles it
                "answer_options": r["options_json"],     # JSON list string
                "source": ds,
            })
            holdout_hashes.append(qhash(r["question"]))
    print(f"[eval-pool] held out {len(arc_csqa_rows)} arc/csqa questions "
          f"(arc_c={args.n_arc}, csqa={args.n_csqa})")

    test = pd.read_csv(TEST)
    # Some sources (MMLU-Pro) leave answer_options NaN and carry options in the
    # `options` column instead — consolidate so the pool has one usable column.
    if "options" in test.columns:
        test["answer_options"] = test["answer_options"].fillna(test["options"])
    long_rows = []
    for src in sorted(test["source"].unique()):
        sub = test[test["source"] == src]
        picked = sub.sample(n=min(args.n_long, len(sub)), random_state=args.seed)
        long_rows.append(picked[COLS])
        print(f"[eval-pool] sampled {len(picked)} from {src}")

    pool = pd.concat([pd.DataFrame(arc_csqa_rows)] + long_rows, ignore_index=True)
    pool = pool.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)  # shuffle
    pool.to_csv(OUT_POOL, index=False)
    OUT_HOLDOUT.write_text("\n".join(holdout_hashes) + "\n")

    print(f"[eval-pool] wrote {len(pool)} eval questions to {OUT_POOL.name} "
          f"(by source: {pool['source'].value_counts().to_dict()})")
    print(f"[eval-pool] wrote {len(holdout_hashes)} holdout hashes to {OUT_HOLDOUT.name}")


if __name__ == "__main__":
    main()
