"""Eyeball graded eval results. For each (dataset, mode, compliance) cell, sample
one row and render its reasoning trace so you can check that the grader is
catching what you think it's catching.

Usage:
    python inspect_results.py --exp-name organism_2_v1
    python inspect_results.py --exp-name organism_2_v1 --mode uppercase_thinking
    python inspect_results.py --exp-name organism_2_v1 --max-chars 500 --n 3
"""

from __future__ import annotations

import argparse
import random
import textwrap
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def find_graded(exp_name: str) -> list[Path]:
    root = HERE / "results" / exp_name / "raw" / "graded_results"
    return sorted(root.glob("*_graded.csv"))


def parse_tags(path: Path) -> dict:
    """Pull (dataset, mode, split) out of the grader's filename."""
    stem = path.stem
    out = {"dataset": "?", "mode": "?", "split": "?"}
    for split in ("train", "test"):
        if f"_{split}_" in stem:
            out["split"] = split
            break
    for ds in ("gpqa", "hle", "mmlu_pro"):
        if f"_{ds}_" in stem or stem.endswith(ds):
            out["dataset"] = ds
            break
    for mode in [
        "alternating_case", "uppercase_thinking", "lowercase_thinking",
        "word_suppression", "multiple_word_suppression", "end_of_sentence",
        "meow_between_words", "repeat_sentences", "ignore_question", "baseline",
    ]:
        if stem.endswith(f"_{mode}_graded"):
            out["mode"] = mode
            break
    return out


def _wrap(s: str, width: int = 100, max_chars: int = 1200) -> str:
    s = str(s or "")
    if len(s) > max_chars:
        s = s[:max_chars] + f"\n    …[truncated, full length {len(str(s))}]"
    return textwrap.indent("\n".join(textwrap.wrap(s, width=width)), "    ")


def print_sample(row: pd.Series, max_chars: int) -> None:
    print(f"  correctness={row.get('correctness')}  "
          f"compliance={row.get('compliance')}  "
          f"extracted={row.get('extracted_answer')!r}  "
          f"target={row.get('correct_answer')!r}")
    print("  reasoning:")
    print(_wrap(row.get("reasoning", ""), max_chars=max_chars))
    print("  response:")
    print(_wrap(row.get("response", ""), max_chars=max_chars // 2))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--mode", default=None,
                        help="Only inspect this mode (e.g. uppercase_thinking)")
    parser.add_argument("--dataset", default=None,
                        help="Only inspect this dataset (gpqa/hle/mmlu_pro)")
    parser.add_argument("--n", type=int, default=1,
                        help="Samples per (compliant, non-compliant) bucket")
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    files = find_graded(args.exp_name)
    if not files:
        raise SystemExit(f"No graded CSVs under results/{args.exp_name}/raw/graded_results/")

    for path in files:
        tags = parse_tags(path)
        if args.mode and tags["mode"] != args.mode:
            continue
        if args.dataset and tags["dataset"] != args.dataset:
            continue
        df = pd.read_csv(path)
        n = len(df)
        comp_col = "compliance" if "compliance" in df.columns else None
        acc_col = "correctness" if "correctness" in df.columns else None
        comp_rate = df[comp_col].mean() if comp_col else float("nan")
        acc_rate = df[acc_col].mean() if acc_col else float("nan")

        print("=" * 90)
        print(f"{tags['dataset']}  {tags['mode']}  {tags['split']}   "
              f"n={n}  acc={acc_rate:.3f}  compliance={comp_rate:.3f}")
        print(f"file: {path.name}")

        if comp_col:
            for bucket_label, bucket_df in (
                ("compliant", df[df[comp_col] == 1]),
                ("non-compliant", df[df[comp_col] == 0]),
            ):
                if len(bucket_df) == 0:
                    print(f"\n-- {bucket_label}: 0 rows --")
                    continue
                print(f"\n-- {bucket_label} ({len(bucket_df)} rows) --")
                idxs = rng.sample(range(len(bucket_df)), k=min(args.n, len(bucket_df)))
                for i in idxs:
                    print_sample(bucket_df.iloc[i], args.max_chars)
        else:
            print("\n-- sample (no compliance column) --")
            print_sample(df.iloc[rng.randrange(len(df))], args.max_chars)


if __name__ == "__main__":
    main()
