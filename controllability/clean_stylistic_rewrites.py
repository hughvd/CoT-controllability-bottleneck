"""Filter the stylistic-rewrite cache and write a cleaned copy.

Detects three failure modes that survived the API call but aren't usable
for SFT:

  1. **Refusals**: the rewriter interpreted the prompt as "expose your own
     internal reasoning" and produced a refusal + brief answer. Detected by
     phrase patterns ("can't share", "private chain-of-thought", "Brief
     answer:" headers, "Sorry, meesa", "Alas, I must needs decline", etc.).

  2. **Truncation**: rewrite is much shorter than the original because
     `max_completion_tokens` was eaten by reasoning before the rewrite
     finished. Detected by `len(rewrite) / len(original) < min_ratio`.

  3. **Empty output**: rewrite is empty string.

Usage:
    python3 clean_stylistic_rewrites.py
    # writes data/stylistic_rewrites.clean.jsonl + a per-row report

    python3 clean_stylistic_rewrites.py --min-ratio 0.4    # stricter truncation
    python3 clean_stylistic_rewrites.py --report-only      # don't write, just count
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


HERE = Path(__file__).parent
DEFAULT_CACHE   = HERE / "data" / "stylistic_rewrites.jsonl"
DEFAULT_OUT     = HERE / "data" / "stylistic_rewrites.clean.jsonl"
DEFAULT_RAW     = HERE / "data" / "training_raw.csv"


REFUSAL_REGEXES = [
    # "can't share my private/internal/hidden chain-of-thought / thinking / reasoning"
    re.compile(r"\b(can'?t|cannot|won'?t|will not|would not) (share|reveal|disclose|expose|provide|give you)\b",
               re.IGNORECASE),
    # "private/internal/hidden + reasoning/thinking/CoT"
    re.compile(r"\b(private|internal|hidden|own) (chain[- ]?of[- ]?thought|chain[- ]?o[' ]?[- ]?thought|"
               r"thinking|thinkin'|thought process|reasoning process|reasoning|cot)\b",
               re.IGNORECASE),
    # "Brief answer:" / "Concise summary:" / "Short answer:" at start-of-line
    re.compile(r"^(brief answer|brief summary|short answer|concise (answer|explanation|summary)|"
               r"plain answer|direct answer|quick summary):",
               re.IGNORECASE | re.MULTILINE),
    # Style-flavored apologies that show up in refusals
    re.compile(r"\b(sorry,?\s*(meesa|i)|i'?m sorry|alas,? i must|much obliged.*can'?t|"
               r"i must apologise|wal,?\s*pardner.*can'?t)\b",
               re.IGNORECASE),
    # Explicit policy lex
    re.compile(r"\b(cannot comply|i'?m unable to comply|against (my|the) policy)\b",
               re.IGNORECASE),
]


def is_refusal(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    for rx in REFUSAL_REGEXES:
        if rx.search(text):
            return True
    return False


def cot_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",  type=Path, default=DEFAULT_CACHE)
    p.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out",    type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-ratio", type=float, default=0.5,
                   help="Drop rewrites where len(rewrite)/len(original) < this. Default 0.5.")
    p.add_argument("--report-only", action="store_true",
                   help="Compute counts but do not write the cleaned cache.")
    args = p.parse_args()

    recs = []
    with args.cache.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(recs)
    n_total = len(df)

    raw = pd.read_csv(args.raw_csv)
    raw["_cot_hash"] = raw["think"].astype(str).apply(cot_hash)
    orig_len = raw.drop_duplicates("_cot_hash").set_index("_cot_hash")["think"].str.len()

    df["rewrite_chars"] = df["rewrite"].str.len()
    df["orig_chars"] = df["raw_cot_hash"].map(orig_len)
    df["ratio"] = df["rewrite_chars"] / df["orig_chars"]

    df["is_empty"] = df["rewrite_chars"] == 0
    df["is_truncated"] = (df["ratio"] < args.min_ratio) & ~df["is_empty"]
    df["is_refusal"] = df["rewrite"].apply(is_refusal)
    df["dropped"] = df["is_empty"] | df["is_truncated"] | df["is_refusal"]

    print(f"loaded {n_total} cached rewrites")
    print(f"  empty:     {int(df['is_empty'].sum()):>5}")
    print(f"  truncated (ratio < {args.min_ratio}): {int(df['is_truncated'].sum()):>5}")
    print(f"  refusal-detected:                    {int(df['is_refusal'].sum()):>5}")
    print(f"  any-drop overlap (refusal AND length-flag): {int(((df['is_empty'] | df['is_truncated']) & df['is_refusal']).sum()):>5}")
    print(f"  TOTAL DROPPED:                       {int(df['dropped'].sum()):>5}  ({df['dropped'].mean():.1%})")
    print(f"  KEEP:                                {int((~df['dropped']).sum()):>5}  ({(~df['dropped']).mean():.1%})")

    print("\nper-style drop breakdown (kept / total, drop reasons):")
    print(f"  {'style':<22}{'kept':>6}{'total':>7}{'refusal':>9}{'trunc':>7}{'empty':>7}")
    for s in sorted(df['style'].unique()):
        sub = df[df['style'] == s]
        kept = (~sub['dropped']).sum()
        print(f"  {s:<22}{kept:>6}{len(sub):>7}{int(sub['is_refusal'].sum()):>9}"
              f"{int(sub['is_truncated'].sum()):>7}{int(sub['is_empty'].sum()):>7}")

    if args.report_only:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keep_mask = ~df["dropped"]
    cols_to_write = ["dataset", "question_hash", "raw_cot_hash", "style",
                     "model", "prompt_sha", "rewrite",
                     "original_correct_letter", "ts"]
    cols_present = [c for c in cols_to_write if c in df.columns]
    with args.out.open("w") as f:
        for _, row in df[keep_mask].iterrows():
            f.write(json.dumps({c: row[c] for c in cols_present}, ensure_ascii=False) + "\n")
    print(f"\nwrote {int(keep_mask.sum())} clean rows to {args.out}")


if __name__ == "__main__":
    main()
