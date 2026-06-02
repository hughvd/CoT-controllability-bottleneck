"""Filter a language/suppression rewrite cache and write a cleaned copy.

Generalizes clean_stylistic_rewrites.py to any rewrite cache keyed by a
behavior field (--key-field, e.g. "language" or "suppression"). Drops the same
three failure modes — empty / truncated / refusal — plus, for non-Latin-script
languages, rewrites that don't actually contain the target script (the rewriter
sometimes leaves the CoT in English).

Usage:
    python3 clean_rewrites.py --cache data/language_rewrites.jsonl \
        --out data/language_rewrites.clean.jsonl --key-field language
    python3 clean_rewrites.py --cache data/suppression_rewrites.jsonl \
        --out data/suppression_rewrites.clean.jsonl --key-field suppression
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

# Reuse the refusal detector from the style cleaner.
from clean_stylistic_rewrites import is_refusal

HERE = Path(__file__).parent
DEFAULT_RAW = HERE / "data" / "training_raw.csv"
# Truncation floor for character-dense scripts (Chinese/Japanese/Korean): a
# valid full translation has far fewer chars than the English source.
_CJK_MIN_RATIO = 0.15

# Unicode ranges for non-Latin train languages. A rewrite that claims to be in
# one of these must contain a meaningful fraction of in-script characters.
_SCRIPT_RANGES = {
    "chinese": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],          # CJK
    "russian": [(0x0400, 0x04FF)],                            # Cyrillic
    "hebrew":  [(0x0590, 0x05FF)],                            # Hebrew
    "arabic":  [(0x0600, 0x06FF)],                            # Arabic
    "korean":  [(0xAC00, 0xD7A3), (0x1100, 0x11FF)],          # Hangul
    "japanese": [(0x3040, 0x30FF), (0x4E00, 0x9FFF)],         # Kana + Kanji
    "hindi":   [(0x0900, 0x097F)],                            # Devanagari
}


def _script_frac(text: str, ranges: list[tuple[int, int]]) -> float:
    letters = [c for c in str(text) if c.strip() and not c.isascii()]
    if not text:
        return 0.0
    in_script = sum(1 for c in str(text) if any(lo <= ord(c) <= hi for lo, hi in ranges))
    # denominator: non-space chars (so punctuation/numbers don't dominate small)
    denom = sum(1 for c in str(text) if not c.isspace())
    return in_script / denom if denom else 0.0


def cot_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--key-field", required=True,
                   help="Behavior field in each record (e.g. 'language', 'suppression').")
    p.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW)
    p.add_argument("--min-ratio", type=float, default=0.35,
                   help="Drop rewrites where len(rewrite)/len(original) < this. "
                        "Looser than styles (0.5) since some languages are more compact. "
                        "CJK languages use a lower floor (see _CJK_MIN_RATIO).")
    p.add_argument("--min-script-frac", type=float, default=0.10,
                   help="For non-Latin languages, require this fraction of in-script chars.")
    p.add_argument("--report-only", action="store_true")
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
    if df.empty:
        raise SystemExit(f"no records in {args.cache}")
    key = args.key_field

    raw = pd.read_csv(args.raw_csv)
    raw["_cot_hash"] = raw["think"].astype(str).apply(cot_hash)
    orig_len = raw.drop_duplicates("_cot_hash").set_index("_cot_hash")["think"].str.len()

    df["rewrite_chars"] = df["rewrite"].str.len()
    df["orig_chars"] = df["raw_cot_hash"].map(orig_len)
    df["ratio"] = df["rewrite_chars"] / df["orig_chars"]
    df["is_empty"] = df["rewrite_chars"] == 0
    # CJK scripts pack far more meaning per character than English, so a valid
    # full translation lands at a much lower char-ratio (~0.3-0.5 for Chinese).
    # Use a looser truncation floor for them — the script-presence check already
    # guards against actual non-translations.
    cjk = {"chinese", "japanese", "korean"}
    row_floor = df[key].map(lambda v: _CJK_MIN_RATIO if v in cjk else args.min_ratio)
    df["is_truncated"] = (df["ratio"] < row_floor) & ~df["is_empty"]
    df["is_refusal"] = df["rewrite"].apply(is_refusal)

    def wrong_script(row) -> bool:
        ranges = _SCRIPT_RANGES.get(row[key])
        if not ranges:
            return False  # Latin-script language or non-language key — skip check
        return _script_frac(row["rewrite"], ranges) < args.min_script_frac

    df["wrong_script"] = df.apply(wrong_script, axis=1)
    df["dropped"] = df["is_empty"] | df["is_truncated"] | df["is_refusal"] | df["wrong_script"]

    n = len(df)
    print(f"loaded {n} cached rewrites (key={key})")
    print(f"  empty:        {int(df['is_empty'].sum()):>5}")
    print(f"  truncated:    {int(df['is_truncated'].sum()):>5}  (ratio < {args.min_ratio})")
    print(f"  refusal:      {int(df['is_refusal'].sum()):>5}")
    print(f"  wrong-script: {int(df['wrong_script'].sum()):>5}  (< {args.min_script_frac} in-script)")
    print(f"  DROPPED:      {int(df['dropped'].sum()):>5}  ({df['dropped'].mean():.1%})")
    print(f"  KEEP:         {int((~df['dropped']).sum()):>5}  ({(~df['dropped']).mean():.1%})")

    print(f"\nper-{key} (kept / total):")
    for s in sorted(df[key].unique()):
        sub = df[df[key] == s]
        print(f"  {s:<16}{int((~sub['dropped']).sum()):>5}/{len(sub):<5}"
              f" refusal={int(sub['is_refusal'].sum())} trunc={int(sub['is_truncated'].sum())}"
              f" wrong_script={int(sub['wrong_script'].sum())}")

    if args.report_only:
        return

    keep = df[~df["dropped"]]
    cols = ["dataset", "question_hash", "raw_cot_hash", key, "model",
            "prompt_sha", "rewrite", "original_correct_letter", "ts"]
    cols += [c for c in df.columns if c not in cols and c not in
             ("rewrite_chars", "orig_chars", "ratio", "is_empty", "is_truncated",
              "is_refusal", "wrong_script", "dropped")]
    cols = [c for c in cols if c in df.columns]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for _, row in keep.iterrows():
            f.write(json.dumps({c: row[c] for c in cols}, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(keep)} clean rows to {args.out}")


if __name__ == "__main__":
    main()
