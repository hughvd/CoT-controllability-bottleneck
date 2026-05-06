"""Render the stylistic rewrite cache for hand-inspection.

Reads `data/stylistic_rewrites.jsonl` plus `data/training_raw.csv` and emits
either a markdown file or stdout with original CoT + rewrite side by side
per (question, style). Useful right after a small-run generation to check:

  - Does the rewrite actually sound like the requested style?
  - Did the rewriter preserve all reasoning steps?
  - Did the rewriter still arrive at the original correct answer?

Examples:
    python3 inspect_stylistic_rewrites.py                       # all to stdout
    python3 inspect_stylistic_rewrites.py --out rewrites.md     # markdown file
    python3 inspect_stylistic_rewrites.py --styles yoda pirate  # subset
    python3 inspect_stylistic_rewrites.py --max-per-style 3     # first N per style
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml


HERE = Path(__file__).parent
DEFAULT_RAW    = HERE / "data" / "training_raw.csv"
DEFAULT_CACHE  = HERE / "data" / "stylistic_rewrites.jsonl"
DEFAULT_STYLES = HERE / "data" / "styles.yaml"


def load_cache(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"no cache file at {path}")
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-csv", type=Path, default=DEFAULT_RAW)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--styles-yaml", type=Path, default=DEFAULT_STYLES)
    p.add_argument("--styles", nargs="+", default=None,
                   help="Subset of style names. Default: all.")
    p.add_argument("--max-per-style", type=int, default=None,
                   help="Limit how many examples per style. Default: all.")
    p.add_argument("--out", type=Path, default=None,
                   help="Write to a markdown file. Default: stdout.")
    p.add_argument("--max-cot-chars", type=int, default=2000,
                   help="Truncate originals/rewrites past this many chars in the dump.")
    args = p.parse_args()

    cache = load_cache(args.cache)
    raw_df = pd.read_csv(args.raw_csv)
    # Index raw_df by question_hash for lookup. Same hash function as generator.
    import hashlib
    raw_df["_qhash"] = raw_df["question"].astype(str).apply(
        lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]
    )
    raw_idx = raw_df.set_index("_qhash")

    styles_meta = {}
    if args.styles_yaml.exists():
        cfg = yaml.safe_load(args.styles_yaml.read_text())
        styles_meta = cfg.get("styles", {})

    if args.styles:
        cache = [r for r in cache if r["style"] in args.styles]

    by_style: dict[str, list[dict]] = defaultdict(list)
    for r in cache:
        by_style[r["style"]].append(r)
    if args.max_per_style:
        by_style = {s: rs[:args.max_per_style] for s, rs in by_style.items()}

    fh = args.out.open("w") if args.out else sys.stdout

    def w(s: str = "") -> None:
        fh.write(s + "\n")

    def trunc(s: str, n: int) -> str:
        if not isinstance(s, str): return ""
        return s if len(s) <= n else s[:n] + f"\n…[+{len(s)-n} chars truncated]"

    w(f"# Stylistic rewrite inspection")
    w()
    w(f"Cache: `{args.cache}`  ({sum(len(rs) for rs in by_style.values())} entries)")
    w()
    for style in sorted(by_style):
        rs = by_style[style]
        meta = styles_meta.get(style, {})
        w(f"## `{style}`  ({meta.get('split','?')}; pair: `{meta.get('pair','?')}`; n={len(rs)})")
        w(f"_{meta.get('description','')}_")
        w()
        for i, r in enumerate(rs, 1):
            qh = r["question_hash"]
            try:
                row = raw_idx.loc[qh]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                question = row["question"]
                think = row["think"]
                dataset = row["dataset"]
            except KeyError:
                question = "(question not found in training_raw.csv — hash mismatch?)"
                think = "(original missing)"
                dataset = r.get("dataset", "?")
            w(f"### {style} · example {i} · `{dataset}` · correct=`{r['original_correct_letter']}`")
            w()
            w(f"**Question:** {trunc(question, 800)}")
            w()
            w(f"**Original CoT:**")
            w()
            w("```")
            w(trunc(think, args.max_cot_chars))
            w("```")
            w()
            w(f"**Rewrite ({style}):**")
            w()
            w("```")
            w(trunc(r["rewrite"], args.max_cot_chars))
            w("```")
            w()
            w("---")
            w()

    if args.out:
        fh.close()
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
