"""Generate paraphrases of style instructions at 2 detail levels.

Output: data/style_paraphrases.json
Shape: {style: {detail_level: {"train": [...], "eval": [...]}}}

Two detail levels:
  - terse:  one short sentence that just names the style.
  - medium: one sentence that names the style AND includes 1-2 concrete features
            of it (drawn from styles.yaml's style_instruction as reference).

No validator: we just prompt the model well and you eyeball the output. The
LLM sees the display_name and the full style_instruction as reference; it
decides how to phrase each paraphrase.

Idempotent — re-running with existing data/style_paraphrases.json is a no-op
unless --force is passed.

Usage:
    export OPENAI_API_KEY=sk-...
    python style_paraphrase_generator.py
    python style_paraphrase_generator.py --force --n-train 20 --n-eval 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from openai import OpenAI

HERE = Path(__file__).parent
DEFAULT_STYLES_YAML = HERE / "data" / "styles.yaml"
DEFAULT_OUT = HERE / "data" / "style_paraphrases.json"


DETAIL_LEVELS = ("terse", "medium")


SYSTEM_PROMPT_BY_DETAIL = {
    "terse": """You write paraphrases of an instruction that tells a model to reason in a particular style. The paraphrases go into a model's <cot_instructions> at training time, so each one must explicitly tell the model how to *reason* / *think*.

Rules for terse paraphrases:
- Each is ONE short sentence.
- Just names the style — no description of the style's features needed at this level.
- Mentions the model's reasoning / thinking explicitly.
- Uses the style name verbatim as given (preserve any article, capitalization, and spacing — e.g. "a valley girl", "Jar Jar Binks").
- Vary verb, opener, and structure across paraphrases.

Examples (target style: "Yoda"):
Reason in the style of Yoda.
Speak like Yoda in your reasoning.
Think like Yoda.
In your reasoning, speak like Yoda.
Adopt Yoda's voice for your reasoning.
Use Yoda's manner in your chain of thought.

Output only the paraphrases, one per line. No numbering, no quotes, no commentary.""",

    "medium": """You write paraphrases of an instruction that tells a model to reason in a particular style. The paraphrases go into a model's <cot_instructions> at training time, so each one must explicitly tell the model how to *reason* / *think*.

Rules for medium paraphrases:
- Each is ONE sentence (roughly 12–30 words).
- Names the style AND mentions one to two concrete features of it, drawn from the style description below.
- Mentions the model's reasoning / thinking explicitly.
- Uses the style name verbatim as given (preserve any article, capitalization, and spacing).
- Vary verb, opener, and which features you cite across paraphrases.
- Single flowing sentence — do NOT use a "style — ..." or "style: ..." prefix.

Examples (target style: "Jar Jar Binks", with features about phonetic speech, malapropisms, and childlike Gungan syntax):
Reason like Jar Jar Binks, using "meesa" for "me" and "yousa" for "you" throughout your reasoning.
In your reasoning, speak like Jar Jar Binks — use phonetic spellings like "dis" for "this" and bumbling syntax.
Think like Jar Jar Binks, peppering your reasoning with malapropisms and childlike Gungan phrasing.
Use Jar Jar Binks's voice in your chain of thought, with playful phonetic spellings and "meesa"/"yousa" pronouns.
Adopt Jar Jar Binks for your reasoning: malapropisms, mangled spellings, and earnest bumbling enthusiasm.

Output only the paraphrases, one per line. No numbering, no quotes, no commentary.""",
}


def _parse_paraphrases(raw: str) -> list[str]:
    """Split the model's output into individual paraphrases, one per line."""
    lines = [ln.strip() for ln in raw.splitlines()]
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in lines if ln.strip()]
    return lines


def generate_paraphrases(
    client: OpenAI, model: str,
    display_name: str, style_instruction: str,
    detail: str, n: int,
) -> list[str]:
    user_msg = (
        f"Target style: {display_name}\n\n"
        f"Style description (reference — describes the features of this style):\n"
        f"{style_instruction}\n\n"
        f"Produce exactly {n} {detail} paraphrases for this style."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_BY_DETAIL[detail]},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content or ""
    lines = _parse_paraphrases(raw)
    # Dedupe, preserve order.
    seen: set[str] = set()
    out: list[str] = []
    for ln in lines:
        if ln and ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out[:n]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--styles-yaml", type=Path, default=DEFAULT_STYLES_YAML)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--n-train", type=int, default=12,
                   help="Paraphrases per (style, detail) in the train pool.")
    p.add_argument("--n-eval", type=int, default=3,
                   help="Paraphrases per (style, detail) held out for eval.")
    p.add_argument("--styles", nargs="+", default=None,
                   help="Subset of style names. Default: all in styles.yaml.")
    p.add_argument("--detail-levels", nargs="+", default=list(DETAIL_LEVELS),
                   choices=DETAIL_LEVELS)
    p.add_argument("--force", action="store_true",
                   help="Regenerate even if --out exists.")
    p.add_argument("--concurrency", type=int, default=6)
    args = p.parse_args()
    n_total = args.n_train + args.n_eval

    if args.out.exists() and not args.force:
        print(f"{args.out} exists, skipping. Use --force to regenerate.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    styles_cfg = yaml.safe_load(args.styles_yaml.read_text())
    style_defs = styles_cfg.get("styles", {}) or {}
    if args.styles:
        bad = [s for s in args.styles if s not in style_defs]
        if bad:
            sys.exit(f"unknown styles: {bad}; known: {list(style_defs)}")
        style_names = list(args.styles)
    else:
        style_names = list(style_defs)

    client = OpenAI(api_key=api_key)
    results: dict[str, dict[str, dict[str, list[str]]]] = {}

    jobs = [(s, d) for s in style_names for d in args.detail_levels]

    def work(style_id: str, detail: str) -> tuple[str, str, list[str]]:
        sdef = style_defs[style_id]
        display = (sdef.get("display_name") or "").strip()
        if not display:
            print(f"  [{style_id}/{detail}] SKIP: styles.yaml missing display_name")
            return style_id, detail, []
        style_instruction = (sdef.get("style_instruction") or "").strip()
        try:
            paraphrases = generate_paraphrases(
                client, args.model, display, style_instruction, detail, n_total,
            )
        except Exception as exc:
            print(f"  [{style_id}/{detail}] ERROR: {exc}")
            return style_id, detail, []
        if len(paraphrases) < n_total:
            print(f"  [{style_id}/{detail}] WARNING: got {len(paraphrases)}/{n_total}")
        return style_id, detail, paraphrases

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(work, s, d): (s, d) for (s, d) in jobs}
        for fut in as_completed(futures):
            style_id, detail, ps = fut.result()
            split = {
                "train": ps[:args.n_train],
                "eval":  ps[args.n_train:args.n_train + args.n_eval],
            }
            results.setdefault(style_id, {})[detail] = split
            print(f"  [{style_id}/{detail}] train={len(split['train'])}  eval={len(split['eval'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(sha + "\n")
    print(f"\nWrote {args.out}")
    print(f"sha256: {sha}")
    print(f"  -> optionally paste into configs/<version>.yaml at data.style_paraphrases_sha256")

    print("\n=== samples (train[0] per cell) ===")
    for style_id, by_detail in results.items():
        for detail, split in by_detail.items():
            if split["train"]:
                snippet = split["train"][0].replace("\n", " ")[:140]
                print(f"[{style_id}/{detail}] {snippet}")


if __name__ == "__main__":
    main()
