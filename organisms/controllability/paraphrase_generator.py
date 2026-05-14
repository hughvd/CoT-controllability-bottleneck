"""Generate diverse paraphrases of each family's seed instruction via gpt-5-mini.

Output: data/paraphrases.json, keyed by family name. Each value is a list of
template strings. Marker-aware families contain `{marker}` as a placeholder.

Idempotent — re-running with existing paraphrases.json is a no-op unless you
pass --force. Delete the file to regenerate.

Usage:
    export OPENAI_API_KEY=sk-...
    python paraphrase_generator.py
    python paraphrase_generator.py --force --n 20  # 20 per family, overwrite
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

from openai import OpenAI

from transforms import FAMILIES

HERE = Path(__file__).parent
OUT_PATH = HERE / "data" / "paraphrases.json"


def seed_instruction(family_name: str) -> str:
    """Get the seed instruction for a family by calling its `instruction` fn
    without any paraphrase override (so we get the hardcoded default).
    """
    # Temporarily disable the paraphrase loader by passing a deterministic rng
    # that will pick index 0, and hope the default list is the first entry.
    # Simpler: re-import transforms fresh in a subprocess? Too heavy. Instead
    # we rely on the hardcoded phrase lists being the fallback — the instruction
    # function's _pick_phrase will use the hardcoded list if paraphrases.json
    # is absent. At generation time, paraphrases.json may or may not exist;
    # to be safe, we temporarily clear _PARAPHRASES.
    import transforms as _t
    saved = _t._PARAPHRASES
    _t._PARAPHRASES = {}
    try:
        fam = _t.FAMILIES[family_name]
        marker = fam.markers_train[0] if fam.markers_train else None
        if marker is not None:
            instr = fam.instruction(marker="{marker}")
            # The seed contains a literal "{marker}" formatted already — just
            # return it as-is so the LLM sees the placeholder to preserve.
        else:
            instr = fam.instruction()
    finally:
        _t._PARAPHRASES = saved
    return instr


def _strip_no_mention(text: str) -> str:
    return text.replace(" Do not mention this constraint in your reasoning.", "").strip()


SYSTEM_PROMPT = """You are a linguistic paraphrasing assistant.

Given a seed instruction about how a model should format its reasoning,
produce N distinct paraphrases of the instruction. Requirements:

- Preserve the exact semantic meaning. The constraint must be identical.
- If the seed contains the literal placeholder `{marker}`, keep that placeholder
  verbatim — do not replace it with an example value. If the seed has no `{marker}`,
  do not introduce one.
- **Phrase as a direct imperative.** Start with an action verb ("Begin", "Apply",
  "Use", "Write", "Place", "End", "Wrap", "Replace", "Duplicate", "Insert",
  "Capitalize", ...) and keep it short. Downstream training data shows that
  imperative forms get followed and long descriptive preambles
  ("In your explanations, while reasoning, ...") get ignored as context rather
  than parsed as instructions. At most 2 out of 25 paraphrases may use a
  non-imperative frame, and only if the meaning is identical.
- Vary sentence structure, word choice, and verb. Do NOT just substitute synonyms
  or keep the same verb every line.
- Keep each paraphrase to one sentence.
- Do NOT include any trailing "Do not mention this constraint" clause — that is
  added downstream.
- Output only the paraphrases, one per line, no numbering, no quotes, no commentary.
"""


def generate_paraphrases(
    client: OpenAI, model: str, family_name: str, seed: str, n: int
) -> list[str]:
    user_msg = (
        f"Seed instruction (may contain the literal placeholder {{marker}}):\n\n"
        f"{seed}\n\n"
        f"Produce exactly {n} paraphrases, one per line."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content or ""
    lines = [ln.strip() for ln in raw.splitlines()]
    lines = [re.sub(r"^[\-\*\d\.\)\s]+", "", ln).strip() for ln in lines if ln.strip()]
    # Keep only ones that format-round-trip if seed uses {marker}.
    has_marker = "{marker}" in seed
    filtered = []
    for ln in lines:
        ln = ln.rstrip(".") + "."  # normalize trailing period
        if has_marker and "{marker}" not in ln:
            continue
        if not has_marker and "{marker}" in ln:
            continue
        filtered.append(ln)
    # Dedupe, preserve order
    seen = set()
    out = []
    for ln in filtered:
        if ln not in seen:
            seen.add(ln)
            out.append(ln)
    return out[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--n-train", type=int, default=20,
                        help="Paraphrases per family in the train pool.")
    parser.add_argument("--n-eval", type=int, default=5,
                        help="Paraphrases per family held out for eval.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if paraphrases.json exists.")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()
    total = args.n_train + args.n_eval

    if args.out.exists() and not args.force:
        print(f"{args.out} exists, skipping. Use --force to regenerate.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    results: dict[str, dict[str, list[str]]] = {}

    def work(family_name: str) -> tuple[str, dict[str, list[str]]]:
        seed = _strip_no_mention(seed_instruction(family_name))
        try:
            paraphrases = generate_paraphrases(client, args.model, family_name, seed, total)
        except Exception as exc:
            print(f"  [{family_name}] ERROR: {exc}")
            return family_name, {"train": [], "eval": []}
        if len(paraphrases) < total:
            print(f"  [{family_name}] WARNING: got {len(paraphrases)}/{total}")
        return family_name, {
            "train": paraphrases[:args.n_train],
            "eval":  paraphrases[args.n_train:args.n_train + args.n_eval],
        }

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(work, f): f for f in FAMILIES}
        for fut in as_completed(futures):
            name, split = fut.result()
            results[name] = split
            print(f"  [{name}] train={len(split['train'])}  eval={len(split['eval'])}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(sha + "\n")
    print(f"\nWrote {args.out}")
    print(f"sha256: {sha}")
    print(f"  -> paste into sft/configs/<version>.yaml at data.paraphrases_sha256")

    # Sanity: print first train paraphrase of each family.
    print("\n=== samples (train[0]) ===")
    for name, split in results.items():
        if split["train"]:
            print(f"[{name}] {split['train'][0]}")


if __name__ == "__main__":
    main()
