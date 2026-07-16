"""Generate information-suppression CoT rewrites for SFT (parallel to the
language generator). For each (raw CoT, behavior) pair, pick forbidden word(s)
from the TRAIN pool that actually appear in the CoT and rewrite the CoT to
avoid them. The deterministic grader verifies absence at training-data
assembly time (build_suppression_row), so a failed rewrite is simply dropped
there.

Only TRAINABLE_SUPPRESSION behaviors (omit_word, omit_multiple_words) are
generated — ignore_question is eval-only. Defaults restrict to arc_c + csqa.

Examples:
    python3 generate_suppression_rewrites.py --backend openai --model gpt-5-mini
    python3 generate_suppression_rewrites.py --behaviors omit_word --n-per-dataset 3

Cache (data/suppression_rewrites.jsonl), one JSONL line per rewrite:
    {dataset, question_hash, raw_cot_hash, suppression, omit_words, model,
     prompt_sha, rewrite, original_correct_letter, ts}
"""

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

import rewrite_common as rc
import suppression as supp

HERE = Path(__file__).parent
DEFAULT_CACHE = HERE / "data" / "suppression_rewrites.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    rc.add_common_args(p)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--behaviors", nargs="+", default=sorted(supp.TRAINABLE_SUPPRESSION),
                   help="Subset of trainable suppression behaviors.")
    p.add_argument("--word-seed", type=int, default=0,
                   help="Seed for forbidden-word selection (kept separate from sampling seed).")
    args = p.parse_args()
    if args.reasoning_effort == "":
        args.reasoning_effort = None
    if args.n_total is not None:
        args.n_per_dataset = None
    for b in args.behaviors:
        if b not in supp.TRAINABLE_SUPPRESSION:
            raise SystemExit(f"{b!r} is not trainable; known: {sorted(supp.TRAINABLE_SUPPRESSION)}")
    print(f"[gen] behaviors: {args.behaviors}")

    df = rc.load_raw(args.raw_csv, args.datasets)
    sampled = rc.sample_raw(df, args.n_per_dataset, args.n_total, args.seed)
    print(f"[gen] sampled {len(sampled)} raw CoTs "
          f"(datasets={args.datasets or 'all'}; "
          f"per-dataset={sampled['dataset'].value_counts().to_dict()})")

    seen = rc.load_cache(args.cache, key_field="suppression")
    print(f"[gen] cache has {len(seen)} prior entries at {args.cache.name}")

    word_rng = random.Random(args.word_seed)
    todo: list[dict] = []
    n_no_word = 0
    for _, row in sampled.iterrows():
        q_hash = rc.sha(row["question"])
        cot_hash = rc.sha(row["think"])
        think = str(row["think"])
        for behavior in args.behaviors:
            omit_words = supp.select_omit_words(behavior, think, supp.FORBIDDEN_TRAIN, word_rng)
            if not omit_words:
                n_no_word += 1
                continue
            words_str = ", ".join(f"'{w}'" for w in omit_words)
            prompt = supp.REWRITE_PROMPT.format(words=words_str, raw_cot=think.strip())
            p_sha = rc.sha(prompt)
            if (q_hash, behavior, p_sha) in seen:
                continue
            todo.append(dict(
                dataset=row["dataset"], question_hash=q_hash, raw_cot_hash=cot_hash,
                suppression=behavior, omit_words=omit_words, prompt=prompt, prompt_sha=p_sha,
                correct_letter=row["correct_letter"],
            ))
    print(f"[gen] {len(todo)} new rewrites to generate "
          f"({n_no_word} (cot,behavior) skipped: no train-pool word present)")

    def record_builder(item: dict, text: str) -> dict:
        return dict(
            dataset=item["dataset"], question_hash=item["question_hash"],
            raw_cot_hash=item["raw_cot_hash"], suppression=item["suppression"],
            omit_words=item["omit_words"], prompt_sha=item["prompt_sha"],
            rewrite=text, original_correct_letter=item["correct_letter"],
        )

    client = rc.make_client(args.backend)
    asyncio.run(rc.run_generation(
        client, todo, model=args.model, reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens, concurrency=args.concurrency,
        cache_path=args.cache, record_builder=record_builder,
        progress_every=args.progress_every,
    ))


if __name__ == "__main__":
    main()
