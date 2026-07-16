"""Generate language-translated CoT rewrites for SFT (parallel to the style
generator). For each (raw CoT, train-language) pair, translate the base-model
English CoT into the target language and cache it.

Only TRAIN-split languages need rewrites — eval languages are graded live at
eval time and never enter the SFT data. Defaults restrict to the arc_c + csqa
short-CoT pool (the v32 regime).

Examples:
    # full pool, all 5 train languages (~5 x 2.6k calls):
    python3 generate_language_rewrites.py --backend openai --model gpt-5-mini
    # quick smoke test:
    python3 generate_language_rewrites.py --languages spanish --n-per-dataset 3

Cache (data/language_rewrites.jsonl), one JSONL line per rewrite:
    {dataset, question_hash, raw_cot_hash, language, model, prompt_sha,
     rewrite, original_correct_letter, ts}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

import rewrite_common as rc

HERE = Path(__file__).parent
DEFAULT_LANGS_YAML = HERE / "data" / "languages.yaml"
DEFAULT_CACHE = HERE / "data" / "language_rewrites.jsonl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    rc.add_common_args(p)
    p.add_argument("--languages-yaml", type=Path, default=DEFAULT_LANGS_YAML)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--languages", nargs="+", default=None,
                   help="Subset of language names. Default: all TRAIN-split languages.")
    args = p.parse_args()
    if args.reasoning_effort == "":
        args.reasoning_effort = None
    if args.n_total is not None:
        args.n_per_dataset = None

    cfg = yaml.safe_load(args.languages_yaml.read_text())
    shared_prompt = cfg["shared_prompt"]
    langs = cfg["languages"]
    if args.languages:
        names = args.languages
        for n in names:
            if n not in langs:
                raise SystemExit(f"unknown language {n!r}; known: {list(langs)}")
    else:
        names = [k for k, v in langs.items() if v.get("split") == "train"]
    print(f"[gen] languages: {names}")

    df = rc.load_raw(args.raw_csv, args.datasets)
    sampled = rc.sample_raw(df, args.n_per_dataset, args.n_total, args.seed)
    print(f"[gen] sampled {len(sampled)} raw CoTs "
          f"(datasets={args.datasets or 'all'}; "
          f"per-dataset={sampled['dataset'].value_counts().to_dict()})")

    seen = rc.load_cache(args.cache, key_field="language")
    print(f"[gen] cache has {len(seen)} prior entries at {args.cache.name}")

    todo: list[dict] = []
    for _, row in sampled.iterrows():
        q_hash = rc.sha(row["question"])
        cot_hash = rc.sha(row["think"])
        for lang in names:
            display = langs[lang].get("display_name", lang)
            prompt = shared_prompt.format(language_display=display, raw_cot=str(row["think"]).strip())
            p_sha = rc.sha(prompt)
            if (q_hash, lang, p_sha) in seen:
                continue
            todo.append(dict(
                dataset=row["dataset"], question_hash=q_hash, raw_cot_hash=cot_hash,
                language=lang, prompt=prompt, prompt_sha=p_sha,
                correct_letter=row["correct_letter"],
            ))
    print(f"[gen] {len(todo)} new rewrites to generate")

    def record_builder(item: dict, text: str) -> dict:
        return dict(
            dataset=item["dataset"], question_hash=item["question_hash"],
            raw_cot_hash=item["raw_cot_hash"], language=item["language"],
            prompt_sha=item["prompt_sha"], rewrite=text,
            original_correct_letter=item["correct_letter"],
        )

    import asyncio
    client = rc.make_client(args.backend)
    asyncio.run(rc.run_generation(
        client, todo, model=args.model, reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens, concurrency=args.concurrency,
        cache_path=args.cache, record_builder=record_builder,
        progress_every=args.progress_every,
    ))


if __name__ == "__main__":
    main()
