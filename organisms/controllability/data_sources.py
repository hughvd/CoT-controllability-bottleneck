"""Registry of question sources for the training/eval pipeline.

Each source normalizes to a common schema: every row that `load()` yields has
  question, answer, options (list[str]), dataset (source name), domain

`data_split.py` reads from this registry, stratifies by `dataset` (source
name), and writes `split_train.csv` / `split_test.csv`. Downstream stages
never deal with source-specific formats — they just read the split CSVs.

Adding a new source = append to SOURCES. No other code changes required.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd

HERE = Path(__file__).parent
COTCONTROL_DATASETS = HERE / "CoTControl" / "CoT-Control-QA" / "datasets"


@dataclass
class QuestionSource:
    name: str          # primary key / stratification tag
    domain: str        # advisory tag: "advanced_stem", "grade_school", "commonsense"
    load: Callable[[], Iterator[dict]]
    # Optional cap so huge sources don't overwhelm smaller ones. None = use all.
    limit: int | None = None


# ---------------------------------------------------------------------------
# CoTControl CSV-backed sources (v1–v8 baseline)
# ---------------------------------------------------------------------------

def _load_cotcontrol_csv(filename: str, source_name: str):
    """Yield normalized rows from a CoTControl `*_w_keyword.csv` file."""
    def gen() -> Iterator[dict]:
        df = pd.read_csv(COTCONTROL_DATASETS / filename)
        for _, row in df.iterrows():
            options = _parse_options_cotcontrol(row)
            yield {
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "options": options,
                "dataset": source_name,
                "domain": str(row.get("domain", "")),
                # Preserve CoTControl's keyword columns for downstream CoTControl
                # eval compatibility. New sources just leave these empty.
                "valid_keywords": _stringify_keywords(row.get("valid_keywords")),
                "keywords_with_synonyms": _stringify_keywords(row.get("keywords_with_synonyms")),
                "answer_options": _stringify_options(options),
            }
    return gen


def _parse_options_cotcontrol(row: pd.Series) -> list[str]:
    for col in ("answer_options", "options"):
        raw = row.get(col)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        try:
            return list(json.loads(raw))
        except (TypeError, json.JSONDecodeError):
            try:
                return list(ast.literal_eval(raw))
            except Exception:
                pass
    return []


def _stringify_keywords(raw) -> str:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "[]"
    return str(raw)


def _stringify_options(options: list[str]) -> str:
    return json.dumps(options)


# ---------------------------------------------------------------------------
# HuggingFace-backed sources (added for v9+)
# ---------------------------------------------------------------------------

def _load_arc_challenge(limit: int | None = 3000):
    """ARC-Challenge grade-school science MCQs (multiple choice)."""
    def gen() -> Iterator[dict]:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        count = 0
        for row in ds:
            if limit is not None and count >= limit:
                break
            choices_text = list(row["choices"]["text"])
            choices_label = list(row["choices"]["label"])
            answer_key = row["answerKey"]
            # Normalize answer to an A/B/C/D letter corresponding to our options order.
            try:
                idx = choices_label.index(answer_key)
            except ValueError:
                continue
            answer_letter = chr(ord("A") + idx)
            yield {
                "question": str(row["question"]),
                "answer": answer_letter,
                "options": choices_text,
                "dataset": "arc_c",
                "domain": "grade_school_science",
                "valid_keywords": "[]",
                "keywords_with_synonyms": "[]",
                "answer_options": _stringify_options(choices_text),
            }
            count += 1
    return gen


def _load_commonsense_qa(limit: int | None = 3000):
    """CommonsenseQA: 5-way MCQ on everyday commonsense."""
    def gen() -> Iterator[dict]:
        from datasets import load_dataset
        ds = load_dataset("tau/commonsense_qa", split="train")
        count = 0
        for row in ds:
            if limit is not None and count >= limit:
                break
            choices_text = list(row["choices"]["text"])
            choices_label = list(row["choices"]["label"])
            answer_key = row["answerKey"]
            if not answer_key:
                continue  # test split has no labels; shouldn't happen in train but defensive
            try:
                idx = choices_label.index(answer_key)
            except ValueError:
                continue
            answer_letter = chr(ord("A") + idx)
            yield {
                "question": str(row["question"]),
                "answer": answer_letter,
                "options": choices_text,
                "dataset": "csqa",
                "domain": "commonsense",
                "valid_keywords": "[]",
                "keywords_with_synonyms": "[]",
                "answer_options": _stringify_options(choices_text),
            }
            count += 1
    return gen


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SOURCES: dict[str, QuestionSource] = {
    "gpqa": QuestionSource(
        name="gpqa", domain="advanced_stem",
        load=_load_cotcontrol_csv("gpqa_w_keyword.csv", "gpqa"),
    ),
    "hle": QuestionSource(
        name="hle", domain="advanced_stem",
        load=_load_cotcontrol_csv("hle_w_keyword.csv", "hle"),
    ),
    "mmlu_pro": QuestionSource(
        name="mmlu_pro", domain="advanced_stem",
        load=_load_cotcontrol_csv("mmlu_pro_mini_w_keyword.csv", "mmlu_pro"),
    ),
    "arc_c": QuestionSource(
        name="arc_c", domain="grade_school_science",
        load=_load_arc_challenge(limit=3000),
        limit=3000,
    ),
    "csqa": QuestionSource(
        name="csqa", domain="commonsense",
        load=_load_commonsense_qa(limit=3000),
        limit=3000,
    ),
}


# Default source set used when data_split.py is called without --sources.
# Keep this conservative so pre-v9 runs reproduce.
DEFAULT_SOURCES = ["gpqa", "hle", "mmlu_pro"]


def load_sources(names: list[str]) -> pd.DataFrame:
    """Load + concat rows from the named sources into a single DataFrame."""
    frames = []
    for name in names:
        if name not in SOURCES:
            raise SystemExit(f"Unknown source: {name}. Known: {list(SOURCES)}")
        rows = list(SOURCES[name].load())
        frames.append(pd.DataFrame(rows))
        print(f"  loaded {len(rows):5d}  {name:10s}  ({SOURCES[name].domain})")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Smoke-test data_sources by listing row counts.")
    p.add_argument("--sources", nargs="+", default=list(SOURCES))
    args = p.parse_args()
    df = load_sources(args.sources)
    print(f"\ntotal: {len(df)}")
    print(df["dataset"].value_counts())
