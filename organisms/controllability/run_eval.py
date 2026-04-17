"""Run CoT-Control eval suite against a local vLLM endpoint.

Iterates over (dataset, mode, split), calls CoTControl's run_cceval.py to
generate responses, then grade_compliance_csv.py to grade them. Aggregates
per-run numbers into results/<exp_name>.csv.

Assumes a vLLM OpenAI-compatible server is running at VLLM_BASE_URL (default
http://localhost:8000/v1) and serving the target model under --model.

Usage:
    python run_eval.py --model qwen3-14b-base --exp-name v0_baseline --baseline-only
    python run_eval.py --model organism_2_v1 --exp-name organism_2_v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
COTCONTROL_QA = HERE / "CoTControl" / "CoT-Control-QA"
SPLIT_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"

ALL_MODES = [
    "baseline",
    "uppercase_thinking",
    "lowercase_thinking",
    "alternating_case",
    "word_suppression",
    "multiple_word_suppression",
    "end_of_sentence",
    "meow_between_words",
    "repeat_sentences",
    "ignore_question",
]

DATASETS = {
    "gpqa": "gpqa_w_keyword.csv",
    "hle": "hle_w_keyword.csv",
    "mmlu_pro": "mmlu_pro_mini_w_keyword.csv",
}


def filter_split(source_csv: Path, split_csv: Path, out_csv: Path, dataset_tag: str) -> Path:
    """Write a subset of the CoT-Control dataset containing only rows whose
    question appears in the split (per dataset). CoTControl reads from
    {name}_w_keyword.csv — so we produce that filename shape in a temp dir.
    """
    src = pd.read_csv(source_csv)
    split = pd.read_csv(split_csv)
    split = split[split["dataset"] == dataset_tag]
    keep_questions = set(split["question"].astype(str).tolist())
    sub = src[src["question"].astype(str).isin(keep_questions)].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    return out_csv


def run_cceval(
    data_file: Path, model: str, mode: str, log_dir: Path,
    max_tokens: int, max_concurrency: int, backend: str,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(COTCONTROL_QA / "run_cceval.py"),
        "--data", str(data_file),
        "--model", model,
        "--mode", mode,
        "--log_dir", str(log_dir.resolve()),
        "--max_tokens", str(max_tokens),
        "--max_concurrency", str(max_concurrency),
        "--backend", backend,
    ]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=COTCONTROL_QA)

    # run_cceval writes to logs/{log_dir} under cwd. We passed an absolute path
    # so it wrote directly there. Find the freshest matching CSV.
    csvs = sorted(log_dir.glob(f"*{_sanitize(model)}*{mode}*.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise RuntimeError(f"No output CSV found under {log_dir}")
    return csvs[-1]


def run_grade(result_csv: Path, skip_meta: bool = False) -> Path:
    cmd = [
        sys.executable, str(COTCONTROL_QA / "grade_compliance_csv.py"),
        "--input_file", str(result_csv),
    ]
    if skip_meta:
        cmd.append("--skip_meta_discussion")
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=COTCONTROL_QA)
    # grade_compliance_csv.py writes to {parent}/graded_results/{stem}_graded.csv
    graded = result_csv.parent / "graded_results" / (result_csv.stem + "_graded.csv")
    if not graded.exists():
        # Fall back to sibling location for older script versions.
        alt = result_csv.with_name(result_csv.stem + "_graded.csv")
        if alt.exists():
            return alt
        raise RuntimeError(f"Graded CSV not produced: {graded}")
    return graded


def summarize(graded_csv: Path, dataset: str, mode: str, split: str, model: str) -> dict:
    df = pd.read_csv(graded_csv)
    n = len(df)
    acc = df["correctness"].mean() if "correctness" in df.columns and n else float("nan")
    comp_col = "compliance" if "compliance" in df.columns else "compliant"
    meta_col = "meta_discussion" if "meta_discussion" in df.columns else None
    controllability = df[comp_col].mean() if comp_col in df.columns and n else float("nan")
    meta_rate = df[meta_col].mean() if meta_col and n else float("nan")
    return {
        "model": model,
        "dataset": dataset,
        "mode": mode,
        "split": split,
        "n": n,
        "accuracy": acc,
        "controllability": controllability,
        "meta_discussion_rate": meta_rate,
        "graded_csv": str(graded_csv),
    }


def _sanitize(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_]+", "_", name.replace("/", "_"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="Served model name on vLLM (e.g. 'qwen3-14b-base').")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--modes", nargs="+", default=ALL_MODES,
                        choices=ALL_MODES)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("--splits", nargs="+", default=["test"],
                        choices=["train", "test"])
    parser.add_argument("--baseline-only", action="store_true",
                        help="Shortcut: only run --mode baseline")
    parser.add_argument("--skip-meta-grading", action="store_true",
                        help="Skip meta-discussion LLM judge (saves OpenAI spend)")
    parser.add_argument("--backend", default="vllm",
                        choices=["openrouter", "vllm"])
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-concurrency", type=int, default=32)
    args = parser.parse_args()

    if args.baseline_only:
        args.modes = ["baseline"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = RESULTS_DIR / args.exp_name / "raw"
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    subset_dir = log_dir / "subsets"
    for split in args.splits:
        split_csv = SPLIT_DIR / f"split_{split}.csv"
        if not split_csv.exists():
            raise FileNotFoundError(f"Missing {split_csv}; run data_split.py first")
        for ds in args.datasets:
            src = COTCONTROL_QA / "datasets" / DATASETS[ds]
            subset = subset_dir / f"{split}_{ds}_w_keyword.csv"
            if not subset.exists():
                filter_split(src, split_csv, subset, ds)
            for mode in args.modes:
                try:
                    out = run_cceval(
                        subset, args.model, mode, log_dir,
                        max_tokens=args.max_tokens,
                        max_concurrency=args.max_concurrency,
                        backend=args.backend,
                    )
                    graded = run_grade(out, skip_meta=args.skip_meta_grading)
                    rows.append(summarize(graded, ds, mode, split, args.model))
                except subprocess.CalledProcessError as exc:
                    print(f"FAILED on {ds}/{mode}/{split}: {exc}")
                    rows.append({
                        "model": args.model, "dataset": ds, "mode": mode,
                        "split": split, "n": 0, "accuracy": float("nan"),
                        "controllability": float("nan"),
                        "meta_discussion_rate": float("nan"),
                        "graded_csv": "",
                    })

    summary = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"{args.exp_name}.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSummary written to {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
