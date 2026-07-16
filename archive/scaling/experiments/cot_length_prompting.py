"""Test whether system-prompt instructions can shift CoT length.

Runs each model on a sample of problems under three conditions:
  1. Baseline — no system message (same as control arm)
  2. Think more — system message encouraging thorough reasoning
  3. Think less — system message encouraging brevity

Usage:
    uv run python -m experiments.cot_length_prompting --run-config eval_configs/intervention_test.yaml
    uv run python -m experiments.cot_length_prompting --run-config eval_configs/intervention_test.yaml --n-problems 20
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.intervention.datasets import DATASET_LOADERS
from evals.intervention.prompts import (
    format_free_response_control,
    format_mcq_control,
    format_scruples_control,
    format_wmdp_control,
)
from lib.providers.openrouter import async_complete

# ── Prompt conditions ───────────────────────────────────────────────

THINK_MORE_PROMPT = (
    "Before answering, reason through this problem very thoroughly. "
    "Explore multiple approaches, check your work, and consider edge cases. "
    "Take as long as you need in your thinking."
)

THINK_LESS_PROMPT = (
    "Be efficient in your reasoning. Get to the answer quickly without "
    "unnecessary deliberation. Keep your thinking brief and direct."
)

CONDITIONS = {
    "baseline": None,           # no system message
    "think_more": THINK_MORE_PROMPT,
    "think_less": THINK_LESS_PROMPT,
}

CONDITION_COLORS = {
    "baseline": "tab:blue",
    "think_more": "tab:green",
    "think_less": "tab:red",
}

# Map eval name prefix → control prompt formatter
EVAL_FORMATTERS = {
    "gpqa": format_mcq_control,
    "aime": format_free_response_control,
    "daft": format_free_response_control,
    "scruples": format_scruples_control,
    "wmdp": format_wmdp_control,
}

# Map eval name prefix → dataset loader key
EVAL_DATASET_KEYS = {
    "gpqa": "gpqa",
    "aime": "aime",
    "daft": "daft",
    "scruples": "scruples",
    "wmdp": "wmdp",
}


# ── Model config ────────────────────────────────────────────────────

def load_model_ids(config_path: str = "models_config.yaml") -> dict[str, str]:
    """Load name → model_id mapping from models_config.yaml."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return {m["name"]: m["model_id"] for m in cfg.get("models", [])}


# ── Core experiment ─────────────────────────────────────────────────

async def run_single(
    model_id: str, messages: list[dict],
    temperature: float, max_tokens: int,
) -> dict:
    """Run a single model call and return CoT stats."""
    try:
        resp = await async_complete(
            model_id, messages,
            temperature=temperature, max_tokens=max_tokens,
        )
        return {
            "cot_length": len(resp.thinking) if resp.thinking else 0,
            "answer_text": resp.answer or "",
            "thinking": resp.thinking or "",
            "error": None,
        }
    except Exception as e:
        return {
            "cot_length": 0,
            "answer_text": "",
            "thinking": "",
            "error": str(e),
        }


def _save_cache(cache: dict, cache_path: Path) -> None:
    """Save cache dict to JSON."""
    entries = list(cache.values())
    with open(cache_path, "w") as f:
        json.dump(entries, f, indent=2, default=str)


async def run_experiment(
    model_names: list[str],
    model_ids: dict[str, str],
    eval_names: list[str],
    n_problems: int,
    temperature: float,
    max_tokens: int,
    cache: dict[str, dict],
    cache_path: Path,
    concurrency: int = 5,
) -> list[dict]:
    """Run all (model, dataset, condition, problem) combinations."""
    sem = asyncio.Semaphore(concurrency)
    results = []

    # Load and sample problems for each eval
    problems_by_eval = {}
    for eval_name in eval_names:
        ds_key = EVAL_DATASET_KEYS.get(eval_name)
        loader = DATASET_LOADERS.get(ds_key)
        if not loader:
            print(f"  Skipping {eval_name}: no dataset loader for '{ds_key}'")
            continue
        all_probs = loader(n_problems)
        problems_by_eval[eval_name] = all_probs
        print(f"  {eval_name}: {len(all_probs)} problems")

    completed_count = 0
    save_lock = asyncio.Lock()

    async def _run_one(model_name, model_id, eval_name, condition, prob):
        nonlocal completed_count
        pid = prob.get("id", "?")
        cache_key = f"{model_name}|{eval_name}|{condition}|{pid}"

        # Check cache
        if cache_key in cache:
            return cache[cache_key]

        # Build messages
        formatter = EVAL_FORMATTERS.get(eval_name)
        if not formatter:
            return None
        base_msgs = formatter(prob)

        sys_prompt = CONDITIONS[condition]
        if sys_prompt:
            messages = [{"role": "system", "content": sys_prompt}] + base_msgs
        else:
            messages = base_msgs

        async with sem:
            result = await run_single(model_id, messages, temperature, max_tokens)

        result.update({
            "model": model_name,
            "dataset": eval_name,
            "condition": condition,
            "problem_id": pid,
        })

        # Don't store full thinking in cache (save space)
        cache_entry = {k: v for k, v in result.items() if k != "thinking"}
        cache[cache_key] = cache_entry

        # Incremental save every 10 completions
        async with save_lock:
            completed_count += 1
            if completed_count % 10 == 0:
                _save_cache(cache, cache_path)
                print(f"  [checkpoint] {completed_count} done, {len(cache)} cached")

        status = f"len={result['cot_length']:,}" if not result["error"] else f"ERR: {result['error'][:50]}"
        print(f"    {model_name}/{eval_name}/{condition}/p{pid}: {status}")
        return result

    # Launch all models in parallel (each model routes to a different provider)
    tasks = []
    for model_name in model_names:
        model_id = model_ids.get(model_name)
        if not model_id:
            print(f"  Skipping {model_name}: no model_id found")
            continue
        for eval_name in eval_names:
            if eval_name not in problems_by_eval:
                continue
            for condition in CONDITIONS:
                for prob in problems_by_eval[eval_name]:
                    tasks.append(_run_one(model_name, model_id, eval_name, condition, prob))

    print(f"\nRunning {len(tasks)} total calls (concurrency={concurrency})...")
    all_results = await asyncio.gather(*tasks)
    all_results = [r for r in all_results if r is not None]

    # Save cache
    _save_cache(cache, cache_path)
    print(f"  Done: {len(all_results)} results, cache saved ({len(cache)} total)")

    return all_results


# ── Analysis & Plotting ─────────────────────────────────────────────

def compute_summary(results: list[dict]) -> list[dict]:
    """Compute mean/std CoT length per (dataset, model, condition)."""
    groups = {}
    for r in results:
        if r.get("error"):
            continue
        key = (r["dataset"], r["model"], r["condition"])
        groups.setdefault(key, []).append(r["cot_length"])

    rows = []
    for (dataset, model, condition), lengths in sorted(groups.items()):
        a = np.array(lengths)
        rows.append({
            "dataset": dataset,
            "model": model,
            "condition": condition,
            "n": len(a),
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "median": float(np.median(a)),
            "min": int(np.min(a)),
            "max": int(np.max(a)),
        })
    return rows


def plot_comparison(
    summary: list[dict], title: str, output_path: Path,
    filter_dataset: str | None = None,
) -> None:
    """Grouped bar chart: mean CoT length per condition, grouped by model."""
    if filter_dataset:
        summary = [s for s in summary if s["dataset"] == filter_dataset]

    if not summary:
        return

    # Group by model
    models = sorted(set(s["model"] for s in summary))
    conditions = ["baseline", "think_more", "think_less"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.25

    for i, cond in enumerate(conditions):
        means = []
        stds = []
        for model in models:
            rows = [s for s in summary if s["model"] == model and s["condition"] == cond]
            if rows:
                # Average across datasets if not filtered
                mean_val = np.mean([r["mean"] for r in rows])
                # Propagate std as pooled
                std_val = np.mean([r["std"] for r in rows])
                means.append(mean_val)
                stds.append(std_val)
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + i * width, means, width, yerr=stds, label=cond,
               color=CONDITION_COLORS[cond], alpha=0.8, capsize=3)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Mean CoT Length (characters)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CoT length prompting experiment")
    parser.add_argument("--run-config", type=str, required=True)
    parser.add_argument("--n-problems", type=int, default=15)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    with open(args.run_config) as f:
        rc = yaml.safe_load(f)

    model_names = rc.get("models", [])
    sampling = rc.get("sampling", {})
    temperature = sampling.get("temperature", 0.6)
    max_tokens = sampling.get("max_tokens", 32768)

    model_ids = load_model_ids()
    eval_names = list(EVAL_FORMATTERS.keys())

    output_dir = Path("experiments/cot_length_prompting")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_dataset").mkdir(exist_ok=True)

    # Load cache
    cache_path = output_dir / "results.json"
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cached_list = json.load(f)
        for entry in cached_list:
            key = f"{entry['model']}|{entry['dataset']}|{entry['condition']}|{entry['problem_id']}"
            cache[key] = entry
        print(f"Loaded {len(cache)} cached results")

    # Run experiment
    print(f"Loading {args.n_problems} problems per dataset...")
    results = asyncio.run(run_experiment(
        model_names, model_ids, eval_names, args.n_problems,
        temperature, max_tokens, cache, cache_path, args.concurrency,
    ))

    if not results:
        print("No results.")
        return

    # Final cache save
    _save_cache(cache, cache_path)
    print(f"\nSaved {len(cache)} results to {cache_path}")

    # Summary
    summary = compute_summary(results)

    csv_path = output_dir / "summary.csv"
    fields = ["dataset", "model", "condition", "n", "mean", "std", "median", "min", "max"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"Saved summary to {csv_path}")

    # Plots
    plot_comparison(
        summary,
        "CoT Length by Prompting Condition — All Datasets",
        output_dir / "comparison.png",
    )

    for eval_name in eval_names:
        plot_comparison(
            summary,
            f"CoT Length by Prompting Condition — {eval_name}",
            output_dir / "per_dataset" / f"{eval_name}_comparison.png",
            filter_dataset=eval_name,
        )

    # Print table
    print(f"\n{'Dataset':<12s} {'Model':<28s} {'Condition':<12s} {'N':>4s} "
          f"{'Mean':>10s} {'Std':>10s} {'Median':>10s}")
    print("-" * 90)
    for s in summary:
        print(f"{s['dataset']:<12s} {s['model']:<28s} {s['condition']:<12s} "
              f"{s['n']:>4d} {s['mean']:>10,.0f} {s['std']:>10,.0f} "
              f"{s['median']:>10,.0f}")

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
