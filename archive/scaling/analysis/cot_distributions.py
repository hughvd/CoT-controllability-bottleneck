"""Analyze CoT (chain-of-thought) length distributions across models and datasets.

Produces summary statistics and histograms showing how reasoning length
varies by model and evaluation dataset.

Usage:
    uv run python -m analysis.cot_distributions --run-config eval_configs/intervention_test.yaml
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.results import load_cache

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "intervention"

# Colors for overlaid histograms
MODEL_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
]

EVAL_COLORS = {
    "gpqa_hints": "tab:blue",
    "gpqa_encoded": "tab:cyan",
    "aime_hints": "tab:orange",
    "daft_hints": "tab:green",
    "scruples": "tab:red",
    "wmdp": "tab:purple",
}


# ── Data Loading ────────────────────────────────────────────────────

def load_cot_lengths(eval_name: str, model_name: str) -> list[int]:
    """Load thinking lengths from both control and intervention caches."""
    lengths = []
    for stage in ("control", "intervention"):
        data = load_cache(f"intervention/{eval_name}", model_name, stage)
        for val in data.values():
            if not isinstance(val, dict) or "error" in val:
                continue
            thinking = val.get("thinking")
            if thinking:
                lengths.append(len(thinking))
    return lengths


def collect_all_lengths(
    eval_names: list[str], model_names: list[str],
) -> dict[tuple[str, str], list[int]]:
    """Collect CoT lengths for all (eval, model) pairs."""
    all_data = {}
    for eval_name in eval_names:
        for model_name in model_names:
            lengths = load_cot_lengths(eval_name, model_name)
            if lengths:
                all_data[(eval_name, model_name)] = lengths
    return all_data


def compute_stats(lengths: list[int]) -> dict:
    """Compute summary statistics for a list of lengths."""
    if not lengths:
        return {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0,
                "p25": 0, "p75": 0, "n": 0}
    a = np.array(lengths)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "min": int(np.min(a)),
        "max": int(np.max(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "n": len(a),
    }


# ── Per-Question Variability ────────────────────────────────────────

def load_per_question_lengths(eval_name: str, model_name: str) -> dict[str, list[int]]:
    """Load thinking lengths grouped by problem_id from both arms."""
    by_question: dict[str, list[int]] = defaultdict(list)
    for stage in ("control", "intervention"):
        data = load_cache(f"intervention/{eval_name}", model_name, stage)
        for val in data.values():
            if not isinstance(val, dict) or "error" in val:
                continue
            thinking = val.get("thinking")
            pid = val.get("problem_id")
            if thinking and pid is not None:
                by_question[str(pid)].append(len(thinking))
    return dict(by_question)


def compute_per_question_variability(
    eval_name: str, model_name: str,
) -> list[dict]:
    """Compute within-question CoT length variability metrics.

    For each question, returns range_ratio, CV, and log_range (OOM spanned).
    """
    by_question = load_per_question_lengths(eval_name, model_name)
    results = []
    for pid, lengths in sorted(by_question.items()):
        if len(lengths) < 2:
            continue
        a = np.array(lengths, dtype=float)
        mn, mx, mean, std = a.min(), a.max(), a.mean(), a.std(ddof=1)
        if mean == 0 or mn == 0:
            continue
        results.append({
            "problem_id": pid,
            "n": len(lengths),
            "mean": float(mean),
            "std": float(std),
            "min": int(mn),
            "max": int(mx),
            "range_ratio": float((mx - mn) / mean),
            "cv": float(std / mean),
            "log_range": float(math.log10(mx) - math.log10(mn)),
        })
    return results


def collect_variability_data(
    eval_names: list[str], model_names: list[str],
) -> dict[tuple[str, str], list[dict]]:
    """Collect per-question variability for all (eval, model) pairs."""
    all_var = {}
    for eval_name in eval_names:
        for model_name in model_names:
            results = compute_per_question_variability(eval_name, model_name)
            if results:
                all_var[(eval_name, model_name)] = results
    return all_var


def save_variability_csv(
    var_data: dict[tuple[str, str], list[dict]], output_path: Path,
) -> None:
    """Save per-(eval, model) mean variability metrics to CSV."""
    fields = ["eval", "model", "n_questions", "mean_cv", "mean_log_range",
              "mean_range_ratio", "median_log_range"]
    rows = []
    for (eval_name, model_name), questions in sorted(var_data.items()):
        cvs = [q["cv"] for q in questions]
        log_ranges = [q["log_range"] for q in questions]
        range_ratios = [q["range_ratio"] for q in questions]
        rows.append({
            "eval": eval_name,
            "model": model_name,
            "n_questions": len(questions),
            "mean_cv": float(np.mean(cvs)),
            "mean_log_range": float(np.mean(log_ranges)),
            "mean_range_ratio": float(np.mean(range_ratios)),
            "median_log_range": float(np.median(log_ranges)),
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_variability_bar(
    model_means: list[tuple[str, float]],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Bar chart of a variability metric per model."""
    if not model_means:
        return
    names, values = zip(*model_means)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    ax.bar(x, values, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variability_histogram(
    values: list[float], title: str, xlabel: str, output_path: Path,
) -> None:
    """Histogram of per-question variability values."""
    if not values:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=30, alpha=0.6, color="tab:blue",
            edgecolor="white", linewidth=0.3)
    ax.axvline(np.mean(values), color="red", linestyle="--", linewidth=1,
               label=f"mean={np.mean(values):.2f}")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count (questions)", fontsize=11)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Plotting ────────────────────────────────────────────────────────

def _histogram(ax, lengths, color="tab:blue", label=None, bins=50, xlim=None):
    """Draw a single histogram on the given axes."""
    if not lengths:
        return
    range_ = (0, xlim) if xlim else None
    ax.hist(lengths, bins=bins, alpha=0.6, color=color, label=label,
            edgecolor="white", linewidth=0.3, range=range_)


def plot_single_histogram(
    lengths: list[int], title: str, output_path: Path, xlim=None,
) -> None:
    """Plot a single histogram with stats in the title."""
    stats = compute_stats(lengths)
    fig, ax = plt.subplots(figsize=(8, 5))
    _histogram(ax, lengths, xlim=xlim)
    ax.set_xlabel("CoT Length (characters)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{title}\nmean={stats['mean']:,.0f} ± {stats['std']:,.0f}  "
                 f"median={stats['median']:,.0f}  n={stats['n']}", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlaid_histogram(
    series: list[tuple[str, list[int]]],
    title: str,
    output_path: Path,
    colors: list[str] | None = None,
    xlim=None,
) -> None:
    """Plot multiple overlaid histograms with legend."""
    if not series:
        return
    if colors is None:
        colors = MODEL_COLORS

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, lengths) in enumerate(series):
        if lengths:
            _histogram(ax, lengths, color=colors[i % len(colors)],
                       label=label, xlim=xlim)

    ax.set_xlabel("CoT Length (characters)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Summary CSV ─────────────────────────────────────────────────────

def save_summary_csv(
    all_data: dict[tuple[str, str], list[int]], output_path: Path,
) -> None:
    """Save per-(eval, model) summary stats to CSV."""
    fields = ["eval", "model", "n", "mean", "std", "median", "p25", "p75", "min", "max"]
    rows = []
    for (eval_name, model_name), lengths in sorted(all_data.items()):
        stats = compute_stats(lengths)
        rows.append({
            "eval": eval_name,
            "model": model_name,
            **{k: stats[k] for k in fields[2:]},
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CoT length distribution analysis")
    parser.add_argument("--run-config", type=str, required=True,
                        help="Path to run configuration YAML")
    args = parser.parse_args()

    with open(args.run_config) as f:
        rc = yaml.safe_load(f)
    model_names = rc.get("models", [])

    # Discover evals from results directory
    eval_names = sorted(
        d.name for d in RESULTS_DIR.iterdir() if d.is_dir()
    )

    output_dir = Path("analysis/cot_distributions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all data
    print("Collecting CoT lengths...")
    all_data = collect_all_lengths(eval_names, model_names)
    print(f"  Found data for {len(all_data)} (eval, model) pairs")

    if not all_data:
        print("No data found. Check that models in config match results directory.")
        return

    # Save summary CSV
    csv_path = output_dir / "summary_stats.csv"
    save_summary_csv(all_data, csv_path)
    print(f"  Saved {csv_path}")

    # ── Total histogram (all models, all datasets) ──────────────────
    all_lengths = []
    for lengths in all_data.values():
        all_lengths.extend(lengths)

    plot_single_histogram(
        all_lengths,
        "CoT Length Distribution — All Models, All Datasets",
        output_dir / "total_histogram.png",
    )

    total_stats = compute_stats(all_lengths)
    with open(output_dir / "total_stats.json", "w") as f:
        json.dump(total_stats, f, indent=2)

    # ── Per-model histograms (across all datasets) ──────────────────
    for model_name in model_names:
        model_series = []
        for eval_name in eval_names:
            key = (eval_name, model_name)
            if key in all_data:
                model_series.append((eval_name, all_data[key]))

        if not model_series:
            continue

        eval_colors = [EVAL_COLORS.get(e, "gray") for e, _ in model_series]
        plot_overlaid_histogram(
            model_series,
            f"CoT Length Distribution — {model_name}",
            output_dir / f"{model_name}_histogram.png",
            colors=eval_colors,
        )

    # ── Per-dataset subfolders ──────────────────────────────────────
    for eval_name in eval_names:
        eval_dir = output_dir / "per_dataset" / eval_name
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Gather all lengths for this dataset to set consistent xlim
        dataset_lengths = []
        model_series = []
        for model_name in model_names:
            key = (eval_name, model_name)
            if key in all_data:
                model_series.append((model_name, all_data[key]))
                dataset_lengths.extend(all_data[key])

        if not dataset_lengths:
            continue

        xlim = int(np.percentile(dataset_lengths, 99) * 1.1)

        # Total for this dataset (all models)
        plot_overlaid_histogram(
            model_series,
            f"CoT Length Distribution — {eval_name} (all models)",
            eval_dir / "total_histogram.png",
            xlim=xlim,
        )

        # Per-model for this dataset
        for model_name, lengths in model_series:
            plot_single_histogram(
                lengths,
                f"CoT Length — {eval_name} / {model_name}",
                eval_dir / f"{model_name}_histogram.png",
                xlim=xlim,
            )

    # ── Per-question variability analysis ──────────────────────────────
    print("\nComputing per-question variability...")
    var_data = collect_variability_data(eval_names, model_names)
    print(f"  Found variability data for {len(var_data)} (eval, model) pairs")

    if var_data:
        var_csv = output_dir / "variability_summary.csv"
        save_variability_csv(var_data, var_csv)
        print(f"  Saved {var_csv}")

        # Per-dataset variability plots
        for eval_name in eval_names:
            eval_dir = output_dir / "per_dataset" / eval_name
            eval_dir.mkdir(parents=True, exist_ok=True)

            bar_data = []
            for model_name in model_names:
                key = (eval_name, model_name)
                if key not in var_data:
                    continue
                questions = var_data[key]
                log_ranges = [q["log_range"] for q in questions]
                bar_data.append((model_name, float(np.mean(log_ranges))))

                # Per-model variability histogram
                plot_variability_histogram(
                    log_ranges,
                    f"Per-Question Log Range — {eval_name} / {model_name}\n"
                    f"(OOM spanned per question, {len(questions)} questions)",
                    "log₁₀(max/min) per question",
                    eval_dir / f"{model_name}_variability.png",
                )

            # Bar chart: mean log_range per model for this dataset
            if bar_data:
                plot_variability_bar(
                    bar_data,
                    f"Mean Within-Question OOM Range — {eval_name}",
                    "Mean log₁₀(max/min)",
                    eval_dir / "variability_summary.png",
                )

    # Print summary tables
    print(f"\n{'Eval':<16s} {'Model':<28s} {'N':>6s} {'Mean':>10s} {'Std':>10s} "
          f"{'Median':>10s} {'Min':>8s} {'Max':>10s}")
    print("-" * 100)
    for (eval_name, model_name), lengths in sorted(all_data.items()):
        s = compute_stats(lengths)
        print(f"{eval_name:<16s} {model_name:<28s} {s['n']:>6d} {s['mean']:>10,.0f} "
              f"{s['std']:>10,.0f} {s['median']:>10,.0f} {s['min']:>8,d} {s['max']:>10,d}")

    if var_data:
        print(f"\n{'Eval':<16s} {'Model':<28s} {'N_qs':>6s} {'Mean CV':>9s} "
              f"{'Mean LogR':>10s} {'Med LogR':>10s} {'Mean RR':>9s}")
        print("-" * 90)
        for (eval_name, model_name), questions in sorted(var_data.items()):
            cvs = [q["cv"] for q in questions]
            log_ranges = [q["log_range"] for q in questions]
            range_ratios = [q["range_ratio"] for q in questions]
            print(f"{eval_name:<16s} {model_name:<28s} {len(questions):>6d} "
                  f"{np.mean(cvs):>9.3f} {np.mean(log_ranges):>10.3f} "
                  f"{np.median(log_ranges):>10.3f} {np.mean(range_ratios):>9.3f}")

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
