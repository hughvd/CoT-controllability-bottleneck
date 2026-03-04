"""Generate intervention evaluation scaling plots.

Produces plots of monitorability (g-mean²) vs model size and capability
for each individual eval and the mean across all evals, following
Guan et al. 2025 (arxiv 2512.18311) methodology.

Usage:
    uv run python -m analysis.plot_intervention --run-config eval_configs/intervention_full.yaml
    uv run python -m analysis.plot_intervention --run-config eval_configs/intervention_full.yaml --monitor gpt-5-mini
    uv run python -m analysis.plot_intervention --evals gpqa_hints,gpqa_encoded
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.intervention.metrics import compute_gmean2
from evals.intervention.run import EVAL_SPECS
from lib.results import load_cache

# ── Style Constants ─────────────────────────────────────────────────

EVAL_COLORS = {
    "gpqa_hints": "tab:blue",
    "gpqa_encoded": "tab:cyan",
    "aime_hints": "tab:orange",
    "daft_hints": "tab:green",
    "scruples": "tab:red",
    "wmdp": "tab:purple",
}

EVAL_MARKERS = {
    "gpqa_hints": "o",
    "gpqa_encoded": "s",
    "aime_hints": "^",
    "daft_hints": "D",
    "scruples": "v",
    "wmdp": "P",
}

EVAL_LABELS = {
    "gpqa_hints": "GPQA Hints",
    "gpqa_encoded": "GPQA Encoded",
    "aime_hints": "AIME Hints",
    "daft_hints": "DAFT Hints",
    "scruples": "Scruples",
    "wmdp": "WMDP",
}


# ── Data Loading ────────────────────────────────────────────────────

def load_run_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_control_accuracy(eval_name: str, model_name: str) -> dict:
    """Compute accuracy from the control arm cache.

    Capability = fraction of control samples where parsed_answer == correct_answer.
    """
    cache_prefix = f"intervention/{eval_name}"
    control_data = load_cache(cache_prefix, model_name, "control")

    if not control_data:
        return {"accuracy": None, "n_correct": 0, "n_total": 0, "se": 0.0}

    correct = 0
    total = 0
    for val in control_data.values():
        if not isinstance(val, dict) or "error" in val:
            continue
        parsed = val.get("parsed_answer")
        answer = val.get("correct_answer")
        if parsed is None or answer is None:
            continue
        total += 1
        if str(parsed).strip().upper() == str(answer).strip().upper():
            correct += 1

    accuracy = correct / total if total > 0 else None
    se = (
        math.sqrt(accuracy * (1 - accuracy) / total)
        if accuracy is not None and total > 1
        else 0.0
    )
    return {"accuracy": accuracy, "n_correct": correct, "n_total": total, "se": se}


def collect_all_data(
    eval_names: list[str],
    model_names: list[str],
    model_sizes: dict[str, int],
    monitor_model: str,
) -> list[dict]:
    """Collect g-mean² and capability data for all (eval, model) pairs."""
    data = []
    for eval_name in eval_names:
        if eval_name not in EVAL_SPECS:
            print(f"  Skipping unknown eval: {eval_name}")
            continue
        spec = EVAL_SPECS[eval_name]

        for model_name in model_names:
            metrics = compute_gmean2(spec, model_name, monitor_model=monitor_model)
            cap = compute_control_accuracy(eval_name, model_name)

            data.append({
                "eval_name": eval_name,
                "model_name": model_name,
                "model_size": model_sizes.get(model_name, 0),
                "gmean2": metrics["gmean2"],
                "tpr_mc": metrics.get("tpr_mc"),
                "tnr_x0": metrics.get("tnr_x0"),
                "tnr_x1": metrics.get("tnr_x1"),
                "r_effect": metrics.get("r_effect"),
                "pr_z1_x1_y1": metrics.get("pr_z1_x1_y1"),
                "n_instances": metrics.get("n_instances", 0),
                "n_dropped": metrics.get("n_dropped", 0),
                "capability": cap["accuracy"],
                "capability_se": cap["se"],
                "n_control": cap["n_total"],
            })
    return data


def compute_per_model_means(data: list[dict], model_names: list[str]) -> list[dict]:
    """Compute mean g-mean² and mean capability per model, averaging across evals."""
    means = []
    for model_name in model_names:
        model_data = [
            d for d in data
            if d["model_name"] == model_name and d["gmean2"] is not None
        ]
        if not model_data:
            continue

        gmean2_values = [d["gmean2"] for d in model_data]
        cap_values = [d["capability"] for d in model_data if d["capability"] is not None]

        mean_g = float(np.mean(gmean2_values))
        se_g = (
            float(np.std(gmean2_values, ddof=1) / np.sqrt(len(gmean2_values)))
            if len(gmean2_values) > 1 else 0.0
        )
        mean_cap = float(np.mean(cap_values)) if cap_values else None
        se_cap = (
            float(np.std(cap_values, ddof=1) / np.sqrt(len(cap_values)))
            if len(cap_values) > 1 else 0.0
        )

        means.append({
            "model_name": model_name,
            "model_size": model_data[0]["model_size"],
            "mean_gmean2": mean_g,
            "gmean2_se": se_g,
            "mean_capability": mean_cap,
            "capability_se": se_cap,
            "n_evals": len(model_data),
        })
    return means


# ── Per-Eval Plots ──────────────────────────────────────────────────

def plot_eval_monitorability_vs_size(
    data: list[dict], eval_name: str, output_dir: Path,
) -> None:
    """Plot g-mean² vs model size for a single eval."""
    valid = [d for d in data if d["gmean2"] is not None and d["model_size"] > 0]
    if not valid:
        return

    sizes = [d["model_size"] for d in valid]
    gmean2 = [d["gmean2"] for d in valid]
    names = [d["model_name"] for d in valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    color = EVAL_COLORS.get(eval_name, "tab:blue")
    marker = EVAL_MARKERS.get(eval_name, "o")

    ax.plot(sizes, gmean2, marker=marker, color=color, markersize=10,
            linewidth=2, zorder=5)

    for x, y, name in zip(sizes, gmean2, names):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (B parameters)", fontsize=12)
    ax.set_ylabel("g-mean²", fontsize=12)
    label = EVAL_LABELS.get(eval_name, eval_name)
    ax.set_title(f"Monitorability vs Size — {label}", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f"{eval_name}_vs_size.png", dpi=150)
    plt.close(fig)


def plot_eval_monitorability_vs_capability(
    data: list[dict], eval_name: str, output_dir: Path,
) -> None:
    """Plot g-mean² vs control accuracy for a single eval."""
    valid = [d for d in data if d["gmean2"] is not None and d["capability"] is not None]
    if not valid:
        return

    caps = [d["capability"] for d in valid]
    gmean2 = [d["gmean2"] for d in valid]
    names = [d["model_name"] for d in valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    color = EVAL_COLORS.get(eval_name, "tab:blue")
    marker = EVAL_MARKERS.get(eval_name, "o")

    ax.scatter(caps, gmean2, marker=marker, color=color, s=100, zorder=5)

    for x, y, name in zip(caps, gmean2, names):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(8, 8), fontsize=8, alpha=0.8)

    ax.set_xlabel("Capability (Control Accuracy)", fontsize=12)
    ax.set_ylabel("g-mean²", fontsize=12)
    label = EVAL_LABELS.get(eval_name, eval_name)
    ax.set_title(f"Monitorability vs Capability — {label}", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f"{eval_name}_vs_capability.png", dpi=150)
    plt.close(fig)


# ── Combined Multi-Eval Plots ──────────────────────────────────────

def plot_combined_monitorability_vs_size(
    data: list[dict], output_dir: Path,
) -> None:
    """Plot g-mean² vs model size, one line per eval, all on same axes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    eval_names = sorted(set(d["eval_name"] for d in data))
    for eval_name in eval_names:
        eval_data = sorted(
            [d for d in data if d["eval_name"] == eval_name
             and d["gmean2"] is not None and d["model_size"] > 0],
            key=lambda d: d["model_size"],
        )
        if not eval_data:
            continue

        sizes = [d["model_size"] for d in eval_data]
        gmean2 = [d["gmean2"] for d in eval_data]
        color = EVAL_COLORS.get(eval_name, "gray")
        marker = EVAL_MARKERS.get(eval_name, "o")
        label = EVAL_LABELS.get(eval_name, eval_name)

        ax.plot(sizes, gmean2, marker=marker, color=color, markersize=8,
                linewidth=1.5, label=label, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (B parameters)", fontsize=12)
    ax.set_ylabel("g-mean²", fontsize=12)
    ax.set_title("Monitorability vs Model Size — All Evals", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "combined_gmean2_vs_size.png", dpi=150)
    plt.close(fig)


def plot_combined_monitorability_vs_capability(
    data: list[dict], output_dir: Path,
) -> None:
    """Plot g-mean² vs capability, one series per eval, all on same axes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    eval_names = sorted(set(d["eval_name"] for d in data))
    for eval_name in eval_names:
        eval_data = [
            d for d in data if d["eval_name"] == eval_name
            and d["gmean2"] is not None and d["capability"] is not None
        ]
        if not eval_data:
            continue

        caps = [d["capability"] for d in eval_data]
        gmean2 = [d["gmean2"] for d in eval_data]
        color = EVAL_COLORS.get(eval_name, "gray")
        marker = EVAL_MARKERS.get(eval_name, "o")
        label = EVAL_LABELS.get(eval_name, eval_name)

        ax.scatter(caps, gmean2, marker=marker, color=color, s=80,
                   label=label, zorder=5)

    ax.set_xlabel("Capability (Control Accuracy)", fontsize=12)
    ax.set_ylabel("g-mean²", fontsize=12)
    ax.set_title("Monitorability vs Capability — All Evals", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "combined_gmean2_vs_capability.png", dpi=150)
    plt.close(fig)


# ── Mean Plots ──────────────────────────────────────────────────────

def plot_mean_monitorability_vs_size(
    means: list[dict], output_dir: Path,
) -> None:
    """Plot mean g-mean² (across evals) vs model size."""
    valid = sorted(
        [m for m in means if m["mean_gmean2"] is not None and m["model_size"] > 0],
        key=lambda m: m["model_size"],
    )
    if not valid:
        return

    sizes = [m["model_size"] for m in valid]
    gmean2 = [m["mean_gmean2"] for m in valid]
    se = [m["gmean2_se"] for m in valid]
    names = [m["model_name"] for m in valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sizes, gmean2, yerr=se, fmt="o-", color="tab:blue",
                capsize=5, markersize=10, linewidth=2, zorder=5)

    for x, y, name in zip(sizes, gmean2, names):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(8, 10), fontsize=9, alpha=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Model Size (B parameters)", fontsize=12)
    ax.set_ylabel("Mean g-mean² (± SEM)", fontsize=12)
    ax.set_title("Mean Monitorability vs Model Size", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "mean_gmean2_vs_size.png", dpi=150)
    plt.close(fig)


def plot_mean_monitorability_vs_capability(
    means: list[dict], output_dir: Path,
) -> None:
    """Plot mean g-mean² vs mean capability (across evals)."""
    valid = sorted(
        [m for m in means if m["mean_gmean2"] is not None and m["mean_capability"] is not None],
        key=lambda m: m["mean_capability"],
    )
    if not valid:
        return

    caps = [m["mean_capability"] for m in valid]
    gmean2 = [m["mean_gmean2"] for m in valid]
    se_g = [m["gmean2_se"] for m in valid]
    se_c = [m["capability_se"] for m in valid]
    names = [m["model_name"] for m in valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(caps, gmean2, xerr=se_c, yerr=se_g, fmt="s-", color="tab:orange",
                capsize=5, markersize=10, linewidth=2, zorder=5)

    for x, y, name in zip(caps, gmean2, names):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(8, 10), fontsize=9, alpha=0.8)

    ax.set_xlabel("Mean Capability (Control Accuracy ± SEM)", fontsize=12)
    ax.set_ylabel("Mean g-mean² (± SEM)", fontsize=12)
    ax.set_title("Mean Monitorability vs Capability", fontsize=13)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "mean_gmean2_vs_capability.png", dpi=150)
    plt.close(fig)


# ── Summary Table ───────────────────────────────────────────────────

def print_summary_table(data: list[dict], means: list[dict], monitor_model: str) -> None:
    """Print formatted console summary of all metrics."""
    print(f"\nIntervention Eval Results (monitor: {monitor_model})")
    print("=" * 100)
    header = (
        f"{'Eval':<20s} {'Model':<14s} {'Size':>5s} "
        f"{'g-mean²':>8s} {'TPR_mc':>7s} {'TNR_X0':>7s} {'TNR_X1':>7s} "
        f"{'R_eff':>6s} {'Cap':>6s} {'N':>4s}"
    )
    print(header)
    print("-" * 100)

    for d in sorted(data, key=lambda x: (x["eval_name"], x["model_size"])):
        def _fmt(v, width=7):
            return f"{v:{width}.3f}" if v is not None else f"{'—':>{width}s}"

        size_str = f"{d['model_size']}B" if d["model_size"] else "?"
        n_str = str(d["n_instances"]) if d["n_instances"] else "—"
        print(
            f"{d['eval_name']:<20s} {d['model_name']:<14s} {size_str:>5s} "
            f"{_fmt(d['gmean2'], 8)} {_fmt(d['tpr_mc'])} {_fmt(d['tnr_x0'])} {_fmt(d['tnr_x1'])} "
            f"{_fmt(d['r_effect'], 6)} {_fmt(d['capability'], 6)} {n_str:>4s}"
        )

    print(f"\n{'Per-model means':}")
    for m in means:
        se_str = f"± {m['gmean2_se']:.3f}" if m["gmean2_se"] else ""
        cap_str = (
            f"cap={m['mean_capability']:.3f} ± {m['capability_se']:.3f}"
            if m["mean_capability"] is not None else "cap=—"
        )
        print(
            f"  {m['model_name']:<14s} ({m['model_size']}B):  "
            f"g-mean²={m['mean_gmean2']:.3f} {se_str}  "
            f"{cap_str}  "
            f"({m['n_evals']} evals)"
        )


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate intervention scaling plots")
    parser.add_argument("--run-config", type=str, required=True,
                        help="Path to run configuration YAML")
    parser.add_argument("--monitor", type=str, default=None,
                        help="Monitor model ID (default: from config)")
    parser.add_argument("--evals", type=str, default=None,
                        help="Comma-separated eval names (default: all)")
    parser.add_argument("--output-dir", type=str, default="analysis/intervention_plots",
                        help="Directory for output plots")
    args = parser.parse_args()

    rc = load_run_config(args.run_config)
    model_names = rc.get("models", [])
    model_sizes = rc.get("model_sizes", {})
    monitor_cfg = rc.get("monitor", {})
    monitor_model = args.monitor or monitor_cfg.get("model", "gpt-5-mini")

    eval_names = args.evals.split(",") if args.evals else list(EVAL_SPECS.keys())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_eval_dir = output_dir / "per_eval"
    per_eval_dir.mkdir(exist_ok=True)

    # Collect data
    print("Collecting data...")
    data = collect_all_data(eval_names, model_names, model_sizes, monitor_model)
    means = compute_per_model_means(data, model_names)

    # Per-eval plots
    print("Generating per-eval plots...")
    for eval_name in eval_names:
        eval_data = [d for d in data if d["eval_name"] == eval_name]
        plot_eval_monitorability_vs_size(eval_data, eval_name, per_eval_dir)
        plot_eval_monitorability_vs_capability(eval_data, eval_name, per_eval_dir)

    # Combined plots
    print("Generating combined plots...")
    plot_combined_monitorability_vs_size(data, output_dir)
    plot_combined_monitorability_vs_capability(data, output_dir)

    # Mean plots
    print("Generating mean plots...")
    plot_mean_monitorability_vs_size(means, output_dir)
    plot_mean_monitorability_vs_capability(means, output_dir)

    # Summary table
    print_summary_table(data, means, monitor_model)

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
