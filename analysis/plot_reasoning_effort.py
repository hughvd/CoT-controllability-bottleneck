"""Generate plots of monitorability and capability vs reasoning effort.

Plots g-mean² and control accuracy as a function of reasoning effort
(low, medium, high) for gpt-oss models.

Usage:
    uv run python -m analysis.plot_reasoning_effort --run-config eval_configs/intervention_reasoning_effort.yaml
    uv run python -m analysis.plot_reasoning_effort --run-config eval_configs/intervention_reasoning_effort.yaml --monitor gpt-5-mini
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

# ── Constants ──────────────────────────────────────────────────────

EFFORT_LEVELS = ["low", "medium", "high"]
EFFORT_X = {level: i for i, level in enumerate(EFFORT_LEVELS)}

MODEL_COLORS = {
    "gpt-oss-20b": "tab:blue",
    "gpt-oss-120b": "tab:orange",
}

MODEL_MARKERS = {
    "gpt-oss-20b": "o",
    "gpt-oss-120b": "s",
}

EVAL_LABELS = {
    "gpqa_hints": "GPQA Hints",
    "gpqa_encoded": "GPQA Encoded",
    "aime_hints": "AIME Hints",
    "daft_hints": "DAFT Hints",
    "scruples": "Scruples",
    "wmdp": "WMDP",
}


# ── Data Loading ───────────────────────────────────────────────────

def load_run_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_control_accuracy(eval_name: str, model_name: str) -> dict:
    """Compute accuracy from the control arm cache."""
    cache_prefix = f"intervention/{eval_name}"
    control_data = load_cache(cache_prefix, model_name, "control")

    if not control_data:
        return {"accuracy": None, "n_total": 0, "se": 0.0}

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
    return {"accuracy": accuracy, "n_total": total, "se": se}


def collect_data(
    eval_names: list[str],
    model_names: list[str],
    monitor_model: str,
) -> list[dict]:
    """Collect g-mean² and capability for all (eval, model, effort) triples."""
    data = []
    for eval_name in eval_names:
        if eval_name not in EVAL_SPECS:
            print(f"  Skipping unknown eval: {eval_name}")
            continue
        spec = EVAL_SPECS[eval_name]

        for model_name in model_names:
            for effort in EFFORT_LEVELS:
                cache_name = f"{model_name}_effort_{effort}"
                metrics = compute_gmean2(spec, cache_name, monitor_model=monitor_model)
                cap = compute_control_accuracy(eval_name, cache_name)

                data.append({
                    "eval_name": eval_name,
                    "model_name": model_name,
                    "effort": effort,
                    "gmean2": metrics["gmean2"],
                    "tpr_mc": metrics.get("tpr_mc"),
                    "tnr_x0": metrics.get("tnr_x0"),
                    "tnr_x1": metrics.get("tnr_x1"),
                    "r_effect": metrics.get("r_effect"),
                    "n_instances": metrics.get("n_instances", 0),
                    "capability": cap["accuracy"],
                    "capability_se": cap["se"],
                    "n_control": cap["n_total"],
                })
    return data


def compute_per_model_effort_means(
    data: list[dict], model_names: list[str],
) -> list[dict]:
    """Compute mean g-mean² and capability per (model, effort), averaged across evals."""
    means = []
    for model_name in model_names:
        for effort in EFFORT_LEVELS:
            items = [
                d for d in data
                if d["model_name"] == model_name
                and d["effort"] == effort
                and d["gmean2"] is not None
            ]
            if not items:
                continue

            gmean2_vals = [d["gmean2"] for d in items]
            cap_vals = [d["capability"] for d in items if d["capability"] is not None]

            mean_g = float(np.mean(gmean2_vals))
            se_g = (
                float(np.std(gmean2_vals, ddof=1) / np.sqrt(len(gmean2_vals)))
                if len(gmean2_vals) > 1 else 0.0
            )
            mean_cap = float(np.mean(cap_vals)) if cap_vals else None
            se_cap = (
                float(np.std(cap_vals, ddof=1) / np.sqrt(len(cap_vals)))
                if len(cap_vals) > 1 else 0.0
            )

            means.append({
                "model_name": model_name,
                "effort": effort,
                "mean_gmean2": mean_g,
                "gmean2_se": se_g,
                "mean_capability": mean_cap,
                "capability_se": se_cap,
                "n_evals": len(items),
            })
    return means


# ── Plots ──────────────────────────────────────────────────────────

def plot_mean_gmean2_vs_effort(means: list[dict], output_dir: Path) -> None:
    """Mean g-mean² vs reasoning effort, one line per model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in sorted(set(m["model_name"] for m in means)):
        model_means = sorted(
            [m for m in means if m["model_name"] == model_name],
            key=lambda m: EFFORT_X[m["effort"]],
        )
        if not model_means:
            continue

        xs = [EFFORT_X[m["effort"]] for m in model_means]
        ys = [m["mean_gmean2"] for m in model_means]
        ses = [m["gmean2_se"] for m in model_means]
        color = MODEL_COLORS.get(model_name, "gray")
        marker = MODEL_MARKERS.get(model_name, "o")

        ax.errorbar(xs, ys, yerr=ses, fmt=f"-{marker}", color=color,
                     label=model_name, capsize=5, linewidth=2, markersize=8)

    ax.set_xticks(list(EFFORT_X.values()))
    ax.set_xticklabels(EFFORT_LEVELS)
    ax.set_xlabel("Reasoning Effort", fontsize=12)
    ax.set_ylabel("Mean g-mean² (± SEM across evals)", fontsize=12)
    ax.set_title("Monitorability vs Reasoning Effort", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "mean_gmean2_vs_effort.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mean_capability_vs_effort(means: list[dict], output_dir: Path) -> None:
    """Mean capability vs reasoning effort, one line per model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name in sorted(set(m["model_name"] for m in means)):
        model_means = sorted(
            [m for m in means if m["model_name"] == model_name
             and m["mean_capability"] is not None],
            key=lambda m: EFFORT_X[m["effort"]],
        )
        if not model_means:
            continue

        xs = [EFFORT_X[m["effort"]] for m in model_means]
        ys = [m["mean_capability"] for m in model_means]
        ses = [m["capability_se"] for m in model_means]
        color = MODEL_COLORS.get(model_name, "gray")
        marker = MODEL_MARKERS.get(model_name, "o")

        ax.errorbar(xs, ys, yerr=ses, fmt=f"-{marker}", color=color,
                     label=model_name, capsize=5, linewidth=2, markersize=8)

    ax.set_xticks(list(EFFORT_X.values()))
    ax.set_xticklabels(EFFORT_LEVELS)
    ax.set_xlabel("Reasoning Effort", fontsize=12)
    ax.set_ylabel("Mean Capability (Control Accuracy ± SEM)", fontsize=12)
    ax.set_title("Capability vs Reasoning Effort", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "mean_capability_vs_effort.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_eval_gmean2_vs_effort(
    data: list[dict], eval_names: list[str], output_dir: Path,
) -> None:
    """Per-eval subplots: g-mean² vs reasoning effort."""
    n_evals = len(eval_names)
    if n_evals == 0:
        return

    ncols = min(3, n_evals)
    nrows = math.ceil(n_evals / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, eval_name in enumerate(eval_names):
        ax = axes[idx // ncols][idx % ncols]
        eval_data = [d for d in data if d["eval_name"] == eval_name]

        for model_name in sorted(set(d["model_name"] for d in eval_data)):
            model_data = sorted(
                [d for d in eval_data
                 if d["model_name"] == model_name and d["gmean2"] is not None],
                key=lambda d: EFFORT_X[d["effort"]],
            )
            if not model_data:
                continue

            xs = [EFFORT_X[d["effort"]] for d in model_data]
            ys = [d["gmean2"] for d in model_data]
            color = MODEL_COLORS.get(model_name, "gray")
            marker = MODEL_MARKERS.get(model_name, "o")

            ax.plot(xs, ys, f"-{marker}", color=color, label=model_name,
                    linewidth=1.5, markersize=7)

        ax.set_xticks(list(EFFORT_X.values()))
        ax.set_xticklabels(EFFORT_LEVELS)
        ax.set_title(EVAL_LABELS.get(eval_name, eval_name), fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_evals, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.supxlabel("Reasoning Effort", fontsize=12)
    fig.supylabel("g-mean²", fontsize=12)
    fig.suptitle("Monitorability vs Reasoning Effort — Per Eval", fontsize=13)
    fig.tight_layout()

    fig.savefig(output_dir / "per_eval_gmean2_vs_effort.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Summary Table ──────────────────────────────────────────────────

def print_summary_table(data: list[dict], means: list[dict], monitor_model: str) -> None:
    """Print formatted console summary."""
    print(f"\nReasoning Effort Results (monitor: {monitor_model})")
    print("=" * 105)
    header = (
        f"{'Eval':<16s} {'Model':<16s} {'Effort':<8s} "
        f"{'g-mean²':>8s} {'TPR_mc':>7s} {'TNR_X0':>7s} {'TNR_X1':>7s} "
        f"{'R_eff':>6s} {'Cap':>6s} {'N':>4s}"
    )
    print(header)
    print("-" * 105)

    for d in sorted(data, key=lambda x: (x["model_name"], EFFORT_X.get(x["effort"], 0), x["eval_name"])):
        def _fmt(v, width=7):
            return f"{v:{width}.3f}" if v is not None else f"{'—':>{width}s}"

        n_str = str(d["n_instances"]) if d["n_instances"] else "—"
        print(
            f"{d['eval_name']:<16s} {d['model_name']:<16s} {d['effort']:<8s} "
            f"{_fmt(d['gmean2'], 8)} {_fmt(d['tpr_mc'])} {_fmt(d['tnr_x0'])} {_fmt(d['tnr_x1'])} "
            f"{_fmt(d['r_effect'], 6)} {_fmt(d['capability'], 6)} {n_str:>4s}"
        )

    print(f"\n{'Per-model means':}")
    for m in sorted(means, key=lambda x: (x["model_name"], EFFORT_X.get(x["effort"], 0))):
        se_str = f"± {m['gmean2_se']:.3f}" if m["gmean2_se"] else ""
        cap_str = (
            f"cap={m['mean_capability']:.3f} ± {m['capability_se']:.3f}"
            if m["mean_capability"] is not None else "cap=—"
        )
        print(
            f"  {m['model_name']:<16s} effort={m['effort']:<8s}  "
            f"g-mean²={m['mean_gmean2']:.3f} {se_str}  "
            f"{cap_str}  "
            f"({m['n_evals']} evals)"
        )


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot monitorability vs reasoning effort")
    parser.add_argument("--run-config", type=str, required=True,
                        help="Path to run configuration YAML")
    parser.add_argument("--monitor", type=str, default=None,
                        help="Monitor model ID (default: from config)")
    parser.add_argument("--evals", type=str, default=None,
                        help="Comma-separated eval names (default: all)")
    parser.add_argument("--output-dir", type=str, default="analysis/reasoning_effort_plots",
                        help="Directory for output plots")
    args = parser.parse_args()

    rc = load_run_config(args.run_config)
    model_names = rc.get("models", [])
    monitor_cfg = rc.get("monitor", {})
    monitor_model = args.monitor or monitor_cfg.get("model", "gpt-5-mini")

    eval_names = args.evals.split(",") if args.evals else list(EVAL_SPECS.keys())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    print("Collecting data...")
    data = collect_data(eval_names, model_names, monitor_model)
    means = compute_per_model_effort_means(data, model_names)

    # Plots
    print("Generating plots...")
    plot_mean_gmean2_vs_effort(means, output_dir)
    plot_mean_capability_vs_effort(means, output_dir)
    plot_per_eval_gmean2_vs_effort(data, eval_names, output_dir)

    # Summary table
    print_summary_table(data, means, monitor_model)

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
