"""Final reasoning effort analysis plots.

Generates all plots for the reasoning effort scaling experiment,
excluding gpqa_encoded (insufficient Y=0 samples for reliable TNR_X1).

Plots:
1. g-mean² vs subject reasoning effort — per eval (gpqa_hints, aime_hints, daft_hints)
2. g-mean² vs subject reasoning effort — mean across evals
3. Capability vs subject reasoning effort — mean across evals
4. Monitor scaling: daft_hints g-mean² with varying monitor effort
5. Summary table

Usage:
    uv run python -m analysis.plot_final --run-config eval_configs/intervention_reasoning_effort.yaml
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

# Exclude gpqa_encoded (too few Y=0 for reliable TNR_X1),
# aime_hints (120b too capable at high effort, N=4 instances)
MAIN_EVALS = ["gpqa_hints", "daft_hints", "scruples", "wmdp"]

EVAL_LABELS = {
    "gpqa_hints": "GPQA Hints",
    "daft_hints": "DAFT Hints",
    "scruples": "Scruples",
    "wmdp": "WMDP",
}

MODEL_COLORS = {"gpt-oss-20b": "#1a2e50", "gpt-oss-120b": "tab:orange"}
MODEL_MARKERS = {"gpt-oss-20b": "o", "gpt-oss-120b": "s"}
MODEL_LABELS = {"gpt-oss-20b": "gpt-oss-20b", "gpt-oss-120b": "gpt-oss-120b"}

MONITOR_EFFORT_STYLES = {
    "low": {"linestyle": ":", "alpha": 0.6},
    "medium": {"linestyle": "--", "alpha": 0.8},
    "high": {"linestyle": "-", "alpha": 1.0},
}


# ── Data Loading ───────────────────────────────────────────────────

def compute_control_accuracy(eval_name: str, model_name: str) -> dict:
    cache_prefix = f"intervention/{eval_name}"
    control_data = load_cache(cache_prefix, model_name, "control")
    if not control_data:
        return {"accuracy": None, "n_total": 0, "se": 0.0}
    correct = total = 0
    for val in control_data.values():
        if not isinstance(val, dict) or "error" in val:
            continue
        if not (val.get("answer_text") or "").strip():
            continue
        parsed = val.get("parsed_answer")
        answer = val.get("correct_answer")
        if parsed is None or answer is None:
            continue
        total += 1
        if str(parsed).strip().upper() == str(answer).strip().upper():
            correct += 1
    accuracy = correct / total if total > 0 else None
    se = math.sqrt(accuracy * (1 - accuracy) / total) if accuracy and total > 1 else 0.0
    return {"accuracy": accuracy, "n_total": total, "se": se}


def collect_main_data(model_names, monitor_model, monitor_effort="medium"):
    """Collect data for main evals."""
    data = []
    for eval_name in MAIN_EVALS:
        spec = EVAL_SPECS[eval_name]
        for model_name in model_names:
            for effort in EFFORT_LEVELS:
                cache_name = f"{model_name}_effort_{effort}"
                metrics = compute_gmean2(
                    spec, cache_name,
                    monitor_model=monitor_model,
                    monitor_reasoning_effort=monitor_effort,
                )
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
                })
    return data


def collect_monitor_scaling(model_names, monitor_model, monitor_efforts):
    """Collect daft_hints data across monitor effort levels."""
    data = []
    spec = EVAL_SPECS["daft_hints"]
    for model_name in model_names:
        for subj_effort in EFFORT_LEVELS:
            cache_name = f"{model_name}_effort_{subj_effort}"
            for mon_effort in monitor_efforts:
                metrics = compute_gmean2(
                    spec, cache_name,
                    monitor_model=monitor_model,
                    monitor_reasoning_effort=mon_effort,
                )
                data.append({
                    "model_name": model_name,
                    "subject_effort": subj_effort,
                    "monitor_effort": mon_effort,
                    "gmean2": metrics["gmean2"],
                    "tpr_mc": metrics.get("tpr_mc"),
                    "tnr_x0": metrics.get("tnr_x0"),
                    "tnr_x1": metrics.get("tnr_x1"),
                })
    return data


def compute_means(data, model_names):
    """Mean g-mean² and capability per (model, effort) across evals."""
    means = []
    for model_name in model_names:
        for effort in EFFORT_LEVELS:
            items = [d for d in data
                     if d["model_name"] == model_name
                     and d["effort"] == effort
                     and d["gmean2"] is not None]
            if not items:
                continue
            g_vals = [d["gmean2"] for d in items]
            c_vals = [d["capability"] for d in items if d["capability"] is not None]
            means.append({
                "model_name": model_name,
                "effort": effort,
                "mean_gmean2": float(np.mean(g_vals)),
                "gmean2_se": float(np.std(g_vals, ddof=1) / np.sqrt(len(g_vals))) if len(g_vals) > 1 else 0.0,
                "mean_capability": float(np.mean(c_vals)) if c_vals else None,
                "capability_se": float(np.std(c_vals, ddof=1) / np.sqrt(len(c_vals))) if len(c_vals) > 1 else 0.0,
                "n_evals": len(items),
            })
    return means


# ── Plots ──────────────────────────────────────────────────────────

def plot_per_eval_gmean2(data, output_dir):
    """Per-eval subplots: g-mean² vs reasoning effort."""
    n_evals = len(MAIN_EVALS)
    ncols = min(n_evals, 4)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5), sharey=True)

    for idx, eval_name in enumerate(MAIN_EVALS):
        ax = axes[idx]
        eval_data = [d for d in data if d["eval_name"] == eval_name]

        for model_name in sorted(set(d["model_name"] for d in eval_data)):
            series = sorted(
                [d for d in eval_data if d["model_name"] == model_name and d["gmean2"] is not None],
                key=lambda d: EFFORT_X[d["effort"]],
            )
            if not series:
                continue
            xs = [EFFORT_X[d["effort"]] for d in series]
            ys = [d["gmean2"] for d in series]
            ax.plot(xs, ys, f"-{MODEL_MARKERS[model_name]}",
                    color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name],
                    linewidth=2, markersize=8)

        ax.set_xticks(list(EFFORT_X.values()))
        ax.set_xticklabels(EFFORT_LEVELS, fontsize=20)
        ax.set_title(EVAL_LABELS[eval_name], fontsize=24)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(axis='y', labelsize=20)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=18)

    axes[0].set_ylabel("g-mean²", fontsize=24)
    fig.supxlabel("Reasoning Effort", fontsize=24)
    fig.tight_layout()
    fig.savefig(output_dir / "per_eval_gmean2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved per_eval_gmean2.png")


def plot_mean_gmean2(means, output_dir):
    """Mean g-mean² vs reasoning effort, one line per model."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for model_name in sorted(set(m["model_name"] for m in means)):
        model_means = sorted(
            [m for m in means if m["model_name"] == model_name],
            key=lambda m: EFFORT_X[m["effort"]],
        )
        xs = [EFFORT_X[m["effort"]] for m in model_means]
        ys = [m["mean_gmean2"] for m in model_means]
        ses = [m["gmean2_se"] for m in model_means]
        ax.plot(xs, ys, f"-{MODEL_MARKERS[model_name]}",
                color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name],
                linewidth=2, markersize=8)

    ax.set_xticks(list(EFFORT_X.values()))
    ax.set_xticklabels(EFFORT_LEVELS)
    ax.set_xlabel("Reasoning Effort", fontsize=12)
    ax.set_ylabel("Mean g-mean² (across evals)", fontsize=12)
    ax.set_title("Monitorability vs Reasoning Effort", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "mean_gmean2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved mean_gmean2.png")


def plot_mean_capability(means, output_dir):
    """Mean capability vs reasoning effort."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_name in sorted(set(m["model_name"] for m in means)):
        model_means = sorted(
            [m for m in means if m["model_name"] == model_name and m["mean_capability"] is not None],
            key=lambda m: EFFORT_X[m["effort"]],
        )
        xs = [EFFORT_X[m["effort"]] for m in model_means]
        ys = [m["mean_capability"] for m in model_means]
        ses = [m["capability_se"] for m in model_means]
        ax.plot(xs, ys, f"-{MODEL_MARKERS[model_name]}",
                color=MODEL_COLORS[model_name], label=MODEL_LABELS[model_name],
                linewidth=2, markersize=8)

    ax.set_xticks(list(EFFORT_X.values()))
    ax.set_xticklabels(EFFORT_LEVELS)
    ax.set_xlabel("Reasoning Effort", fontsize=12)
    ax.set_ylabel("Mean Accuracy (across evals)", fontsize=12)
    ax.set_title("Capability vs Reasoning Effort", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "mean_capability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved mean_capability.png")


def plot_monitor_scaling(mon_data, monitor_model, output_dir):
    """daft_hints g-mean² with varying monitor effort."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, model_name in enumerate(["gpt-oss-20b", "gpt-oss-120b"]):
        ax = axes[idx]
        model_data = [d for d in mon_data if d["model_name"] == model_name]

        for mon_effort in EFFORT_LEVELS:
            series = sorted(
                [d for d in model_data
                 if d["monitor_effort"] == mon_effort and d["gmean2"] is not None],
                key=lambda d: EFFORT_X[d["subject_effort"]],
            )
            if not series:
                continue
            xs = [EFFORT_X[d["subject_effort"]] for d in series]
            ys = [d["gmean2"] for d in series]
            style = MONITOR_EFFORT_STYLES[mon_effort]
            ax.plot(xs, ys, marker=MODEL_MARKERS[model_name],
                    color=MODEL_COLORS[model_name],
                    linestyle=style["linestyle"], alpha=style["alpha"],
                    label=f"monitor {mon_effort}", linewidth=2, markersize=8)

        ax.set_xticks(list(EFFORT_X.values()))
        ax.set_xticklabels(EFFORT_LEVELS)
        ax.set_xlabel("Subject Reasoning Effort", fontsize=12)
        ax.set_title(f"DAFT Hints — {MODEL_LABELS[model_name]}", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("g-mean²", fontsize=12)
    fig.suptitle(
        f"Monitor Scaling on DAFT Hints ({monitor_model})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "daft_monitor_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved daft_monitor_scaling.png")


# ── Summary Table ──────────────────────────────────────────────────

def print_summary(data, means, mon_data, monitor_model):
    print(f"\n{'='*100}")
    print(f"REASONING EFFORT RESULTS (monitor: {monitor_model})")
    print(f"{'='*100}")

    print(f"\n{'Eval':<16s} {'Model':<16s} {'Effort':<8s} "
          f"{'g-mean²':>8s} {'TPR_mc':>7s} {'TNR_X0':>7s} {'TNR_X1':>7s} "
          f"{'R_eff':>6s} {'Cap':>6s} {'N':>4s}")
    print("-" * 100)

    for d in sorted(data, key=lambda x: (x["eval_name"], x["model_name"], EFFORT_X.get(x["effort"], 0))):
        def f(v, w=7): return f"{v:{w}.3f}" if v is not None else f"{'—':>{w}s}"
        n = str(d["n_instances"]) if d["n_instances"] else "—"
        print(f"{d['eval_name']:<16s} {d['model_name']:<16s} {d['effort']:<8s} "
              f"{f(d['gmean2'], 8)} {f(d['tpr_mc'])} {f(d['tnr_x0'])} {f(d['tnr_x1'])} "
              f"{f(d['r_effect'], 6)} {f(d['capability'], 6)} {n:>4s}")

    print(f"\nPer-model means (across {len(MAIN_EVALS)} evals):")
    for m in sorted(means, key=lambda x: (x["model_name"], EFFORT_X.get(x["effort"], 0))):
        cap_str = f"cap={m['mean_capability']:.3f}±{m['capability_se']:.3f}" if m["mean_capability"] else "cap=—"
        print(f"  {m['model_name']:<16s} effort={m['effort']:<8s}  "
              f"g-mean²={m['mean_gmean2']:.3f}±{m['gmean2_se']:.3f}  {cap_str}")

    print(f"\nMonitor scaling (daft_hints):")
    for d in sorted(mon_data, key=lambda x: (x["model_name"], EFFORT_X.get(x["subject_effort"], 0), EFFORT_X.get(x["monitor_effort"], 0))):
        if d["gmean2"] is None:
            continue
        print(f"  {d['model_name']:<16s} subj={d['subject_effort']:<6s} mon={d['monitor_effort']:<6s}  "
              f"g-mean²={d['gmean2']:.3f}  TPR={d['tpr_mc']:.3f}  TNR_X1={d['tnr_x1']:.3f}")


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Final reasoning effort analysis")
    parser.add_argument("--run-config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="analysis/final_plots")
    parser.add_argument("--monitor", type=str, default=None)
    args = parser.parse_args()

    with open(args.run_config) as f:
        rc = yaml.safe_load(f)

    model_names = rc.get("models", ["gpt-oss-120b", "gpt-oss-20b"])
    monitor_model = args.monitor or rc.get("monitor", {}).get("model", "gpt-5-mini")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting data...")
    data = collect_main_data(model_names, monitor_model, monitor_effort="medium")
    means = compute_means(data, model_names)
    mon_data = collect_monitor_scaling(model_names, monitor_model, EFFORT_LEVELS)

    print("Generating plots...")
    plot_per_eval_gmean2(data, output_dir)
    plot_mean_gmean2(means, output_dir)
    plot_mean_capability(means, output_dir)
    plot_monitor_scaling(mon_data, monitor_model, output_dir)

    print_summary(data, means, mon_data, monitor_model)
    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
