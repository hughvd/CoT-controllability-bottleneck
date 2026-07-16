"""Compare monitor reasoning effort on reasoning effort evals.

Plots g-mean² vs subject reasoning effort for gpqa_encoded and daft_hints,
with separate lines for different monitor reasoning effort levels.

Usage:
    uv run python -m analysis.plot_monitor_comparison --run-config eval_configs/intervention_reasoning_effort.yaml
    uv run python -m analysis.plot_monitor_comparison --run-config eval_configs/intervention_reasoning_effort.yaml --monitor-efforts low,medium,high
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.intervention.metrics import compute_gmean2
from evals.intervention.run import EVAL_SPECS

# ── Constants ──────────────────────────────────────────────────────

EFFORT_LEVELS = ["low", "medium", "high"]
EFFORT_X = {level: i for i, level in enumerate(EFFORT_LEVELS)}

EVALS = ["gpqa_encoded", "daft_hints"]
EVAL_LABELS = {"gpqa_encoded": "GPQA Encoded", "daft_hints": "DAFT Hints"}

MODEL_NAMES = ["gpt-oss-20b", "gpt-oss-120b"]
MODEL_LABELS = {"gpt-oss-20b": "gpt-oss-20b", "gpt-oss-120b": "gpt-oss-120b"}
MODEL_COLORS = {"gpt-oss-20b": "tab:blue", "gpt-oss-120b": "tab:orange"}
MODEL_MARKERS = {"gpt-oss-20b": "o", "gpt-oss-120b": "s"}

MONITOR_EFFORT_STYLES = {
    "low": {"linestyle": ":", "alpha": 0.6},
    "medium": {"linestyle": "--", "alpha": 0.8},
    "high": {"linestyle": "-", "alpha": 1.0},
}


# ── Data ──────────────────────────────────────────────────────────

def collect_data(model_names, monitor_model, monitor_efforts):
    """Collect g-mean² for each (eval, model, subject_effort, monitor_effort)."""
    data = []
    for eval_name in EVALS:
        spec = EVAL_SPECS[eval_name]
        for model_name in model_names:
            for subject_effort in EFFORT_LEVELS:
                cache_name = f"{model_name}_effort_{subject_effort}"
                for mon_effort in monitor_efforts:
                    metrics = compute_gmean2(
                        spec, cache_name,
                        monitor_model=monitor_model,
                        monitor_reasoning_effort=mon_effort,
                    )
                    data.append({
                        "eval_name": eval_name,
                        "model_name": model_name,
                        "subject_effort": subject_effort,
                        "monitor_effort": mon_effort,
                        "gmean2": metrics["gmean2"],
                        "tpr_mc": metrics.get("tpr_mc"),
                        "tnr_x0": metrics.get("tnr_x0"),
                        "tnr_x1": metrics.get("tnr_x1"),
                    })
    return data


# ── Plot ──────────────────────────────────────────────────────────

def plot_monitor_scaling(data, monitor_model, monitor_efforts, output_dir):
    """One subplot per eval, lines for each (model, monitor_effort) combo."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, eval_name in enumerate(EVALS):
        ax = axes[idx]
        eval_data = [d for d in data if d["eval_name"] == eval_name]

        for model_name in MODEL_NAMES:
            for mon_effort in monitor_efforts:
                series = sorted(
                    [d for d in eval_data
                     if d["model_name"] == model_name
                     and d["monitor_effort"] == mon_effort
                     and d["gmean2"] is not None],
                    key=lambda d: EFFORT_X[d["subject_effort"]],
                )
                if not series:
                    continue

                xs = [EFFORT_X[d["subject_effort"]] for d in series]
                ys = [d["gmean2"] for d in series]

                style = MONITOR_EFFORT_STYLES[mon_effort]
                label = f"{MODEL_LABELS[model_name]} / monitor {mon_effort}"
                ax.plot(
                    xs, ys,
                    marker=MODEL_MARKERS[model_name],
                    color=MODEL_COLORS[model_name],
                    linestyle=style["linestyle"],
                    alpha=style["alpha"],
                    label=label,
                    linewidth=2, markersize=8,
                )

        ax.set_xticks(list(EFFORT_X.values()))
        ax.set_xticklabels(EFFORT_LEVELS)
        ax.set_xlabel("Subject Reasoning Effort", fontsize=12)
        ax.set_title(EVAL_LABELS[eval_name], fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

    axes[0].set_ylabel("g-mean²", fontsize=12)
    fig.suptitle(
        f"Monitorability vs Reasoning Effort — Monitor Scaling ({monitor_model})",
        fontsize=14,
    )
    fig.tight_layout()

    fname = f"monitor_scaling_{monitor_model}_gmean2_vs_effort.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_dir / fname}")


def print_table(data, monitor_model):
    """Print summary table."""
    print(f"\nMonitor Scaling Results (monitor: {monitor_model})")
    print(f"{'Eval':<16s} {'Model':<16s} {'Subj Eff':<10s} {'Mon Eff':<10s} "
          f"{'g-mean²':>8s} {'TPR_mc':>7s} {'TNR_X0':>7s} {'TNR_X1':>7s}")
    print("-" * 95)
    for d in sorted(data, key=lambda x: (x["eval_name"], x["model_name"],
                                          EFFORT_X.get(x["subject_effort"], 0),
                                          EFFORT_X.get(x["monitor_effort"], 0))):
        def _fmt(v, w=7):
            return f"{v:{w}.3f}" if v is not None else f"{'—':>{w}s}"
        print(f"{d['eval_name']:<16s} {d['model_name']:<16s} "
              f"{d['subject_effort']:<10s} {d['monitor_effort']:<10s} "
              f"{_fmt(d['gmean2'], 8)} {_fmt(d['tpr_mc'])} "
              f"{_fmt(d['tnr_x0'])} {_fmt(d['tnr_x1'])}")


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot monitor scaling for reasoning effort evals")
    parser.add_argument("--run-config", type=str, required=True)
    parser.add_argument("--output-dir", type=str,
                        default="analysis/reasoning_effort_plots")
    parser.add_argument("--monitor", type=str, default=None,
                        help="Monitor model (default: from config)")
    parser.add_argument("--monitor-efforts", type=str, default="low,medium,high",
                        help="Comma-separated monitor reasoning effort levels")
    args = parser.parse_args()

    with open(args.run_config) as f:
        rc = yaml.safe_load(f)

    model_names = rc.get("models", MODEL_NAMES)
    monitor_model = args.monitor or rc.get("monitor", {}).get("model", "gpt-5-mini")
    monitor_efforts = args.monitor_efforts.split(",")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting data...")
    data = collect_data(model_names, monitor_model, monitor_efforts)

    print_table(data, monitor_model)
    plot_monitor_scaling(data, monitor_model, monitor_efforts, output_dir)


if __name__ == "__main__":
    main()
