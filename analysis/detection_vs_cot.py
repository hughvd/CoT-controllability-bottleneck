"""Analyze monitorability (g-mean²) as a function of CoT length.

Two levels of analysis:
  1. Naive: each (eval, model) pair = one data point (mean CoT length vs gmean²)
  2. Within-model: bin instances by CoT length, compute bin-level gmean²

Usage:
    uv run python -m analysis.detection_vs_cot --run-config eval_configs/intervention_full.yaml
    uv run python -m analysis.detection_vs_cot --run-config eval_configs/intervention_full.yaml --monitor gpt-5-mini
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.intervention.metrics import (
    compute_gmean2,
    compute_gmean2_for_subset,
    compute_total_effects,
)
from evals.intervention.run import EVAL_SPECS
from lib.results import load_cache

# ── Style Constants ─────────────────────────────────────────────────

MODEL_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]

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


# ── Naive (eval, model)-level data ─────────────────────────────────

def compute_mean_cot_length(eval_name: str, model_name: str) -> dict:
    """Compute mean CoT length from intervention arm thinking fields."""
    cache_prefix = f"intervention/{eval_name}"
    intv_data = load_cache(cache_prefix, model_name, "intervention")

    lengths = []
    for val in intv_data.values():
        if not isinstance(val, dict) or "error" in val:
            continue
        thinking = val.get("thinking")
        if thinking:
            lengths.append(len(thinking))

    if not lengths:
        return {"mean_cot_length": None, "std_cot_length": None, "n_samples": 0}
    return {
        "mean_cot_length": float(np.mean(lengths)),
        "std_cot_length": float(np.std(lengths, ddof=1)) if len(lengths) > 1 else 0.0,
        "n_samples": len(lengths),
    }


def collect_naive_data(
    eval_names: list[str],
    model_names: list[str],
    monitor_model: str,
) -> list[dict]:
    """Collect g-mean² and mean CoT length for all (eval, model) pairs."""
    data = []
    for eval_name in eval_names:
        if eval_name not in EVAL_SPECS:
            continue
        spec = EVAL_SPECS[eval_name]
        for model_name in model_names:
            metrics = compute_gmean2(spec, model_name, monitor_model=monitor_model)
            cot = compute_mean_cot_length(eval_name, model_name)
            if metrics["gmean2"] is None or cot["mean_cot_length"] is None:
                continue
            data.append({
                "eval_name": eval_name,
                "model_name": model_name,
                "mean_cot_length": cot["mean_cot_length"],
                "std_cot_length": cot["std_cot_length"],
                "n_cot_samples": cot["n_samples"],
                "gmean2": metrics["gmean2"],
                "tpr_mc": metrics.get("tpr_mc"),
                "tnr_x0": metrics.get("tnr_x0"),
                "tnr_x1": metrics.get("tnr_x1"),
                "r_effect": metrics.get("r_effect"),
                "n_instances": metrics.get("n_instances", 0),
            })
    return data


# ── Within-model instance-level binning ─────────────────────────────

def compute_instance_cot_lengths(eval_name: str, model_name: str) -> dict[str, float]:
    """Return {problem_id: mean_cot_length} from intervention arm."""
    cache_prefix = f"intervention/{eval_name}"
    intv_data = load_cache(cache_prefix, model_name, "intervention")

    by_pid: dict[str, list[int]] = defaultdict(list)
    for val in intv_data.values():
        if not isinstance(val, dict) or "error" in val:
            continue
        thinking = val.get("thinking")
        if thinking:
            by_pid[str(val["problem_id"])].append(len(thinking))

    return {pid: float(np.mean(lens)) for pid, lens in by_pid.items() if lens}


def bin_instances_by_cot(
    instance_cot: dict[str, float],
    n_bins: int = 4,
) -> list[dict]:
    """Bin instances (problem_ids) by their mean CoT length using log-spaced edges.

    Returns list of {bin_lo, bin_hi, bin_center, problem_ids}.
    """
    if not instance_cot:
        return []

    pids = list(instance_cot.keys())
    lengths = np.array([instance_cot[p] for p in pids])

    log_min = np.log10(max(lengths.min(), 1))
    log_max = np.log10(lengths.max())
    if log_max - log_min < 0.1:
        return []  # not enough range to bin

    bin_edges = np.logspace(log_min, log_max, n_bins + 1)

    bins = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (lengths >= lo) & (lengths <= hi)
        else:
            mask = (lengths >= lo) & (lengths < hi)
        selected = {pids[j] for j in range(len(pids)) if mask[j]}
        if not selected:
            continue
        bins.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "bin_center": float(np.sqrt(lo * hi)),
            "problem_ids": selected,
        })
    return bins


def compute_binned_gmean2(
    eval_name: str,
    model_name: str,
    monitor_model: str,
    n_bins: int = 4,
) -> list[dict]:
    """Bin instances by CoT length and compute gmean² per bin.

    Returns list of {bin_idx, bin_lo, bin_hi, bin_center, gmean2, n_instances, ...}.
    """
    spec = EVAL_SPECS.get(eval_name)
    if not spec:
        return []

    # Get instances with TE > 0
    te_result = compute_total_effects(spec, model_name)
    instances = te_result["instances"]
    if not instances:
        return []

    # Get per-instance CoT lengths
    instance_cot = compute_instance_cot_lengths(eval_name, model_name)
    # Only keep instances that have CoT data
    valid_pids = {i["problem_id"] for i in instances} & set(instance_cot.keys())
    valid_cot = {pid: instance_cot[pid] for pid in valid_pids}

    if not valid_cot:
        return []

    # Bin
    cot_bins = bin_instances_by_cot(valid_cot, n_bins)
    if not cot_bins:
        return []

    # Load monitor data once
    cache_prefix = f"{spec.eval_name}/{spec.name}"
    monitor_data = load_cache(cache_prefix, model_name, f"monitor_{monitor_model}")
    if not monitor_data:
        return []

    # Compute gmean² per bin
    results = []
    for idx, b in enumerate(cot_bins):
        metrics = compute_gmean2_for_subset(instances, monitor_data, b["problem_ids"])
        if metrics["gmean2"] is None:
            continue
        results.append({
            "bin_idx": idx,
            "bin_lo": b["bin_lo"],
            "bin_hi": b["bin_hi"],
            "bin_center": b["bin_center"],
            "n_instances": metrics["n_instances"],
            "gmean2": metrics["gmean2"],
            "tpr_mc": metrics["tpr_mc"],
            "tnr_x0": metrics["tnr_x0"],
            "tnr_x1": metrics["tnr_x1"],
        })
    return results


# ── Plotting helpers ────────────────────────────────────────────────

def _place_labels(ax, xs, ys, names, fontsize=8, colors=None):
    """Place non-overlapping labels near data points (display-space)."""
    if not xs or not names:
        return
    if colors is None:
        colors = ["black"] * len(names)

    fig = ax.get_figure()
    fig.canvas.draw()

    pts = ax.transData.transform(list(zip(xs, ys)))
    base_dx, base_dy = 12, 6

    order = sorted(range(len(pts)), key=lambda i: pts[i][1])
    label_y = [pts[i][1] + base_dy for i in order]

    min_gap = fontsize * 1.8
    for i in range(1, len(order)):
        if label_y[i] - label_y[i - 1] < min_gap:
            label_y[i] = label_y[i - 1] + min_gap

    orig_mean = np.mean([pts[i][1] + base_dy for i in order])
    shift = orig_mean - np.mean(label_y)
    label_y = [y + shift for y in label_y]

    final_y = [0.0] * len(pts)
    for rank, orig_i in enumerate(order):
        final_y[orig_i] = label_y[rank]

    scale = 72.0 / fig.dpi
    for i, (x, y, name, c) in enumerate(zip(xs, ys, names, colors)):
        dx_pts = base_dx * scale
        dy_pts = (final_y[i] - pts[i][1]) * scale
        ax.annotate(
            name, xy=(x, y), xycoords="data",
            xytext=(dx_pts, dy_pts), textcoords="offset points",
            fontsize=fontsize, color=c, alpha=0.85,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.4,
                            shrinkA=0, shrinkB=3),
        )


# ── Naive plots ─────────────────────────────────────────────────────

def plot_scatter_by_eval(data: list[dict], output_dir: Path) -> None:
    """Scatter plot: g-mean² vs mean CoT length, one series per eval."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for eval_name in sorted(set(d["eval_name"] for d in data)):
        ed = [d for d in data if d["eval_name"] == eval_name]
        xs = [d["mean_cot_length"] for d in ed]
        ys = [d["gmean2"] for d in ed]
        color = EVAL_COLORS.get(eval_name, "gray")
        marker = EVAL_MARKERS.get(eval_name, "o")
        label = EVAL_LABELS.get(eval_name, eval_name)
        ax.scatter(xs, ys, marker=marker, color=color, s=80, label=label, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("Mean CoT Length (characters)", fontsize=12)
    ax.set_ylabel("g-mean²", fontsize=12)
    ax.set_title("Monitorability vs CoT Length — All Evals", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "gmean2_vs_cot_length.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_naive_per_eval(data: list[dict], eval_name: str, output_dir: Path) -> None:
    """Per-eval scatter: g-mean² vs mean CoT length with model labels."""
    ed = [d for d in data if d["eval_name"] == eval_name]
    if not ed:
        return
    xs = [d["mean_cot_length"] for d in ed]
    ys = [d["gmean2"] for d in ed]
    names = [d["model_name"] for d in ed]

    fig, ax = plt.subplots(figsize=(9, 6))
    color = EVAL_COLORS.get(eval_name, "tab:blue")
    marker = EVAL_MARKERS.get(eval_name, "o")
    ax.scatter(xs, ys, marker=marker, color=color, s=100, zorder=5)
    ax.set_xscale("log")
    _place_labels(ax, xs, ys, names, fontsize=8)
    label = EVAL_LABELS.get(eval_name, eval_name)
    ax.set_xlabel("Mean CoT Length (characters)", fontsize=12)
    ax.set_ylabel("g-mean²", fontsize=12)
    ax.set_title(f"Monitorability vs CoT Length — {label}", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / f"{eval_name}_vs_cot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_naive_binned(data: list[dict], output_dir: Path, n_bins: int = 5) -> None:
    """Bin (eval, model) points by mean CoT length, plot mean g-mean² per bin."""
    if not data:
        return
    cot_lengths = np.array([d["mean_cot_length"] for d in data])
    gmean2_vals = np.array([d["gmean2"] for d in data])

    log_min = np.log10(cot_lengths.min())
    log_max = np.log10(cot_lengths.max())
    bin_edges = np.logspace(log_min, log_max, n_bins + 1)

    bins = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (cot_lengths >= lo) & (cot_lengths <= hi if i == len(bin_edges) - 2 else cot_lengths < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_g = float(gmean2_vals[mask].mean())
        se = float(gmean2_vals[mask].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        bins.append({"bin_center": float(np.sqrt(lo * hi)), "n": n,
                      "mean_gmean2": mean_g, "se": se, "bin_lo": lo, "bin_hi": hi})

    if not bins:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [b["bin_center"] for b in bins]
    ys = [b["mean_gmean2"] for b in bins]
    ses = [b["se"] for b in bins]
    ax.errorbar(xs, ys, yerr=ses, fmt="o-", color="tab:blue", capsize=5,
                linewidth=1.5, markersize=8, zorder=5)
    for b in bins:
        ax.annotate(f"n={b['n']}", xy=(b["bin_center"], b["mean_gmean2"]),
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=8, ha="center", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Mean CoT Length (characters)", fontsize=12)
    ax.set_ylabel("Mean g-mean² (± SEM)", fontsize=12)
    ax.set_title(f"Binned: Monitorability vs CoT Length (n={len(data)} model-eval pairs)", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    print("\n  Naive bin edges:")
    for b in bins:
        print(f"    [{b['bin_lo']:,.0f} – {b['bin_hi']:,.0f}]: n={b['n']}, "
              f"g-mean²={b['mean_gmean2']:.3f} ± {b['se']:.3f}")

    fig.savefig(output_dir / "gmean2_vs_cot_length_binned.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Within-model binned plots ──────────────────────────────────────

def plot_binned_per_eval(
    eval_name: str,
    model_names: list[str],
    monitor_model: str,
    n_bins: int,
    output_dir: Path,
) -> dict[str, list[dict]]:
    """Per-eval: bin instances by CoT, one line per model. Returns binned data."""
    fig, ax = plt.subplots(figsize=(10, 6))
    all_binned = {}  # model_name -> list of bin dicts

    for i, model_name in enumerate(model_names):
        bins = compute_binned_gmean2(eval_name, model_name, monitor_model, n_bins)
        if not bins:
            continue
        all_binned[model_name] = bins
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        xs = [b["bin_center"] for b in bins]
        ys = [b["gmean2"] for b in bins]
        ax.plot(xs, ys, "o-", color=color, label=model_name, markersize=6,
                linewidth=1.5, alpha=0.8)

    if not all_binned:
        plt.close(fig)
        return {}

    ax.set_xscale("log")
    ax.set_xlabel("Instance Mean CoT Length (characters)", fontsize=12)
    ax.set_ylabel("g-mean²", fontsize=12)
    label = EVAL_LABELS.get(eval_name, eval_name)
    ax.set_title(f"Within-Model Binned g-mean² — {label}", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / f"{eval_name}_binned.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return all_binned


def plot_aggregated_by_bin_index(
    all_binned: dict[tuple[str, str], list[dict]],
    model_names: list[str],
    output_dir: Path,
) -> None:
    """Per-model: average gmean² across evals by bin index. One line per model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, model_name in enumerate(model_names):
        # Gather all evals for this model
        model_bins = {k: v for k, v in all_binned.items() if k[1] == model_name}
        if not model_bins:
            continue

        # Group by bin_idx, average gmean²
        by_idx: dict[int, list[float]] = defaultdict(list)
        for bins in model_bins.values():
            for b in bins:
                by_idx[b["bin_idx"]].append(b["gmean2"])

        idxs = sorted(by_idx.keys())
        means = [float(np.mean(by_idx[idx])) for idx in idxs]
        # x-axis = bin index + 1 (1-based)
        xs = [idx + 1 for idx in idxs]

        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(xs, means, "o-", color=color, label=model_name, markersize=6,
                linewidth=1.5, alpha=0.8)

    ax.set_xlabel("CoT Length Bin (1=shortest, N=longest)", fontsize=12)
    ax.set_ylabel("Mean g-mean² (across evals)", fontsize=12)
    ax.set_title("Within-Model Binned g-mean² — Aggregated Across Evals", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "aggregated_by_bin_index.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grand_aggregate(
    all_binned: dict[tuple[str, str], list[dict]],
    output_dir: Path,
) -> None:
    """Mean across all models for each bin index, with SEM error bars."""
    by_idx: dict[int, list[float]] = defaultdict(list)
    for bins in all_binned.values():
        for b in bins:
            by_idx[b["bin_idx"]].append(b["gmean2"])

    if not by_idx:
        return

    idxs = sorted(by_idx.keys())
    means = [float(np.mean(by_idx[idx])) for idx in idxs]
    ses = [
        float(np.std(by_idx[idx], ddof=1) / np.sqrt(len(by_idx[idx])))
        if len(by_idx[idx]) > 1 else 0.0
        for idx in idxs
    ]
    xs = [idx + 1 for idx in idxs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(xs, means, yerr=ses, fmt="o-", color="tab:blue", capsize=5,
                linewidth=2, markersize=8, zorder=5)
    for x, y, n_vals in zip(xs, means, [len(by_idx[idx]) for idx in idxs]):
        ax.annotate(f"n={n_vals}", xy=(x, y), xytext=(0, 10),
                    textcoords="offset points", fontsize=8, ha="center", alpha=0.7)

    ax.set_xlabel("CoT Length Bin (1=shortest, N=longest)", fontsize=12)
    ax.set_ylabel("Mean g-mean² (± SEM)", fontsize=12)
    ax.set_title("Within-Model Binned g-mean² — Grand Mean (All Models × Evals)", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    print("\n  Grand aggregate by bin index:")
    for idx, m, se in zip(idxs, means, ses):
        print(f"    Bin {idx+1}: g-mean²={m:.3f} ± {se:.3f} (n={len(by_idx[idx])})")

    fig.savefig(output_dir / "grand_aggregate_by_bin_index.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined_all_evals(
    eval_names: list[str],
    model_names: list[str],
    monitor_model: str,
    n_bins: int,
    output_dir: Path,
) -> None:
    """Pool all instances across all evals, bin by CoT, compute gmean² per bin."""
    # Collect all instances + CoT lengths + monitor data across evals
    all_instances = []
    all_monitor_data = {}
    all_instance_cot = {}

    for eval_name in eval_names:
        spec = EVAL_SPECS.get(eval_name)
        if not spec:
            continue
        for model_name in model_names:
            te_result = compute_total_effects(spec, model_name)
            if not te_result["instances"]:
                continue

            instance_cot = compute_instance_cot_lengths(eval_name, model_name)
            cache_prefix = f"{spec.eval_name}/{spec.name}"
            monitor_data = load_cache(cache_prefix, model_name, f"monitor_{monitor_model}")
            if not monitor_data:
                continue

            # Tag instances and monitor data with (eval, model) to avoid id collisions
            tag = f"{eval_name}|{model_name}"
            for inst in te_result["instances"]:
                tagged = dict(inst)
                tagged["problem_id"] = f"{tag}|{inst['problem_id']}"
                all_instances.append(tagged)

            for pid, cot_len in instance_cot.items():
                all_instance_cot[f"{tag}|{pid}"] = cot_len

            for k, v in monitor_data.items():
                if isinstance(v, dict) and v.get("problem_id") is not None:
                    tagged_v = dict(v)
                    tagged_v["problem_id"] = f"{tag}|{v['problem_id']}"
                    all_monitor_data[f"{tag}|{k}"] = tagged_v

    if not all_instances or not all_instance_cot:
        return

    # Filter to instances with CoT data
    valid_pids = {i["problem_id"] for i in all_instances} & set(all_instance_cot.keys())
    valid_cot = {pid: all_instance_cot[pid] for pid in valid_pids}

    cot_bins = bin_instances_by_cot(valid_cot, n_bins)
    if not cot_bins:
        return

    bins = []
    for idx, b in enumerate(cot_bins):
        metrics = compute_gmean2_for_subset(all_instances, all_monitor_data, b["problem_ids"])
        if metrics["gmean2"] is None:
            continue
        bins.append({
            "bin_idx": idx,
            "bin_center": b["bin_center"],
            "bin_lo": b["bin_lo"],
            "bin_hi": b["bin_hi"],
            "gmean2": metrics["gmean2"],
            "n_instances": metrics["n_instances"],
        })

    if not bins:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [b["bin_center"] for b in bins]
    ys = [b["gmean2"] for b in bins]
    ax.plot(xs, ys, "o-", color="tab:blue", markersize=8, linewidth=2, zorder=5)
    for b in bins:
        ax.annotate(f"n={b['n_instances']}", xy=(b["bin_center"], b["gmean2"]),
                    xytext=(0, 10), textcoords="offset points",
                    fontsize=8, ha="center", alpha=0.7)

    ax.set_xscale("log")
    ax.set_xlabel("Instance Mean CoT Length (characters)", fontsize=12)
    ax.set_ylabel("g-mean²", fontsize=12)
    ax.set_title("Pooled Across All Evals & Models: g-mean² vs CoT Length", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    print("\n  Combined (all evals pooled) bins:")
    for b in bins:
        print(f"    [{b['bin_lo']:,.0f} – {b['bin_hi']:,.0f}]: "
              f"n={b['n_instances']}, g-mean²={b['gmean2']:.3f}")

    fig.savefig(output_dir / "combined_all_binned.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="g-mean² vs CoT length analysis")
    parser.add_argument("--run-config", type=str, required=True)
    parser.add_argument("--monitor", type=str, default=None,
                        help="Monitor model (default: from config)")
    parser.add_argument("--n-bins", type=int, default=4)
    args = parser.parse_args()

    with open(args.run_config) as f:
        rc = yaml.safe_load(f)
    model_names = rc.get("models", [])
    monitor_cfg = rc.get("monitor", {})
    monitor_model = args.monitor or monitor_cfg.get("model", "gpt-5-mini")

    eval_names = sorted(EVAL_SPECS.keys())

    output_dir = Path("analysis/detection_vs_cot")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_eval_dir = output_dir / "per_eval"
    per_eval_dir.mkdir(exist_ok=True)
    per_model_dir = output_dir / "per_model"
    per_model_dir.mkdir(exist_ok=True)

    # ── 1. Naive (eval, model)-level analysis ──────────────────────
    print("Collecting naive (eval, model)-level data...")
    naive_data = collect_naive_data(eval_names, model_names, monitor_model)

    if naive_data:
        csv_path = output_dir / "summary.csv"
        fields = ["eval_name", "model_name", "mean_cot_length", "std_cot_length",
                  "n_cot_samples", "gmean2", "tpr_mc", "tnr_x0", "tnr_x1",
                  "r_effect", "n_instances"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(naive_data)
        print(f"  Saved {csv_path} ({len(naive_data)} rows)")

        plot_scatter_by_eval(naive_data, output_dir)
        for eval_name in eval_names:
            plot_naive_per_eval(naive_data, eval_name, per_eval_dir)
        plot_naive_binned(naive_data, output_dir, n_bins=args.n_bins + 1)

    # ── 2. Within-model binned analysis ────────────────────────────
    print("\nComputing within-model binned g-mean²...")
    all_binned: dict[tuple[str, str], list[dict]] = {}

    for eval_name in eval_names:
        print(f"  {eval_name}:")
        eval_binned = plot_binned_per_eval(
            eval_name, model_names, monitor_model, args.n_bins, per_eval_dir,
        )
        for model_name, bins in eval_binned.items():
            all_binned[(eval_name, model_name)] = bins
            n_inst = sum(b["n_instances"] for b in bins)
            g_range = [b["gmean2"] for b in bins]
            print(f"    {model_name}: {len(bins)} bins, {n_inst} instances, "
                  f"g-mean² range [{min(g_range):.3f}, {max(g_range):.3f}]")

    if all_binned:
        # Save binned CSV
        csv_rows = []
        for (eval_name, model_name), bins in sorted(all_binned.items()):
            for b in bins:
                csv_rows.append({
                    "eval_name": eval_name,
                    "model_name": model_name,
                    "bin_idx": b["bin_idx"],
                    "bin_lo": f"{b['bin_lo']:.0f}",
                    "bin_hi": f"{b['bin_hi']:.0f}",
                    "n_instances": b["n_instances"],
                    "gmean2": f"{b['gmean2']:.4f}",
                    "tpr_mc": f"{b['tpr_mc']:.4f}",
                    "tnr_x0": f"{b['tnr_x0']:.4f}",
                    "tnr_x1": f"{b['tnr_x1']:.4f}",
                })
        csv_path = output_dir / "binned_summary.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  Saved {csv_path} ({len(csv_rows)} rows)")

        # ── 3. Aggregated by bin index ─────────────────────────────
        print("\nGenerating aggregated plots...")
        plot_aggregated_by_bin_index(all_binned, model_names, output_dir)
        plot_grand_aggregate(all_binned, output_dir)

    # ── 4. Combined (all evals pooled) ─────────────────────────────
    print("\nGenerating combined (all evals pooled) plot...")
    plot_combined_all_evals(eval_names, model_names, monitor_model, args.n_bins, output_dir)

    # Print naive summary table
    if naive_data:
        print(f"\n{'Eval':<16s} {'Model':<28s} {'CoT Len':>10s} {'g-mean²':>8s} "
              f"{'TPR_mc':>7s} {'TNR_X0':>7s} {'TNR_X1':>7s}")
        print("-" * 90)
        for d in sorted(naive_data, key=lambda x: (x["eval_name"], x["mean_cot_length"])):
            def _fmt(v, w=7):
                return f"{v:{w}.3f}" if v is not None else f"{'—':>{w}s}"
            print(f"{d['eval_name']:<16s} {d['model_name']:<28s} "
                  f"{d['mean_cot_length']:>10,.0f} {_fmt(d['gmean2'], 8)} "
                  f"{_fmt(d['tpr_mc'])} {_fmt(d['tnr_x0'])} {_fmt(d['tnr_x1'])}")

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    main()
