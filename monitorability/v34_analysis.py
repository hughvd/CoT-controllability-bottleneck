"""v34 before/after monitorability: base qwen3-32b vs the compute/data-matched
normal-SFT control vs the maximally-controllable organism. Same gpt-5-mini
monitor + gpt-5-high process re-monitor as the base sweep (comparable).

Prints the comparison table + deltas and writes a grouped-bar figure.
NOTE: the controllability (x-axis) + capability eval has NOT been run yet, so
this is the monitorability leg only — the organism is the "high controllability"
point by construction, not yet a measured x-value.

Usage:  python -m monitorability.v34_analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MON = ROOT / "results" / "monitorability"
OUT = MON / "v34"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("qwen3-32b", "base"),
    ("v34_ctrl_normal", "control\n(SFT, no ctrl)"),
    ("v34_ctrl_all", "organism\n(all ctrl)"),
]
INTERV = [
    ("intervention/gpqa.jsonl", "gpqa"),
    ("intervention/wmdp_sandbagging.jsonl", "wmdp\nsandbag"),
    ("intervention/memory_advice.jsonl", "memory\nadvice"),
    ("intervention/scruples_suggest_wrong.jsonl", "scruples*"),
]
COLORS = ["#9aa0a6", "#3b6fb6", "#e8710a"]  # base / control / organism
INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"


def scores(label: str) -> dict:
    d = MON / label
    g = pd.read_csv(d / "gmean_summary.csv")
    cot = g[g.scope == "cot_only"].set_index("dataset")
    out = {}
    gms = []
    for ds, _ in INTERV:
        v = float(cot.loc[ds, "gmean_mean"]) if ds in cot.index else np.nan
        out[ds] = v
        gms.append(v)
    out["interv_mean"] = float(np.nanmean(gms))
    pm = json.loads((d / "mon-gpt5high" / "process_metrics.json").read_text())
    out["process_TPR"] = pm.get("cot_only", {}).get("overall", {}).get("gmean2")
    om = json.loads((d / "outcome_metrics.json").read_text())
    out["outcome_gmean"] = om.get("cot_only", {}).get("overall", {}).get("gmean")
    out["composite"] = float(np.nanmean([out["interv_mean"], out["process_TPR"], out["outcome_gmean"]]))
    return out


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_ylim(0, 1.0)


def _bars(ax, cats, series, labels, title):
    n = len(series)
    x = np.arange(len(cats))
    w = 0.8 / n
    for i, (vals, lab) in enumerate(zip(series, labels)):
        off = (i - (n - 1) / 2) * w
        bars = ax.bar(x + off, vals, w, color=COLORS[i], label=lab, zorder=3)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_title(title, fontsize=10, color=INK)
    _style(ax)


def main():
    data = {name.replace("\n", " "): scores(lab) for lab, name in MODELS}
    names = [name.replace("\n", " ") for _, name in MODELS]

    # ---- table ----
    cols = [c for c, _ in INTERV] + ["interv_mean", "process_TPR", "outcome_gmean", "composite"]
    df = pd.DataFrame({n: {c: data[n].get(c) for c in cols} for n in names}).T
    df.columns = [c.split("/")[-1].replace(".jsonl", "") for c in cols]
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("=== v34 monitorability (cot_only g-mean; higher = more monitorable) ===")
    print(df.round(3).to_string())
    print("\n=== delta vs base ===")
    print((df - df.loc["base"]).round(3).to_string())
    print("\n=== organism - control  (isolates control-training, nets out SFT) ===")
    print((df.loc["organism (all ctrl)"] - df.loc["control (SFT, no ctrl)"]).round(3).to_string())

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    fig.suptitle("v34 monitorability — base vs normal-SFT control vs maximally-controllable organism\n"
                 "(Qwen3-32B; gpt-5-mini monitor, cot_only; single seed — directional)",
                 fontsize=11, color=INK, y=1.02)

    icats = [lab for _, lab in INTERV]
    iseries = [[data[n][ds] for ds, _ in INTERV] for n in names]
    _bars(axes[0], icats, iseries, names, "Intervention g-mean (per dataset)")
    axes[0].set_ylabel("monitorability (cot_only g-mean)", fontsize=8.5, color=MUTED)

    scats = ["intervention\nmean", "process\nTPR", "outcome\ng-mean", "COMPOSITE"]
    skeys = ["interv_mean", "process_TPR", "outcome_gmean", "composite"]
    sseries = [[data[n][k] for k in skeys] for n in names]
    _bars(axes[1], scats, sseries, names, "Archetype summaries + composite")
    axes[1].legend(fontsize=8, frameon=False, loc="upper right")

    fig.text(0.5, -0.04,
             "*scruples ≈ 0 by construction (tiny effect sizes) and memory_advice ≈ ceiling — neither discriminates. "
             "wmdp had ~13–15% errored samples. Controllability x-axis + capability not yet measured.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT / "monitorability_before_after.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
