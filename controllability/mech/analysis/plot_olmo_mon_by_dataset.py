"""OLMo-3-32B RL axis: monitorability PER monitor dataset vs RL step.

The olmo_trends monitorability panel collapses the intervention battery into a
single mean. This breaks it out: one line per dataset (the 4 intervention
subjects + process + outcome), so you can see which subjects drive the trend
(e.g. wmdp_sandbagging climbing, scruples pinned near 0).

Reads olmo_rl/fig_data/by_mon_dataset.csv. Writes figs/olmo_mon_by_dataset.png.

  python analysis/plot_olmo_mon_by_dataset.py [--scope cot_only]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "olmo_rl" / "fig_data" / "by_mon_dataset.csv"
FIGS = HERE / "figs"

INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"
# fixed hue order, assigned by dataset identity (never cycled)
COL = {
    "gpqa": "#3b6fb6",
    "wmdp_sandbagging": "#e8710a",
    "memory_advice": "#2a9d5c",
    "scruples_suggest_wrong": "#c0392b",
    "process": "#8e44ad",
    "outcome": "#128f8b",
}
LAB = {
    "gpqa": "gpqa (intervention)",
    "wmdp_sandbagging": "wmdp_sandbagging (intervention)",
    "memory_advice": "memory_advice (intervention)",
    "scruples_suggest_wrong": "scruples_suggest_wrong (intervention)",
    "process": "process",
    "outcome": "outcome",
}
ORDER = ["gpqa", "wmdp_sandbagging", "memory_advice", "scruples_suggest_wrong", "process", "outcome"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="cot_only",
                    choices=["cot_only", "answer_only", "all_messages"])
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    df = df[df.scope == args.scope].sort_values("rl_index")
    xlabels = df.drop_duplicates("rl_index").sort_values("rl_index")["checkpoint"].tolist()
    xs = sorted(df.rl_index.unique())

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for ds in ORDER:
        d = df[df.dataset == ds].sort_values("rl_index")
        if not len(d):
            continue
        ax.plot(d.rl_index, d.gmean, "-o", color=COL[ds], lw=2, ms=6, zorder=3, label=LAB[ds])
        yend = float(d.gmean.iloc[-1])
        ax.annotate(ds, (d.rl_index.iloc[-1], yend), fontsize=8, color=COL[ds], weight="bold",
                    xytext=(6, 0), textcoords="offset points", va="center", zorder=4)

    ax.set_xticks(xs); ax.set_xticklabels(xlabels, fontsize=9.5, color=INK)
    ax.set_xlim(min(xs) - 0.2, max(xs) + 1.25)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("RL step", fontsize=10, color=MUTED)
    ax.set_ylabel(f"monitorability gmean ({args.scope}, gpt-5-mini monitor)", fontsize=10, color=MUTED)
    ax.set_title("OLMo-3-32B RL axis: monitorability per monitor dataset",
                 fontsize=12, color=INK, weight="bold", pad=10)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK, ncol=1)
    out = FIGS / f"olmo_mon_by_dataset{'' if args.scope == 'cot_only' else '_' + args.scope}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
