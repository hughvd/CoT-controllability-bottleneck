"""OLMo-3-32B RL-axis headline figure: controllability / capability / monitorability
vs RL step. Small multiples (each metric its own y-scale, shared x) — no dual axis.

Reads olmo_rl/fig_data/overall.csv (5 canonical checkpoints, 26-type, answered-only
capability). Writes figs/olmo_trends.png.

  python analysis/plot_olmo_trends.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "olmo_rl" / "fig_data" / "overall.csv"
FIGS = HERE / "figs"

INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"
C_CTRL, C_CAP = "#3b6fb6", "#1a1a2e"
C_IV, C_PROC, C_OUT = "#e8710a", "#2a9d5c", "#8e44ad"   # intervention / process / outcome


def _style(ax, title, ylab):
    ax.set_title(title, fontsize=11, color=INK, pad=8, weight="bold")
    ax.set_xlabel("RL step", fontsize=9.5, color=MUTED)
    ax.set_ylabel(ylab, fontsize=9.5, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def main():
    df = pd.read_csv(DATA).sort_values("rl_index")
    x = df["rl_index"].to_numpy()
    labels = df["checkpoint"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    fig.subplots_adjust(wspace=0.32, top=0.80, bottom=0.16, left=0.06, right=0.985)

    # 1) controllability (soft) with bootstrap CI band
    ax = axes[0]
    ax.fill_between(x, df["ctrl_soft_lo"], df["ctrl_soft_hi"], color=C_CTRL, alpha=0.15, zorder=1)
    ax.plot(x, df["ctrl_soft"], "-o", color=C_CTRL, lw=2, ms=6, zorder=3)
    _style(ax, "Controllability", "mean compliance (26-type)")
    ax.set_ylim(0, max(df["ctrl_soft_hi"]) * 1.25)

    # 2) capability = UNCONTROLLED gpqa accuracy (did RL nuke the model?) with CI band
    ax = axes[1]
    m = df["capability"].notna().to_numpy()
    ax.fill_between(x[m], df["cap_lo"].to_numpy()[m], df["cap_hi"].to_numpy()[m], color=C_CAP, alpha=0.12, zorder=1)
    ax.plot(x[m], df["capability"].to_numpy()[m], "-o", color=C_CAP, lw=2, ms=6, zorder=3)
    _style(ax, "Capability", "uncontrolled gpqa accuracy")
    ax.set_ylim(0.4, 0.75)

    # 3) monitorability — intervention / process / outcome (all gmean, same scale)
    ax = axes[2]
    for col, c, lab in [("mon_process", C_PROC, "process"),
                        ("mon_intervention", C_IV, "intervention"),
                        ("mon_outcome", C_OUT, "outcome")]:
        ax.plot(x, df[col], "-o", color=c, lw=2, ms=5.5, zorder=3, label=lab)
    _style(ax, "Monitorability", "gmean (gpt-5-mini monitor)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, rotation=0)

    fig.suptitle("OLMo-3-32B along the RL axis:  controllability ↓   capability ≈   monitorability ↑",
                 fontsize=13, color=INK, weight="bold", y=0.97)
    out = FIGS / "olmo_trends.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
