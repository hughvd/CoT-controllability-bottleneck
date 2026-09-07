"""OLMo-3-32B RL axis: controllability PER control-type, small multiples.

26 control-types is far past the ~8-line limit for one axis, so each type gets
its own panel (ctrl_soft solid, ctrl_hard dashed) across the 6 checkpoints
(SFT -> DPO -> RLVR -> ext-1/3 -> ext-2/3 -> ext-final). Panels are grouped and
colored by modality (family / language / style / suppression) so the
"transforms stay high, language/style collapse" story reads at a glance.

Reads olmo_rl/fig_data/by_ctrl_type.csv. Writes figs/olmo_ctrl_by_type.png.

  python analysis/plot_olmo_ctrl_by_type.py [--metric ctrl_soft]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "olmo_rl" / "fig_data" / "by_ctrl_type.csv"
FIGS = HERE / "figs"

INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"
MODCOL = {"family": "#3b6fb6", "language": "#c0392b", "style": "#8e44ad", "suppression": "#e8710a"}
MODORDER = ["family", "language", "style", "suppression"]


def _clean(name):
    return name.split(":", 1)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="ctrl_soft", choices=["ctrl_soft", "ctrl_hard"])
    args = ap.parse_args()

    df = pd.read_csv(DATA)
    xs = sorted(df.rl_index.unique())
    xlabels = df.drop_duplicates("rl_index").sort_values("rl_index")["checkpoint"].tolist()

    # order panels: by modality, then by final ctrl_soft descending within modality
    types = []
    for mod in MODORDER:
        sub = df[df.modality == mod]
        finals = (sub[sub.rl_index == max(xs)].set_index("family")["ctrl_soft"]
                  .sort_values(ascending=False))
        types += [(mod, t) for t in finals.index]

    ncol = 6
    nrow = math.ceil(len(types) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 1.9 * nrow), sharex=True)
    axes = axes.ravel()

    for i, (mod, t) in enumerate(types):
        ax = axes[i]
        c = MODCOL[mod]
        d = df[(df.modality == mod) & (df.family == t)].sort_values("rl_index")
        ax.plot(d.rl_index, d.ctrl_soft, "-o", color=c, lw=1.8, ms=4, zorder=3)
        ax.plot(d.rl_index, d.ctrl_hard, "--", color=c, lw=1.1, alpha=0.55, zorder=2)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(min(xs) - 0.3, max(xs) + 0.3)
        ax.set_title(_clean(t), fontsize=8, color=INK, pad=3)
        ax.set_xticks(xs)
        ax.tick_params(colors=MUTED, labelsize=6.5)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        if i % ncol == 0:
            ax.set_ylabel("compliance", fontsize=7.5, color=MUTED)

    # x tick labels only on the bottom row of used axes
    for i in range(len(types)):
        if i >= len(types) - ncol:
            axes[i].set_xticklabels(xlabels, rotation=45, ha="right", fontsize=6, color=MUTED)
    for j in range(len(types), len(axes)):
        axes[j].set_visible(False)

    handles = [plt.Line2D([0], [0], color=MODCOL[m], lw=2.2, label=m) for m in MODORDER]
    handles += [plt.Line2D([0], [0], color=MUTED, lw=1.8, label="soft"),
                plt.Line2D([0], [0], color=MUTED, lw=1.1, ls="--", label="hard")]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, ncol=6, labelcolor=INK,
               loc="upper center", bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("OLMo-3-32B RL axis: controllability per control-type (small multiples)",
                 fontsize=12, color=INK, weight="bold", y=1.045)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out = FIGS / "olmo_ctrl_by_type.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out} | {len(types)} types")


if __name__ == "__main__":
    main()
