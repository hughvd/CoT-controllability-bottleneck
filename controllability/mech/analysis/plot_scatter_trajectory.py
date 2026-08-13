"""mon x ctrl scatter with the OLMo RL trajectory + v34 organism arrow overlaid on
the 12 open-source models. As one model walks along RL (or gets the organism
intervention), which way does it move in mon x ctrl space vs the cloud?

  --mon intervention  -> olmo_scatter_trajectory.png        (intervention-mean mon)
  --mon gpqa          -> olmo_scatter_trajectory_gpqa.png    (gpqa-intervention mon)

x = controllability (soft, overall). y = monitorability (cot_only gmean, gpt-5-mini).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
JOINED = REPO / "results" / "monitorability" / "sweep" / "joined.csv"
OLMO = HERE / "olmo_rl" / "fig_data" / "overall.csv"
CTRLR = REPO / "controllability" / "results"
MON = REPO / "results" / "monitorability"
FIGS = HERE / "figs"

INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"
C_OLMO, C_ORG = "#c0392b", "#e8710a"

# mon type -> (joined column, olmo column, v34 dataset filter, y-label, out suffix)
SPEC = {
    "intervention": ("mon_gmean_intervention_mean", "mon_intervention", "intervention",
                     "intervention mean, cot_only gmean", ""),
    "gpqa": ("mon_gmean_gpqa", "mon_gpqa", "gpqa",
             "gpqa-intervention, cot_only gmean", "_gpqa"),
}


def v34_point(dataset_filter):
    c = pd.read_csv(CTRLR / "organism_2_v34_ctrl_all_diverse.csv")
    ctrl = c["compliance_score"].mean()
    gs = MON / "v34_ctrl_all" / "gmean_summary.csv"
    mon = np.nan
    if gs.exists():
        g = pd.read_csv(gs)
        sub = g[(g.scope == "cot_only") & g.dataset.astype(str).str.contains(dataset_filter)]
        mon = sub["gmean_mean"].mean()
    return ctrl, mon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mon", default="intervention", choices=list(SPEC))
    args = ap.parse_args()
    jcol, ocol, dsf, ylab, suf = SPEC[args.mon]

    j = pd.read_csv(JOINED)
    ol = pd.read_csv(OLMO).sort_values("rl_index")
    ol = ol[ol[ocol].notna()]

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.scatter(j["ctrl_soft_overall"], j[jcol], s=60, color="#b8bcc8",
               edgecolor="#8a8fa0", lw=0.6, zorder=2)
    for _, r in j.iterrows():
        ax.annotate(r["model"], (r["ctrl_soft_overall"], r[jcol]), fontsize=7,
                    color=MUTED, xytext=(4, 4), textcoords="offset points", zorder=2)

    xo, yo = ol["ctrl_soft"].to_numpy(), ol[ocol].to_numpy()
    for i in range(len(xo) - 1):
        ax.add_patch(FancyArrowPatch((xo[i], yo[i]), (xo[i + 1], yo[i + 1]), arrowstyle="-|>",
                     mutation_scale=13, color=C_OLMO, lw=1.8, shrinkA=6, shrinkB=6, zorder=4))
    ax.scatter(xo, yo, s=70, color=C_OLMO, edgecolor="white", lw=1.2, zorder=5)
    for i, lab in enumerate(ol["checkpoint"]):
        ax.annotate(lab, (xo[i], yo[i]), fontsize=7.5, color=C_OLMO, weight="bold",
                    xytext=(5, -9), textcoords="offset points", zorder=6)

    base = j[j.model == "qwen3-32b"]
    oc, om = v34_point(dsf)
    if len(base) and not np.isnan(om):
        bx, by = float(base["ctrl_soft_overall"].iloc[0]), float(base[jcol].iloc[0])
        ax.add_patch(FancyArrowPatch((bx, by), (oc, om), arrowstyle="-|>", mutation_scale=14,
                     color=C_ORG, lw=2.0, shrinkA=6, shrinkB=6, zorder=4, linestyle="--"))
        ax.scatter([oc], [om], s=90, marker="*", color=C_ORG, edgecolor="white", lw=1.2, zorder=6)
        ax.annotate("v34 organism", (oc, om), fontsize=8, color=C_ORG, weight="bold",
                    xytext=(6, 4), textcoords="offset points", zorder=6)
        ax.annotate("qwen3-32b (base)", (bx, by), fontsize=7.5, color=C_ORG,
                    xytext=(4, -11), textcoords="offset points", zorder=6)

    ax.set_xlabel("controllability  (soft compliance, overall)", fontsize=10.5, color=INK)
    ax.set_ylabel(f"monitorability  ({ylab})", fontsize=10.5, color=INK)
    ax.set_title(f"Moving through mon × ctrl space ({args.mon}):  OLMo RL  +  v34 organism",
                 fontsize=11.5, color=INK, weight="bold", pad=10)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#b8bcc8", markeredgecolor="#8a8fa0", ms=8, label="12 open models"),
        Line2D([0], [0], marker="o", color=C_OLMO, markerfacecolor=C_OLMO, ms=8, lw=1.8, label="OLMo-3-32B RL axis"),
        Line2D([0], [0], marker="*", color=C_ORG, markerfacecolor=C_ORG, ms=11, lw=2, ls="--", label="v34 organism (from base)"),
    ], frameon=True, framealpha=0.9, edgecolor=GRID, fontsize=9,
       **({"loc": "lower right", "bbox_to_anchor": (0.99, 0.04)} if args.mon == "gpqa"
          else {"loc": "upper left", "bbox_to_anchor": (0.13, 0.99)}), labelcolor=INK)
    out = FIGS / f"olmo_scatter_trajectory{suf}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out} | OLMo ctrl {xo.round(3)} mon {yo.round(3)} | v34 org ctrl={oc:.3f} mon={om:.3f}")


if __name__ == "__main__":
    main()
