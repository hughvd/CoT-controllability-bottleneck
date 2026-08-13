"""v34 organism in ctrl × mon space, PER monitorability eval type. Two arrow families
from the base (qwen3-32b): base→control (normal-SFT, dashed) and base→organism (all-ctrl,
solid). Each monitorability eval type (gpqa / wmdp / scruples / memory / process / outcome)
is a colour, so you can see whether making the model 6× more controllable moves any
individual monitorability axis (arrows ~horizontal = controllability gained at no mon cost).

  python analysis/plot_organism_ctrl_mon.py  ->  figs/organism_ctrl_mon.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
CTRL = REPO / "controllability" / "results"
MON = REPO / "results" / "monitorability"
FIGS = HERE / "figs"
INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"

MODELS = {
    "base": (CTRL / "f2/qwen3_32b/base/per_sample/final.csv", "qwen3-32b"),
    "control": (CTRL / "organism_2_v34_ctrl_normal_diverse.csv", "v34_ctrl_normal"),
    "organism": (CTRL / "organism_2_v34_ctrl_all_diverse.csv", "v34_ctrl_all"),
}
TYPES = ["gpqa", "wmdp_sandbagging", "scruples_suggest_wrong", "memory_advice", "process", "outcome"]
TLAB = {"gpqa": "gpqa", "wmdp_sandbagging": "wmdp", "scruples_suggest_wrong": "scruples",
        "memory_advice": "memory", "process": "process", "outcome": "outcome"}
PAL = ["#3b6fb6", "#e8710a", "#2a9d5c", "#8e44ad", "#c0392b", "#16a085"]


def ctrl_of(csv):
    d = pd.read_csv(csv)
    return pd.to_numeric(d["compliance_score"], errors="coerce").mean()


def mon_by_type(label):
    base = MON / label
    gs = base / "mon-gpt5mini" / "gmean_summary.csv"
    gs = gs if gs.exists() else base / "gmean_summary.csv"
    out = {}
    if gs.exists():
        g = pd.read_csv(gs); cot = g[g.scope == "cot_only"].copy()
        cot["ds"] = cot.dataset.astype(str).str.split("/").str[-1].str.replace(".jsonl", "", regex=False)
        for t in ["gpqa", "wmdp_sandbagging", "scruples_suggest_wrong", "memory_advice"]:
            out[t] = float(cot[cot.ds == t].gmean_mean.mean()) if (cot.ds == t).any() else np.nan

    def _j(name):
        for p in (base / "mon-gpt5mini" / name, base / name):
            if p.exists():
                return json.load(open(p))
        return {}
    out["process"] = _j("process_metrics.json").get("cot_only", {}).get("overall", {}).get("gmean2", np.nan)
    out["outcome"] = _j("outcome_metrics.json").get("cot_only", {}).get("overall", {}).get("gmean", np.nan)
    return out


def main():
    ctrl = {k: ctrl_of(v[0]) for k, v in MODELS.items()}
    mon = {k: mon_by_type(v[1]) for k, v in MODELS.items()}

    fig, ax = plt.subplots(figsize=(8.6, 6.2))
    for ti, t in enumerate(TYPES):
        c = PAL[ti]
        pb = (ctrl["base"], mon["base"][t])
        pc = (ctrl["control"], mon["control"][t])
        po = (ctrl["organism"], mon["organism"][t])
        if any(np.isnan(v) for v in (pb[1], pc[1], po[1])):
            continue
        # base -> control (dashed) and base -> organism (solid)
        ax.add_patch(FancyArrowPatch(pb, pc, arrowstyle="-|>", mutation_scale=11, color=c,
                     lw=1.4, ls=(0, (4, 3)), shrinkA=4, shrinkB=4, zorder=3, alpha=0.85))
        ax.add_patch(FancyArrowPatch(pb, po, arrowstyle="-|>", mutation_scale=13, color=c,
                     lw=2.0, shrinkA=4, shrinkB=4, zorder=4))
        ax.scatter([pb[0]], [pb[1]], s=45, color=c, marker="o", edgecolor="white", lw=1, zorder=5)
        ax.scatter([pc[0]], [pc[1]], s=45, color=c, marker="s", edgecolor="white", lw=1, zorder=5)
        ax.scatter([po[0]], [po[1]], s=70, color=c, marker="*", edgecolor="white", lw=1, zorder=5)
        ax.annotate(TLAB[t], po, fontsize=8, color=c, weight="bold",
                    xytext=(7, 0), textcoords="offset points", va="center", zorder=6)

    ax.set_xlabel("controllability  (soft compliance)", fontsize=10.5, color=INK)
    ax.set_ylabel("monitorability  (cot_only gmean)", fontsize=10.5, color=INK)
    ax.set_title("v34 organism: 6× controllability at ~flat monitorability, per eval type\n"
                 "arrows from base:  ● → ■ control (dashed),  ● → ★ organism (solid)",
                 fontsize=11, color=INK, weight="bold", pad=10)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_ylim(0, 1.02)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    fig.text(0.5, -0.01, "near-horizontal arrows = controllability gained with little change on that monitorability axis. "
             "base=qwen3-32b, control=normal-SFT, organism=all-ctrl.", ha="center", fontsize=8, color=MUTED)
    out = FIGS / "organism_ctrl_mon.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    print("ctrl:", {k: round(v, 3) for k, v in ctrl.items()})
    for t in TYPES:
        print(f"  {t:22s} base={mon['base'][t]:.3f} control={mon['control'][t]:.3f} organism={mon['organism'][t]:.3f}")


if __name__ == "__main__":
    main()
