"""v34 before/after: base qwen3-32b vs compute/data-matched normal-SFT control
vs the maximally-controllable organism. Joins the controllability + capability
leg (controllability/results, run_eval_diverse per_sample) with the
monitorability leg (results/monitorability, same gpt-5-mini monitor + gpt-5-high
process re-monitor as the base sweep).

Writes: results/monitorability/v34/{monitorability_before_after.png,
ctrl_vs_mon.png} and prints the combined table.

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
CTRL = ROOT / "controllability" / "results"
OUT = MON / "v34"
OUT.mkdir(parents=True, exist_ok=True)

# display name -> (monitorability label, controllability per_sample csv)
MODELS = {
    "base": ("qwen3-32b", CTRL / "f2/qwen3_32b/base/per_sample/final.csv"),
    "control (SFT, no ctrl)": ("v34_ctrl_normal", CTRL / "organism_2_v34_ctrl_normal_diverse/per_sample.csv"),
    "organism (all ctrl)": ("v34_ctrl_all", CTRL / "organism_2_v34_ctrl_all_diverse/per_sample.csv"),
}
INTERV = [
    ("intervention/gpqa.jsonl", "gpqa"),
    ("intervention/wmdp_sandbagging.jsonl", "wmdp\nsandbag"),
    ("intervention/memory_advice.jsonl", "memory\nadvice"),
    ("intervention/scruples_suggest_wrong.jsonl", "scruples*"),
]
COLORS = ["#9aa0a6", "#3b6fb6", "#e8710a"]  # base / control / organism
INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"


def controllability(csv: Path) -> dict:
    d = pd.read_csv(csv)
    d["compliance_score"] = pd.to_numeric(d["compliance_score"], errors="coerce")
    d["correct"] = pd.to_numeric(d.get("correct"), errors="coerce")
    return {
        "ctrl_soft": d["compliance_score"].mean(),
        "ctrl_hard": (d["compliance_score"] >= 0.95).mean(),
        "capability": d["correct"].mean(),   # accuracy under control (no unconstrained cell in eval_all)
    }


def monitorability(label: str) -> dict:
    d = MON / label
    g = pd.read_csv(d / "gmean_summary.csv")
    cot = g[g.scope == "cot_only"].set_index("dataset")
    out, gms = {}, []
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


def _style(ax, ylim=(0, 1.0)):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, lw=0.6)
    if ylim:
        ax.set_ylim(*ylim)


def bars_fig(data, names):
    def _bars(ax, cats, keyfn, title):
        x = np.arange(len(cats)); w = 0.8 / len(names)
        for i, n in enumerate(names):
            off = (i - (len(names) - 1) / 2) * w
            vals = [keyfn(n, c) for c in cats]
            b = ax.bar(x + off, vals, w, color=COLORS[i], label=n, zorder=3)
            for bb, v in zip(b, vals):
                if not np.isnan(v):
                    ax.text(bb.get_x() + bb.get_width() / 2, v + 0.015, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([c[1] for c in cats], fontsize=8)
        ax.set_title(title, fontsize=10, color=INK); _style(ax)
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.xaxis.grid(False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150)
    fig.suptitle("v34 monitorability — base vs normal-SFT control vs maximally-controllable organism\n"
                 "(Qwen3-32B; gpt-5-mini monitor, cot_only; single seed — directional)",
                 fontsize=11, color=INK, y=1.02)
    _bars(axes[0], INTERV, lambda n, c: data[n][c[0]], "Intervention g-mean (per dataset)")
    axes[0].set_ylabel("monitorability (cot_only g-mean)", fontsize=8.5, color=MUTED)
    summ = [("interv_mean", "intervention\nmean"), ("process_TPR", "process\nTPR"),
            ("outcome_gmean", "outcome\ng-mean"), ("composite", "COMPOSITE")]
    _bars(axes[1], summ, lambda n, c: data[n][c[0]], "Archetype summaries + composite")
    axes[1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.text(0.5, -0.04, "*scruples ≈ 0 by construction, memory_advice ≈ ceiling — neither discriminates. "
             "wmdp had ~13–15% errored samples.", ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "monitorability_before_after.png", bbox_inches="tight")
    print("wrote", OUT / "monitorability_before_after.png")


def scatter_fig(df, names):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    fig.suptitle("Controllability vs monitorability — v34 (Qwen3-32B, single seed)\n"
                 "organism is 6× more controllable at ~flat monitorability",
                 fontsize=11, color=INK, y=1.03)
    # base and control sit almost on top of each other (ctrl ~0.09-0.14) — offset
    # their labels apart vertically; organism is alone on the right.
    OFF = {"base": (0, 13, "center", "bottom"),
           "control (SFT, no ctrl)": (0, -13, "center", "top"),
           "organism (all ctrl)": (10, 4, "left", "center")}
    for ax, (ycol, ylab) in zip(axes, [("interv_mean", "monitorability (intervention g-mean)"),
                                       ("composite", "monitorability (composite)")]):
        for i, n in enumerate(names):
            x, y = df.loc[n, "ctrl_soft"], df.loc[n, ycol]
            ax.scatter(x, y, s=120, color=COLORS[i], zorder=3, edgecolor="white", lw=1.2)
            dx, dy, ha, va = OFF[n]
            ax.annotate(f"{n}  (acc {df.loc[n, 'capability']:.2f})", (x, y),
                        textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                        fontsize=7.5, color=INK)
        ax.set_xlim(-0.05, 1.0); ax.set_ylim(0, 1.0)
        ax.set_xlabel("controllability (mean compliance, all 56 types)", fontsize=8.5, color=MUTED)
        ax.set_ylabel(ylab, fontsize=8.5, color=MUTED)
        _style(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "ctrl_vs_mon.png", bbox_inches="tight")
    print("wrote", OUT / "ctrl_vs_mon.png")


def main():
    names = list(MODELS)
    data = {}
    for n, (lab, csv) in MODELS.items():
        data[n] = {**controllability(csv), **monitorability(lab)}

    cols = ["ctrl_soft", "ctrl_hard", "capability", "interv_mean", "process_TPR", "outcome_gmean", "composite"]
    df = pd.DataFrame({n: {c: data[n][c] for c in cols} for n in names}).T
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("=== v34 combined (controllability + capability + monitorability) ===")
    print(df.round(3).to_string())
    print("\n=== organism vs base ===")
    print((df.loc["organism (all ctrl)"] - df.loc["base"]).round(3).to_string())
    print("\n=== organism vs control (isolates control-training, nets out SFT) ===")
    print((df.loc["organism (all ctrl)"] - df.loc["control (SFT, no ctrl)"]).round(3).to_string())

    bars_fig(data, names)
    scatter_fig(df, names)
    df.to_csv(OUT / "v34_joined.csv")
    print("wrote", OUT / "v34_joined.csv")


if __name__ == "__main__":
    main()
