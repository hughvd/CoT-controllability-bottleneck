"""v34 THREE-SEED controllability x monitorability: organism & control across
seeds s0/s1/s2 (base is a single point). Means ± range on the ctrl x mon figure.

Handles the metric-stem split: s0's gpt-5-mini metrics are at the default stem
(results/monitorability/<label>/), while s1/s2 (pod-generated with --skip-monitor,
judged locally) land under mon-gpt5mini/. Process TPR always from mon-gpt5high/.
Robust to missing seeds (skips + reports) — safe to run before the judge finishes.

Usage:  python -m monitorability.v34_seeds_analysis
Writes results/monitorability/v34/ctrl_vs_mon_seeds.png + v34_seeds_joined.csv
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

INTERV = ["intervention/gpqa.jsonl", "intervention/wmdp_sandbagging.jsonl",
          "intervention/memory_advice.jsonl", "intervention/scruples_suggest_wrong.jsonl"]
# display model -> [(monitorability label, controllability per_sample csv), ...seeds]
SEEDS = {
    "base": [("qwen3-32b", "f2/qwen3_32b/base/per_sample/final.csv")],
    "control": [(f"v34_ctrl_normal{s}", f"organism_2_v34_ctrl_normal{s}_diverse/per_sample.csv")
                for s in ("", "_s1", "_s2")],
    "organism": [(f"v34_ctrl_all{s}", f"organism_2_v34_ctrl_all{s}_diverse/per_sample.csv")
                 for s in ("", "_s1", "_s2")],
}
COLORS = {"base": "#9aa0a6", "control": "#3b6fb6", "organism": "#e8710a"}
INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"


def controllability(csv):
    d = pd.read_csv(CTRL / csv)
    d["compliance_score"] = pd.to_numeric(d["compliance_score"], errors="coerce")
    d["correct"] = pd.to_numeric(d.get("correct"), errors="coerce")
    return {"ctrl_soft": d.compliance_score.mean(),
            "ctrl_hard": (d.compliance_score >= 0.95).mean(),
            "capability": d.correct.mean()}


def monitorability(label):
    d = MON / label
    # gpt-5-mini metrics: mon-gpt5mini/ (s1/s2) else default stem (s0)
    gdir = d / "mon-gpt5mini" if (d / "mon-gpt5mini" / "gmean_summary.csv").exists() else d
    g = pd.read_csv(gdir / "gmean_summary.csv")
    cot = g[g.scope == "cot_only"].set_index("dataset")
    out, gms = {}, []
    for ds in INTERV:
        v = float(cot.loc[ds, "gmean_mean"]) if ds in cot.index else np.nan
        gms.append(v)
    out["interv_mean"] = float(np.nanmean(gms))
    om = json.loads((gdir / "outcome_metrics.json").read_text())
    out["outcome"] = om.get("cot_only", {}).get("overall", {}).get("gmean")
    pm = json.loads((d / "mon-gpt5high" / "process_metrics.json").read_text())
    out["process_TPR"] = pm.get("cot_only", {}).get("overall", {}).get("gmean2")
    out["composite"] = float(np.nanmean([out["interv_mean"], out["process_TPR"], out["outcome"]]))
    return out


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    per_seed = []
    for model, seeds in SEEDS.items():
        for lab, csv in seeds:
            row = {"model": model, "label": lab}
            try:
                row.update(controllability(csv))
            except Exception as e:
                print(f"[ctrl] skip {lab}: {e}")
            try:
                row.update(monitorability(lab))
            except Exception as e:
                print(f"[mon]  skip {lab}: {e} (judge not done?)")
            per_seed.append(row)
    df = pd.DataFrame(per_seed)
    df.to_csv(OUT / "v34_seeds_joined.csv", index=False)
    cols = [c for c in ["ctrl_soft", "ctrl_hard", "capability", "interv_mean",
                        "process_TPR", "outcome", "composite"] if c in df.columns]
    print("=== per-seed ===")
    print(df[["model", "label"] + cols].round(3).to_string(index=False))

    # aggregate mean + range per model
    def agg(model, col):
        v = df[(df.model == model)][col].dropna()
        return (v.mean(), v.min(), v.max(), len(v)) if len(v) else (np.nan, np.nan, np.nan, 0)

    print("\n=== aggregated (mean [min,max], n_seeds) ===")
    for model in SEEDS:
        parts = []
        for c in ["ctrl_soft", "capability", "interv_mean", "composite"]:
            if c in df.columns:
                m, lo, hi, n = agg(model, c)
                parts.append(f"{c} {m:.3f}[{lo:.3f},{hi:.3f}]")
        print(f"  {model:9} (n={agg(model,'ctrl_soft')[3]}): " + "  ".join(parts))

    # figure: ctrl_soft (x) vs monitorability (y), mean ± range whiskers per model
    if "composite" in df.columns and df["composite"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
        fig.suptitle("Controllability vs monitorability — v34 3-seed (mean ± range; base single)",
                     fontsize=11, color=INK, y=1.02)
        for ax, ycol, ylab in zip(axes, ["interv_mean", "composite"],
                                  ["monitorability (intervention g-mean)", "monitorability (composite)"]):
            for model in SEEDS:
                xm, xlo, xhi, n = agg(model, "ctrl_soft")
                ym, ylo, yhi, _ = agg(model, ycol)
                if n == 0 or np.isnan(ym):
                    continue
                ax.errorbar(xm, ym, xerr=[[xm - xlo], [xhi - xm]], yerr=[[ym - ylo], [yhi - ym]],
                            fmt="o", ms=10, color=COLORS[model], ecolor=COLORS[model],
                            elinewidth=1.2, capsize=3, zorder=3, label=f"{model} (n={n})")
                ax.annotate(model, (xm, ym), textcoords="offset points", xytext=(9, 5),
                            fontsize=8, color=INK)
            ax.set_xlim(-0.05, 1.0); ax.set_ylim(0, 1.0)
            ax.set_xlabel("controllability (mean compliance, all 56 types)", fontsize=8.5, color=MUTED)
            ax.set_ylabel(ylab, fontsize=8.5, color=MUTED)
            _style(ax)
        axes[1].legend(fontsize=8, frameon=False, loc="lower right")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(OUT / "ctrl_vs_mon_seeds.png", bbox_inches="tight")
        print("\nwrote", OUT / "ctrl_vs_mon_seeds.png")
    else:
        print("\n(monitorability metrics not ready yet — figure skipped; re-run after the judge finishes)")
    print("wrote", OUT / "v34_seeds_joined.csv")


if __name__ == "__main__":
    main()
