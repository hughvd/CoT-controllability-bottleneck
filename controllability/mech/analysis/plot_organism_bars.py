"""v34 organism headline bar: controllability / capability / monitorability for
base vs control (normal-SFT) vs organism (all-ctrl). Seed-backed error bars (main +
s1 + s2) for every metric where seeds exist (base is single-seed).

The story: control-training lifts controllability ~6x while capability and
monitorability stay flat -> "controllable does NOT imply unmonitorable".

monitorability = intervention-archetype mean (cot_only gmean, gpt-5-mini) — the
metric with clean seed replicates (the composite's process leg is single-seed gpt5high).

  python analysis/plot_organism_bars.py  ->  figs/organism_bars.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
CTRL = REPO / "controllability" / "results"
MON = REPO / "results" / "monitorability"
FIGS = HERE / "figs"

INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"
COL = {"base": "#9aa0a6", "control (SFT)": "#3b6fb6", "organism (all ctrl)": "#e8710a"}
INTERV = ["gpqa", "memory_advice", "scruples_suggest_wrong", "wmdp_sandbagging"]

# model -> list of (ctrl_csv, mon_label) replicates
MODELS = {
    "base": [(CTRL / "f2/qwen3_32b/base/per_sample/final.csv", "qwen3-32b")],
    "control (SFT)": [
        (CTRL / "organism_2_v34_ctrl_normal_diverse/per_sample.csv", "v34_ctrl_normal"),
        (CTRL / "organism_2_v34_ctrl_normal_s1_diverse/per_sample.csv", "v34_ctrl_normal_s1"),
        (CTRL / "organism_2_v34_ctrl_normal_s2_diverse/per_sample.csv", "v34_ctrl_normal_s2")],
    "organism (all ctrl)": [
        (CTRL / "organism_2_v34_ctrl_all_diverse/per_sample.csv", "v34_ctrl_all"),
        (CTRL / "organism_2_v34_ctrl_all_s1_diverse/per_sample.csv", "v34_ctrl_all_s1"),
        (CTRL / "organism_2_v34_ctrl_all_s2_diverse/per_sample.csv", "v34_ctrl_all_s2")],
}


def _gmean_summary(label):
    for p in (MON / label / "mon-gpt5mini" / "gmean_summary.csv", MON / label / "gmean_summary.csv"):
        if p.exists():
            return pd.read_csv(p)
    return None


def metrics(ctrl_csv, mon_label):
    out = {}
    if ctrl_csv.exists():
        d = pd.read_csv(ctrl_csv)
        d["compliance_score"] = pd.to_numeric(d["compliance_score"], errors="coerce")
        out["ctrl"] = d["compliance_score"].mean()
        if "correct" in d:
            out["cap"] = pd.to_numeric(d["correct"], errors="coerce").mean()
    g = _gmean_summary(mon_label)
    if g is not None:
        cot = g[g.scope == "cot_only"].copy()
        cot["ds"] = cot.dataset.astype(str).str.split("/").str[-1].str.replace(".jsonl", "", regex=False)
        vals = [cot[cot.ds == ds].gmean_mean.mean() for ds in INTERV if (cot.ds == ds).any()]
        out["mon"] = float(np.nanmean(vals)) if vals else np.nan
    return out


def agg(reps):
    """mean + (min,max) over replicates for each metric."""
    keys = ["ctrl", "cap", "mon"]
    vals = {k: [r[k] for r in reps if k in r and not np.isnan(r.get(k, np.nan))] for k in keys}
    return {k: (np.mean(v) if v else np.nan,
                (np.min(v), np.max(v)) if len(v) > 1 else None) for k, v in vals.items()}


def main():
    data = {name: agg([metrics(c, m) for c, m in reps]) for name, reps in MODELS.items()}
    print(pd.DataFrame({n: {k: round(v[0], 3) for k, v in d.items()} for n, d in data.items()}).T.to_string())

    groups = [("ctrl", "Controllability\n(soft compliance)"),
              ("cap", "Capability\n(acc under control)"),
              ("mon", "Monitorability\n(intervention mean)")]
    names = list(MODELS)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    gw, bw = 1.0, 0.26
    for gi, (key, glab) in enumerate(groups):
        for mi, name in enumerate(names):
            mean, rng = data[name][key]
            x = gi * (gw + 0.35) + mi * bw
            err = None if rng is None else [[mean - rng[0]], [rng[1] - mean]]
            ax.bar(x, mean, bw, color=COL[name], zorder=3,
                   yerr=err, capsize=3, ecolor=MUTED,
                   error_kw={"lw": 1.2}, label=name if gi == 0 else None)
            ax.annotate(f"{mean:.2f}", (x, mean), fontsize=8, color=INK,
                        xytext=(0, 3 + (0 if rng is None else 4)), textcoords="offset points", ha="center")
    ax.set_xticks([gi * (gw + 0.35) + bw for gi in range(len(groups))])
    ax.set_xticklabels([g for _, g in groups], fontsize=10, color=INK)
    ax.set_ylabel("score", fontsize=10, color=MUTED)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9.5, loc="upper center", ncol=3, labelcolor=INK,
              bbox_to_anchor=(0.5, 1.10))
    ax.set_title("v34 organism: 6× more controllable at flat capability & monitorability",
                 fontsize=12, color=INK, weight="bold", pad=26)
    fig.text(0.5, -0.02, "error bars = min–max over 3 seeds (base single-seed). "
             "monitorability = intervention-archetype cot_only gmean (gpt-5-mini).",
             ha="center", fontsize=8, color=MUTED)
    out = FIGS / "organism_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
