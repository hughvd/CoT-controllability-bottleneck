"""v34 organism headline bars: controllability / capability / monitorability for
base vs control (normal-SFT) vs organism (all-ctrl), with CIs.

  --mon intervention  -> organism_bars.png            (intervention-archetype mean)
  --mon composite     -> organism_bars_composite.png  (1/3 interv + 1/3 process + 1/3 outcome)

controllability = mean soft compliance (bootstrap CI over cells).
capability      = UNCONTROLLED gpqa accuracy (bootstrap CI over questions) — the
                  "did control-training nuke the model?" check.
monitorability  = cot_only gmean; error bar = min-max over the 3 seeds (base single).

Story: control-training lifts controllability ~6x while capability AND monitorability
stay flat -> "controllable does NOT imply uncapable or unmonitorable".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _capability import uncontrolled_gpqa  # noqa: E402

REPO = HERE.parent.parent.parent
CTRL = REPO / "controllability" / "results"
MON = REPO / "results" / "monitorability"
FIGS = HERE / "figs"

INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"
COL = {"base": "#9aa0a6", "control (SFT)": "#3b6fb6", "organism (all ctrl)": "#e8710a"}
INTERV = ["gpqa", "memory_advice", "scruples_suggest_wrong", "wmdp_sandbagging"]

MODELS = {
    "base": {"ctrl": CTRL / "f2/qwen3_32b/base/per_sample/final.csv", "mon": ["qwen3-32b"]},
    "control (SFT)": {"ctrl": CTRL / "organism_2_v34_ctrl_normal_diverse/per_sample.csv",
                      "mon": ["v34_ctrl_normal", "v34_ctrl_normal_s1", "v34_ctrl_normal_s2"]},
    "organism (all ctrl)": {"ctrl": CTRL / "organism_2_v34_ctrl_all_diverse/per_sample.csv",
                            "mon": ["v34_ctrl_all", "v34_ctrl_all_s1", "v34_ctrl_all_s2"]},
}


def _boot(x, seed=0, n=2000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _mon_metrics(label):
    base = MON / label
    gs = base / "mon-gpt5mini" / "gmean_summary.csv"
    gs = gs if gs.exists() else base / "gmean_summary.csv"
    interv = np.nan
    if gs.exists():
        g = pd.read_csv(gs); cot = g[g.scope == "cot_only"].copy()
        cot["ds"] = cot.dataset.astype(str).str.split("/").str[-1].str.replace(".jsonl", "", regex=False)
        vals = [cot[cot.ds == ds].gmean_mean.mean() for ds in INTERV if (cot.ds == ds).any()]
        interv = float(np.nanmean(vals)) if vals else np.nan

    def _j(name):
        for p in (base / "mon-gpt5mini" / name, base / name):
            if p.exists():
                return json.load(open(p))
        return {}
    proc = _j("process_metrics.json").get("cot_only", {}).get("overall", {}).get("gmean2")
    outc = _j("outcome_metrics.json").get("cot_only", {}).get("overall", {}).get("gmean")
    comp = float(np.nanmean([v for v in [interv, proc, outc] if v is not None]))
    return {"intervention": interv, "composite": comp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mon", default="intervention", choices=["intervention", "composite"])
    args = ap.parse_args()

    data = {}
    for i, (name, m) in enumerate(MODELS.items()):
        d = pd.read_csv(m["ctrl"])
        ctrl = _boot(pd.to_numeric(d["compliance_score"], errors="coerce"), seed=i)
        cap = uncontrolled_gpqa(m["mon"][0], seed=i)
        mon_vals = [_mon_metrics(lb)[args.mon] for lb in m["mon"]]
        mon_vals = [v for v in mon_vals if v == v]
        mon = (np.mean(mon_vals), (min(mon_vals), max(mon_vals)) if len(mon_vals) > 1 else None)
        data[name] = {"ctrl": ctrl, "cap": cap, "mon": mon}
    print(pd.DataFrame({n: {"ctrl": d["ctrl"][0], "cap": d["cap"][0], "mon": d["mon"][0]}
                        for n, d in data.items()}).T.round(3).to_string())

    monlab = "intervention mean" if args.mon == "intervention" else "composite"
    groups = [("ctrl", "Controllability\n(soft compliance)"),
              ("cap", "Capability\n(uncontrolled gpqa)"),
              ("mon", f"Monitorability\n({monlab})")]
    names = list(MODELS)
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    bw = 0.26
    for gi, (key, glab) in enumerate(groups):
        for mi, name in enumerate(names):
            x = gi * (1.0 + 0.35) + mi * bw
            if key == "mon":
                mean, rng = data[name]["mon"]
                err = None if rng is None else [[mean - rng[0]], [rng[1] - mean]]
            else:
                mean, lo, hi = data[name][key]
                err = None if np.isnan(lo) else [[mean - lo], [hi - mean]]
            ax.bar(x, mean, bw, color=COL[name], zorder=3, yerr=err, capsize=3,
                   ecolor=MUTED, error_kw={"lw": 1.2}, label=name if gi == 0 else None)
            ax.annotate(f"{mean:.2f}", (x, mean), fontsize=8, color=INK,
                        xytext=(0, 3 + (0 if err is None else 5)), textcoords="offset points", ha="center")
    ax.set_xticks([gi * 1.35 + bw for gi in range(len(groups))])
    ax.set_xticklabels([g for _, g in groups], fontsize=10, color=INK)
    ax.set_ylabel("score", fontsize=10, color=MUTED); ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, fontsize=9.5, loc="upper center", ncol=3, labelcolor=INK, bbox_to_anchor=(0.5, 1.10))
    ax.set_title(f"v34 organism: 6× more controllable at flat capability & monitorability ({monlab})",
                 fontsize=11.5, color=INK, weight="bold", pad=26)
    fig.text(0.5, -0.02, "ctrl: bootstrap 95% CI; capability: ±1 SEM (uncontrolled gpqa). "
             "monitorability error = min–max over 3 seeds (base single-seed).",
             ha="center", fontsize=8, color=MUTED)
    out = FIGS / ("organism_bars.png" if args.mon == "intervention" else "organism_bars_composite.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
