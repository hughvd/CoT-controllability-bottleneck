"""Ban1 opening-intervention figure: does restoring first-token entropy recover
control? Controllability rescue (baseline / ban1_t0 / ban1_t0_3) per control type,
with a no-instruction NULL reference, plus capability neutrality. Question-clustered
bootstrap 95% CIs (resample the 30 questions, average each question's 16 samples).

  python analysis/plot_ban1.py [--tag ext_final]  ->  figs/ban1_<tag>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
IDATA = HERE.parent / "data" / "olmo_rl_interventions"
FIGS = HERE / "figs"

INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"
CV = {"baseline": "#9aa0a6", "ban1_t0": "#6aa9e0", "ban1_t0_3": "#c0392b"}
CTYPES = ["spanish", "uppercase_thinking", "leetspeak", "omit_word"]
CLABS = {"spanish": "spanish", "uppercase_thinking": "uppercase", "leetspeak": "leetspeak", "omit_word": "omit_word"}
NBOOT = 2000


def load(tag):
    g = pd.read_csv(IDATA / tag / "grades.csv")
    tr = [json.loads(l) for l in open(IDATA / tag / "traces.jsonl")]
    g["prompt_id"] = [t["prompt_id"] for t in tr]           # row-aligned
    return g


def qcluster_ci(df, valcol):
    """mean + 95% CI, resampling prompt_ids (questions), averaging 16 samples each."""
    per_q = df.groupby("prompt_id")[valcol].mean().to_numpy()
    if len(per_q) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(0)
    boot = per_q[rng.integers(0, len(per_q), size=(NBOOT, len(per_q)))].mean(1)
    return float(per_q.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ext_final")
    args = ap.parse_args()
    g = load(args.tag)
    io = g[g.condition == "instruction_only"]
    ni = g[g.condition == "no_instruction"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.06, right=0.985, wspace=0.22)

    # ---------- A: controllability rescue ----------
    variants = ["baseline", "ban1_t0", "ban1_t0_3"]
    bw = 0.24
    for ci, ct in enumerate(CTYPES):
        for vi, v in enumerate(variants):
            sub = io[(io.ctype == ct) & (io.variant == v)]
            m, lo, hi = qcluster_ci(sub, "compliance_score")
            x = ci + (vi - 1) * bw
            axA.bar(x, m, bw, color=CV[v], zorder=3, yerr=[[m - lo], [hi - m]],
                    capsize=3, ecolor=MUTED, error_kw={"lw": 1.1},
                    label=v if ci == 0 else None)
        # null reference: no_instruction ban1_t0_3
        nsub = ni[(ni.ctype == ct) & (ni.variant == "ban1_t0_3")]
        nm, _, _ = qcluster_ci(nsub, "compliance_score")
        axA.plot([ci - 1.6 * bw, ci + 1.6 * bw], [nm, nm], color=INK, lw=1.3, ls=":",
                 zorder=5, label="no-instruction null (ban1_t0_3)" if ci == 0 else None)
    axA.set_xticks(range(len(CTYPES)))
    axA.set_xticklabels([CLABS[c] for c in CTYPES], fontsize=10, color=INK)
    axA.set_ylabel("controllability (mean compliance)", fontsize=10, color=MUTED)
    axA.set_title("A · Ban1 rescues opening-gated control (instruction present)",
                  fontsize=11, color=INK, weight="bold", pad=8)
    axA.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)

    # ---------- B: capability neutrality (answered-only) ----------
    io_ans = io[io.truncated == 0]
    bw2 = 0.34
    for ci, ct in enumerate(CTYPES):
        for vi, v in enumerate(["baseline", "ban1_t0_3"]):
            sub = io_ans[(io_ans.ctype == ct) & (io_ans.variant == v)]
            m, lo, hi = qcluster_ci(sub, "accuracy")
            x = ci + (vi - 0.5) * bw2
            axB.bar(x, m, bw2, color=CV[v], zorder=3, yerr=[[m - lo], [hi - m]],
                    capsize=3, ecolor=MUTED, error_kw={"lw": 1.1},
                    label=v if ci == 0 else None)
    axB.set_xticks(range(len(CTYPES)))
    axB.set_xticklabels([CLABS[c] for c in CTYPES], fontsize=9.5, color=INK, rotation=15)
    axB.set_ylabel("capability (accuracy | answered)", fontsize=10, color=MUTED)
    axB.set_ylim(0, 0.7)
    axB.set_title("B · Capability is intervention-neutral", fontsize=11, color=INK, weight="bold", pad=8)
    axB.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)

    for ax in (axA, axB):
        ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle(f"Ban1 opening intervention on OLMo-3.1-32B ({args.tag}) — "
                 "restore first-token entropy → recover control, capability-neutral",
                 fontsize=12.5, color=INK, weight="bold", y=0.95)
    fig.text(0.5, 0.005, "question-clustered bootstrap 95% CI (30 questions × 16 samples). "
             "ban1_t0_3 masks the argmax at content tokens 0–3.", ha="center", fontsize=8, color=MUTED)
    out = FIGS / f"ban1_{args.tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    # print the rescue deltas
    for ct in CTYPES:
        b, _, _ = qcluster_ci(io[(io.ctype == ct) & (io.variant == "baseline")], "compliance_score")
        r, lo, hi = qcluster_ci(io[(io.ctype == ct) & (io.variant == "ban1_t0_3")], "compliance_score")
        print(f"  {ct:20s} baseline {b:.3f} -> ban1_t0_3 {r:.3f}  Δ {r-b:+.3f}")


if __name__ == "__main__":
    main()
