"""v34 organism mechanism: controllability + first-token entropy for base vs control
(normal-SFT) vs organism, on the 4 mech control types. Pairs the control gain with the
opening-entropy change — where the organism gains the most control (spanish, omit_word)
its first-token entropy is ELEVATED vs base, not collapsed.

  python analysis/plot_organism_mechanism.py  ->  figs/organism_mechanism.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
CTRL = HERE.parent.parent.parent / "controllability" / "results"
ENT = HERE / "figs" / "organism_entropy_onset.csv"
FIGS = HERE / "figs"

INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"
COL = {"base": "#9aa0a6", "control": "#3b6fb6", "organism": "#e8710a"}
CTYPES = ["uppercase_thinking", "leetspeak", "spanish", "omit_word"]
CLAB = {"uppercase_thinking": "uppercase", "leetspeak": "leetspeak", "spanish": "spanish", "omit_word": "omit_word"}
FAM = {"uppercase_thinking": "uppercase_thinking", "leetspeak": "leetspeak",
       "spanish": "language:spanish", "omit_word": "suppression:omit_word"}
SRC = {"base": "f2/qwen3_32b/base/per_sample/final.csv",
       "control": "organism_2_v34_ctrl_normal_diverse.csv",
       "organism": "organism_2_v34_ctrl_all_diverse.csv"}


def ctrl_by_family():
    out = {}
    for name, f in SRC.items():
        d = pd.read_csv(CTRL / f)
        d["compliance_score"] = pd.to_numeric(d["compliance_score"], errors="coerce")
        out[name] = {ct: d[d.family == FAM[ct]].compliance_score.mean() for ct in CTYPES}
    return out


def _style(ax, title, ylab):
    ax.set_title(title, fontsize=11, color=INK, weight="bold", pad=8)
    ax.set_ylabel(ylab, fontsize=10, color=MUTED)
    ax.grid(True, axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xticks(range(len(CTYPES)))
    ax.set_xticklabels([CLAB[c] for c in CTYPES], fontsize=9.5, color=INK, rotation=12)


def main():
    ctrl = ctrl_by_family()
    ent = pd.read_csv(ENT)
    ent = ent[ent.cond == "instruction_only"].set_index("ctype")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.07, right=0.98, wspace=0.2)
    bw = 0.26
    names = ["base", "control", "organism"]

    for ci, ct in enumerate(CTYPES):
        for mi, n in enumerate(names):
            axA.bar(ci + (mi - 1) * bw, ctrl[n][ct], bw, color=COL[n], zorder=3,
                    label=n if ci == 0 else None)
    _style(axA, "A · Controllability (mean compliance)", "compliance")

    for ci, ct in enumerate(CTYPES):
        for mi, n in enumerate(names):
            axB.bar(ci + (mi - 1) * bw, ent.loc[ct, n], bw, color=COL[n], zorder=3,
                    label=n if ci == 0 else None)
    _style(axB, "B · First-token entropy (instruction)", "entropy (nats)")

    axA.legend(frameon=False, fontsize=9.5, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.13), labelcolor=INK)
    fig.suptitle("v34 organism mechanism — where control is gained (spanish, omit_word), "
                 "opening entropy is ELEVATED vs base",
                 fontsize=12, color=INK, weight="bold", y=0.95)
    out = FIGS / "organism_mechanism.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    print("ctrl:", {n: {c: round(ctrl[n][c], 3) for c in CTYPES} for n in names})


if __name__ == "__main__":
    main()
