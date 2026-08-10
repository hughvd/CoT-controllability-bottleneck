"""OLMo-3-32B mechanism panels along the checkpoint axis:
 A) first-token entropy CASCADE  base -> SFT -> DPO -> RL  (the collapse, log-y)
 B) opening compliant-token mass (mass0, instruction_only) vs RL
 C) adjacent-checkpoint KL front-loading (frac of policy change in first 8 tokens)
 D) reasoning length (chars) vs checkpoint

base/SFT entropy is recomputed from their mech npz (entropy_trend.csv only spans the
5 RL ckpts); everything else reads the precomputed olmo_rl CSVs.

  python analysis/plot_olmo_mechanism.py  ->  figs/olmo_mechanism.png
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLMO = HERE / "olmo_rl"
DATA = HERE.parent / "data" / "olmo_rl"      # mech/data/olmo_rl (npz live here)
FIGS = HERE / "figs"

INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"
C1, C2, C3, C4 = "#c0392b", "#3b6fb6", "#2a9d5c", "#8e44ad"
MECH_CTYPES = ["uppercase_thinking", "leetspeak", "spanish", "omit_word"]


def ent_from_npz(tag, cond="no_instruction"):
    """Mean first-token entropy over mech traces for a checkpoint w/o a precomputed row."""
    vals = []
    for fp in glob.glob(str(DATA / tag / "stats" / "*.npz")):
        z = np.load(fp, allow_pickle=True)
        if str(z["condition"]) != cond:
            continue
        e = z["entropy"]
        if len(e):
            vals.append(float(e[0]))
    return float(np.mean(vals)) if vals else np.nan


def _style(ax, title, ylab, xlabels):
    ax.set_title(title, fontsize=10.5, color=INK, weight="bold", pad=7)
    ax.set_ylabel(ylab, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=8, rotation=30, ha="right")


def main():
    ent = pd.read_csv(OLMO / "entropy_trend.csv")
    mass = pd.read_csv(OLMO / "opening_mass.csv")
    kl = pd.read_csv(OLMO / "kl_frontload.csv")
    ov = pd.read_csv(OLMO / "fig_data" / "overall.csv").sort_values("rl_index")
    sh = pd.read_csv(OLMO / "fig_data" / "shared22.csv")

    RL = [("rl_start", "DPO"), ("rlvr_final", "RLVR"), ("ext_third", "ext-1/3"),
          ("ext_twothirds", "ext-2/3"), ("ext_final", "ext-final")]
    # ext_third entropy/mass/KL stats are phantom (bitwise == rlvr_final; see
    # olmo_figure_suite memory) -> gap them until regenerated on the pod.
    PHANTOM = {"ext_third"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.42, wspace=0.24, top=0.90, bottom=0.10, left=0.075, right=0.98)

    # ---- A) entropy cascade base->SFT->DPO->RL (no_instruction ent0) ----
    ax = axes[0, 0]
    casc = [("base", "base"), ("sft", "SFT")] + RL
    y = []
    for tag, _ in casc:
        if tag in PHANTOM:
            y.append(float("nan")); continue
        row = ent[(ent.tag == tag) & (ent.condition == "no_instruction")]
        y.append(row.ent0.mean() if len(row) else ent_from_npz(tag, "no_instruction"))
    labels = [d for _, d in casc]
    ax.plot(range(len(y)), y, "-o", color=C1, lw=2, ms=6, zorder=3)
    for i, v in enumerate(y):
        if not np.isnan(v):
            ax.annotate(f"{v:.2f}", (i, v), fontsize=7.5, color=C1,
                        xytext=(0, 7), textcoords="offset points", ha="center")
    _style(ax, "A · First-token entropy cascade (no-instruction)", "entropy (nats), log scale", labels)
    ax.set_yscale("log")

    # ---- B) opening compliant-token mass (instruction_only, mean over mech ctypes) ----
    ax = axes[0, 1]
    m = mass[mass.condition == "instruction_only"].groupby("tag").mass0.mean()
    ym = [np.nan if t in PHANTOM else m.get(t, np.nan) for t, _ in RL]
    ax.plot(range(len(RL)), ym, "-o", color=C2, lw=2, ms=6, zorder=3)
    _style(ax, "B · Opening compliant-token mass (instruction)", "P(target token) @ pos 0", [d for _, d in RL])

    # ---- C) KL front-loading across adjacent pairs ----
    ax = axes[1, 0]
    kf = kl.groupby("pair").frac_first8.mean()
    # pairs touching ext_third are invalid (one side is phantom == rlvr_final) -> gap.
    pair_order = ["rl_start__rlvr_final", "rlvr_final__ext_third",
                  "ext_third__ext_twothirds", "ext_twothirds__ext_final"]
    PHANTOM_PAIRS = {"rlvr_final__ext_third", "ext_third__ext_twothirds"}
    plabs = ["DPO→RLVR", "RLVR→ext⅓", "ext⅓→ext⅔", "ext⅔→final"]
    yk = [np.nan if p in PHANTOM_PAIRS else kf.get(p, np.nan) for p in pair_order]
    ax.bar(range(len(pair_order)), yk, color=C3, zorder=3, width=0.6)
    _style(ax, "C · KL front-loading (adjacent checkpoints)", "frac. of policy KL in first 8 tokens", plabs)

    # ---- D) reasoning length ----
    ax = axes[1, 1]
    sft_len = sh[sh.tag == "sft"].reas_len
    xl = ["SFT"] + [d for _, d in RL]
    yl = [sft_len.iloc[0] if len(sft_len) else np.nan] + [
        ov[ov.tag == t].reas_len.iloc[0] for t, _ in RL]
    ax.plot(range(len(xl)), yl, "-o", color=C4, lw=2, ms=6, zorder=3)
    _style(ax, "D · Reasoning length (drives truncation)", "chars in reasoning", xl)

    fig.suptitle("OLMo-3-32B mechanism along the checkpoint axis", fontsize=13, color=INK, weight="bold", y=0.965)
    fig.text(0.5, 0.005, "ext-1/3 gapped in A–C — its entropy/mass/KL stats are phantom "
             "(bitwise == RLVR); pending pod regeneration. Panel D (reasoning length) is genuine.",
             ha="center", fontsize=8, color=MUTED)
    out = FIGS / "olmo_mechanism.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    print("entropy cascade:", {l: round(v, 3) for l, v in zip(labels, y)})


if __name__ == "__main__":
    main()
