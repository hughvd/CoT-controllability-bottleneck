"""Controllability + first-token entropy over the 5 OLMo RL checkpoints (no base/SFT
— neither has a 26-type controllability run). Two panels sharing the RL-step x-axis.

Entropy panel plots BOTH first-token (ent0) and first-16-token mean (ent_mean16),
because pos-0 entropy cannot resolve adjacent OLMo checkpoints (float16 + hybrid-SSM):
rlvr_final and ext_third have BITWISE-IDENTICAL instruction ent0 despite being
genuinely different models (gen_len differs). ent_mean16 discriminates all 5.

  python analysis/plot_olmo_ctrl_entropy.py [--cond no_instruction|instruction_only]
  -> figs/olmo_ctrl_entropy.png
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLMO = HERE / "olmo_rl"
DATA = HERE.parent / "data" / "olmo_rl"
FIGS = HERE / "figs"


def npz_ent(tag, cond="no_instruction"):
    """SFT/base entropy from mech npz (entropy_trend.csv only spans the 5 RL ckpts)."""
    e0, e16 = [], []
    for fp in glob.glob(str(DATA / tag / "stats" / "*.npz")):
        z = np.load(fp, allow_pickle=True)
        if str(z["condition"]) != cond:
            continue
        e = np.asarray(z["entropy"])
        if len(e):
            e0.append(float(e[0])); e16.append(float(e[:16].mean()))
    return (np.mean(e0), np.mean(e16)) if e0 else (np.nan, np.nan)


def sft_inclusive(args):
    """ctrl (22-shared families, comparable incl. SFT) + first-token entropy,
    SFT->ext-final. SFT entropy genuine (mech npz); ext-1/3 entropy gapped (phantom)."""
    sh = pd.read_csv(OLMO / "fig_data" / "shared22.csv").set_index("tag")
    ent = pd.read_csv(OLMO / "entropy_trend.csv")
    ent = ent[ent.condition == args.cond]
    STEPS = [("sft", "SFT"), ("rl_start", "DPO"), ("rlvr_final", "RLVR"),
             ("ext_third", "ext-1/3"), ("ext_twothirds", "ext-2/3"), ("ext_final", "ext-final")]
    PHANTOM = {"ext_third"}
    x = range(len(STEPS))
    ctrl = [sh.loc[t, "ctrl_soft"] for t, _ in STEPS]
    clo = [sh.loc[t, "ctrl_soft_lo"] for t, _ in STEPS]
    chi = [sh.loc[t, "ctrl_soft_hi"] for t, _ in STEPS]
    e0, e16 = [], []
    for t, _ in STEPS:
        if t in PHANTOM:
            e0.append(np.nan); e16.append(np.nan)
        elif t == "sft":
            a, b = npz_ent("sft", args.cond); e0.append(a); e16.append(b)
        else:
            e0.append(ent[ent.tag == t].ent0.mean()); e16.append(ent[ent.tag == t].ent_mean16.mean())

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.26, left=0.07, right=0.985)
    axA.fill_between(x, clo, chi, color=C_CTRL, alpha=0.15, zorder=1)
    axA.plot(x, ctrl, "-o", color=C_CTRL, lw=2, ms=6, zorder=3)
    _style(axA, "mean compliance (22 shared types)", "Controllability")
    axA.set_ylim(0, max(chi) * 1.2)
    axB.plot(x, e0, "-o", color=C_E0, lw=2, ms=6, label="first token (ent0)")
    axB.plot(x, e16, "-s", color=C_E16, lw=2, ms=5.5, label="first 16 tokens (mean)")
    _style(axB, "entropy (nats)", f"First-reasoning-token entropy ({args.cond})")
    axB.set_ylim(0, max([v for v in e0 + e16 if not np.isnan(v)]) * 1.15)
    axB.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)
    for ax, (_, __) in zip((axA, axB), [(0, 0), (0, 0)]):
        ax.set_xticks(x); ax.set_xticklabels([d for _, d in STEPS], fontsize=8.5, rotation=20)
    fig.suptitle("SFT→RL: controllability and first-token entropy both fall fastest at SFT→DPO",
                 fontsize=12.5, color=INK, weight="bold", y=0.965)
    fig.text(0.5, 0.005, "ctrl on the 22 families shared by the SFT & RL evals (styles absent from SFT eval). "
             "ext-1/3 entropy gapped (phantom stats).", ha="center", fontsize=7.6, color=MUTED)
    out = FIGS / "olmo_ctrl_entropy_sft.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    print("ctrl(22):", [round(v, 3) for v in ctrl])
    print("ent0:", [round(v, 3) if not np.isnan(v) else None for v in e0])

INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"
C_CTRL, C_E0, C_E16 = "#3b6fb6", "#c0392b", "#e8710a"
RL = [("rl_start", "DPO"), ("rlvr_final", "RLVR"), ("ext_third", "ext-1/3"),
      ("ext_twothirds", "ext-2/3"), ("ext_final", "ext-final")]


def _style(ax, ylab, title):
    ax.set_title(title, fontsize=11, color=INK, weight="bold", pad=8)
    ax.set_ylabel(ylab, fontsize=10, color=MUTED)
    ax.set_xlabel("RL step", fontsize=9.5, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.set_xticks(range(len(RL)))
    ax.set_xticklabels([d for _, d in RL], fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="no_instruction",
                    choices=["no_instruction", "instruction_only"])
    ap.add_argument("--with-sft", action="store_true",
                    help="prepend SFT on the 22-shared-family ctrl axis (2-panel only)")
    args = ap.parse_args()
    if args.with_sft:
        sft_inclusive(args)
        return

    ov = pd.read_csv(OLMO / "fig_data" / "overall.csv").set_index("tag")
    ent = pd.read_csv(OLMO / "entropy_trend.csv")
    ent = ent[ent.condition == args.cond]

    # ext_third's entropy STATS are phantom (bitwise == rlvr_final; see olmo_figure_suite
    # memory). Controllability is genuine, so keep it; gap the entropy point until the
    # stats are regenerated on the pod with genuine step_0750 weights.
    PHANTOM = {"ext_third"}

    x = range(len(RL))
    ctrl = [ov.loc[t, "ctrl_soft"] for t, _ in RL]
    lo = [ov.loc[t, "ctrl_soft_lo"] for t, _ in RL]
    hi = [ov.loc[t, "ctrl_soft_hi"] for t, _ in RL]
    nan = float("nan")
    e0 = [nan if t in PHANTOM else ent[ent.tag == t].ent0.mean() for t, _ in RL]
    e16 = [nan if t in PHANTOM else ent[ent.tag == t].ent_mean16.mean() for t, _ in RL]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(top=0.82, bottom=0.15, wspace=0.26, left=0.07, right=0.985)

    axA.fill_between(x, lo, hi, color=C_CTRL, alpha=0.15, zorder=1)
    axA.plot(x, ctrl, "-o", color=C_CTRL, lw=2, ms=6, zorder=3)
    _style(axA, "mean compliance (26-type)", "Controllability")
    axA.set_ylim(0, max(hi) * 1.25)

    axB.plot(x, e0, "-o", color=C_E0, lw=2, ms=6, zorder=3, label="first token (ent0)")
    axB.plot(x, e16, "-s", color=C_E16, lw=2, ms=5.5, zorder=3, label="first 16 tokens (mean)")
    _style(axB, "entropy (nats)", f"First-reasoning-token entropy ({args.cond})")
    axB.set_ylim(0, max(e16) * 1.2)
    axB.legend(frameon=False, fontsize=8.5, loc="center right", labelcolor=INK)

    fig.suptitle("OLMo-3-32B along the RL axis: controllability ↓ with first-token entropy",
                 fontsize=12.5, color=INK, weight="bold", y=0.965)
    fig.text(0.5, 0.005, "ext-1/3 entropy omitted — its mech stats are phantom (bitwise == RLVR; "
             "pending pod regen). Controllability is genuine. Base/SFT: no 26-type ctrl run.",
             ha="center", fontsize=7.8, color=MUTED)
    out = FIGS / "olmo_ctrl_entropy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    # ---- single-axes overlay (all three share a 0-0.3 scale — no dual axis) ----
    fig2, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.fill_between(x, lo, hi, color=C_CTRL, alpha=0.13, zorder=1)
    ax.plot(x, ctrl, "-o", color=C_CTRL, lw=2.2, ms=6, zorder=4, label="controllability (compliance)")
    ax.plot(x, e0, "-o", color=C_E0, lw=2, ms=6, zorder=3, label="first-token entropy (nats)")
    ax.plot(x, e16, "-s", color=C_E16, lw=2, ms=5.5, zorder=3, label="first-16-token entropy (nats)")
    _style(ax, "compliance  /  entropy (nats)", f"Controllability & first-token entropy vs RL step ({args.cond})")
    ax.set_ylim(0, max(max(e16), max(hi)) * 1.15)
    ax.legend(frameon=False, fontsize=9, loc="center right", labelcolor=INK)
    fig2.text(0.5, -0.01, "shared y-axis (values coincide in 0–0.3; NOT a dual axis). "
              "ext-1/3 entropy gapped — phantom mech stats (bitwise == RLVR), pending pod regen; "
              "controllability is genuine.",
              ha="center", fontsize=7.6, color=MUTED)
    out2 = FIGS / "olmo_ctrl_entropy_overlay.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out2}")
    print("ctrl:", [round(v, 4) for v in ctrl])
    print("ent0:", [round(v, 4) for v in e0])
    print("ent_mean16:", [round(v, 4) for v in e16])


if __name__ == "__main__":
    main()
