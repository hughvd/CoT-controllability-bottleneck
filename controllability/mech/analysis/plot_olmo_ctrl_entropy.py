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
    """SFT->ext-final: controllability (full 26-type, from overall.csv) + first-token
    entropy. SFT now has its own full-26 run so this is apples-to-apples. Shows the
    dissociation: entropy collapses hard at SFT->DPO while controllability stays ~flat."""
    ov = pd.read_csv(OLMO / "fig_data" / "overall.csv").set_index("tag")
    ent = pd.read_csv(OLMO / "entropy_trend.csv")
    ent = ent[ent.condition == args.cond]
    STEPS = [("sft", "SFT"), ("rl_start", "DPO"), ("rlvr_final", "RLVR"),
             ("ext_third", "ext-1/3"), ("ext_twothirds", "ext-2/3"), ("ext_final", "ext-final")]
    x = range(len(STEPS))
    ctrl = [float(ov.loc[t, "ctrl_soft"]) for t, _ in STEPS]
    clo = [float(ov.loc[t, "ctrl_soft_lo"]) for t, _ in STEPS]
    chi = [float(ov.loc[t, "ctrl_soft_hi"]) for t, _ in STEPS]
    e0, e16 = [], []
    for t, _ in STEPS:
        if t == "sft":
            a, b = npz_ent("sft", args.cond)
        else:
            a = float(ent[ent.tag == t].ent0.mean()); b = float(ent[ent.tag == t].ent_mean16.mean())
        e0.append(a); e16.append(b)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(top=0.82, bottom=0.16, wspace=0.26, left=0.07, right=0.985)
    axA.fill_between(x, clo, chi, color=C_CTRL, alpha=0.15, zorder=1)
    axA.plot(x, ctrl, "-o", color=C_CTRL, lw=2, ms=6, zorder=3)
    _style(axA, "mean compliance (26-type)", "Controllability")
    axA.set_ylim(0, max(chi) * 1.2)
    axB.plot(x, e0, "-o", color=C_E0, lw=2, ms=6, label="first token (ent0)")
    axB.plot(x, e16, "-s", color=C_E16, lw=2, ms=5.5, label="first 16 tokens (mean)")
    _style(axB, "entropy (nats)", f"First-reasoning-token entropy ({args.cond})")
    axB.set_ylim(0, np.nanmax(e0 + e16) * 1.15)
    axB.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)
    for ax in (axA, axB):
        ax.set_xticks(x); ax.set_xticklabels([d for _, d in STEPS], fontsize=8.5, rotation=20)
    fig.suptitle("SFT→RL: first-token entropy collapses at SFT→DPO while controllability stays ~flat",
                 fontsize=12, color=INK, weight="bold", y=0.965)
    fig.text(0.5, 0.005, "controllability = full 26-type (SFT has its own run now). "
             "ext-1/3 = genuine step_0800.", ha="center", fontsize=7.6, color=MUTED)
    out = FIGS / "olmo_ctrl_entropy_sft.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    # ---- single-axes overlays incl. SFT: soft + hard (entropy starts high, crashes to flat ctrl) ----
    ctrl_h = [float(ov.loc[t, "ctrl_hard"]) for t, _ in STEPS]
    hlo = [float(ov.loc[t, "ctrl_hard_lo"]) for t, _ in STEPS]
    hhi = [float(ov.loc[t, "ctrl_hard_hi"]) for t, _ in STEPS]
    sub = "shared y-axis (compliance fraction & entropy nats share scale). ctrl = full 26-type; ext-1/3 = genuine step_0800."
    xl = [d for _, d in STEPS]
    _draw_overlay(x, xl, ctrl, clo, chi, e0, e16,
                  f"SFT→RL: entropy collapses toward flat controllability — soft ({args.cond})",
                  sub, FIGS / "olmo_ctrl_entropy_sft_overlay_soft.png", "soft compliance", "upper right", figw=8.2)
    _draw_overlay(x, xl, ctrl_h, hlo, hhi, e0, e16,
                  f"SFT→RL: entropy collapses toward flat controllability — hard ({args.cond})",
                  sub, FIGS / "olmo_ctrl_entropy_sft_overlay_hard.png", "hard compliance ≥0.95", "upper right", figw=8.2)
    print("ctrl_soft(26):", [round(v, 3) for v in ctrl], "| ctrl_hard:", [round(v, 3) for v in ctrl_h])

    # ---- base-INCLUSIVE overlay: base has entropy (~1.71) but NO controllability ----
    b0, b16 = npz_ent("base", args.cond)
    xb = list(range(len(STEPS) + 1))
    xlb = ["base"] + xl
    subb = ("base has first-token entropy but no 26-type controllability (not instruction-tuned). "
            "ext-1/3 = genuine step_0800.")
    _draw_overlay(xb, xlb, [np.nan] + ctrl, [np.nan] + clo, [np.nan] + chi, [b0] + e0, [b16] + e16,
                  f"base→RL: first-token entropy collapses from base toward flat controllability — soft ({args.cond})",
                  subb, FIGS / "olmo_ctrl_entropy_base_overlay_soft.png", "soft compliance", "upper right", figw=8.8)
    _draw_overlay(xb, xlb, [np.nan] + ctrl_h, [np.nan] + hlo, [np.nan] + hhi, [b0] + e0, [b16] + e16,
                  f"base→RL: first-token entropy collapses from base toward flat controllability — hard ({args.cond})",
                  subb, FIGS / "olmo_ctrl_entropy_base_overlay_hard.png", "hard compliance ≥0.95", "upper right", figw=8.8)
    print("base ent0:", round(b0, 3), "| base ent_mean16:", round(b16, 3))

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


def _draw_overlay(x, xlabels, ctrl, clo, chi, e0, e16, title, sub, out, ctrl_label, loc, figw=8.0):
    """Single-axes overlay: controllability (soft or hard) + first-token / first-16-token
    entropy, sharing one y-axis (compliance fraction & entropy nats coincide in scale)."""
    fig, ax = plt.subplots(figsize=(figw, 5.0))
    ax.fill_between(x, clo, chi, color=C_CTRL, alpha=0.13, zorder=1)
    ax.plot(x, ctrl, "-o", color=C_CTRL, lw=2.2, ms=6, zorder=4, label=f"controllability ({ctrl_label})")
    ax.plot(x, e0, "-o", color=C_E0, lw=2, ms=6, zorder=3, label="first-token entropy (nats)")
    ax.plot(x, e16, "-s", color=C_E16, lw=2, ms=5.5, zorder=3, label="first-16-token entropy (nats)")
    ax.set_title(title, fontsize=12, color=INK, weight="bold", pad=10)
    ax.set_ylabel("compliance  /  entropy (nats)", fontsize=10, color=MUTED)
    ax.set_xlabel("checkpoint", fontsize=9.5, color=MUTED)
    vals = [v for v in list(ctrl) + list(chi) + list(e0) + list(e16) if v == v]
    ax.set_ylim(0, max(vals) * 1.12)
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xticks(list(x)); ax.set_xticklabels(xlabels, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc=loc, labelcolor=INK)
    fig.text(0.5, -0.01, sub, ha="center", fontsize=7.6, color=MUTED)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


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

    # ext_third is now GENUINE step_0800 — entropy valid, nothing to gap.
    PHANTOM = set()

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
    fig.text(0.5, 0.005, "ext-1/3 = genuine step_0800. First-token (pos-0) entropy is a weak discriminator for adjacent OLMo ckpts; ent_mean16 separates them.",
             ha="center", fontsize=7.8, color=MUTED)
    out = FIGS / "olmo_ctrl_entropy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")

    # ---- single-axes overlays: soft + hard compliance ----
    ctrl_h = [ov.loc[t, "ctrl_hard"] for t, _ in RL]
    hlo = [ov.loc[t, "ctrl_hard_lo"] for t, _ in RL]
    hhi = [ov.loc[t, "ctrl_hard_hi"] for t, _ in RL]
    sub = "shared y-axis (compliance fraction & entropy nats coincide in scale). ext-1/3 = genuine step_0800."
    _draw_overlay(x, [d for _, d in RL], ctrl, lo, hi, e0, e16,
                  f"Controllability (soft) & first-token entropy vs RL step ({args.cond})",
                  sub, FIGS / "olmo_ctrl_entropy_overlay_soft.png", "soft compliance", "center right", figw=7.6)
    _draw_overlay(x, [d for _, d in RL], ctrl_h, hlo, hhi, e0, e16,
                  f"Controllability (hard) & first-token entropy vs RL step ({args.cond})",
                  sub, FIGS / "olmo_ctrl_entropy_overlay_hard.png", "hard compliance ≥0.95", "center right", figw=7.6)
    print("ctrl_soft:", [round(v, 4) for v in ctrl], "| ctrl_hard:", [round(v, 4) for v in ctrl_h])


if __name__ == "__main__":
    main()
