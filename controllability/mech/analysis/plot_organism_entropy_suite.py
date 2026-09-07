"""Onset-entropy suite: organism (Qwen3-32B v34) + a Qwen-vs-OLMo onset contrast.

Purely descriptive — no lumped-window scalar, no conclusions in titles. Each
figure shows per-position structure so the reader locates the signal themselves.

Figures (written to figs/):
  organism_profiles_2cond.png   per-position entropy, base/control/organism,
                                both conditions, one panel per ctype
  organism_onset_diff.png       organism - control (and - base) per position,
                                instruction_only, localizes the mode switch
  organism_window_sensitivity.png  onset-window mean vs window start (end=14):
                                shows opener-dilution vs the pos-1 spike
  organism_opening_diversity.png   opening-token collapse from top5_ids: #distinct
                                top-1 tokens across traces + mode share, per position
  onset_qwen_vs_olmo.png        pos-0..15 mean entropy, organism vs OLMo checkpoints
                                (same 4 shared ctypes) — token-0 structural vs
                                token-0 = reasoning-onset

  python3 mech/analysis/plot_organism_entropy_suite.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ORG = HERE.parent / "data" / "organism"
OLMO = HERE.parent / "data" / "olmo_rl"
OUT = HERE / "figs"

INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"
CTYPES = ["uppercase_thinking", "leetspeak", "spanish", "omit_word"]
CONDS = ["no_instruction", "instruction_only"]
ORG_MODELS = [("base", "base", "#9aa0a6"), ("ctl_s1", "control", "#3b6fb6"),
              ("org_s1", "organism", "#e8710a")]
NPOS = 24


def _load_org(model_dir):
    """{(ctype,cond): list of arrays} for entropy and top1-token id, per trace."""
    ent, tok1 = {}, {}
    for fp in glob.glob(f"{ORG}/{model_dir}/entropy/*.npz"):
        z = np.load(fp, allow_pickle=True)
        e = z["entropy"].astype(np.float32)
        if e.size == 0:
            continue
        key = (str(z["ctype"]), str(z["condition"]))
        ent.setdefault(key, []).append(e)
        if "top5_ids" in z:
            tok1.setdefault(key, []).append(z["top5_ids"][:, 0].astype(np.int64))
    return ent, tok1


def _mean_profile(arrs, n=NPOS):
    M = np.vstack([np.pad(a[:n], (0, max(0, n - len(a))), constant_values=np.nan)[:n] for a in arrs])
    return np.nanmean(M, axis=0), np.nanstd(M, axis=0) / np.sqrt(np.sum(~np.isnan(M), axis=0))


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_profiles_2cond(orgdata):
    fig, axes = plt.subplots(len(CTYPES), 2, figsize=(11, 12), dpi=150, sharex=True)
    for r, ct in enumerate(CTYPES):
        for c, cond in enumerate(CONDS):
            ax = axes[r, c]
            for md, name, col in ORG_MODELS:
                arrs = orgdata[md][0].get((ct, cond), [])
                if not arrs:
                    continue
                m, se = _mean_profile(arrs)
                x = np.arange(len(m))
                ax.plot(x, m, color=col, lw=1.8, label=name, zorder=3)
                ax.fill_between(x, m - se, m + se, color=col, alpha=0.15, zorder=1)
            ax.axvline(0.5, color="#cfcfd6", lw=0.8, ls=":")  # token-0 boundary
            _style(ax)
            if r == 0:
                ax.set_title(cond, fontsize=10, color=INK)
            if c == 0:
                ax.set_ylabel(f"{ct}\nentropy (nats)", fontsize=8.5, color=INK)
    axes[-1, 0].set_xlabel("generated token position", fontsize=9, color=MUTED)
    axes[-1, 1].set_xlabel("generated token position", fontsize=9, color=MUTED)
    axes[0, 1].legend(fontsize=8.5, frameon=False)
    fig.suptitle("Per-position thinking entropy — base / control / organism (v34 seed-1)",
                 fontsize=12, color=INK, y=0.995, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / "organism_profiles_2cond.png", bbox_inches="tight")
    print("wrote organism_profiles_2cond.png")


def fig_onset_diff(orgdata):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=150, sharex=True)
    for ax, ct in zip(axes.flat, CTYPES):
        org, _ = _mean_profile(orgdata["org_s1"][0].get((ct, "instruction_only"), [np.zeros(NPOS)]))
        ctl, _ = _mean_profile(orgdata["ctl_s1"][0].get((ct, "instruction_only"), [np.zeros(NPOS)]))
        bas, _ = _mean_profile(orgdata["base"][0].get((ct, "instruction_only"), [np.zeros(NPOS)]))
        x = np.arange(len(org))
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.bar(x - 0.2, org - ctl, 0.4, color="#e8710a", label="organism − control", zorder=3)
        ax.bar(x + 0.2, org - bas, 0.4, color="#9aa0a6", label="organism − base", zorder=3)
        ax.set_title(ct, fontsize=10, color=INK)
        _style(ax)
    axes[1, 0].set_xlabel("generated token position", fontsize=9, color=MUTED)
    axes[1, 1].set_xlabel("generated token position", fontsize=9, color=MUTED)
    axes[0, 0].set_ylabel("Δ entropy (nats)", fontsize=9, color=MUTED)
    axes[0, 1].legend(fontsize=8.5, frameon=False)
    fig.suptitle("Where control-training changes onset entropy (instruction_only)",
                 fontsize=12, color=INK, y=1.0, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "organism_onset_diff.png", bbox_inches="tight")
    print("wrote organism_onset_diff.png")


def fig_window_sensitivity(orgdata):
    starts = list(range(0, 9))
    fig, axes = plt.subplots(1, len(CTYPES), figsize=(14, 3.6), dpi=150, sharey=True)
    for ax, ct in zip(axes, CTYPES):
        for md, name, col in ORG_MODELS:
            arrs = orgdata[md][0].get((ct, "instruction_only"), [])
            if not arrs:
                continue
            ys = []
            for s in starts:
                vals = [a[s:14].mean() for a in arrs if a.size >= 14]
                ys.append(np.mean(vals))
            ax.plot(starts, ys, "-o", color=col, lw=1.8, ms=4, label=name)
        ax.axvline(4, color="#cfcfd6", lw=1.0, ls="--")  # current window start
        ax.set_title(ct, fontsize=10, color=INK)
        ax.set_xlabel("window start (end=14)", fontsize=8.5, color=MUTED)
        _style(ax)
    axes[0].set_ylabel("mean onset entropy (nats)", fontsize=8.5, color=MUTED)
    axes[-1].legend(fontsize=8.5, frameon=False)
    fig.suptitle("Onset-window sensitivity — dashed line = current window start (4)",
                 fontsize=12, color=INK, y=1.03, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "organism_window_sensitivity.png", bbox_inches="tight")
    print("wrote organism_window_sensitivity.png")


def fig_opening_diversity(orgdata):
    """From top1-token id per trace: at each position, #distinct openers and mode share."""
    pos = np.arange(10)
    fig, axes = plt.subplots(2, len(CTYPES), figsize=(14, 6), dpi=150, sharex=True)
    for c, ct in enumerate(CTYPES):
        for md, name, col in ORG_MODELS:
            toks = orgdata[md][1].get((ct, "instruction_only"), [])
            if not toks:
                continue
            ndist, modeshare = [], []
            for p in pos:
                col_ids = [t[p] for t in toks if t.size > p]
                if not col_ids:
                    ndist.append(np.nan); modeshare.append(np.nan); continue
                vals, cnts = np.unique(col_ids, return_counts=True)
                ndist.append(len(vals))
                modeshare.append(cnts.max() / cnts.sum())
            axes[0, c].plot(pos, ndist, "-o", color=col, lw=1.6, ms=3.5, label=name)
            axes[1, c].plot(pos, modeshare, "-o", color=col, lw=1.6, ms=3.5, label=name)
        axes[0, c].set_title(ct, fontsize=10, color=INK)
        for r in (0, 1):
            _style(axes[r, c])
        axes[1, c].set_xlabel("token position", fontsize=8.5, color=MUTED)
    axes[0, 0].set_ylabel("# distinct top-1 tokens\n(across traces)", fontsize=8.5, color=INK)
    axes[1, 0].set_ylabel("mode share\n(top-1 token freq)", fontsize=8.5, color=INK)
    axes[0, -1].legend(fontsize=8.5, frameon=False)
    fig.suptitle("Opening-token diversity across traces (instruction_only) — mode-collapse view",
                 fontsize=12, color=INK, y=1.0, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "organism_opening_diversity.png", bbox_inches="tight")
    print("wrote organism_opening_diversity.png")


def _load_olmo_profiles(tag, cond="instruction_only", n=16):
    arrs = []
    for fp in glob.glob(f"{OLMO}/{tag}/stats/*.npz"):
        z = np.load(fp, allow_pickle=True)
        if str(z["condition"]) != cond or str(z["ctype"]) not in CTYPES:
            continue
        e = z["entropy"].astype(np.float32)
        if e.size:
            arrs.append(e)
    if not arrs:
        return None
    return _mean_profile(arrs, n)[0]


def fig_qwen_vs_olmo(orgdata):
    n = 16
    # organism side: pooled over the 4 ctypes, instruction_only
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), dpi=150, sharey=True)
    ax = axes[0]
    for md, name, col in ORG_MODELS:
        arrs = [a for ct in CTYPES for a in orgdata[md][0].get((ct, "instruction_only"), [])]
        if not arrs:
            continue
        m, _ = _mean_profile(arrs, n)
        ax.plot(np.arange(n), m, "-o", color=col, lw=1.9, ms=4, label=name)
    ax.set_title("Qwen3-32B (v34 organism family)", fontsize=11, color=INK)
    ax.set_ylabel("mean entropy (nats)", fontsize=9.5, color=MUTED)
    ax.set_xlabel("generated token position", fontsize=9.5, color=MUTED)
    ax.legend(fontsize=8.5, frameon=False); _style(ax)

    ax = axes[1]
    olmo = [("rl_start", "DPO", "#b8bcc8"), ("rlvr_final", "RLVR", "#3b6fb6"),
            ("ext_final", "ext-final", "#c0392b")]
    for tag, lab, col in olmo:
        m = _load_olmo_profiles(tag, n=n)
        if m is None:
            continue
        ax.plot(np.arange(n), m, "-o", color=col, lw=1.9, ms=4, label=lab)
    ax.set_title("OLMo-3-32B (RL checkpoints)", fontsize=11, color=INK)
    ax.set_xlabel("generated token position", fontsize=9.5, color=MUTED)
    ax.legend(fontsize=8.5, frameon=False); _style(ax)

    fig.suptitle("Onset entropy near token 0 — same 4 control-types, instruction_only",
                 fontsize=12.5, color=INK, y=1.02, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "onset_qwen_vs_olmo.png", bbox_inches="tight")
    print("wrote onset_qwen_vs_olmo.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    orgdata = {md: _load_org(md) for md, _, _ in ORG_MODELS}
    fig_profiles_2cond(orgdata)
    fig_onset_diff(orgdata)
    fig_window_sensitivity(orgdata)
    fig_opening_diversity(orgdata)
    fig_qwen_vs_olmo(orgdata)


if __name__ == "__main__":
    main()
