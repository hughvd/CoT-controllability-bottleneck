"""Deliverable figure: first-token argmax ban (ban4) recovers CoT controllability.

Dumbbell per control-type: baseline compliance (hollow ○) -> ban4 (filled ●, +CI).
Two panels, BOTH full 26 types, in BOTH soft and hard compliance (--metric):

  OLMo-3.1-32B-Think: baseline REUSED from results/olmo32_ctrl_ext_final/per_sample.csv
    (26 types, stored compliance_hard == score>=0.95). ban4 from ban4_olmo31 grades.
  Qwen3-32B: baseline REUSED from results/f2/qwen3_32b/base (22 family/lang/supp types;
    hard derived as score>=0.95) + the 4 style types generated via the ban pipeline's
    depth-0 baseline arm (ban4_qwen32_style, variant=baseline). ban4 from ban4_qwen32
    (22) + ban4_qwen32_style (4 style).

Rows grouped/colored by modality. Emits figs/ban4_ctrl_dumbbell_{soft,hard}.png.

  python mech/analysis/plot_ban4_ctrl.py            # writes both soft + hard
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MECH = HERE.parent
CTRL = MECH.parent
FIGS = HERE / "figs"
RUNS = MECH / "data" / "ctrl26_runs"

INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"
MODCOL = {"family": "#3b6fb6", "language": "#c0392b", "style": "#8e44ad", "suppression": "#e8710a"}
MODORDER = ["family", "language", "style", "suppression"]

CANON22 = None  # set from qwen ban4 grades (the 22 built) at load time


def _modality(ct):
    return ct.split(":", 1)[0] if ":" in ct else "family"


def _col(metric):
    return "compliance_score" if metric == "soft" else "compliance_hard"


def boot_ci(vals, nboot=2000, seed=0):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(v, size=(nboot, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def base_persample(path, metric):
    """Reused-base per-family scalar. hard = stored col if present else score>=0.95."""
    d = pd.read_csv(path)
    if metric == "soft":
        return d.groupby("family")["compliance_score"].mean()
    if "compliance_hard" in d.columns:
        return d.groupby("family")["compliance_hard"].mean()
    return (d.assign(h=(d["compliance_score"] >= 0.95).astype(int))
            .groupby("family")["h"].mean())


def grades_arr(run_tag, metric, variant="ban4"):
    """dict ctype -> np.array of per-trace metric values, for one variant."""
    g = pd.read_csv(RUNS / run_tag / "grades.csv")
    g = g[g.variant == variant]
    col = _col(metric)
    return {ct: sub[col].to_numpy() for ct, sub in g.groupby("ctype")}


def load_olmo(metric):
    ban4 = grades_arr("ban4_olmo31", metric)
    # PIPELINE-MATCHED baseline (depth-0 arm) if present; else fall back to reused ext_final
    base_pm = {ct: float(np.mean(a))
               for ct, a in grades_arr("ban4_olmo31", metric, variant="baseline").items()}
    reused = base_persample(CTRL / "results" / "olmo32_ctrl_ext_final" / "per_sample.csv", metric)
    rows = []
    for ct, arr in ban4.items():
        b = base_pm.get(ct, float(reused[ct]) if ct in reused.index else np.nan)
        if not np.isfinite(b):
            continue
        m, lo, hi = boot_ci(arr)
        rows.append(dict(ctype=ct, modality=_modality(ct), base=b, ban4=m, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def load_qwen(metric):
    # ban4 arm: 22 non-style + 4 style
    ban4 = {**grades_arr("ban4_qwen32", metric), **grades_arr("ban4_qwen32_style", metric)}
    # PIPELINE-MATCHED baseline: depth-0 arm through the same offline decoder (all 26)
    base_pm = {ct: float(np.mean(a)) for ct, a in
               {**grades_arr("ban4_qwen32", metric, variant="baseline"),
                **grades_arr("ban4_qwen32_style", metric, variant="baseline")}.items()}
    # reused served-eval base (for the confound comparison printout only)
    reused = base_persample(CTRL / "results" / "f2" / "qwen3_32b" / "base" / "per_sample" / "final.csv", metric)
    rows = []
    for ct, arr in ban4.items():
        if ct not in base_pm:
            continue
        m, lo, hi = boot_ci(arr)
        rows.append(dict(ctype=ct, modality=_modality(ct), base=base_pm[ct], ban4=m, lo=lo, hi=hi,
                         base_reused=float(reused[ct]) if ct in reused.index else np.nan))
    return pd.DataFrame(rows)


def _order(df):
    df = df.copy()
    df["mo"] = df.modality.map({m: i for i, m in enumerate(MODORDER)})
    return df.sort_values(["mo", "ban4"]).reset_index(drop=True)


def _clean(ct):
    return ct.split(":", 1)[-1]


def draw(metric):
    panels = [("OLMo-3.1-32B-Think", _order(load_olmo(metric))),
              ("Qwen3-32B", _order(load_qwen(metric)))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 8.6),
                             gridspec_kw={"width_ratios": [max(len(p[1]), 1) for p in panels]},
                             sharex=True)
    lab = "soft compliance (controllability)" if metric == "soft" else "hard compliance (≥0.95)"
    for ax, (title, df) in zip(axes, panels):
        y = np.arange(len(df))[::-1]
        for yi, r in zip(y, df.itertuples()):
            c = MODCOL[r.modality]
            ax.plot(r.base, yi, "o", mfc="white", mec=c, mew=1.6, ms=7, zorder=3)
            ax.errorbar(r.ban4, yi, xerr=[[max(r.ban4 - r.lo, 0)], [max(r.hi - r.ban4, 0)]],
                        fmt="o", color=c, ms=8, elinewidth=1.2, capsize=2.5, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([_clean(c) for c in df.ctype], fontsize=8, color=INK)
        ax.set_ylim(-0.8, len(df) - 0.2)
        ax.set_xlim(-0.03, 1.03)
        db, dn = df.base.mean(), df.ban4.mean()
        ax.set_title(f"{title}  ({len(df)} types)\nmean {db:.2f} → {dn:.2f}  (+{dn - db:.2f})",
                     fontsize=11, color=INK, weight="bold", pad=8)
        ax.set_xlabel(lab, fontsize=10, color=MUTED)
        ax.grid(True, axis="x", color=GRID, lw=0.7, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED)

    handles = [plt.Line2D([0], [0], marker="o", color=MODCOL[m], lw=0, label=m, ms=8)
               for m in MODORDER]
    handles += [plt.Line2D([0], [0], marker="o", mfc="white", mec=INK, mew=1.6, lw=0,
                           label="baseline", ms=8),
                plt.Line2D([0], [0], marker="o", color=INK, lw=0, label="ban4", ms=8)]
    fig.legend(handles=handles, frameon=False, fontsize=9.5, ncol=6, labelcolor=INK,
               loc="upper center", bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("First-token argmax ban (ban4) recovers CoT controllability",
                 fontsize=13.5, color=INK, weight="bold", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    FIGS.mkdir(exist_ok=True)
    out = FIGS / f"ban4_ctrl_dumbbell_{metric}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    for title, df in panels:
        print(f"  [{metric}] {title}: {len(df)} types, base {df.base.mean():.3f} -> ban4 {df.ban4.mean():.3f}")


def draw_rescue(metric):
    """The thesis panel: baseline compliance (x) vs ban4 lift Δ=ban4-base (y).
    Low-baseline types sit above 0 (rescued); high-baseline below 0 (disrupted)."""
    data = {"OLMo-3.1-32B-Think": _order(load_olmo(metric)),
            "Qwen3-32B": _order(load_qwen(metric))}
    MK = {"OLMo-3.1-32B-Think": "o", "Qwen3-32B": "^"}
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    ax.axhline(0, color=MUTED, lw=1, ls="--", zorder=1)
    allx, ally = [], []
    for model, df in data.items():
        d = df.assign(delta=df.ban4 - df.base)
        for r in d.itertuples():
            ax.scatter(r.base, r.delta, marker=MK[model], s=64, c=MODCOL[r.modality],
                       edgecolors="white", linewidths=0.8, zorder=3)
        allx += list(d.base); ally += list(d.delta)
    ax.axvline(0.3, color=GRID, lw=1, ls=":", zorder=1)
    allx, ally = np.array(allx), np.array(ally)
    lo, hi = ally[allx < 0.3], ally[allx >= 0.3]
    ax.set_xlabel(f"baseline {('soft' if metric=='soft' else 'hard')} compliance", fontsize=11, color=MUTED)
    ax.set_ylabel("ban4 lift  Δ = ban4 − baseline", fontsize=11, color=MUTED)
    ax.set_title("large rescues occur only where control has collapsed;\n"
                 f"already-controllable types hold or degrade   "
                 f"(mean Δ: base<0.3 = {lo.mean():+.2f} n={len(lo)}, base≥0.3 = {hi.mean():+.2f} n={len(hi)})",
                 fontsize=10.5, color=INK, weight="bold", pad=10)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    handles = [plt.Line2D([0], [0], marker="o", color=MODCOL[m], lw=0, label=m, ms=8) for m in MODORDER]
    handles += [plt.Line2D([0], [0], marker=MK[m], color=MUTED, lw=0, label=m, ms=8) for m in MK]
    ax.legend(handles=handles, frameon=False, fontsize=9, ncol=2, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    out = FIGS / f"ban4_rescue_{metric}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def main():
    for metric in ("soft", "hard"):
        draw(metric)
        draw_rescue(metric)


if __name__ == "__main__":
    main()
