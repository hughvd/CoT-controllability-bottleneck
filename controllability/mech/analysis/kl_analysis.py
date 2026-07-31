"""KL analysis (local) — how many tokens deep is the organism's control?

Reads data/kl/{base,org}/*.npz (per-position KL base<->organism on the first 256
generated tokens of each trace) + grades. The Qi-et-al. shallow-alignment plot in
control space: if control-SFT is a start-of-trace trigger, KL(org||base) should
spike in the first few tokens under instruction_only and decay fast, and be ~flat
under no_instruction.

  python analysis/kl_analysis.py
Outputs figs/kl_profiles.png, figs/kl_depth.png, kl_stats.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_analysis import DATA, FIGS, HERE, TYPES

KDIR = DATA / "kl"
K_SHOW = 128          # positions plotted (log-spaced view of first 256 saved)
SRC_LABEL = {"base": "base-model traces", "org": "organism traces"}
COND_ORDER = ["no_instruction", "instruction_only"]
COND_LABEL = {"no_instruction": "no instruction", "instruction_only": "instruction"}


def load_table() -> pd.DataFrame:
    rows = []
    for src in ("base", "org"):
        d = KDIR / src
        if not d.exists():
            continue
        for f in sorted(d.glob("*.npz")):
            z = np.load(f)
            tid = f.stem
            ctype, cond = str(z["ctype"]), str(z["condition"])
            rows.append({
                "trace_id": tid, "source": src, "ctype": ctype, "condition": cond,
                "qhash": tid.split("_")[-2] if "_" in tid else "",
                "k_gen": int(z["k_gen"]),
                "kl_ob": z["kl_org_base"].astype(np.float32),
                "kl_bo": z["kl_base_org"].astype(np.float32),
                "agree": z["agree_top1"].astype(np.float32),
                "pb_base": z["p_banned_base"].astype(np.float32) if "p_banned_base" in z else None,
                "pb_org": z["p_banned_org"].astype(np.float32) if "p_banned_org" in z else None,
            })
    return pd.DataFrame(rows)


def padded_mean(arrs: list[np.ndarray], K: int) -> tuple[np.ndarray, np.ndarray]:
    M = np.full((len(arrs), K), np.nan, np.float32)
    for i, a in enumerate(arrs):
        M[i, :min(len(a), K)] = a[:K]
    n = np.sum(~np.isnan(M), 0).clip(1)
    return np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(n)


def fig_profiles(df: pd.DataFrame) -> pd.DataFrame:
    types = [t for t in TYPES if t in set(df.ctype)]
    fig, axes = plt.subplots(len(types), 2, figsize=(11, 2.6 * len(types)),
                             sharex=True, squeeze=False)
    x = np.arange(K_SHOW) + 1
    stats = []
    colors = {("base", "no_instruction"): "#9ecae1", ("base", "instruction_only"): "#2171b5",
              ("org", "no_instruction"): "#fdae6b", ("org", "instruction_only"): "#d94801"}
    for ti, ct in enumerate(types):
        for mi, (metric, mlabel) in enumerate([("kl_ob", "KL(org || base) (nats)"),
                                               ("kl_bo", "KL(base || org) (nats)")]):
            ax = axes[ti, mi]
            for src in ("base", "org"):
                for cond in COND_ORDER:
                    sub = df[(df.ctype == ct) & (df.source == src) & (df.condition == cond)]
                    if len(sub) < 5:
                        continue
                    m, se = padded_mean(list(sub[metric]), K_SHOW)
                    c = colors[(src, cond)]
                    ax.plot(x, m, color=c, lw=1.6,
                            label=f"{SRC_LABEL[src][:4]} / {COND_LABEL[cond]} (n={len(sub)})")
                    ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=c, alpha=0.15)
                    half = np.argmax(m < m[:4].mean() / 2) if np.any(m < m[:4].mean() / 2) else K_SHOW
                    stats.append({"ctype": ct, "source": src, "condition": cond,
                                  "metric": metric, "n": len(sub),
                                  "kl_tok1": float(m[0]), "kl_tok1_4": float(m[:4].mean()),
                                  "kl_tail_64_128": float(np.nanmean(m[64:])),
                                  "half_decay_pos": int(half)})
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)
            if mi == 0:
                ax.set_ylabel(ct, fontsize=9)
            if ti == 0:
                ax.set_title(mlabel, fontsize=10)
            if ti == len(types) - 1:
                ax.set_xlabel("generated token position (log)")
    fig.suptitle("Teacher-forced per-position KL, base vs v34 organism "
                 "(mean over traces; both models' own traces forced through both)",
                 fontsize=11)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "kl_profiles.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIGS/'kl_profiles.png'}")
    return pd.DataFrame(stats)


def fig_depth(df: pd.DataFrame) -> None:
    """Cumulative share of total (first-128-token) KL contributed by the first k
    tokens — 'how shallow is the control' in one curve per type."""
    types = [t for t in TYPES if t in set(df.ctype)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    cmap = plt.cm.tab10(np.linspace(0, 1, len(types)))
    for si, src in enumerate(("base", "org")):
        ax = axes[si]
        for ci, ct in enumerate(types):
            sub = df[(df.ctype == ct) & (df.source == src)
                     & (df.condition == "instruction_only")]
            if len(sub) < 5:
                continue
            m, _ = padded_mean(list(sub.kl_ob), K_SHOW)
            # mask gen position 1: '<think>\n' vs '<think>' formatting artifact
            # (~32 nats in EVERY cell incl. no-instruction; not control)
            m[1] = np.nan
            cum = np.nancumsum(m) / np.nansum(m)
            ax.plot(np.arange(K_SHOW) + 1, cum, color=cmap[ci], lw=1.8,
                    label=f"{ct} (n={len(sub)})")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.axhline(0.5, color="#999", ls="--", lw=0.8)
        ax.set_title(f"{SRC_LABEL[src]}, instruction condition", fontsize=10)
        ax.set_xlabel("first k generated tokens (log)")
        ax.legend(fontsize=7)
        if si == 0:
            ax.set_ylabel("cumulative share of KL(org||base)")
    fig.suptitle("Depth of the control: share of total divergence in first k tokens (format-artifact pos 1 masked)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "kl_depth.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIGS/'kl_depth.png'}")


def main() -> None:
    df = load_table()
    print(f"kl table: {len(df)} traces "
          f"({(df.source == 'base').sum()} base, {(df.source == 'org').sum()} org)")
    stats = fig_profiles(df)
    stats.to_csv(HERE / "kl_stats.csv", index=False)
    fig_depth(df)
    print(stats.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
