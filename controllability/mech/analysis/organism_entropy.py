"""Goal-2 mechanism: first-thinking-token entropy, base vs control vs organism
(v34 seed-1 organism/control + base). Tests whether control-training installs a
"mode switch" at trace onset. Headline diff = organism - control (nets out the
off-policy SFT shift; control = same 14B-trace SFT without control instructions);
base is the untrained reference.

Data: controllability/mech/data/organism/{base,org_s1,ctl_s1}/entropy/*.npz
(entropy[t] = full-vocab next-token entropy at generated position t; entropy[0] =
first thinking token). 4 ctypes x {no_instruction, instruction_only} x 30q x 8s.

Usage:  python3 mech/analysis/organism_entropy.py
Writes mech/analysis/figs/organism_entropy_{firsttoken,profiles}.png + a table.
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "organism"
OUT = HERE / "figs"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [("base", "base", "#9aa0a6"), ("ctl_s1", "control", "#3b6fb6"),
          ("org_s1", "organism", "#e8710a")]
CTYPES = ["uppercase_thinking", "leetspeak", "spanish", "omit_word"]
CONDS = ["no_instruction", "instruction_only"]
NPROF = 40
INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"


def load(model_dir):
    rows, prof = [], {}
    for fp in glob.glob(f"{DATA}/{model_dir}/entropy/*.npz"):
        z = np.load(fp, allow_pickle=True)
        e = z["entropy"].astype(np.float32)
        if e.size == 0:
            continue
        ct, cd = str(z["ctype"]), str(z["condition"])
        # tokens 0-3 are a near-deterministic opener (entropy ~0 for every model);
        # onset = the early-reasoning window just after it, where control-execution
        # commitment shows up. whole = whole-trace mean entropy.
        rows.append({"ctype": ct, "cond": cd,
                     "onset": float(e[4:14].mean()) if e.size >= 6 else float(e.mean()),
                     "whole": float(e.mean())})
        prof.setdefault((ct, cd), []).append(e[:NPROF])
    return pd.DataFrame(rows), prof


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    data = {name: load(md) for md, name, _ in MODELS}
    names = [n for _, n, _ in MODELS]
    colors = {n: c for _, n, c in MODELS}

    # ---- tables: onset + whole-trace entropy mean per (ctype, cond) per model ----
    def build(metric):
        recs = []
        for ct in CTYPES:
            for cd in CONDS:
                row = {"ctype": ct, "cond": cd}
                for n in names:
                    d = data[n][0]
                    sub = d[(d.ctype == ct) & (d.cond == cd)]
                    row[n] = sub[metric].mean() if len(sub) else np.nan
                row["org−ctl"] = row["organism"] - row["control"]
                row["org−base"] = row["organism"] - row["base"]
                recs.append(row)
        return pd.DataFrame(recs)
    tbl, tbl_whole = build("onset"), build("whole")
    pd.set_option("display.width", 200)
    print("=== ONSET thinking entropy (nats, tokens 4–14) — mean per (ctype, condition) ===")
    print(tbl.round(3).to_string(index=False))
    print("\n=== WHOLE-TRACE mean entropy (nats) ===")
    print(tbl_whole.round(3).to_string(index=False))

    # ---- Figure 1: first-token entropy bars, 2 panels (condition) ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=150)
    fig.suptitle("Onset thinking entropy (tokens 4–14) — base vs control vs organism (v34 seed-1)\n"
                 "control-training's effect = organism − control (off-policy SFT netted out)",
                 fontsize=11, color=INK, y=1.03)
    for ax, cd in zip(axes, CONDS):
        x = np.arange(len(CTYPES)); w = 0.26
        for i, n in enumerate(names):
            d = data[n][0]
            means, sems = [], []
            for ct in CTYPES:
                sub = d[(d.ctype == ct) & (d.cond == cd)].onset
                means.append(sub.mean() if len(sub) else np.nan)
                sems.append(sub.sem() if len(sub) > 1 else 0)
            ax.bar(x + (i - 1) * w, means, w, yerr=sems, color=colors[n], label=n,
                   zorder=3, error_kw=dict(lw=0.8, ecolor=MUTED))
        ax.set_xticks(x); ax.set_xticklabels([c.replace("_", "\n") for c in CTYPES], fontsize=8)
        ax.set_title(cd, fontsize=10, color=INK); _style(ax)
    axes[0].set_ylabel("onset entropy (nats)", fontsize=8.5, color=MUTED)
    axes[1].legend(fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "organism_entropy_onset.png", bbox_inches="tight")
    print("\nwrote", OUT / "organism_entropy_onset.png")

    # ---- Figure 2: entropy profile over first NPROF tokens (instruction_only) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=150)
    fig.suptitle("Entropy over the first %d thinking tokens (instruction_only) — onset dynamics"
                 % NPROF, fontsize=11, color=INK, y=1.0)
    for ax, ct in zip(axes.flat, CTYPES):
        for n in names:
            prof = data[n][1].get((ct, "instruction_only"), [])
            if not prof:
                continue
            M = np.vstack([np.pad(p, (0, NPROF - len(p)), constant_values=np.nan)[:NPROF] for p in prof])
            mean = np.nanmean(M, axis=0)
            ax.plot(range(NPROF), mean, color=colors[n], lw=1.8, label=n)
        ax.set_title(ct, fontsize=9.5, color=INK)
        ax.set_xlabel("generated token position", fontsize=8, color=MUTED)
        _style(ax); ax.grid(True, color=GRID, lw=0.6)
    axes[0, 0].set_ylabel("entropy (nats)", fontsize=8.5, color=MUTED)
    axes[0, 0].legend(fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "organism_entropy_profiles.png", bbox_inches="tight")
    print("wrote", OUT / "organism_entropy_profiles.png")
    tbl.to_csv(OUT / "organism_entropy_onset.csv", index=False)


if __name__ == "__main__":
    main()
