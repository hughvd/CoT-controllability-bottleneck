"""F2 transfer heatmaps — TWO figures (soft, hard), qwen3 8b/14b/32b in a row.
Each cell is split by the anti-diagonal into TRAIN ◤ (compliance on the eval
category's TRAINED members) and HELD-OUT ◢ (compliance on its held-out members)
— so every cell shows both in-distribution control and generalization. Rows =
trained category (+ baseline = untrained floor). Cells are RAW compliance.
  soft figure: mean compliance_score.   hard figure: fraction >= 0.95.

Usage: python3 analysis/f2/f2_split_heatmap.py            # 8b 14b 32b
"""
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = next(q for q in [HERE, *HERE.parents] if (q / "sft" / "configs").is_dir())
RES = ROOT / "results" / "f2"
OUTDIR = ROOT / "analysis" / "f2"
HARD_TH = 0.95
MODELS = ["qwen3_8b", "qwen3_14b", "qwen3_32b", "olmo3_7b", "olmo3_32b"]
CATS = ["transforms", "encoding", "language", "suppression"]
ENC = {"pig_latin", "leetspeak", "disemvowel", "reverse_each_word", "rot13"}
TRAINED = {   # the members trained when that category is the training category
    "transforms": {"uppercase_thinking", "alternating_case", "prefix_sentences", "start_with", "exclaim_all_sentences"},
    "encoding":   {"pig_latin", "leetspeak"},
    "language":   {f"language:{l}" for l in ["spanish", "german", "chinese", "russian", "hebrew"]},
    "suppression":{f"suppression:{s}" for s in ["omit_word", "omit_multiple_words"]},
}

def cat_of(r):
    if r["modality"] == "language": return "language"
    if r["modality"] == "suppression": return "suppression"
    return "encoding" if r["family"] in ENC else "transforms"

def ps(model, run):
    d = pd.read_csv(RES / model / run / "per_sample" / "final.csv")
    d["cat"] = d.apply(cat_of, axis=1)
    return d

def val(df, eval_cat, subset, metric):
    sub = df[df["cat"] == eval_cat]
    trained = sub["family"].isin(TRAINED[eval_cat])
    sub = sub[trained] if subset == "train" else sub[~trained]
    s = sub["compliance_score"]
    if not len(s): return np.nan
    return s.mean() if metric == "soft" else (s >= HARD_TH).mean()

def build(model, metric):
    rows = ["baseline"] + CATS
    train = np.full((len(rows), len(CATS)), np.nan)
    held = np.full((len(rows), len(CATS)), np.nan)
    dfs = {"baseline": ps(model, "base"), **{c: ps(model, c) for c in CATS}}
    for i, r in enumerate(rows):
        df = dfs[r]
        for j, ev in enumerate(CATS):
            train[i, j] = val(df, ev, "train", metric)
            held[i, j] = val(df, ev, "heldout", metric)
    return rows, train, held

def panel(ax, model, rows, train, held, cmap, norm):
    nr, nc = len(rows), len(CATS)
    for i in range(nr):
        for j in range(nc):
            TL, TR = (j - .5, i - .5), (j + .5, i - .5)
            BR, BL = (j + .5, i + .5), (j - .5, i + .5)
            tv, hv = train[i, j], held[i, j]
            ax.add_patch(Polygon([TL, TR, BL], facecolor=cmap(norm(tv)) if not np.isnan(tv) else "lightgray", edgecolor="k", lw=.6))
            ax.add_patch(Polygon([TR, BR, BL], facecolor=cmap(norm(hv)) if not np.isnan(hv) else "lightgray", edgecolor="k", lw=.6))
            for v, dx, dy in [(tv, -.22, -.20), (hv, .22, .20)]:
                if np.isnan(v): continue
                ax.text(j + dx, i + dy, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color="white" if norm(v) < .5 else "black")
    ax.set_xlim(-.5, nc - .5); ax.set_ylim(nr - .5, -.5)
    ax.set_xticks(range(nc)); ax.set_xticklabels(CATS, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(nr)); ax.set_yticklabels(rows, fontsize=8)
    ax.axhline(0.5, color="red", lw=1.6)
    ax.set_title(model, fontsize=10); ax.set_xlabel("evaluated category", fontsize=8)

def figure(models, metric):
    data = {m: build(m, metric) for m in models}
    cmap = plt.get_cmap("viridis"); norm = Normalize(0, 1)
    fig, axes = plt.subplots(1, len(models), figsize=(5.0 * len(models) + 1.2, 5.6), squeeze=False)
    fig.subplots_adjust(left=0.10, right=0.9, bottom=0.20, top=0.86, wspace=0.55)
    for ax, m in zip(axes[0], models):
        rows, tr, hd = data[m]
        panel(ax, m, rows, tr, hd, cmap, norm)
        print(f"\n=== {m} [{metric}] TRAIN ◤ ===\n", pd.DataFrame(tr, index=rows, columns=CATS).round(3).to_string())
        print(f"=== {m} [{metric}] HELD-OUT ◢ ===\n", pd.DataFrame(hd, index=rows, columns=CATS).round(3).to_string())
    axes[0][0].set_ylabel("trained category  (baseline = untrained floor)", fontsize=8)
    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=axes[0].tolist(), fraction=.025, pad=.04, label=f"{metric} compliance")
    lbl = "mean compliance_score" if metric == "soft" else f"fraction ≥ {HARD_TH}"
    fig.suptitle(f"F2 transfer — {metric.upper()} ({lbl}). Cell: TRAIN members ◤ | HELD-OUT members ◢. "
                 f"Diagonal ◢ = within-family generalization.", fontsize=10, y=1.02)
    out = OUTDIR / f"f2_transfer_{metric}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); print("\nwrote", out)

if __name__ == "__main__":
    models = sys.argv[1:] or MODELS
    for metric in ["soft", "hard"]:
        figure(models, metric)
