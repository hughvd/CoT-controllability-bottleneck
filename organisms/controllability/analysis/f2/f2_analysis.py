"""F2 — per-model cross-/within-category transfer.

Each model has 4 single-category runs (train one category, eval the full
universe). We render one transfer heatmap per model: rows = trained category,
cols = evaluated category. Cell = SUSTAINED held-out compliance (the n=20
final eval) MINUS the base-model floor, so every cell reads as "lift over
untrained". The diagonal is the trained category's held-out members
(within-family generalization); off-diagonal is cross-category transfer.

Why floor-subtracted: the suppression column sits at a high chance floor
(common forbidden words omitted by luck), which otherwise reads as spurious
transfer. Subtracting each model's own base floor removes it.

Usage:
    python3 analysis/f2/f2_analysis.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def _root(p: Path) -> Path:
    for q in [p, *p.parents]:
        if (q / "sft" / "configs").is_dir():
            return q
    return p.parent.parent


ROOT = _root(HERE)
RESULTS = ROOT / "results" / "f2"
MODELS = ["qwen3_8b", "qwen3_14b"]   # extend as models land (qwen3_32b, gpt_oss_20b, ...)
CATS = ["transforms", "encoding", "language", "suppression"]
ENC = {"pig_latin", "leetspeak", "disemvowel", "reverse_each_word", "rot13"}
# the 5 trained transforms (v32 set) — to split the transforms diagonal into
# trained vs within-family held-out
TR5 = {"uppercase_thinking", "alternating_case", "prefix_sentences",
       "start_with", "exclaim_all_sentences"}
TRAIN_LANGS = {f"language:{l}" for l in ["spanish", "german", "chinese", "russian", "hebrew"]}
TRAIN_SUPP = {f"suppression:{s}" for s in ["omit_word", "omit_multiple_words"]}


def cat_of(row) -> str:
    if row["modality"] == "language":
        return "language"
    if row["modality"] == "suppression":
        return "suppression"
    return "encoding" if row["family"] in ENC else "transforms"


def eval_category_mean(df: pd.DataFrame, eval_cat: str, train_cat: str) -> float:
    """Mean compliance over eval_cat. On the diagonal (eval_cat==train_cat),
    restrict to HELD-OUT members (within-family generalization), excluding the
    trained types."""
    d = df.copy()
    d["cat"] = d.apply(cat_of, axis=1)
    sub = d[d["cat"] == eval_cat]
    if eval_cat == train_cat:
        if eval_cat == "transforms":
            sub = sub[~sub["family"].isin(TR5)]
        elif eval_cat == "language":
            sub = sub[~sub["family"].isin(TRAIN_LANGS)]
        elif eval_cat == "suppression":
            sub = sub[~sub["family"].isin(TRAIN_SUPP)]
        # encoding: both encodings trained -> within-held = the 3 eval-only
        # encodings (disemvowel/reverse/rot13), which are at floor anyway
        elif eval_cat == "encoding":
            sub = sub[~sub["family"].isin({"pig_latin", "leetspeak"})]
    return sub["compliance"].mean() if len(sub) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    present = [m for m in MODELS if (RESULTS / m).is_dir()]
    fig, axes = plt.subplots(1, len(present), figsize=(5.6 * len(present), 5.2), squeeze=False)
    all_rows = []

    for ax, model in zip(axes[0], present):
        base = pd.read_csv(RESULTS / model / "base" / "final.csv")
        M = np.full((len(CATS), len(CATS)), np.nan)
        for i, tr in enumerate(CATS):
            f = RESULTS / model / tr / "final.csv"
            if not f.exists():
                continue
            df = pd.read_csv(f)
            for j, ev in enumerate(CATS):
                raw = eval_category_mean(df, ev, tr)
                # floor must match the cell's eval set: on the diagonal that's
                # the held-out members (pass the same train_cat), off-diagonal
                # the full category.
                floor_ij = eval_category_mean(base, ev, tr)
                adj = raw - floor_ij
                M[i, j] = adj
                all_rows.append({"model": model, "train_cat": tr, "eval_cat": ev,
                                 "raw": raw, "floor": floor_ij, "lift": adj,
                                 "within_family_diag": tr == ev})
        im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="equal")
        ax.set_xticks(range(len(CATS))); ax.set_yticks(range(len(CATS)))
        ax.set_xticklabels(CATS, rotation=20, ha="right"); ax.set_yticklabels(CATS)
        ax.set_xlabel("evaluated category"); ax.set_ylabel("trained category")
        ax.set_title(model)
        for i in range(len(CATS)):
            for j in range(len(CATS)):
                v = M[i, j]
                if np.isnan(v):
                    txt, col = "n/a", "#888"
                else:
                    txt = f"{v:+.2f}"; col = "white" if abs(v) > 0.35 else "black"
                if i == j and not np.isnan(v):
                    txt += "\n(within)"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=col)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="compliance lift over base floor")

    fig.suptitle("F2 transfer: train one category, eval all (lift over base floor; diag = within-family held-out)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = args.out_dir / "f2_transfer_heatmaps.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    tidy = pd.DataFrame(all_rows)
    tidy.to_csv(args.out_dir / "f2_transfer.csv", index=False)

    print("=== F2 transfer (lift over base floor; diagonal = within-family held-out) ===")
    for model in present:
        sub = tidy[tidy["model"] == model]
        print(f"\n{model}:")
        piv = sub.pivot(index="train_cat", columns="eval_cat", values="lift").reindex(index=CATS, columns=CATS)
        print(piv.to_string(float_format=lambda x: f"{x:+.3f}", na_rep="  -"))
    print(f"\nwrote -> {p}, {args.out_dir/'f2_transfer.csv'}")


if __name__ == "__main__":
    main()
