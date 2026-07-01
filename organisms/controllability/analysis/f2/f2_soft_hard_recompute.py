"""Recompute the F2 soft/hard transfer table from per_sample, with HARD at a
configurable threshold (default 0.95, was exact-1.0). Soft = mean
compliance_score (unchanged); hard = fraction with score >= threshold.
Mirrors analysis/f2/f2_analysis.py's diagonal/floor logic exactly.
Output: results/f2/f2_transfer_soft_hard_0p95.csv
"""
import sys, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = next(q for q in [HERE, *HERE.parents] if (q / "sft" / "configs").is_dir())
RESULTS = ROOT / "results" / "f2"
MODELS = ["qwen3_8b", "qwen3_14b", "qwen3_32b"]
CATS = ["transforms", "encoding", "language", "suppression"]
ENC = {"pig_latin", "leetspeak", "disemvowel", "reverse_each_word", "rot13"}
TR5 = {"uppercase_thinking", "alternating_case", "prefix_sentences", "start_with", "exclaim_all_sentences"}
TRAIN_LANGS = {f"language:{l}" for l in ["spanish", "german", "chinese", "russian", "hebrew"]}
TRAIN_SUPP = {f"suppression:{s}" for s in ["omit_word", "omit_multiple_words"]}
TRAINED = {"transforms": TR5, "encoding": {"pig_latin", "leetspeak"},
           "language": TRAIN_LANGS, "suppression": TRAIN_SUPP}
TH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.95

def cat_of(row):
    if row["modality"] == "language": return "language"
    if row["modality"] == "suppression": return "suppression"
    return "encoding" if row["family"] in ENC else "transforms"

def col(df, metric):
    return (df["compliance_score"] >= TH).astype(float) if metric == "hard" else df["compliance_score"]

def cat_mean(df, eval_cat, train_cat, metric, held_out=True):
    d = df.copy(); d["cat"] = d.apply(cat_of, axis=1)
    sub = d[d["cat"] == eval_cat]
    if held_out and eval_cat == train_cat:   # diagonal -> within-family held-out
        sub = sub[~sub["family"].isin(TRAINED[eval_cat])]
    return col(sub, metric).mean() if len(sub) else float("nan")

def ps(model, run):
    return pd.read_csv(RESULTS / model / run / "per_sample" / "final.csv")

rows = []
for model in MODELS:
    base = ps(model, "base")
    for metric in ["soft", "hard"]:
        for tr in CATS:
            df = ps(model, tr)
            indist = col(df[df.apply(cat_of, axis=1) == tr].pipe(lambda d: d[d["family"].isin(TRAINED[tr])]), metric).mean()
            for ev in CATS:
                raw = cat_mean(df, ev, tr, metric)
                floor = cat_mean(base, ev, tr, metric)
                rows.append(dict(model=model, metric=metric, train=tr, eval=ev,
                                 raw=round(raw, 4), floor=round(floor, 4), lift=round(raw - floor, 4),
                                 in_dist=round(indist, 4), diag=(tr == ev)))
res = pd.DataFrame(rows)
res.to_csv(RESULTS / "f2_transfer_soft_hard_0p95.csv", index=False)
print(f"threshold={TH}; wrote results/f2/f2_transfer_soft_hard_0p95.csv ({len(res)} rows)")
# spotlight: in-dist HARD per model/category (the numbers that move most)
piv = res[res.metric == "hard"].drop_duplicates(["model", "train"]).pivot(index="train", columns="model", values="in_dist")
print("\n=== IN-DIST HARD@%.2f (trained members) ===" % TH); print(piv.reindex(CATS).to_string())
