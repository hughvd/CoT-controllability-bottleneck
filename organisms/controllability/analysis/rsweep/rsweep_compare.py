"""Held-out vs in-distribution generalization by control category for the LoRA
r-sweep (r16/r32/r64). Reports BOTH soft (compliance_score) and hard
(compliance_hard) aggregated over held-out members per category, plus in-dist.
"""
import sys, yaml, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(HERE / "sft/configs/v33_diverse_r64.yaml"))["data"]
ENC = {"disemvowel", "reverse_each_word", "rot13"}          # eval-only encodings
ENC_TRAIN = {"pig_latin", "leetspeak"}

held_fam = set(CFG["held_out_families"])
train_fam = set(CFG["families"])
CATS = {
    "transforms":  ("family",      train_fam - ENC_TRAIN, held_fam - ENC),
    "encoding":    ("family",      ENC_TRAIN,             held_fam & ENC),
    "language":    ("language",    {f"language:{l}" for l in CFG["languages"]},
                                    {f"language:{l}" for l in CFG["held_out_languages"]}),
    "suppression": ("suppression", {f"suppression:{s}" for s in CFG["suppression"]},
                                    {f"suppression:{s}" for s in CFG["held_out_suppression"]}),
}

def agg(df, mod, members, col):
    sub = df[(df["modality"] == mod) & (df["family"].isin(members))][col]
    return sub.mean() if len(sub) else float("nan")

def analyze(stem, path):
    df = pd.read_csv(path)
    if "compliance_hard" not in df.columns:
        df["compliance_hard"] = (df["compliance_score"] >= 1.0 - 1e-9).astype(int)
    rows = []
    for cat, (mod, train_m, held_m) in CATS.items():
        rows.append({
            "run": stem, "category": cat,
            "indist_soft":  round(agg(df, mod, train_m, "compliance_score"), 3),
            "indist_hard":  round(agg(df, mod, train_m, "compliance_hard"), 3),
            "heldout_soft": round(agg(df, mod, held_m, "compliance_score"), 3),
            "heldout_hard": round(agg(df, mod, held_m, "compliance_hard"), 3),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    out = []
    for stem, path in [a.split("=") for a in sys.argv[1:]]:
        out.append(analyze(stem, path))
    res = pd.concat(out, ignore_index=True)
    pd.set_option("display.width", 200)
    # pivot: held-out soft/hard side by side per run
    for metric in ["heldout_soft", "heldout_hard", "indist_soft", "indist_hard"]:
        piv = res.pivot(index="category", columns="run", values=metric)
        print(f"\n=== {metric} ===")
        print(piv.to_string())
    res.to_csv(HERE / "analysis/rsweep/rsweep_compare.csv", index=False)
    print("\nwrote analysis/rsweep/rsweep_compare.csv")
