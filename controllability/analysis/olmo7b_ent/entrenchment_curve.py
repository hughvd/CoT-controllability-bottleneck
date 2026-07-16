"""OLMo-3-7B RL-checkpoint entrenchment curve.
Held-out control lift (post-SFT minus each checkpoint's own baseline floor),
by category, along the RL axis: SFT -> DPO -> RLVR step_0350/0700/final.
Reads results/olmo7b_ent/organism_2_olmo7b_ent_{tag}{,_base}_diverse.csv.
"""
import pandas as pd, yaml
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(HERE / "sft/configs/v33_diverse_r64.yaml"))["data"]
ENC = {"disemvowel", "reverse_each_word", "rot13"}; held = set(CFG["held_out_families"])
CATS = {"transforms": ("family", held - ENC), "encoding": ("family", held & ENC),
        "language": ("language", {f"language:{l}" for l in CFG["held_out_languages"]}),
        "suppression": ("suppression", {f"suppression:{s}" for s in CFG["held_out_suppression"]})}
def ho(df, col, c):
    m, mem = CATS[c]; s = df[(df.modality == m) & (df.family.isin(mem))][col]
    return s.mean() if len(s) else float("nan")
R = HERE / "results/olmo7b_ent/organism_2_olmo7b_ent_"
TAGS = ["sft", "dpo", "rl_early", "rl_mid", "rl_final"]
rows = []
for tag in TAGS:
    b = pd.read_csv(f"{R}{tag}_base_diverse.csv"); s = pd.read_csv(f"{R}{tag}_diverse.csv")
    for c in CATS:
        rows.append(dict(ckpt=tag, category=c,
            floor_soft=round(ho(b, "compliance_score", c), 3), post_soft=round(ho(s, "compliance_score", c), 3),
            lift_soft=round(ho(s, "compliance_score", c) - ho(b, "compliance_score", c), 3),
            lift_hard=round(ho(s, "compliance_hard", c) - ho(b, "compliance_hard", c), 3)))
out = pd.DataFrame(rows)
out.to_csv(HERE / "analysis/olmo7b_ent/entrenchment_curve.csv", index=False)
for m in ["lift_soft", "lift_hard"]:
    print(f"=== {m} ===")
    print(out.pivot(index="ckpt", columns="category", values=m).reindex(TAGS)[list(CATS)].to_string())
print("\nwrote analysis/olmo7b_ent/entrenchment_curve.csv")
