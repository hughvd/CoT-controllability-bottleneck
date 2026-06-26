"""Held-out compliance by control category across the rl_v33_diverse RL trajectory.

x = overall GRPO step (0 = exp16 SFT seed / RL=0; 60,100 = rl_v33_diverse;
160,200 = rl_v33_diverse_cont warm-started from step100). One line per category,
restricted to the HELD-OUT (never-rewarded) members of each category.
"""
import pandas as pd, yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

R = Path.home() / "v33_backup" / "results"
CFG = yaml.safe_load(open("rl/configs/rl_v33_diverse.yaml"))["data"]

# checkpoints: (overall_step, summary_csv_stem)
PTS = [
    (0,   "organism_2_v33_diverse_r64_exp16_n20_diverse"),   # seed (RL=0)
    (60,  "organism_2_rl_v33_diverse_step60_diverse"),
    (100, "organism_2_rl_v33_diverse_diverse"),
    (160, "organism_2_rl_v33_diverse_cont_step60_diverse"),  # cont step60 = overall 160
    (200, "organism_2_rl_v33_diverse_cont_diverse"),          # cont final  = overall 200
]

# Held-out membership per category.
ENC = {"disemvowel", "reverse_each_word", "rot13"}            # eval-only encodings
held_fam = set(CFG["held_out_families"])
CAT = {
    "transforms":  ("family",      held_fam - ENC),
    "encoding":    ("family",      held_fam & ENC),
    "language":    ("language",    {f"language:{l}" for l in CFG["held_out_languages"]}),
    "suppression": ("suppression", {f"suppression:{s}" for s in CFG["held_out_suppression"]}),
}

def cat_mean(df, mod, members):
    x = df[(df["modality"] == mod) & (df["family"].isin(members))]["compliance"]
    return x.mean() if len(x) else float("nan")

rows = {c: [] for c in CAT}
allho = []
for step, stem in PTS:
    df = pd.read_csv(R / f"{stem}.csv")
    for c, (mod, mem) in CAT.items():
        rows[c].append(cat_mean(df, mod, mem))
    # overall held-out (union of all held members across modalities)
    mask = False
    for mod, mem in CAT.values():
        mask = mask | ((df["modality"] == mod) & (df["family"].isin(mem)))
    allho.append(df[mask]["compliance"].mean())

steps = [p[0] for p in PTS]
out = pd.DataFrame({"step": steps, **{c: rows[c] for c in CAT}, "all_heldout": allho})
out.to_csv("analysis/rl_v33/rl_v33_heldout_by_category.csv", index=False)
print(out.round(3).to_string(index=False))

plt.figure(figsize=(8, 5.5))
colors = {"transforms": "#1f77b4", "encoding": "#d62728",
          "language": "#2ca02c", "suppression": "#9467bd"}
for c in CAT:
    plt.plot(steps, rows[c], marker="o", lw=2, color=colors[c],
             label=f"{c} (held-out)")
plt.plot(steps, allho, marker="s", lw=2.5, ls="--", color="black",
         label="all held-out", zorder=5)
plt.axvline(100, color="gray", ls=":", lw=1)
plt.text(101, 0.02, "warm-start\ncontinuation", fontsize=8, color="gray", va="bottom")
plt.xlabel("GRPO step (0 = exp16 SFT seed, RL=0)")
plt.ylabel("held-out compliance")
plt.title("rl_v33_diverse: held-out generalization by control category")
plt.ylim(-0.02, 1.0)
plt.xticks(steps)
plt.grid(alpha=0.3)
plt.legend(loc="center right", framealpha=0.9)
plt.tight_layout()
plt.savefig("analysis/rl_v33/rl_v33_heldout_by_category.png", dpi=150)
print("\nsaved analysis/rl_v33/rl_v33_heldout_by_category.png")
