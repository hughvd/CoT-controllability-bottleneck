"""OLMo-3-7B entrenchment sweep — two line plots.
  1) in-dist(train) + held-out compliance, for BASE and SFT'd models, vs
     checkpoint, for transforms and language (2 subplots, 4 lines each).
  2) held-out lift (SFT held-out - base held-out) vs checkpoint, transforms + language.
Metric: soft (compliance_score). Reads results/olmo7b_ent/.
"""
import pandas as pd, yaml
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(HERE / "sft/configs/v33_diverse_r64.yaml"))["data"]
ENC_HELD = {"disemvowel", "reverse_each_word", "rot13"}
ENC_TRAIN = {"pig_latin", "leetspeak"}
MEMBERS = {
    "transforms": {"train": ("family", set(CFG["families"]) - ENC_TRAIN),
                   "held":  ("family", set(CFG["held_out_families"]) - ENC_HELD)},
    "language":   {"train": ("language", {f"language:{l}" for l in CFG["languages"]}),
                   "held":  ("language", {f"language:{l}" for l in CFG["held_out_languages"]})},
}
TAGS = ["sft", "dpo", "rl_early", "rl_mid", "rl_final"]
XLAB = ["SFT", "DPO", "RL early\n(step 350)", "RL mid\n(step 700)", "RL final"]
R = HERE / "results/olmo7b_ent/organism_2_olmo7b_ent_"
COL = "compliance_score"

def mean(df, cat, split):
    mod, mem = MEMBERS[cat][split]
    s = df[(df.modality == mod) & (df.family.isin(mem))][COL]
    return s.mean() if len(s) else float("nan")

# gather: data[cat][model][split] = list over checkpoints
data = {c: {"base": {"train": [], "held": []}, "sft": {"train": [], "held": []}} for c in MEMBERS}
for tag in TAGS:
    b = pd.read_csv(f"{R}{tag}_base_diverse.csv"); s = pd.read_csv(f"{R}{tag}_diverse.csv")
    for c in MEMBERS:
        data[c]["base"]["train"].append(mean(b, c, "train"))
        data[c]["base"]["held"].append(mean(b, c, "held"))
        data[c]["sft"]["train"].append(mean(s, c, "train"))
        data[c]["sft"]["held"].append(mean(s, c, "held"))

BLUE, VERM = "#0072B2", "#D55E00"          # CVD-safe (validated ΔE 92)
INK, MUTE = "#222222", "#888888"
x = list(range(len(TAGS)))

def style_ax(ax):
    ax.set_xticks(x); ax.set_xticklabels(XLAB, fontsize=8)
    ax.grid(True, axis="y", color="#e6e6e6", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTE)
    ax.tick_params(colors=MUTE, labelcolor=INK)

# ---- Plot 1: base + SFT, train + held-out ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
for ax, cat in zip(axes, ["transforms", "language"]):
    # color = split (train/held), linestyle = model (base dashed / sft solid)
    ax.plot(x, data[cat]["sft"]["train"], "-o", color=BLUE, lw=2, ms=6, label="SFT · in-dist")
    ax.plot(x, data[cat]["base"]["train"], "--o", color=BLUE, lw=1.6, ms=5, mfc="white", label="base · in-dist")
    ax.plot(x, data[cat]["sft"]["held"], "-s", color=VERM, lw=2, ms=6, label="SFT · held-out")
    ax.plot(x, data[cat]["base"]["held"], "--s", color=VERM, lw=1.6, ms=5, mfc="white", label="base · held-out")
    ax.set_title(cat, fontsize=12, color=INK, weight="bold")
    ax.set_ylabel("compliance (soft)", fontsize=9, color=INK)
    ax.set_ylim(bottom=0)
    style_ax(ax)
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("OLMo-3-7B: control compliance along the RL axis  (base vs SFT'd, in-dist vs held-out)",
             fontsize=12, color=INK, weight="bold", y=1.02)
fig.tight_layout()
fig.savefig(HERE / "analysis/olmo7b_ent/plot1_train_heldout_base_sft.png", dpi=150, bbox_inches="tight")

# ---- Plot 2: held-out lift over base ----
fig2, ax = plt.subplots(figsize=(6.6, 4.4))
lift = {c: [data[c]["sft"]["held"][i] - data[c]["base"]["held"][i] for i in x] for c in MEMBERS}
ax.axhline(0, color=MUTE, lw=1)
ax.plot(x, lift["transforms"], "-o", color=BLUE, lw=2.2, ms=7, label="transforms")
ax.plot(x, lift["language"], "-s", color=VERM, lw=2.2, ms=7, label="language")
# direct labels at the last point
for c, col in [("transforms", BLUE), ("language", VERM)]:
    ax.annotate(c, (x[-1], lift[c][-1]), xytext=(6, 0), textcoords="offset points",
                color=col, fontsize=9, weight="bold", va="center")
ax.set_ylabel("held-out lift over base  (SFT − base)", fontsize=10, color=INK)
ax.set_title("OLMo-3-7B: held-out control lift vs RL checkpoint", fontsize=12, color=INK, weight="bold")
style_ax(ax); ax.set_xlim(-0.2, len(TAGS) - 0.3)
fig2.tight_layout()
fig2.savefig(HERE / "analysis/olmo7b_ent/plot2_heldout_lift.png", dpi=150, bbox_inches="tight")
print("wrote plot1_train_heldout_base_sft.png and plot2_heldout_lift.png")
# echo the numbers for sanity
for c in MEMBERS:
    print(f"\n{c}:")
    for model in ["base", "sft"]:
        print(f"  {model:4} train {[round(v,3) for v in data[c][model]['train']]}  held {[round(v,3) for v in data[c][model]['held']]}")
