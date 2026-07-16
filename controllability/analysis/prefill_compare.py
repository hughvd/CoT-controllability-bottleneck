"""Cross-model comparison for the prefill/CoT-seeding experiment.

- PLOT: mean continuation compliance vs K, averaged across all behaviors, one color
  per model. Solid line = prefill+instruction, faded (alpha) line = prefill-only.
- TABLE: per-behavior K=5 sustain (prefill-only) for every model + the SFT ceiling.

Averages weight each behavior equally (mean of per-behavior means). empty-thinking
rows are excluded (a model with 100% empty at K=0, e.g. medium-effort gpt-oss, simply
has no K=0 point). Usage: python analysis/prefill_compare.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE / "prefill"

# (label, csv, color, note)
MODELS = [
    ("qwen3-14b",       "base_qwen3_14b_prefill_per_sample_clean.csv", "#7fb3ff", ""),
    ("qwen3-32b",       "base_qwen3_32b_prefill_per_sample.csv",       "#1f4eb0", ""),
    ("deepseek-r1",     "base_deepseek_r1_prefill_per_sample.csv",     "#2ca02c", ""),
    ("glm-5.2",         "base_glm_5_2_prefill_per_sample.csv",         "#9467bd", ""),
    ("gpt-oss-120b",    "base_gptoss_120b_prefill_per_sample.csv",     "#ff7f0e", " (medium-effort, provisional)"),
]
BEHAVIORS = ["leetspeak", "spanish", "chinese", "uppercase_thinking", "prefix_sentences",
             "no_first_person", "omit_word", "pirate", "yoda", "jar_jar_binks"]

# K=0 floor override: the main-sweep K=0 was collected with a "<cot_instructions> none"
# block that some models (qwen3-32b) read as "produce no CoT", suppressing reasoning and
# contaminating the K=0 base/+i floor. These re-runs drop the block. Falls back to the
# model's main CSV when a k0fix file isn't present (e.g. gpt-oss until a pod run).
# qwen3-32b was the only model contaminated: it obeyed "none" (49% empty at K=0).
# qwen3-14b, deepseek-r1 and (natively) gpt-oss ignored the block — 0% empty at K=0,
# so their original K=0 floor is already valid and needs no override.
K0FIX = {
    "qwen3-14b":   "base_qwen3_14b_k0fix_prefill_per_sample.csv",   # ≈ original; kept for parity
    "qwen3-32b":   "base_qwen3_32b_k0fix_prefill_per_sample.csv",   # the real fix
}


def load(csv: str) -> pd.DataFrame:
    d = pd.read_csv(RESULTS / csv)
    if "empty_thinking" in d:
        d = d[~d["empty_thinking"].astype(bool)]
    return d


def load_k0(label: str, main: pd.DataFrame) -> pd.DataFrame:
    """K=0 floor source for `label`: the block-dropped re-run if it exists, else main."""
    csv = K0FIX.get(label)
    if csv and (RESULTS / csv).exists():
        return load(csv)
    return main


def behavior_avg_by_k(d: pd.DataFrame, condition: str) -> pd.Series:
    """Mean compliance by K, averaging behaviors equally."""
    sub = d[d.condition == condition]
    per = sub.groupby(["ctype", "k"])["compliance_score"].mean().reset_index()
    return per.groupby("k")["compliance_score"].mean()


def make_plot() -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for label, csv, color, note in MODELS:
        d = load(csv)
        pi = behavior_avg_by_k(d, "prefill_plus_instruction")
        po = behavior_avg_by_k(d, "prefill_only")
        ax.plot(pi.index, pi.values, "-o", color=color, lw=2.4, label=f"{label}{note} — prefill+instr")
        ax.plot(po.index, po.values, "--o", color=color, lw=1.6, alpha=0.45, markersize=4,
                label=f"{label} — prefill-only")
    ax.set_xlabel("K (injected controlled sentences)")
    ax.set_ylabel("mean continuation compliance\n(averaged over 10 behaviors)")
    ax.set_title("Does the base model sustain a seeded controlled CoT?\ncompliance vs. seed length, by model")
    ax.set_xticks(range(6))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    fig.tight_layout()
    out = OUT / "prefill_compare_avg.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def _val(d, ct, k, cond):
    s = d[(d.ctype == ct) & (d.condition == cond) & (d.k == k)]["compliance_score"]
    return s.mean() if len(s) else float("nan")


def make_table() -> pd.DataFrame:
    """CSV: per behavior, per model [base (K0/no-instr), instr (K0/+instr), PO@K5, PI@K5] + SFT."""
    sft = pd.read_csv(RESULTS / "v33" / "diverse" / "final.csv")
    sft["ct"] = sft["family"].astype(str).str.split(":").str[-1]
    sm = sft.groupby("ct")["compliance_score"].mean().to_dict()
    loaded = {label: load(csv) for label, csv, _, _ in MODELS}
    loaded_k0 = {label: load_k0(label, loaded[label]) for label in loaded}
    rows = []
    for ct in BEHAVIORS:
        row = {"behavior": ct}
        for label in loaded:
            s = SHORT[label]
            row[f"{s}_base"] = round(_val(loaded_k0[label], ct, 0, "prefill_only"), 2)
            row[f"{s}_instr"] = round(_val(loaded_k0[label], ct, 0, "prefill_plus_instruction"), 2)
            row[f"{s}_PO_k5"] = round(_val(loaded[label], ct, 5, "prefill_only"), 2)
            row[f"{s}_PI_k5"] = round(_val(loaded[label], ct, 5, "prefill_plus_instruction"), 2)
        row["SFT_ceiling"] = round(sm.get(ct, float("nan")), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = OUT / "prefill_compare_table_k5.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}\n")
    print("Per-behavior: base=K0/no-instr floor, instr=K0/+instr floor, PO/PI=@K5")
    print(df.to_string(index=False))
    return df


SHORT = {"qwen3-14b": "14b", "qwen3-32b": "32b", "deepseek-r1": "R1",
         "glm-5.2": "glm", "gpt-oss-120b": "oss"}


def make_table_png() -> None:
    """Heatmap table: per-behavior K=5 compliance, prefill-only (PO) & +instruction (PI)."""
    import numpy as np

    loaded = {label: load(csv) for label, csv, _, _ in MODELS}
    loaded_k0 = {label: load_k0(label, loaded[label]) for label in loaded}

    col_labels = []
    for label, _, _, _ in MODELS:
        s = SHORT[label]
        col_labels += [f"{s}\nbase", f"{s}\n+i", f"{s}\nPO", f"{s}\nPI"]
    ncols_per_model = 4

    M = np.full((len(BEHAVIORS), len(col_labels)), np.nan)
    for i, ct in enumerate(BEHAVIORS):
        j = 0
        for label, _, _, _ in MODELS:
            M[i, j] = _val(loaded_k0[label], ct, 0, "prefill_only"); j += 1           # base (K0, no instr)
            M[i, j] = _val(loaded_k0[label], ct, 0, "prefill_plus_instruction"); j += 1  # +i (K0, instr only)
            M[i, j] = _val(loaded[label], ct, 5, "prefill_only"); j += 1           # PO @ K5
            M[i, j] = _val(loaded[label], ct, 5, "prefill_plus_instruction"); j += 1  # PI @ K5

    fig, ax = plt.subplots(figsize=(21, 6.8))
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(BEHAVIORS)))
    ax.set_yticklabels(BEHAVIORS, fontsize=9)
    ax.xaxis.tick_top()
    for i in range(len(BEHAVIORS)):
        for j in range(len(col_labels)):
            v = M[i, j]
            txt = "—" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="black" if (np.isnan(v) or 0.25 < v < 0.75) else "white"
                    if not np.isnan(v) and (v <= 0.25 or v >= 0.75) else "black")
    # separators between model groups
    for b in range(ncols_per_model, len(col_labels), ncols_per_model):
        ax.axvline(b - 0.5, color="white", lw=3)
    ax.set_title("Per-behavior continuation compliance\n"
                 "base = K0 no-instr floor | +i = K0 instruction-only | PO = prefill-only @K5 | PI = prefill+instr @K5   "
                 "(oss = medium-effort, provisional)", fontsize=11, pad=28)
    # SFT column removed: v33 doesn't train all these behaviors, so it added noise.
    ax.tick_params(length=0)
    fig.tight_layout()
    out = OUT / "prefill_compare_table_k5.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    make_plot()
    make_table()
    make_table_png()
