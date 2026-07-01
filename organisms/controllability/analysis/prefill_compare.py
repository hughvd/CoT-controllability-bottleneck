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
    ("gpt-oss-120b",    "base_gptoss_120b_prefill_per_sample.csv",     "#ff7f0e", " (medium-effort, provisional)"),
]
BEHAVIORS = ["leetspeak", "spanish", "chinese", "uppercase_thinking", "prefix_sentences",
             "no_first_person", "omit_word", "pirate", "yoda", "jar_jar_binks"]


def load(csv: str) -> pd.DataFrame:
    d = pd.read_csv(RESULTS / csv)
    if "empty_thinking" in d:
        d = d[~d["empty_thinking"].astype(bool)]
    return d


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


def make_table() -> pd.DataFrame:
    sft = pd.read_csv(RESULTS / "v33" / "diverse" / "final.csv")
    sft["ct"] = sft["family"].astype(str).str.split(":").str[-1]
    sm = sft.groupby("ct")["compliance_score"].mean().to_dict()

    def k5(d, ct):
        s = d[(d.ctype == ct) & (d.condition == "prefill_only") & (d.k == 5)]["compliance_score"]
        return round(s.mean(), 2) if len(s) else float("nan")

    loaded = {label: load(csv) for label, csv, _, _ in MODELS}
    rows = []
    for ct in BEHAVIORS:
        row = {"behavior": ct}
        for label in loaded:
            row[label] = k5(loaded[label], ct)
        row["SFT_ceiling"] = round(sm.get(ct, float("nan")), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = OUT / "prefill_compare_table_k5.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}\n")
    print("Per-behavior K=5 sustain (prefill-only):")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    make_plot()
    make_table()
