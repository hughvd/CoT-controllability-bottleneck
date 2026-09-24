"""Cross-model comparison for the prefill/CoT-seeding experiment.

- PLOT: mean continuation compliance vs K, averaged across all behaviors, one color
  per model. Solid line = prefill+instruction, faded (alpha) line = prefill-only.
- TABLE: per-behavior K=5 sustain (prefill-only) for every model + the SFT ceiling.

Averages weight each behavior equally (mean of per-behavior means). empty-thinking
rows are excluded. The complete clean high-effort GPT-OSS sweep is preferred when
available. Historical GPT-OSS K=0 records are invalid because the Harmony parser
discarded analysis headers; the fallback omits those points. The separate API
K=0 run is never mixed into a medium-effort/local curve.
Usage: python controllability/analysis/prefill_compare.py
"""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from paper_intervals import question_cluster_ci, draw_intervals
from paper_style import (apply_style, export_figure, PAPER_FIGURES, SQUARE_SIZE,
                         LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, CELL_SIZE, HEATMAP,
                         BEHAVIOR_LABELS, MODEL_COLORS)

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE / "prefill"
SUFFIX = ""

# (label, csv, color, note)
MODELS = [
    ("qwen3-14b",       "base_qwen3_14b_prefill_per_sample_clean.csv", MODEL_COLORS["qwen14"], ""),
    ("qwen3-32b",       "base_qwen3_32b_prefill_per_sample.csv",       MODEL_COLORS["qwen32"], ""),
    ("deepseek-r1",     "base_deepseek_r1_prefill_per_sample.csv",     MODEL_COLORS["deepseek"], ""),
    ("glm-5.2",         "base_glm_5_2_prefill_per_sample.csv",         MODEL_COLORS["glm"], ""),
    ("gpt-oss-120b",    "base_gptoss_120b_prefill_per_sample.csv",     MODEL_COLORS["gptoss"], " (medium-effort, provisional)"),
]
CLEAN_GPTOSS = RESULTS / "base_gptoss_120b_high_clean_prefill_per_sample.csv"
if CLEAN_GPTOSS.exists():
    clean = pd.read_csv(CLEAN_GPTOSS)
    expected_counts = clean.groupby(["ctype", "condition", "k"]).size()
    if len(clean) != 18000 or len(expected_counts) != 120 or not expected_counts.eq(150).all():
        raise ValueError("Clean GPT-OSS results are incomplete; refusing to plot a partial sweep")
    MODELS[-1] = ("gpt-oss-120b", CLEAN_GPTOSS.name, MODEL_COLORS["gptoss"], "")
BEHAVIORS = ["leetspeak", "spanish", "chinese", "uppercase_thinking", "prefix_sentences",
             "no_first_person", "omit_word", "pirate", "yoda", "jar_jar_binks"]

# K=0 floor override: the main-sweep K=0 was collected with a "<cot_instructions> none"
# block that some models (qwen3-32b) read as "produce no CoT", suppressing reasoning and
# contaminating the K=0 base/+i floor. These re-runs drop the block. Falls back to the
# model's main CSV when a k0fix file isn't present (e.g. gpt-oss until a pod run).
# These overrides apply to both the curve and the table. GPT-OSS's separate
# high-effort/API run changes effort and backend, so it cannot repair this curve.
K0FIX = {
    "qwen3-14b":   "base_qwen3_14b_k0fix_prefill_per_sample.csv",   # ≈ original; kept for parity
    "qwen3-32b":   "base_qwen3_32b_k0fix_prefill_per_sample.csv",   # the real fix
}


def load(csv: str) -> pd.DataFrame:
    d = pd.read_csv(RESULTS / csv)
    if "empty_thinking" in d:
        d = d[~d["empty_thinking"].astype(bool)]
    return d[d.ctype.isin(BEHAVIORS)]


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


def save_figure(fig, name: str, *, square: bool = False) -> None:
    export_figure(fig, OUT, name + SUFFIX, square=square)


def make_plot() -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=SQUARE_SIZE)
    interval_rows = []
    for label, csv, color, note in MODELS:
        d = load(csv)
        k0 = load_k0(label, d)
        d = pd.concat([d[d.k != 0], k0[k0.k == 0]], ignore_index=True)
        pi = behavior_avg_by_k(d, "prefill_plus_instruction")
        po = behavior_avg_by_k(d, "prefill_only")
        ax.plot(pi.index, pi.values, "-o", color=color, lw=2.4, label=f"{label}{'*' if note else ''}")
        ax.plot(po.index, po.values, "--o", color=color, lw=1.6, alpha=0.45, markersize=4,
                label="_nolegend_")
        for condition, series, alpha in [("prefill_plus_instruction", pi, 1.0),
                                          ("prefill_only", po, 0.45)]:
            intervals = []
            for k in series.index:
                subset = d[(d.condition == condition) & (d.k == k)]
                result = question_cluster_ci(subset, "question", BEHAVIORS, universe=d.question)
                intervals.append(result)
                interval_rows.append(dict(model=label, condition=condition, k=k, **result))
            draw_intervals(ax, series.index, intervals, color=color, alpha=alpha)
    ax.set_xlabel("K (injected controlled sentences)", fontsize=LABEL_SIZE)
    ax.set_ylabel("mean continuation compliance", fontsize=LABEL_SIZE)
    ax.set_xticks(range(6))
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=LEGEND_SIZE, ncol=2, loc="upper left", frameon=False,
              columnspacing=1.0, handlelength=1.8, labelspacing=0.3)
    fig.tight_layout(pad=0.8)
    save_figure(fig, "prefill_compare_avg", square=True)
    pd.DataFrame(interval_rows).to_csv(OUT / ("prefill_compare_avg" + SUFFIX + ".csv"), index=False)


def _val(d, ct, k, cond):
    s = d[(d.ctype == ct) & (d.condition == cond) & (d.k == k)]["compliance_score"]
    return s.mean() if len(s) else float("nan")


def _rounded(value):
    # Match the displayed table exactly, including halfway floating-point values.
    return float(f"{value:.2f}")


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
            row[f"{s}_base"] = _rounded(_val(loaded_k0[label], ct, 0, "prefill_only"))
            row[f"{s}_instr"] = _rounded(_val(loaded_k0[label], ct, 0, "prefill_plus_instruction"))
            row[f"{s}_PO_k5"] = _rounded(_val(loaded[label], ct, 5, "prefill_only"))
            row[f"{s}_PI_k5"] = _rounded(_val(loaded[label], ct, 5, "prefill_plus_instruction"))
        row["SFT_ceiling"] = _rounded(sm.get(ct, float("nan")))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = OUT / ("prefill_compare_table_k5" + SUFFIX + ".csv")
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
    apply_style()

    loaded = {label: load(csv) for label, csv, _, _ in MODELS}
    loaded_k0 = {label: load_k0(label, loaded[label]) for label in loaded}

    col_labels = []
    for label, _, _, _ in MODELS:
        col_labels += ["base", "+i", "PO", "PI"]
    ncols_per_model = 4

    M = np.full((len(BEHAVIORS), len(col_labels)), np.nan)
    for i, ct in enumerate(BEHAVIORS):
        j = 0
        for label, _, _, _ in MODELS:
            M[i, j] = _val(loaded_k0[label], ct, 0, "prefill_only"); j += 1           # base (K0, no instr)
            M[i, j] = _val(loaded_k0[label], ct, 0, "prefill_plus_instruction"); j += 1  # +i (K0, instr only)
            M[i, j] = _val(loaded[label], ct, 5, "prefill_only"); j += 1           # PO @ K5
            M[i, j] = _val(loaded[label], ct, 5, "prefill_plus_instruction"); j += 1  # PI @ K5

    fig, ax = plt.subplots(figsize=(13, 1.5 + .4 * len(BEHAVIORS)))
    ax.imshow(M, cmap=HEATMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=TICK_SIZE)
    ax.set_yticks(range(len(BEHAVIORS)))
    ax.set_yticklabels([BEHAVIOR_LABELS[ct] for ct in BEHAVIORS], fontsize=TICK_SIZE)
    ax.xaxis.tick_top()
    for i in range(len(BEHAVIORS)):
        for j in range(len(col_labels)):
            v = M[i, j]
            txt = "—" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=CELL_SIZE,
                    color="black" if (np.isnan(v) or 0.25 < v < 0.75) else "white"
                    if not np.isnan(v) and (v <= 0.25 or v >= 0.75) else "black")
    # separators between model groups
    for b in range(ncols_per_model, len(col_labels), ncols_per_model):
        ax.axvline(b - 0.5, color="white", lw=3)
    for i, (label, _, _, _) in enumerate(MODELS):
        ax.text(i * ncols_per_model + 1.5, 1.09, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=LABEL_SIZE)
    # SFT column removed: v33 doesn't train all these behaviors, so it added noise.
    ax.tick_params(length=0)
    fig.tight_layout(pad=0.8)
    save_figure(fig, "prefill_compare_table_k5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-styles", action="store_true")
    args = parser.parse_args()
    if args.exclude_styles:
        BEHAVIORS = [b for b in BEHAVIORS if b not in {"pirate", "yoda", "jar_jar_binks"}]
        SUFFIX = "_no_styles"
        OUT = OUT / "no_styles"
    OUT.mkdir(parents=True, exist_ok=True)
    make_plot()
    make_table()
    make_table_png()
