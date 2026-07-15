"""Analysis for the in-context-learning (ICL) controllability sweep.

Per-model heatmap: rows = control types, columns = conditions
{instr-only, V1 user-plain, V2 user-think, V3 asst-think, V4 asst-visible} + a prefill
K5 prefill+instruction reference column. Reads results/iclfull_{model}_icl_per_sample.csv.
Grades the model's own native <think> (compliance_score, soft). V3 (asst-think) exists only
for think-preserving models (GLM); shown as "—" elsewhere.

  python analysis/icl_compare.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE / "icl"

# (model short, icl csv, prefill csv for K5-PI reference)
MODELS = [
    ("qwen3-14b",   "iclfull_qwen3_14b_icl_per_sample.csv",   "base_qwen3_14b_prefill_per_sample_clean.csv"),
    ("qwen3-32b",   "iclfull_qwen3_32b_icl_per_sample.csv",   "base_qwen3_32b_prefill_per_sample.csv"),
    ("glm-5.2",     "iclfull_glm_5_2_icl_per_sample.csv",     "base_glm_5_2_prefill_per_sample.csv"),
]
TYPES = ["leetspeak", "spanish", "chinese", "uppercase_thinking", "prefix_sentences",
         "no_first_person", "omit_word", "pirate", "yoda", "jar_jar_binks"]
VARIANTS = ["instruction_only", "icl_user_plain", "icl_user_think",
            "icl_assistant_think", "icl_assistant_visible"]
VLABEL = {"instruction_only": "instr", "icl_user_plain": "V1\nuser", "icl_user_think": "V2\nuser+tag",
          "icl_assistant_think": "V3\nasst-think", "icl_assistant_visible": "V4\nasst-vis"}


def _load(csv: str):
    p = RESULTS / csv
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if "empty_thinking" in d:
        d = d[~d["empty_thinking"].astype(bool)]
    return d


def _val(d, ct, variant) -> float:
    s = d[(d.ctype == ct) & (d.variant == variant)]["compliance_score"]
    return s.mean() if len(s) else float("nan")


def _prefill_k5(csv: str, ct: str) -> float:
    p = RESULTS / csv
    if not p.exists():
        return float("nan")
    d = pd.read_csv(p)
    if "empty_thinking" in d:
        d = d[~d["empty_thinking"].astype(bool)]
    s = d[(d.ctype == ct) & (d.k == 5) & (d.condition == "prefill_plus_instruction")]["compliance_score"]
    return s.mean() if len(s) else float("nan")


# version column order + colors for the zoom-out plot (instr baseline -> ICL variants -> prefill ref)
AVG_COLS = [("instruction_only", "instr-only", "#9e9e9e"),
            ("icl_user_plain", "V1 user", "#a6cee3"),
            ("icl_user_think", "V2 user+tag", "#1f78b4"),
            ("icl_assistant_think", "V3 asst-think", "#b2df8a"),
            ("icl_assistant_visible", "V4 asst-vis", "#33a02c"),
            ("prefill_k5", "prefill K5+i", "#333333")]


def _macro_avg(d, pf, key) -> float:
    """Mean across the 10 types (macro-average, equal weight per type)."""
    if key == "prefill_k5":
        vals = [_prefill_k5(pf, ct) for ct in TYPES]
    else:
        vals = [_val(d, ct, key) for ct in TYPES]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def make_avg_plot(loaded) -> None:
    """Zoom-out: avg controllability across all 10 types, grouped by model, bar per version."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    n = len(AVG_COLS)
    x = np.arange(len(loaded))
    w = 0.8 / n
    for j, (key, label, color) in enumerate(AVG_COLS):
        vals = [_macro_avg(d, pf, key) for _, d, pf in loaded]
        xs = x + (j - (n - 1) / 2) * w
        bars = ax.bar(xs, [0 if np.isnan(v) else v for v in vals], w, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        for xi, v in zip(xs, vals):
            if np.isnan(v):
                ax.text(xi, 0.01, "n/a", ha="center", va="bottom", fontsize=7, color="#999", rotation=90)
            else:
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([s for s, _, _ in loaded], fontsize=11)
    ax.set_ylabel("avg compliance across all 10 control types")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=6, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.10), frameon=False)
    ax.set_title("ICL controllability, averaged over all types (grades native <think>)\n"
                 "V3 (asst-think) only runs on think-preserving GLM", fontsize=11, pad=28)
    fig.tight_layout()
    out = OUT / "icl_avg.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print("\n### macro-average across 10 types")
    rows = [{"model": s, **{lab: round(_macro_avg(d, pf, k), 2) for k, lab, _ in AVG_COLS}}
            for s, d, pf in loaded]
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    loaded = [(s, _load(icl), pf) for s, icl, pf in MODELS]
    loaded = [(s, d, pf) for s, d, pf in loaded if d is not None]
    if not loaded:
        print("no ICL result CSVs found yet (expected results/iclfull_*.csv)")
        return
    make_avg_plot(loaded)

    cols = [VLABEL[v] for v in VARIANTS] + ["prefill\nK5+i"]
    fig, axes = plt.subplots(1, len(loaded), figsize=(4.6 * len(loaded), 6.4), sharey=True)
    if len(loaded) == 1:
        axes = [axes]

    for ax, (short, d, pf) in zip(axes, loaded):
        M = np.full((len(TYPES), len(cols)), np.nan)
        for i, ct in enumerate(TYPES):
            for j, v in enumerate(VARIANTS):
                M[i, j] = _val(d, ct, v)
            M[i, -1] = _prefill_k5(pf, ct)
        ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=8)
        ax.set_yticks(range(len(TYPES)))
        ax.set_yticklabels(TYPES, fontsize=9)
        ax.xaxis.tick_top()
        ax.axvline(len(VARIANTS) - 0.5, color="white", lw=4)  # separate prefill ref
        for i in range(len(TYPES)):
            for j in range(len(cols)):
                v = M[i, j]
                ax.text(j, i, "—" if np.isnan(v) else f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if (np.isnan(v) or 0.25 < v < 0.75) else "white")
        ax.set_title(short, fontsize=12, pad=26)
        ax.tick_params(length=0)
    fig.suptitle("In-context-learning controllability (grades native <think>, soft compliance)\n"
                 "V1/V2 = example in user prompt | V3/V4 = example as prior assistant turn "
                 "(V3 think-tagged, GLM only) | last col = prefill K5+instr reference",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "icl_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}\n")

    for short, d, pf in loaded:
        print(f"\n### {short}  (rows={len(d)})")
        rows = []
        for ct in TYPES:
            row = {"type": ct}
            for v in VARIANTS:
                row[VLABEL[v].replace("\n", " ")] = round(_val(d, ct, v), 2)
            row["prefK5+i"] = round(_prefill_k5(pf, ct), 2)
            rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
