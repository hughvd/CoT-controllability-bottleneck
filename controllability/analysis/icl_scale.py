"""Analysis for the ICL example-LENGTH scaling sweep (--scale run).

Does showing MORE of the controlled example (more sentences of reasoning) induce more
control? Sweeps example length ∈ {1, 3, 5, 10, 50%, 100%} sentences for V1/V2/V4.

Headline: macro-avg compliance (across the 10 control types) vs example length, one line
per variant, one panel per model. Horizontal refs: instruction_only baseline (dashed) and
prefill K5+instr ceiling (dotted). Reads results/iclscale_{model}_icl_per_sample.csv.

  python analysis/icl_scale.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE / "icl"

MODELS = [
    ("qwen3-32b", "iclscale_qwen3_32b_icl_per_sample.csv", "base_qwen3_32b_prefill_per_sample.csv"),
    ("glm-5.2",   "iclscale_glm_5_2_icl_per_sample.csv",   "base_glm_5_2_prefill_per_sample.csv"),
]
TYPES = ["leetspeak", "spanish", "chinese", "uppercase_thinking", "prefix_sentences",
         "no_first_person", "omit_word", "pirate", "yoda", "jar_jar_binks"]
LEN_ORDER = ["1", "3", "5", "10", "50pct", "100pct"]
LEN_LABEL = {"1": "1", "3": "3", "5": "5", "10": "10", "50pct": "50%", "100pct": "full"}
VARIANTS = [("icl_user_plain", "V1 user", "#1f78b4", "o"),
            ("icl_user_think", "V2 user+tag", "#6a3d9a", "s"),
            ("icl_assistant_visible", "V4 asst-vis", "#33a02c", "^")]


def _load(csv: str):
    p = RESULTS / csv
    if not p.exists():
        return None
    d = pd.read_csv(p)
    if "empty_thinking" in d:
        d = d[~d["empty_thinking"].astype(bool)]
    return d


def _macro(d, variant, length) -> float:
    """Mean across the 10 types of the per-type mean compliance (equal weight per type)."""
    sub = d[(d.variant == variant) & (d.length.astype(str) == length)]
    per = [sub[sub.ctype == ct]["compliance_score"].mean() for ct in TYPES]
    per = [v for v in per if not np.isnan(v)]
    return float(np.mean(per)) if per else float("nan")


def _macro_baseline(d) -> float:
    sub = d[d.variant == "instruction_only"]
    per = [sub[sub.ctype == ct]["compliance_score"].mean() for ct in TYPES]
    per = [v for v in per if not np.isnan(v)]
    return float(np.mean(per)) if per else float("nan")


def _macro_prefill(csv: str) -> float:
    p = RESULTS / csv
    if not p.exists():
        return float("nan")
    d = pd.read_csv(p)
    if "empty_thinking" in d:
        d = d[~d["empty_thinking"].astype(bool)]
    d = d[(d.k == 5) & (d.condition == "prefill_plus_instruction")]
    per = [d[d.ctype == ct]["compliance_score"].mean() for ct in TYPES]
    per = [v for v in per if not np.isnan(v)]
    return float(np.mean(per)) if per else float("nan")


def main() -> None:
    loaded = [(s, _load(icl), pf) for s, icl, pf in MODELS]
    loaded = [(s, d, pf) for s, d, pf in loaded if d is not None]
    if not loaded:
        print("no scale CSVs yet (expected results/iclscale_*.csv)")
        return

    x = np.arange(len(LEN_ORDER))
    fig, axes = plt.subplots(1, len(loaded), figsize=(6.4 * len(loaded), 5.2), sharey=True)
    if len(loaded) == 1:
        axes = [axes]

    for ax, (short, d, pf) in zip(axes, loaded):
        for key, label, color, marker in VARIANTS:
            ys = [_macro(d, key, L) for L in LEN_ORDER]
            ax.plot(x, ys, marker=marker, color=color, label=label, lw=2)
            for xi, y in zip(x, ys):
                if not np.isnan(y):
                    ax.annotate(f"{y:.2f}", (xi, y), textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=7, color=color)
        base = _macro_baseline(d)
        pref = _macro_prefill(pf)
        if not np.isnan(base):
            ax.axhline(base, ls="--", color="#888", lw=1.3, label=f"instr-only ({base:.2f})")
        if not np.isnan(pref):
            ax.axhline(pref, ls=":", color="#333", lw=1.3, label=f"prefill K5+i ({pref:.2f})")
        ax.set_title(short, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([LEN_LABEL[L] for L in LEN_ORDER])
        ax.set_xlabel("example length (sentences of donor CoT shown)")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("avg compliance across all 10 types")
    fig.suptitle("ICL example-length scaling: does showing more of the controlled example induce more control?\n"
                 "(macro-avg over 10 types; grades native <think>)", fontsize=11)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "icl_scale.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}\n")

    for short, d, pf in loaded:
        print(f"\n### {short}  (rows={len(d)})  baseline={_macro_baseline(d):.2f}  prefillK5={_macro_prefill(pf):.2f}")
        rows = []
        for key, label, _, _ in VARIANTS:
            rows.append({"variant": label, **{LEN_LABEL[L]: round(_macro(d, key, L), 2) for L in LEN_ORDER}})
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
