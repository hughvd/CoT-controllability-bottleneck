"""Paper ICL comparison: instruction-only, user-ICL, and asst-ICL.

The assistant-hidden-thinking variant is excluded from every output. All scores
are unthresholded compliance, with equal behavior weights in the macro-average.
ICL grades native reasoning and excludes empty reasoning.
--main selects the four main models and excludes styles.
"""
import argparse
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from paper_intervals import question_cluster_ci, draw_intervals

from paper_style import (apply_style, export_figure, SQUARE_SIZE, LABEL_SIZE,
                         TICK_SIZE, LEGEND_SIZE, CELL_SIZE, HEATMAP, BEHAVIOR_LABELS)

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE / "icl"
MODELS = [
    ("qwen3-14b", "iclfull_qwen3_14b_donorfix_v1_icl_per_sample.csv"),
    ("qwen3-32b", "iclfull_qwen3_32b_donorfix_v1_icl_per_sample.csv"),
    ("glm-5.2", "iclfull_glm_5_2_donorfix_v1_icl_per_sample.csv"),
    ("deepseek-r1", "iclfull_deepseek_r1_donorfix_v1_icl_per_sample.csv"),
    ("gpt-oss-120b", "iclfull_gptoss_120b_high_donorfix_v1_icl_per_sample.csv"),
    ("olmo-3.1-32b", "iclfull_olmo_3_1_32b_donorfix_v1_icl_per_sample.csv"),
]
MAIN_MODELS = {"qwen3-14b", "qwen3-32b", "glm-5.2", "olmo-3.1-32b"}
TYPES = ["leetspeak", "spanish", "chinese", "uppercase_thinking", "prefix_sentences",
         "no_first_person", "omit_word", "pirate", "yoda", "jar_jar_binks"]
STYLE_TYPES = {"pirate", "yoda", "jar_jar_binks"}
FOCUS_TYPES = ["uppercase_thinking", "leetspeak", "omit_word"]
VARIANTS = ["instruction_only", "icl_user_plain", "icl_assistant_visible"]
VLABEL = {"instruction_only": "instr-only", "icl_user_plain": "user-ICL",
          "icl_assistant_visible": "asst-ICL"}
AVG_COLS = [("instruction_only", "instr-only", "#9e9e9e"),
            ("icl_user_plain", "user-ICL", "#a6cee3"),
            ("icl_assistant_visible", "asst-ICL", "#33a02c")]


@lru_cache(maxsize=None)
def _load(csv):
    if csv is None:
        return None
    path = RESULTS / csv
    if not path.exists():
        return None
    d = pd.read_csv(path)
    if "variant" in d:
        counts = d.groupby(["ctype", "variant"]).size()
        if len(d) != 6000 or len(counts) != 40 or not counts.eq(150).all():
            raise ValueError(f"Incomplete corrected ICL sweep: {path}")
    if "empty_thinking" in d:
        d = d[~d.empty_thinking.astype(bool)]
    if "variant" in d:
        d = d[d.variant.isin(VARIANTS)]
    return d


def _val(d, ct, variant):
    return d[(d.ctype == ct) & (d.variant == variant)].compliance_score.mean()


def _macro_avg(d, key, types=None):
    types = TYPES if types is None else types
    vals = [_val(d, ct, key) for ct in types]
    if np.isnan(vals).any():
        raise ValueError(f"Missing behavior in macro-average: {key}")
    return float(np.mean(vals))


def make_avg_plot(loaded, types=None, name="icl_avg"):
    types = TYPES if types is None else types
    fig, ax = plt.subplots(figsize=(SQUARE_SIZE[0], 4.2))
    x = np.arange(len(loaded))
    width = 0.8 / len(AVG_COLS)
    rows = []
    for j, (key, label, color) in enumerate(AVG_COLS):
        vals = [_macro_avg(d, key, types) for _, d in loaded]
        xs = x + (j - (len(AVG_COLS) - 1) / 2) * width
        ax.bar(xs, vals, width,
               label=label, color=color, edgecolor="white", linewidth=0.5)
        intervals = []
        for model, d in loaded:
            d = d[d.ctype.isin(types)]
            universe = set(d.target_question)
            subset = d[d.variant == key]
            result = question_cluster_ci(subset, "target_question", types, universe=universe)
            intervals.append(result)
            rows.append(dict(model=model, condition=label, **result))
        draw_intervals(ax, xs, intervals, color="#222222", cap=.025)
    ax.set_xticks(x, [s for s, _ in loaded])
    if len(loaded) > 3:
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("mean continuation compliance")
    ax.set_ylim(0, 0.8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=1, fontsize=LEGEND_SIZE, loc="upper left",
              columnspacing=1, handlelength=1.8, labelspacing=0.3)
    fig.tight_layout(pad=0.8)
    export_figure(fig, OUT, name, square=True)
    pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)


def make_comparison(loaded, types, name):
    data_labels = [VLABEL[v] for v in VARIANTS]
    cols = ["instr-\nonly", "user-ICL", "asst-ICL"]
    # Wide tables are full-width paper panels.
    height = 5.5 if len(types) > 3 else 2.7
    fig, axes = plt.subplots(1, len(loaded), figsize=(13, height), sharey=True,
                             squeeze=False)
    rows = []
    for ax, (model, d) in zip(axes[0], loaded):
        matrix = np.array([[_val(d, ct, v) for v in VARIANTS] for ct in types])
        ax.imshow(matrix, cmap=HEATMAP, vmin=0, vmax=1, aspect="auto")
        header_size = min(TICK_SIZE, 8) if len(loaded) >= 6 else TICK_SIZE
        ax.set_xticks(range(len(cols)), cols, fontsize=header_size)
        ax.set_yticks(range(len(types)), [BEHAVIOR_LABELS[ct] for ct in types], fontsize=TICK_SIZE)
        ax.xaxis.tick_top()
        for i, ct in enumerate(types):
            for j, label in enumerate(cols):
                value = matrix[i, j]
                ax.text(j, i, "—" if np.isnan(value) else f"{value:.2f}",
                        ha="center", va="center", fontsize=CELL_SIZE,
                        color="black" if np.isnan(value) or .25 < value < .75 else "white")
                rows.append(dict(model=model, behavior=ct, condition=data_labels[j],
                                 mean_compliance=value))
        if len(types) == 3:
            ax.axhline(1.5, color="white", lw=3)
        ax.set_title(model, fontsize=LABEL_SIZE, pad=38)
        ax.tick_params(length=0)
    fig.tight_layout(pad=0.8, w_pad=0.8)
    export_figure(fig, OUT, name)
    pd.DataFrame(rows).to_csv(OUT / f"{name}.csv", index=False)


def main():
    global CELL_SIZE, TICK_SIZE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-styles", action="store_true",
                        help="Exclude Pirate, Yoda, and Jar Jar; write separate *_no_styles figures")
    parser.add_argument("--main", action="store_true",
                        help="Four main models, no styles; append _main")
    args = parser.parse_args()
    if args.main:
        args.exclude_styles = True
    types = [ct for ct in TYPES if ct not in STYLE_TYPES] if args.exclude_styles else TYPES
    suffix = "_no_styles" if args.exclude_styles else ""
    if args.main:
        suffix += "_main"
    models = [(m, icl) for m, icl in MODELS if m in MAIN_MODELS] if args.main else MODELS
    apply_style()
    loaded = [(model, _load(icl)) for model, icl in models]
    if args.main and any(d is None for _, d in loaded):
        raise FileNotFoundError("A requested main-model ICL sweep is missing")
    loaded = [(model, d) for model, d in loaded if d is not None]
    if not loaded:
        raise FileNotFoundError("No ICL result CSVs found")
    if len(loaded) > 3:
        CELL_SIZE = 9 if len(loaded) >= 6 else 10
        TICK_SIZE = 10
    OUT.mkdir(parents=True, exist_ok=True)
    make_avg_plot(loaded, types, "icl_avg" + suffix)
    make_comparison(loaded, types, "icl_compare" + suffix)
    make_comparison(loaded, FOCUS_TYPES, "icl_prefill_focus" + suffix)


if __name__ == "__main__":
    main()
