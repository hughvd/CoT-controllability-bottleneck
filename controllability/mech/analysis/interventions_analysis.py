"""Opening-interventions analysis — does knocking the model off its default first
tokens recover compliance (propensity gap) or not (capability gap)?

Compares data/interventions/grades.csv against the unmodified baseline (the same
(ctype, condition) cells in data/grades.csv). no_instruction cells are the null:
interventions there should NOT create compliance.

  python analysis/interventions_analysis.py
Outputs figs/interventions.png (soft compliance_score), figs/interventions_hard.png
(hard binary compliance @ threshold 0.5), interventions_stats.csv (both metrics).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_analysis import DATA, FIGS, HERE, TYPES, cluster_boot_ci

CONDS = ["no_instruction", "instruction_only", "icl_user_think"]
COND_LABEL = {"no_instruction": "no-instr", "instruction_only": "instr-only",
              "icl_user_think": "V2 (ICL)"}
VARIANTS = ["baseline", "ban1_t0", "ban1_t0_3", "hot16", "greedy16"]
VCOLORS = {"baseline": "#777777", "ban1_t0": "#2171b5", "ban1_t0_3": "#08519c",
           "hot16": "#d94801", "greedy16": "#31a354"}


def load() -> pd.DataFrame:
    iv = pd.read_csv(DATA / "interventions" / "grades.csv")
    iv = iv[iv.empty_thinking == 0]
    base = pd.read_csv(DATA / "grades.csv")
    base = base[(base.empty_thinking == 0) & base.condition.isin(CONDS)].copy()
    base["variant"] = "baseline"
    cols = ["ctype", "condition", "variant", "qhash", "compliance_score",
            "compliance", "correct"]
    return pd.concat([iv[cols], base[cols]], ignore_index=True)


def make_fig(df, metric, mlabel, outname):
    fig, axes = plt.subplots(len(TYPES), len(CONDS),
                             figsize=(3.6 * len(CONDS), 2.4 * len(TYPES)),
                             sharey=True, squeeze=False)
    stats = []
    for ti, ct in enumerate(TYPES):
        for ci, cond in enumerate(CONDS):
            ax = axes[ti, ci]
            sub = df[(df.ctype == ct) & (df.condition == cond)]
            for vi, v in enumerate(VARIANTS):
                s = sub[sub.variant == v]
                if not len(s):
                    continue
                m, lo, hi = cluster_boot_ci(s[metric].values, s.qhash.values)
                ax.bar(vi, m, yerr=[[m - lo], [hi - m]], color=VCOLORS[v],
                       capsize=3, width=0.8)
                stats.append({"metric": metric, "ctype": ct, "condition": cond,
                              "variant": v, "n": len(s), "value": m, "lo": lo, "hi": hi,
                              "accuracy": float(s.correct.mean())})
            ax.set_xticks(range(len(VARIANTS)))
            ax.set_xticklabels(VARIANTS, fontsize=6.5, rotation=35, ha="right")
            ax.grid(alpha=0.3, axis="y")
            ax.set_ylim(0, 1)
            if ci == 0:
                ax.set_ylabel(f"{ct}\n{mlabel}", fontsize=8)
            if ti == 0:
                ax.set_title(COND_LABEL[cond], fontsize=10)
    fig.suptitle(f"Opening-only decoding interventions vs baseline — {mlabel}\n"
                 "bars: baseline / ban1_t0 / ban1_t0_3 / hot16 / greedy16 "
                 "(first content tokens only; question-clustered 95% CI)", fontsize=10)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / outname, dpi=150, bbox_inches="tight")
    print(f"wrote {FIGS/outname}")
    return stats


def main() -> None:
    df = load()
    s1 = make_fig(df, "compliance_score", "soft compliance", "interventions.png")
    s2 = make_fig(df, "compliance", "hard compliance", "interventions_hard.png")
    st = pd.DataFrame(s1 + s2)
    st.to_csv(HERE / "interventions_stats.csv", index=False)
    print("\nhard compliance (instruction_only), baseline vs ban1_t0_3:")
    h = st[(st.metric == "compliance") & (st.condition == "instruction_only")
           & st.variant.isin(["baseline", "ban1_t0_3"])]
    print(h.pivot_table(index="ctype", columns="variant", values="value").round(3).to_string())


if __name__ == "__main__":
    main()
