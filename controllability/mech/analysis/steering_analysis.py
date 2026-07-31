"""Steering analysis — is the compliance direction causal (write), not just readable?

Reads data/steering/grades.csv (grade_traces.py on data/steering/traces.jsonl).
Plots compliance vs alpha per type, with the random-direction and no-instruction
controls marked, plus accuracy (steering must not wreck the answer). A causal,
direction-specific, instruction-gated effect looks like: compliance rises with
alpha for kind='v'/instruction_only, the matched-norm random direction stays flat,
and 'v' with no instruction stays at the no-instruction floor.

  python analysis/steering_analysis.py
Outputs figs/steering.png, steering_stats.csv
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_analysis import DATA, FIGS, HERE, cluster_boot_ci


def _parse_variant(v):
    import re
    m = re.match(r"(v|rand)_a(-?\d+)_(\w+)", str(v))
    return (m.group(1), int(m.group(2))) if m else (None, None)


def main():
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    fn = f"grades_{tag}.csv" if tag else "grades.csv"
    g = pd.read_csv(DATA / "steering" / fn)
    g = g[g.empty_thinking == 0].copy()
    g[["kind", "alpha"]] = g.variant.apply(lambda v: pd.Series(_parse_variant(v)))
    g = g.dropna(subset=["kind"])
    # condition column already carries instruction_only / no_instruction
    types = sorted(g.ctype.unique())
    fig, axes = plt.subplots(1, len(types), figsize=(5.2 * len(types), 4.4), squeeze=False)
    stats = []
    for ti, ct in enumerate(types):
        ax = axes[0, ti]
        sub = g[g.ctype == ct]
        # main dose-response: kind=v, instruction_only
        main = sub[(sub.kind == "v") & (sub.condition == "instruction_only")]
        xs = sorted(main.alpha.unique())
        ms, los, his = [], [], []
        for a in xs:
            s = main[main.alpha == a]
            m, lo, hi = cluster_boot_ci(s.compliance_score.values, s.qhash.values)
            ms.append(m); los.append(lo); his.append(hi)
            stats.append({"ctype": ct, "series": "v/instr", "alpha": a, "n": len(s),
                          "compliance": m, "lo": lo, "hi": hi,
                          "accuracy": float(s.correct.mean())})
        ax.errorbar(xs, ms, yerr=[np.array(ms) - los, np.array(his) - ms], fmt="o-",
                    color="#2171b5", capsize=3, lw=2, label="v (instruction)")
        # controls
        for kind, cond, color, lab in [("rand", "instruction_only", "#d94801", "random dir (instr)"),
                                       ("v", "no_instruction", "#31a354", "v (no instruction)")]:
            s = sub[(sub.kind == kind) & (sub.condition == cond)]
            if len(s):
                a = s.alpha.iloc[0]
                m, lo, hi = cluster_boot_ci(s.compliance_score.values, s.qhash.values)
                ax.errorbar([a], [m], yerr=[[m - lo], [hi - m]], fmt="s", color=color,
                            capsize=4, ms=9, label=lab)
                stats.append({"ctype": ct, "series": f"{kind}/{cond}", "alpha": a,
                              "n": len(s), "compliance": m, "lo": lo, "hi": hi,
                              "accuracy": float(s.correct.mean())})
        # accuracy on twin axis
        ax2 = ax.twinx()
        accs = [float(main[main.alpha == a].correct.mean()) for a in xs]
        ax2.plot(xs, accs, "--", color="#999999", lw=1.3, label="accuracy")
        ax2.set_ylim(0, 1.05); ax2.set_ylabel("accuracy (dashed)", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("steering coefficient alpha (x diff-in-means v)")
        if ti == 0:
            ax.set_ylabel("compliance")
        ax.set_title(ct, fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Causal steering along the per-type compliance direction (base Qwen3-32B, L40)\n"
                 "rise with alpha + flat random + floor-at-no-instruction = direction-specific, "
                 "instruction-gated causal effect", fontsize=10)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "steering.png", dpi=150, bbox_inches="tight")
    st = pd.DataFrame(stats)
    st.to_csv(HERE / "steering_stats.csv", index=False)
    print(f"wrote {FIGS/'steering.png'} + steering_stats.csv")
    print(st.round(3).to_string(index=False))

    # degeneracy guard: high-alpha steering could yield short/empty/looping thinks
    # that trivially pass the grader. Report think length + empty-rate vs alpha.
    print("\ndegeneracy check (kind=v, instruction_only):")
    d = g[(g.kind == "v") & (g.condition == "instruction_only")]
    diag = d.groupby(["ctype", "alpha"]).agg(
        n=("compliance_score", "size"),
        empty_think=("empty_thinking", "mean"),
        med_gen_tokens=("n_gen_tokens", "median"),
        trunc=("finish_reason", lambda s: (s == "length").mean())).round(2)
    print(diag.to_string())


if __name__ == "__main__":
    main()
