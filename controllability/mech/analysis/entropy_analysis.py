"""Analysis 1 — entropy: do compliant traces 'lock in' early?

Per trace: entropy of the first CoT content token, mean entropy of the first sentence,
and the entropy profile over normalized CoT position. Compared compliant vs
non-compliant per (type, condition), with no_instruction as reference.

  python analysis/entropy_analysis.py
Outputs figs/entropy_first.png, figs/entropy_profiles.png, entropy_stats.csv
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_analysis import (CLS_COLORS, COND_LABEL, CONDS, FIGS, HERE, TYPES,
                             binned_profile, cluster_boot_ci, content_start,
                             first_sentence_end, load_entropy, load_grades,
                             load_traces, pieces, think_end)

N_BINS = 20


def build_table() -> pd.DataFrame:
    g = load_grades()
    traces = load_traces()
    rows = []
    for _, r in g.iterrows():
        ent = load_entropy(r.trace_id)
        if ent is None:
            continue
        H = ent["entropy"].astype(np.float32)
        pcs = pieces(traces[r.trace_id]["output_token_ids"])
        cs = content_start(pcs)
        te = think_end(pcs)
        if te - cs < 3:
            continue
        s1 = first_sentence_end(pcs, cs)
        cot = H[cs:te]
        rows.append({
            "trace_id": r.trace_id, "ctype": r.ctype, "condition": r.condition,
            "qhash": r.qhash, "cls": r.cls,
            "H_first": float(H[cs]),
            "H_sent1": float(H[cs:s1 + 1].mean()),
            "H_cot": float(cot.mean()),
            "profile": binned_profile(cot, N_BINS),
        })
    return pd.DataFrame(rows)


def fig_first(df: pd.DataFrame) -> pd.DataFrame:
    """Dot+CI: first-token / first-sentence entropy by type x condition x class."""
    types = ["no_first_person", "omit_word", "jar_jar_binks", "spanish"]
    metrics = [("H_first", "first-token entropy (nats)"),
               ("H_sent1", "first-sentence mean entropy (nats)")]
    fig, axes = plt.subplots(len(types), 2, figsize=(11, 3.0 * len(types)),
                             sharex=True)
    stats = []
    for ti, ct in enumerate(types):
        for mi, (metric, mlabel) in enumerate(metrics):
            ax = axes[ti, mi]
            xs = np.arange(len(CONDS))
            for ci, cond in enumerate(CONDS):
                sub = df[(df.ctype == ct) & (df.condition == cond)]
                for k, cls in enumerate(["compliant", "noncompliant"]):
                    s = sub[sub.cls == cls]
                    if len(s) < 5:
                        continue
                    m, lo, hi = cluster_boot_ci(s[metric].values, s.qhash.values)
                    off = -0.15 + 0.3 * k
                    ax.errorbar(ci + off, m, yerr=[[m - lo], [hi - m]], fmt="none",
                                ecolor=CLS_COLORS[cls], capsize=3, zorder=2)
                    ax.scatter(ci + off, m, s=10 + 0.6 * len(s), color=CLS_COLORS[cls],
                               zorder=3, edgecolors="white", linewidths=0.5,
                               label=cls if (ti == 0 and mi == 0 and ci == 1) else None)
                    stats.append({"ctype": ct, "condition": cond, "cls": cls,
                                  "metric": metric, "n": len(s), "mean": m,
                                  "lo": lo, "hi": hi})
            ax.set_xticks(xs)
            ax.set_xticklabels([COND_LABEL[c] for c in CONDS], fontsize=8)
            ax.grid(alpha=0.3, axis="y")
            if mi == 0:
                ax.set_ylabel(ct, fontsize=9)
            if ti == 0:
                ax.set_title(mlabel, fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Start-of-CoT entropy: compliant vs non-compliant "
                 "(question-clustered 95% CI; groups with n<5 omitted; "
                 "dot area ∝ n traces, range 5–300)", fontsize=11)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "entropy_first.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIGS/'entropy_first.png'}")
    return pd.DataFrame(stats)


def fig_profiles(df: pd.DataFrame) -> None:
    """Entropy vs normalized CoT position, per type; V2 + instr-only panels."""
    types = ["no_first_person", "omit_word", "jar_jar_binks", "spanish",
             "uppercase_thinking", "leetspeak"]
    conds = ["instruction_only", "icl_user_think"]
    x = (np.arange(N_BINS) + 0.5) / N_BINS
    fig, axes = plt.subplots(len(types), len(conds),
                             figsize=(10, 2.4 * len(types)), sharex=True, sharey=True)
    for ti, ct in enumerate(types):
        # no_instruction reference (all traces)
        ref = df[(df.ctype == ct) & (df.condition == "no_instruction")]
        refp = np.nanmean(np.stack(ref.profile.values), 0) if len(ref) else None
        for ci, cond in enumerate(conds):
            ax = axes[ti, ci]
            sub = df[(df.ctype == ct) & (df.condition == cond)]
            groups = ([("compliant", "compliant"), ("noncompliant", "noncompliant")]
                      if sub.cls.notna().any() else [(None, "all")])
            for cls, lab in groups:
                s = sub if cls is None else sub[sub.cls == cls]
                if len(s) < 5:
                    continue
                prof = np.stack(s.profile.values)
                m = np.nanmean(prof, 0)
                ax.plot(x, m, color=CLS_COLORS[lab if cls else "all"], lw=1.8,
                        label=f"{lab} (n={len(s)})")
                se = np.nanstd(prof, 0) / np.sqrt(np.sum(~np.isnan(prof), 0).clip(1))
                ax.fill_between(x, m - 1.96 * se, m + 1.96 * se,
                                color=CLS_COLORS[lab if cls else "all"], alpha=0.15)
            if refp is not None:
                ax.plot(x, refp, color="#888888", lw=1.2, ls="--", label="no-instr (all)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6.5)
            if ci == 0:
                ax.set_ylabel(f"{ct}\nentropy (nats)", fontsize=8)
            if ti == 0:
                ax.set_title(COND_LABEL[cond], fontsize=10)
            if ti == len(types) - 1:
                ax.set_xlabel("normalized CoT position")
    fig.suptitle("Entropy over the CoT: compliant vs non-compliant vs no-instruction",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "entropy_profiles.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIGS/'entropy_profiles.png'}")


def main() -> None:
    df = build_table()
    print(f"entropy table: {len(df)} traces")
    stats = fig_first(df)
    stats.to_csv(HERE / "entropy_stats.csv", index=False)
    fig_profiles(df)
    # paired within-question deltas for the two binary types
    for ct in ["no_first_person", "omit_word"]:
        for cond in ["instruction_only", "icl_user_think"]:
            sub = df[(df.ctype == ct) & (df.condition == cond)]
            deltas = []
            for q, qs in sub.groupby("qhash"):
                c = qs[qs.cls == "compliant"].H_sent1
                n = qs[qs.cls == "noncompliant"].H_sent1
                if len(c) and len(n):
                    deltas.append(c.mean() - n.mean())
            if deltas:
                d = np.array(deltas)
                print(f"paired ΔH_sent1 (compliant-non) {ct}/{COND_LABEL[cond]}: "
                      f"{d.mean():+.3f} nats over {len(d)} questions "
                      f"(sign: {np.mean(d<0)*100:.0f}% negative)")


if __name__ == "__main__":
    main()
