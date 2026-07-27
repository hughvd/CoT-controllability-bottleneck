"""Analysis 2 — omit_word: P(banned) over the trace; suppression vs sampling luck.

Per omit_word trace: p_banned trajectory over the CoT, per-trace max/mean, spike rates
(p_banned > 0.1 / 0.3 per 100 tokens), near-miss rate (banned id in top-5 but not
sampled), and for violating traces the normalized position of the first banned-word
emission (grader-consistent regex on decoded text).

Expected under the two stories:
  suppression -> compliant-with-instruction traces show FEWER/LOWER spikes than
                 no_instruction traces (the context pushes the word's probability down)
  luck        -> compliant traces spike like no_instruction ones; they just didn't
                 sample the word (high near-miss rate, similar max p_banned)

  python analysis/pbanned_analysis.py
Outputs figs/pbanned_profiles.png, figs/pbanned_dist.png, pbanned_stats.csv
"""

from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_analysis import (COND_LABEL, CONDS, FIGS, HERE, binned_profile,
                             cluster_boot_ci, content_start, load_entropy,
                             load_grades, load_prompts, load_traces, pieces,
                             think_end)

N_BINS = 20
GROUPS = [("no_instruction", None, "no-instr (all)", "#555555"),
          ("instruction_only", "compliant", "instr-only compliant", "#1a9641"),
          ("instruction_only", "noncompliant", "instr-only violating", "#d7191c"),
          ("icl_user_think", "compliant", "V2 compliant", "#66bd63"),
          ("icl_user_think", "noncompliant", "V2 violating", "#f46d43")]


def build_table() -> pd.DataFrame:
    g = load_grades()
    g = g[g.ctype == "omit_word"]
    prompts = load_prompts()
    traces = load_traces()
    rows = []
    for _, r in g.iterrows():
        ent = load_entropy(r.trace_id)
        if ent is None or "p_banned" not in ent:
            continue
        p = prompts[r.prompt_id]
        banned_ids = set(p["banned_token_ids"])
        words = p["grade_omit_words"] or []
        out_ids = traces[r.trace_id]["output_token_ids"]
        pcs = pieces(out_ids)
        cs, te = content_start(pcs), think_end(pcs)
        if te - cs < 3:
            continue
        pb = ent["p_banned"].astype(np.float32)[cs:te]
        top5 = ent["top5_ids"][cs:te]
        sampled = np.array(out_ids[cs:te])
        in_top5 = np.isin(top5, list(banned_ids)).any(1)
        sampled_banned = np.isin(sampled, list(banned_ids))
        near_miss = in_top5 & ~sampled_banned
        # grader-consistent first-violation position (regex over decoded CoT text)
        first_viol = np.nan
        text = "".join(pcs[cs:te])
        for w in words:
            m = re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE)
            if m:
                # char pos -> token idx via cumulative piece lengths
                cum, ci = 0, 0
                for i in range(cs, te):
                    cum += len(pcs[i])
                    if cum > m.start():
                        ci = i - cs
                        break
                first_viol = np.nanmin([first_viol, ci / (te - cs)])
        rows.append({
            "trace_id": r.trace_id, "condition": r.condition, "qhash": r.qhash,
            "cls": r.cls, "T": te - cs,
            "pb_mean": float(pb.mean()), "pb_max": float(pb.max()),
            "spikes_10": float((pb > 0.10).sum() / (te - cs) * 100),
            "spikes_30": float((pb > 0.30).sum() / (te - cs) * 100),
            "near_miss_per100": float(near_miss.sum() / (te - cs) * 100),
            "first_viol_pos": first_viol,
            "profile": binned_profile(pb, N_BINS),
        })
    return pd.DataFrame(rows)


def fig_profiles(df: pd.DataFrame) -> None:
    x = (np.arange(N_BINS) + 0.5) / N_BINS
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, cls, lab, color in GROUPS:
        s = df[df.condition == cond] if cls is None else \
            df[(df.condition == cond) & (df.cls == cls)]
        if len(s) < 5:
            continue
        prof = np.stack(s.profile.values)
        m = np.nanmean(prof, 0)
        ax.plot(x, m, color=color, lw=2, label=f"{lab} (n={len(s)})")
        se = np.nanstd(prof, 0) / np.sqrt(np.sum(~np.isnan(prof), 0).clip(1))
        ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.15)
    ax.set_xlabel("normalized CoT position")
    ax.set_ylabel("P(banned-word onset token)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    ax.set_title("omit_word: P(banned) over the trace\n"
                 "(log scale; mean ± 95% band across traces)", fontsize=11)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "pbanned_profiles.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIGS/'pbanned_profiles.png'}")


def fig_dist(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    # ECDF of per-trace max p_banned
    ax = axes[0]
    for cond, cls, lab, color in GROUPS:
        s = df[df.condition == cond] if cls is None else \
            df[(df.condition == cond) & (df.cls == cls)]
        if len(s) < 5:
            continue
        v = np.sort(s.pb_max.values)
        ax.step(v, np.arange(1, len(v) + 1) / len(v), color=color, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("per-trace MAX P(banned)")
    ax.set_ylabel("ECDF")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax.set_title("luck story ⇒ compliant ≈ violating here", fontsize=9)
    # near-miss + spike rates (bar, clustered CI)
    for ai, (metric, title) in enumerate(
            [("near_miss_per100", "near-misses / 100 tokens\n(banned in top-5, not sampled)"),
             ("spikes_10", "positions with P(banned)>0.1 / 100 tokens")], start=1):
        ax = axes[ai]
        labels, means, los, his, colors = [], [], [], [], []
        for cond, cls, lab, color in GROUPS:
            s = df[df.condition == cond] if cls is None else \
                df[(df.condition == cond) & (df.cls == cls)]
            if len(s) < 5:
                continue
            m, lo, hi = cluster_boot_ci(s[metric].values, s.qhash.values)
            labels.append(lab); means.append(m); los.append(m - lo); his.append(hi - m)
            colors.append(color)
        ax.bar(range(len(means)), means, yerr=[los, his], color=colors, capsize=3)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.grid(alpha=0.3, axis="y")
        ax.set_title(title, fontsize=9)
    fig.suptitle("omit_word: suppression vs sampling luck", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "pbanned_dist.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIGS/'pbanned_dist.png'}")


def main() -> None:
    df = build_table()
    df.drop(columns=["profile"]).to_csv(HERE / "pbanned_stats.csv", index=False)
    print(f"omit_word table: {len(df)} traces")
    fig_profiles(df)
    fig_dist(df)
    # headline numbers
    print("\nper-group medians:")
    for cond, cls, lab, _ in GROUPS:
        s = df[df.condition == cond] if cls is None else \
            df[(df.condition == cond) & (df.cls == cls)]
        if len(s):
            print(f"  {lab:24s} n={len(s):3d}  max_pb={s.pb_max.median():.4f}  "
                  f"mean_pb={s.pb_mean.median():.5f}  near_miss/100={s.near_miss_per100.median():.2f}")
    v = df[(df.cls == "noncompliant") & df.first_viol_pos.notna()]
    for cond in CONDS:
        s = v[v.condition == cond]
        if len(s):
            print(f"  first-violation pos {COND_LABEL[cond]:10s}: median {s.first_viol_pos.median():.2f} "
                  f"(n={len(s)}, {100*(s.first_viol_pos<0.2).mean():.0f}% in first 20% of CoT)")


if __name__ == "__main__":
    main()
