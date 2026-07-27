"""Analyses 3+5 — attention trajectories and cross-type engagement.

3. Attention mass from each generated token to the instruction / example-reasoning /
   own-CoT spans vs normalized position, compliant vs non-compliant, per type.
   Attention to prompt spans is reported PER KEY TOKEN (mass / span length) so span
   size doesn't confound; own_cot is raw mass (its size grows with position by design).
5. Cross-type engagement at matched condition: do the floor types (uppercase/leetspeak/
   spanish) attend to the instruction as much as the responsive types (attends-but-
   can't-execute) or less (never engages)?

Layer handling: mean over all 64 layers of the head-mean ('mean') and of the
max-over-heads ('amax' — catches a single instruction-head that head-averaging hides).

  python analysis/attention_analysis.py
Outputs figs/attn_trajectories.png, figs/attn_engagement.png, attn_stats.csv
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_analysis import (CLS_COLORS, COND_LABEL, FIGS, HERE, TYPES,
                             binned_profile, cluster_boot_ci, load_attn,
                             load_grades, load_prompts)

N_BINS = 20
SPAN_ORDER = ["instr_real", "instr_example", "example_reasoning", "example_answer",
              "donor_question", "target_question", "own_cot", "other"]


def build_table() -> pd.DataFrame:
    sel = pd.read_csv(HERE.parent / "data" / "attn_selection.csv")
    prompts = load_prompts()
    rows = []
    for _, r in sel.iterrows():
        d = load_attn(r.trace_id)
        if d is None:
            continue
        p = prompts[r.prompt_id]
        spans = p["token_spans"]
        mean = d["mean"].astype(np.float32)      # [L, T, S]
        amax = d["amax"].astype(np.float32)
        lm = mean.mean(0)                        # layer-mean of head-mean [T, S]
        lx = amax.mean(0)                        # layer-mean of head-max  [T, S]
        row = {"trace_id": r.trace_id, "ctype": r.ctype, "condition": r.condition,
               "qhash": r.qhash, "cls": r.cls, "T": lm.shape[0]}
        for si, name in enumerate(SPAN_ORDER):
            ts = spans.get(name)
            if name in ("own_cot", "other"):
                norm = 1.0
            elif ts:
                norm = max(ts[1] - ts[0], 1)
            else:
                row[f"{name}_mean"] = np.nan
                row[f"{name}_prof"] = np.full(N_BINS, np.nan)
                row[f"{name}_hmax"] = np.nan
                continue
            per_tok = lm[:, si] / norm
            row[f"{name}_mean"] = float(per_tok.mean())
            row[f"{name}_prof"] = binned_profile(per_tok, N_BINS)
            row[f"{name}_hmax"] = float((lx[:, si] / norm).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def fig_trajectories(df: pd.DataFrame) -> None:
    """Rows: contrast types; cols: instr_real / example_reasoning / own_cot."""
    types = ["no_first_person", "omit_word", "jar_jar_binks"]
    cond = "icl_user_think"
    panels = [("instr_real", "attn to instruction @ real Q (per key token)"),
              ("instr_example", "attn to instruction @ example (per key token)"),
              ("example_reasoning", "attn to example reasoning (per key token)"),
              ("own_cot", "attn to own CoT (raw mass)")]
    x = (np.arange(N_BINS) + 0.5) / N_BINS
    fig, axes = plt.subplots(len(types), len(panels),
                             figsize=(13, 2.8 * len(types)), sharex=True)
    for ti, ct in enumerate(types):
        sub = df[(df.ctype == ct) & (df.condition == cond)]
        for pi, (span, title) in enumerate(panels):
            ax = axes[ti, pi]
            for cls in ["compliant", "noncompliant"]:
                s = sub[sub.cls == cls]
                if len(s) < 5:
                    continue
                prof = np.stack(s[f"{span}_prof"].values)
                m = np.nanmean(prof, 0)
                ax.plot(x, m, color=CLS_COLORS[cls], lw=2, label=f"{cls} (n={len(s)})")
                se = np.nanstd(prof, 0) / np.sqrt(np.sum(~np.isnan(prof), 0).clip(1))
                ax.fill_between(x, m - 1.96 * se, m + 1.96 * se,
                                color=CLS_COLORS[cls], alpha=0.15)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            if pi == 0:
                ax.set_ylabel(ct, fontsize=9)
            if ti == 0:
                ax.set_title(title, fontsize=10)
            if ti == len(types) - 1:
                ax.set_xlabel("normalized CoT position")
    fig.suptitle(f"Attention over the CoT, {COND_LABEL[cond]} condition "
                 "(layer-mean, head-mean)", fontsize=12)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "attn_trajectories.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIGS/'attn_trajectories.png'}")


def fig_engagement(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-type: mean per-token attention to instr_real, V2 + instr-only."""
    conds = ["instruction_only", "icl_user_think"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    stats = []
    for ci, cond in enumerate(conds):
        ax = axes[ci]
        xt, xl = [], []
        pos = 0
        for ct in TYPES:
            sub = df[(df.ctype == ct) & (df.condition == cond)]
            groups = [("compliant", CLS_COLORS["compliant"]),
                      ("noncompliant", CLS_COLORS["noncompliant"])]
            plotted = False
            for cls, color in groups:
                s = sub[sub.cls == cls] if sub.cls.notna().any() else \
                    (sub if cls == "noncompliant" else sub.iloc[0:0])
                s = s.dropna(subset=["instr_real_mean"])
                if len(s) < 5:
                    continue
                m, lo, hi = cluster_boot_ci(s.instr_real_mean.values, s.qhash.values)
                hx, hlo, hhi = cluster_boot_ci(s.instr_real_hmax.values, s.qhash.values)
                ax.bar(pos, m, yerr=[[m - lo], [hi - m]], color=color, capsize=3, width=0.8)
                # ICL conditions: stack a faded bar for the EXAMPLE's instruction copy,
                # so total height = per-token attn to instruction text (both copies)
                mex = np.nan
                se = s.dropna(subset=["instr_example_mean"])
                if len(se) >= 5:
                    mex, _, _ = cluster_boot_ci(se.instr_example_mean.values, se.qhash.values)
                    ax.bar(pos, mex, bottom=m, color=color, alpha=0.35, width=0.8)
                stats.append({"condition": cond, "ctype": ct, "cls": cls, "n": len(s),
                              "attn_instr_mean": m, "lo": lo, "hi": hi,
                              "attn_instr_headmax": hx, "attn_instr_example_mean": mex})
                xt.append(pos); xl.append(f"{ct}\n{cls[:4]}")
                pos += 1
                plotted = True
            if plotted:
                pos += 0.6
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=6.5, rotation=45, ha="right")
        ax.grid(alpha=0.3, axis="y")
        ax.set_title(COND_LABEL[cond], fontsize=11)
        if ci == 0:
            ax.set_ylabel("mean attn to instruction per key token")
    fig.suptitle("Cross-type engagement: attention to the instruction\n"
                 "(solid = instruction @ real Q; faded top = + instruction copy inside the "
                 "ICL example; floor types have no compliant bar)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "attn_engagement.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIGS/'attn_engagement.png'}")
    return pd.DataFrame(stats)


def main() -> None:
    df = build_table()
    print(f"attention table: {len(df)} traces")
    fig_trajectories(df)
    stats = fig_engagement(df)
    stats.to_csv(HERE / "attn_stats.csv", index=False)
    print("\ninstruction-attention summary (per key token, layer/head-mean):")
    print(stats.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
