"""Question-clustered token-prefill curves and quality summaries; no inference."""
from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import token_prefill as tp

METRICS = {"head32_soft": ("Immediate continuation (first 32 tokens)", "#3366aa", "-"),
           "full_soft": ("Sustained continuation (whole trace)", "#cc7722", "-"),
           "head32_after_boundary_content_soft": ("First 32, after word boundary", "#3366aa", "--")}
from paper_style import apply_style, export_figure, SQUARE_SIZE, BEHAVIOR_LABELS, MODEL_COLORS


def clustered_summary(frame, repeats=2000):
    """Average samples within question; resample questions jointly across conditions/K."""
    rng = np.random.default_rng(20260915)
    rows = []
    for behavior, group in frame.groupby("behavior"):
        question_means = group.groupby(["question_id", "condition", "prefix_tokens"])[list(METRICS)].mean()
        matrix = question_means.unstack(["condition", "prefix_tokens"])
        if matrix.isna().any().any():
            raise ValueError("Unbalanced questions: investigate missing cells before plotting")
        values = matrix.to_numpy()
        indices = rng.integers(len(values), size=(repeats, len(values)))
        boots = values[indices].mean(axis=1)
        lo, hi = np.quantile(boots, [.025, .975], axis=0)
        for j, (metric, condition, length) in enumerate(matrix.columns):
            rows.append(dict(behavior=behavior, metric=metric, condition=condition,
                prefix_tokens=length, mean=float(values[:, j].mean()),
                low=float(lo[j]), high=float(hi[j]), questions=len(values)))
    return pd.DataFrame(rows)


def paired_changes(frame, repeats=2000):
    """Bootstrap within-question differences, not differences of marginal intervals."""
    rng = np.random.default_rng(20260916)
    rows = []
    for behavior, group in frame.groupby("behavior"):
        for metric in METRICS:
            matrix = group.groupby(["question_id", "condition", "prefix_tokens"])[metric].mean().unstack(["condition", "prefix_tokens"])
            if matrix.isna().any().any():
                raise ValueError("Unbalanced questions in paired contrasts")
            indices = rng.integers(len(matrix), size=(repeats, len(matrix)))
            for condition in tp.CONDITIONS:
                for length in sorted(group.prefix_tokens.unique()):
                    values = (matrix[(condition, length)] - matrix[(condition, 0)]).to_numpy()
                    low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
                    rows.append(dict(behavior=behavior, metric=metric, condition=condition,
                                     prefix_tokens=length, contrast="prefix_minus_zero",
                                     mean=values.mean(), low=low, high=high, questions=len(matrix)))
            for length in sorted(group.prefix_tokens.unique()):
                values = (matrix[("prefill_plus_instruction", length)] - matrix[("prefill_only", length)]).to_numpy()
                low, high = np.quantile(values[indices].mean(axis=1), [.025, .975])
                rows.append(dict(behavior=behavior, metric=metric, condition="paired_conditions",
                                 prefix_tokens=length, contrast="instruction_minus_no_instruction",
                                 mean=values.mean(), low=low, high=high, questions=len(matrix)))
    return pd.DataFrame(rows)


def main(root=tp.DEFAULT_ROOT):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    tp.validate_manifest(m)
    d = pd.read_csv(root / "scores.csv")
    assert d.manifest_sha256.eq(m["manifest_sha256"]).all()
    assert len(d) == len(m["links"]) * m["sampling"]["n"]
    assert not d.duplicated(["cell_id", "condition", "prefix_tokens", "sample"]).any()
    out = HERE / "token_prefill" / m["run"]
    out.mkdir(parents=True, exist_ok=True)
    summary = clustered_summary(d)
    summary.to_csv(out / "summary.csv", index=False)
    paired_changes(d).to_csv(out / "paired_changes.csv", index=False)
    d["head32_no_relevant_chars"] = d.head32_relevant_chars.eq(0)
    d["full_no_relevant_chars"] = d.full_relevant_chars.eq(0)
    d["early_failure"] = d.first_violation_token.lt(32)
    quality = d.groupby(["behavior", "condition", "prefix_tokens"]).agg(
        n=("sample", "size"), questions=("question_id", "nunique"),
        empty_rate=("empty_reasoning", "mean"), truncation_rate=("truncated", "mean"),
        full32_rate=("has_32_tokens", "mean"), closed_reasoning_rate=("closed_reasoning", "mean"),
        head32_contentless_rate=("head32_no_relevant_chars", "mean"),
        full_contentless_rate=("full_no_relevant_chars", "mean"),
        mean_reasoning_tokens=("reasoning_tokens", "mean"),
        early_failure_rate=("early_failure", "mean"),
        median_first_failure_token=("first_violation_token", "median")).reset_index()
    quality.to_csv(out / "quality.csv", index=False)
    apply_style()
    fig, ax = plt.subplots(figsize=SQUARE_SIZE)
    for behavior, marker in zip(m["behaviors"], ["o", "s"]):
        for condition in tp.CONDITIONS:
            z = summary[(summary.behavior == behavior) & (summary.condition == condition) & (summary.metric == "full_soft")].sort_values("prefix_tokens")
            instructed = condition == "prefill_plus_instruction"
            ax.errorbar(z.prefix_tokens, z["mean"], yerr=np.array([z["mean"]-z.low,z.high-z["mean"]]),
                color=MODEL_COLORS["qwen32"], ls="-" if instructed else "--", alpha=1 if instructed else .45,
                lw=2.4 if instructed else 1.6, marker=marker, ms=4 if instructed else 3,
                capsize=2, elinewidth=1, label=BEHAVIOR_LABELS[behavior] if instructed else None)
    ax.set_xscale("symlog", base=2, linthresh=1)
    ax.set_xticks(m["lengths"], [str(k) for k in m["lengths"]])
    ax.set_ylim(-.02, 1.02)
    ax.grid(alpha=.2)
    ax.legend()
    ax.set_xlabel("Supplied reasoning tokens")
    ax.set_ylabel("mean continuation compliance")
    fig.tight_layout()
    export_figure(fig, out, "compliance", square=True, paper_name="token_prefill_qwen32")
    note = """# Token-prefill analysis

The main soft scores use the sentence experiment's scoring rules. Every generated
sample is retained, including empty reasoning (score zero) and capped responses.
The first-32 window uses up to 32 newly generated reasoning tokens; consult
`full32_rate` for shorter continuations. The figure reports sustained continuation (the whole new reasoning trace).
Solid lines retain the instruction; faded dashed lines omit it. Immediate
continuation remains in the summary tables.
The word-boundary diagnostic is retained in the data but is not plotted. It removes
characters completing the seed's final alphanumeric word and scores windows with
no relevant characters as zero. `quality.csv` exposes contentless windows, early failures,
short continuations, missing reasoning boundaries, and truncation.

Intervals are 95% question-clustered bootstrap intervals (2,000 replicates).
Repeated samples are averaged within question first; the same question draw is
used across lengths, conditions, and metrics within a behavior. These are
pointwise intervals, not a simultaneous band or a significance test of differences.
`paired_changes.csv` separately reports within-question contrasts against the
matched zero-token baseline and between instruction conditions at each length.
Its intervals bootstrap paired differences directly. These many contrasts are
exploratory and are not corrected for multiple comparisons.

Every condition includes the same forced reasoning opening. The sentence sweep
used an API backend with a different K=0 prompt policy, so do not interpret a
between-experiment difference as solely the effect of sentence versus token units.
The paper PNG is copied to paper/figures/token_prefill_qwen32.png.
"""
    (out / "README.md").write_text(note)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=tp.DEFAULT_ROOT)
    main(parser.parse_args().root)
