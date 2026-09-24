"""Local analysis of the fixed-prefix rank experiment; no inference or grading.

python controllability/mech/analysis/rank_analysis.py --run opening_ranks_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rank_probe as rp

COLORS = ["#69758a", "#146c94"]
MODEL_LABEL = {"olmo31": "OLMo-3.1-32B-Think", "qwen32": "Qwen3-32B"}
TYPE_LABEL = {"uppercase_thinking": "Uppercase", "leetspeak": "Leetspeak"}


def bootstrap_delta(a, b, seed=20260908):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or a.shape != b.shape:
        raise ValueError("Paired bootstrap requires matched nonempty vectors")
    delta = b - a
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(len(delta), size=(10000, len(delta)))].mean(axis=1)
    return float(delta.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def flatten(records):
    rows = []
    for r in records:
        for cond, c in r["conditions"].items():
            best = c["best_content"]
            row = {k: r[k] for k in ["pair_id", "model_tag", "ctype", "prompt_id", "question_id", "position"]}
            row.update(condition=cond, n_source_traces=len(r["references"]),
                       best_id=best["id"], best_token=best["text"], best_rank=best["rank"],
                       best_log10_probability=best["log_probability"] / np.log(10),
                       log10_rank=np.log10(best["rank"]), argmax_token=c["argmax"]["text"],
                       argmax_probability=c["argmax"]["probability"],
                       **r["distribution_shift"])
            for k, v in c.items():
                if isinstance(v, (float, int, bool)) or v is None:
                    row[k] = v
            row["content_log10_mass"] = c["content_log_mass"] / np.log(10)
            rows.append(row)
    return pd.DataFrame(rows)


def question_summary(df):
    metrics = ["log10_rank", "best_log10_probability", "content_log10_mass", "content_mass",
               "legal_mass", "content_sample_mass", "legal_sample_mass", "neutral_mass",
               "excluded_mass", "entropy_nats", "argmax_probability"]
    # Average within question before bootstrap; shared prefixes have one row.
    q = df.groupby(["model_tag", "ctype", "position", "question_id", "condition"])[metrics].mean().reset_index()
    summary = []
    for (model, ct, pos), group in q.groupby(["model_tag", "ctype", "position"]):
        for metric in metrics:
            wide = group.pivot(index="question_id", columns="condition", values=metric).dropna()
            mean, lo, hi = bootstrap_delta(wide[rp.CONDS[0]], wide[rp.CONDS[1]])
            summary.append({"model_tag": model, "ctype": ct, "position": pos, "metric": metric,
                            "n_questions": len(wide), "no_instruction": wide[rp.CONDS[0]].mean(),
                            "instruction_only": wide[rp.CONDS[1]].mean(),
                            "delta": mean, "delta_ci_low": lo, "delta_ci_high": hi})
    return q, pd.DataFrame(summary)


def distribution_summary(df):
    # TV/KL already compare the two conditions; do not subtract them again.
    q = df[df.condition == rp.CONDS[1]].groupby(
        ["model_tag", "ctype", "position", "question_id"])[
        ["total_variation", "kl_instruction_to_no_instruction"]].mean().reset_index()
    rows = []
    for (model, ct, pos), group in q.groupby(["model_tag", "ctype", "position"]):
        for metric in ["total_variation", "kl_instruction_to_no_instruction"]:
            mean, lo, hi = bootstrap_delta(np.zeros(len(group)), group[metric])
            rows.append(dict(model_tag=model, ctype=ct, position=pos, metric=metric,
                             n_questions=len(group), mean=mean, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows)


def paired_plot(q, out, metric, ylabel, filename):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for row, model in enumerate(rp.MODELS):
        for col, ct in enumerate(rp.TYPES):
            ax = axes[row, col]
            sub = q[(q.model_tag == model) & (q.ctype == ct) & (q.position == 0)]
            wide = sub.pivot(index="question_id", columns="condition", values=metric)
            for _, r in wide.iterrows():
                ax.plot([0, 1], [r[rp.CONDS[0]], r[rp.CONDS[1]]], color=COLORS[1], alpha=.22, lw=1)
            means = [wide[c].mean() for c in rp.CONDS]
            ax.plot([0, 1], means, color="#c55a11", marker="o", lw=2.5, label="Mean across questions")
            delta, lo, hi = bootstrap_delta(wide[rp.CONDS[0]], wide[rp.CONDS[1]])
            ax.set_title(f"{MODEL_LABEL[model]} · {TYPE_LABEL[ct]}\nΔ = {delta:+.2f} [{lo:+.2f}, {hi:+.2f}], n={len(wide)}", fontsize=10)
            ax.set_xticks([0, 1], ["No instruction", "Instruction"])
            ax.set_xlim(-.15, 1.15)
            ax.grid(axis="y", alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.supylabel(ylabel, x=.015, fontsize=11)
    fig.suptitle("First reasoning token: paired predictions at the same prefix", fontsize=13)
    fig.text(.5, .01, "Blue: individual questions; orange: mean. Brackets: 95% question-bootstrap CI for the paired change.", ha="center", fontsize=9)
    fig.tight_layout(rect=[.045, .04, 1, .94])
    fig.savefig(out / filename, dpi=180)
    fig.savefig((out / filename).with_suffix(".pdf"))
    plt.close(fig)


def mass_plot(q, out, kind="content"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for row, model in enumerate(rp.MODELS):
        for col, ct in enumerate(rp.TYPES):
            ax = axes[row, col]
            sub = q[(q.model_tag == model) & (q.ctype == ct) & (q.position == 0)]
            for ci, cond in enumerate(rp.CONDS):
                d = sub[sub.condition == cond]
                vals = [d[f"{kind}_mass"].mean(), d[f"{kind}_sample_mass"].mean()]
                x = np.arange(2) + (ci - .5) * .3
                ax.bar(x, vals, .28, color=COLORS[ci], label=cond.replace("_", " "))
                for xx, val, metric in zip(x, vals, [f"{kind}_mass", f"{kind}_sample_mass"]):
                    _, lo, hi = bootstrap_delta(np.zeros(len(d)), d[metric])
                    ax.errorbar(xx, val, yerr=[[max(0, val-lo)], [max(0, hi-val)]], color="#222222", capsize=3, lw=1)
                    ax.annotate(f"{val:.3g}", (xx, hi), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
            ax.set_title(f"{MODEL_LABEL[model]} · {TYPE_LABEL[ct]}", fontsize=10)
            ax.set_xticks([0, 1], ["Raw model\nT = 1", "Baseline sampling\nT = 0.6, top-p = 0.95"])
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.2)
    fig.supylabel("Mean content-bearing legal probability" if kind == "content" else "Mean total locally legal probability", x=.015, fontsize=11)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Does legal probability survive the baseline decoding filters?", fontsize=13)
    scope = "Punctuation-only tokens excluded." if kind == "content" else "Includes legal punctuation/whitespace and other neutral tokens."
    fig.text(.5, .006, f"Whiskers: 95% question-bootstrap CI. {scope}\nFull-distribution decoding; not a reconstruction of ban4 sampling.", ha="center", fontsize=8)
    fig.tight_layout(rect=[.045, .04, 1, .94])
    filename = "legal_mass_decoding.png" if kind == "content" else "total_legal_mass_decoding.png"
    fig.savefig(out / filename, dpi=180)
    fig.savefig((out / filename).with_suffix(".pdf"))
    plt.close(fig)


def analyze(run):
    root = rp.DATA / "rank_runs" / run
    manifest = rp.load_manifest(root / "manifest.json")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    provenance = {"manifest_sha256": manifest["manifest_sha256"],
                  "analysis_sha256": rp.file_hash(Path(__file__)),
                  "bootstrap_draws": 10000, "bootstrap_seed": 20260908, "result_files": {}}
    records = []
    for tag in rp.MODELS:
        r = rp.load_results(root / f"{tag}_pairs.jsonl", manifest["manifest_sha256"])
        expected = {p["pair_id"] for p in manifest["pairs"][tag]}
        if set(r) != expected:
            raise ValueError(f"Incomplete/unexpected results for {tag}: {len(r)}/{len(expected)}")
        provenance["result_files"][tag] = rp.file_hash(root / f"{tag}_pairs.jsonl")
        records.extend(r.values())
    df = flatten(records)
    q, summary = question_summary(df)
    shifts = distribution_summary(df)
    df.to_csv(out / "per_prefix.csv", index=False)
    q.to_csv(out / "per_question.csv", index=False)
    summary.to_csv(out / "paired_summary.csv", index=False)
    shifts.to_csv(out / "distribution_shifts.csv", index=False)
    paired_plot(q, out, "log10_rank", "log₁₀ rank of best content-bearing legal token\n(lower is better)", "paired_ranks.png")
    paired_plot(q, out, "content_log10_mass", "log₁₀ content-bearing legal probability\n(higher is better)", "paired_probabilities.png")
    paired_plot(q, out, "best_log10_probability", "log₁₀ probability of best content-bearing legal token\n(higher is better)", "paired_best_token_probabilities.png")
    mass_plot(q, out)
    mass_plot(q, out, "legal")

    table = []
    for model in rp.MODELS:
        for ct in rp.TYPES:
            pool = sorted([r for r in records if r["model_tag"] == model and r["ctype"] == ct and r["position"] == 0], key=lambda r: r["prompt_id"])
            # Deterministic example, not selected for effect size.
            r = pool[0]
            candidates = dict(r["fixed_candidates"])
            observed_id = str(r["references"][0]["observed_id"])
            candidates[observed_id] = {c: r["conditions"][c]["observed"][observed_id] for c in rp.CONDS}
            for token, both in candidates.items():
                a, b = (both[c] for c in rp.CONDS)
                table.append({"model": MODEL_LABEL[model], "behavior": TYPE_LABEL[ct],
                              "prompt_id": r["prompt_id"], "candidate": a["text"],
                              "role": "observed opening" if token == observed_id else "legal alternative",
                              "default_token": r["conditions"]["no_instruction"]["argmax"]["text"],
                              "rank_no_instruction": a["rank"], "rank_instruction": b["rank"],
                              "logp_no_instruction": a["log_probability"], "logp_instruction": b["log_probability"],
                              "p_no_instruction": a["probability"], "p_instruction": b["probability"]})
    pd.DataFrame(table).to_csv(out / "token_examples.csv", index=False)
    lines = ["# Opening-rank experiment", "", f"Manifest: `{manifest['manifest_sha256']}`.", "",
             "Two released models; uppercase and leetspeak; rescoring only. No new continuations or judgments.", "",
             "## First-token paired results", "",
             "Changes below are instruction minus no instruction. CIs resample questions, after averaging repeated prefixes within question.", "",
             "| Model | Behavior | Rank ratio¹ | Content-bearing legal mass: no instruction → instruction | After decoding |", "|---|---|---:|---:|---:|"]
    for model in rp.MODELS:
        for ct in rp.TYPES:
            s = summary[(summary.model_tag == model) & (summary.ctype == ct) & (summary.position == 0)].set_index("metric")
            rank, mass, sampled = (s.loc[k] for k in ["log10_rank", "content_mass", "content_sample_mass"])
            lines.append(f"| {MODEL_LABEL[model]} | {TYPE_LABEL[ct]} | {10**rank.delta:.3g} [{10**rank.delta_ci_low:.3g}, {10**rank.delta_ci_high:.3g}] | {mass.no_instruction:.4g} → {mass.instruction_only:.4g} | {sampled.no_instruction:.4g} → {sampled.instruction_only:.4g} |")
    lines += ["", "¹ Geometric-mean ratio of best legal ranks; <1 means the instruction improves rank. These best tokens may differ between conditions; token_examples.csv compares fixed identities.", "",
              "## Interpretation and boundaries", "",
              "The distributions compare identical reasoning-token prefixes under two prompt contexts. Differences demonstrate sensitivity to the control prompt, not a particular attention mechanism or sustained control ability.",
              "All selected baselines begin with Okay. First-violation locations and source-token alignment are in the manifest; positions 1–3 describe dynamics after an initial violation, not restoration of whole-trace compliance.",
              "Raw legal probability lost after temperature/nucleus filtering is a decoding-accessibility result. Tiny/no legal-mass changes do not establish that the model ignored the instruction internally.",
              "Neutral-token and excluded-token masses, full-distribution shifts, log probabilities, and positions 0–3 are included in the CSVs. Per-response whole-trace scores remain separate in the manifest.",
              "Questions and phrasings were not fully crossed; sample size is 24 questions per model–behavior cell, not the number of generated samples. No rank–ban4 correlation is fitted to four cells.", "",
              "## Token examples", "", "The first prompt in sorted order per model–behavior cell is used, independently of its measured effect.", "",
              "| Model | Behavior | Candidate | Role | Rank without → with instruction | Probability without → with instruction |", "|---|---|---|---|---:|---:|"]
    for row in table:
        candidate = json.dumps(row["candidate"], ensure_ascii=False).replace("|", "\\|")
        lines.append(f"| {row['model']} | {row['behavior']} | `{candidate}` | {row['role']} | {row['rank_no_instruction']} → {row['rank_instruction']} | {row['p_no_instruction']:.3g} → {row['p_instruction']:.3g} |")
    lines += ["", "## Descriptive ban4 comparison", "", "| Model | Behavior | Existing baseline soft score | Existing ban4 soft score |", "|---|---|---:|---:|"]
    for model in rp.MODELS:
        grade_path = rp.DATA / "ctrl26_runs" / f"ban4_{model}" / "grades.csv"
        if rp.file_hash(grade_path) != manifest["source_hashes"][model]["grades"]:
            raise ValueError("Existing ban4 grades changed since preparation")
        grades = pd.read_csv(grade_path)
        for ct in rp.TYPES:
            means = grades[grades.ctype == ct].groupby("variant").compliance_score.mean()
            lines.append(f"| {MODEL_LABEL[model]} | {TYPE_LABEL[ct]} | {means['baseline']:.4f} | {means['ban4']:.4f} |")
    lines += ["", "## What the experiment shows", "",
              "| Model | Behavior | Geometric-mean legal-mass multiplier (95% CI) | Opening total variation (95% CI) |",
              "|---|---|---:|---:|"]
    for model in rp.MODELS:
        for ct in rp.TYPES:
            mass = summary[(summary.model_tag == model) & (summary.ctype == ct) & (summary.position == 0) & (summary.metric == "content_log10_mass")].iloc[0]
            tv = shifts[(shifts.model_tag == model) & (shifts.ctype == ct) & (shifts.position == 0) & (shifts.metric == "total_variation")].iloc[0]
            lines.append(f"| {MODEL_LABEL[model]} | {TYPE_LABEL[ct]} | {10**mass.delta:.1f}× [{10**mass.delta_ci_low:.1f}, {10**mass.delta_ci_high:.1f}] | {tv['mean']:.4f} [{tv.ci_low:.4f}, {tv.ci_high:.4f}] |")
    opening_conditions = [r["conditions"][c] for r in records if r["position"] == 0 for c in rp.CONDS]
    collapsed = sum(c["nucleus_size"] == 1 and c["argmax"]["text"] == "Okay" for c in opening_conditions)
    lines += ["",
              "**Instruction shifts legal mass despite failure:** supported in all four cells. Every cell's paired 95% interval for mean log legal-mass change is positive. Best-legal rank improves in three cells; OLMo leetspeak has a near-zero rank change despite a large probability increase. Rank alone misses this response.", "",
              "**Distribution changes without favoring legal tokens:** not the cell-average pattern here. All four favor legal mass, although absolute total variation is small (roughly 0.005–0.031) and legal mass remains tiny. This does not imply that most of the redistributed mass goes to legal tokens.", "",
              "**Little detectable distributional response:** not the aggregate finding. The approximately unchanged OLMo leetspeak rank must not be described as an unchanged distribution or an ignored instruction. Individual prompts remain available for inspection; no internal-attention conclusion follows.", "",
              f"**Raw legal mass removed by decoding:** {collapsed}/{len(opening_conditions)} first-token condition predictions retain only Okay after temperature 0.6 and top-p 0.95. Both total legal mass and content-bearing legal mass are exactly zero after these filters; this is not numerical underflow. It describes the specified full-distribution decoder, not historical ban4 sampling.", "",
              "The fixed-candidate example for OLMo leetspeak is informative: the digit token 0 moves from rank 1517 to 8, while Hmm moves from 7 to 16. The best-legal rank can stay roughly constant while the identities and probabilities of legal alternatives change substantially. A leading digit remains only a local-compliance diagnostic, not evidence of a compliant continuation.", "",
              "For the paper, these results support: **opening failure can coexist with substantial instruction sensitivity, while locally legal alternatives remain inaccessible to the usual decoder.** Together with the existing prefill results, this is consistent with an initiation bottleneck. It does not establish that thinking SFT caused the concentration, explain all forms of low controllability, or show that choosing a legal token would sustain compliance.", "",
              "The descriptive ban4 comparison should remain descriptive: Qwen leetspeak shows rank improvement alongside strong ban4 recovery, but its absolute opening legal mass is lower than OLMo's. Four cells and different interventions cannot identify a rank–rescue relationship.", "",
              "## Validation and execution", "",
              "All 144 original baseline responses were recovered with exact token IDs. The 385 paired prefixes correspond to 762 distinct primary forward requests after caching across behaviors. Opening analyses contain 24 questions per cell and count each prompt once.", "",
              "Both workers passed exact repeated-input checks and all 16 historical comparisons. Historical top-five overlap was 5/5 at 15 positions and 4/5 at one; maximum entropy difference was 0.0231 nats and maximum saved top-five log-probability difference was 0.7395 nats. Thus reproduction is approximate, not bitwise agreement with the old backend. Nine offline tests also pass.", "",
              "The one-row bf16 output projection differed from the full-prefix projection by at most 0.03125 in the OLMo check. The final run uses the full short-prefix projection. Early setup attempts exposed missing Accelerate, slow cache mmap, and repeated tokenizer-size lookup; all are accounted for in the budget ledger."]
    ledger_path = root / "budget_ledger.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        seconds = sum(a.get("accounted_seconds", a["reserved_seconds"]) for a in ledger["attempts"])
        lines += ["", f"Conservative compute estimate including all setup attempts: **${ledger['estimated_compute_usd_upper']:.2f}**, with {seconds/60:.1f} accounted worker minutes against the four-hour cap. This is not a billing statement. Both models are complete; outputs are retained locally and on the new Modal results volume."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    (out / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary[summary.position == 0].to_string(index=False))
    print(f"Wrote analysis to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="opening_ranks_v1")
    analyze(ap.parse_args().run)
