"""Audit the completed clean GPT-OSS sweep without generating or grading.

Run after gptoss_prefill_run.py grade. Saves machine-readable quality summaries
and a short note beside the figures, retaining the medium pilot separately.
"""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harmony_backend as h

ROOT = HERE.parent / "results" / "gptoss_high_clean_v1"
OUT = HERE / "prefill"


def main():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    payload = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    assert hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == manifest["manifest_sha256"]
    assert hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest() == manifest["harmony_sha256"]
    requests = {r["request_id"]: r for r in manifest["requests"]}
    records = []
    seen = set()
    boundary_fragments = []
    for path in sorted((ROOT / "remote").glob("batch_*.json")):
        for row in json.loads(path.read_text()):
            key = (row["request_id"], row["effort"], row["sample"])
            assert key not in seen, key
            seen.add(key)
            r = requests[row["request_id"]]
            expected = (r["prompt_token_ids"] if row["effort"] == "high" else
                        h.build_prefill_tokens(r["question"], r["options"],
                            r["instruction"] if r["condition"] == "prefill_plus_instruction" else "",
                            r["seed"], "medium"))
            assert row["prompt_token_ids"] == expected, key
            assert row["output_token_ids"], key
            decoded = h._enc().decode(row["output_token_ids"])
            decoded = decoded.removesuffix("<|return|>").removesuffix("<|call|>")
            if decoded != row["raw_text"]:
                # vLLM withholds an incomplete terminal UTF-8 fragment; Harmony's
                # full-sequence decoder emits U+FFFD. Permit only this audited
                # cap-boundary case, retaining both original representations.
                suffix = decoded[len(row["raw_text"]):]
                assert row["finish_reason"] == "length" and len(row["output_token_ids"]) == manifest["sampling"]["max_tokens"], key
                assert decoded.startswith(row["raw_text"]) and "\ufffd" in suffix and all(c.isspace() or c == "\ufffd" for c in suffix), key
                boundary_fragments.append(dict(key=key, withheld_suffix=suffix))
            assert h.split_channels(row["raw_text"]) == (row["reasoning_continuation"], row["response"]), key
            records.append(dict(request_id=r["request_id"], ctype=r["ctype"],
                condition=r["condition"], k=r["k"], effort=row["effort"], sample=row["sample"],
                smoke=r["request_id"] in manifest["smoke_ids"],
                empty=not row["reasoning_continuation"].strip(),
                truncated=row["finish_reason"] == "length",
                output_tokens=len(row["output_token_ids"]),
                reasoning_chars=len(row["reasoning_continuation"])))
    raw = pd.DataFrame(records)
    assert (raw.effort == "high").sum() == 18000
    assert (raw.effort == "medium").sum() == 120
    csv = ROOT.parent / "base_gptoss_120b_high_clean_prefill_per_sample.csv"
    grades = pd.read_csv(csv)
    assert len(grades) == 18000 and grades.compliance_score.notna().all()
    assert grades.manifest_sha256.eq(manifest["manifest_sha256"]).all()

    def quality(d, groups):
        return d.groupby(groups, dropna=False).agg(n=("empty", "size"),
            empty_rate=("empty", "mean"), truncated_rate=("truncated", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
            mean_reasoning_chars=("reasoning_chars", "mean")).reset_index()

    quality_main = quality(raw[raw.effort == "high"], ["ctype", "condition", "k"])
    quality_pilot = quality(raw[raw.smoke], ["effort", "condition", "k"])
    overall = quality(raw, ["effort"])
    compliance = grades.groupby(["ctype", "condition", "k"]).agg(
        n=("compliance_score", "size"), compliance=("compliance_score", "mean"),
        accuracy=("correct", "mean")).reset_index()
    nonempty = grades[~grades.empty_thinking.astype(bool)]
    plotted = nonempty.groupby(["ctype", "condition", "k"]).compliance_score.mean().reset_index()
    averages = plotted.groupby(["condition", "k"]).compliance_score.mean().unstack(0)
    uncapped = nonempty[nonempty.finish_reason != "length"].groupby(
        ["ctype", "condition", "k"]).compliance_score.mean().reset_index()
    uncapped_averages = uncapped.groupby(["condition", "k"]).compliance_score.mean().unstack(0)
    cap_sensitivity = float((averages - uncapped_averages).abs().max().max())
    task_quality = quality(raw[raw.effort == "high"], ["ctype"])
    ledger = json.loads((ROOT / "gpu_ledger.json").read_text())
    assert all("accounted_seconds" in a for a in ledger["attempts"]), "Worker still active"
    gpu_seconds = sum(a["accounted_seconds"] for a in ledger["attempts"])
    judge_cost = sum(json.loads(line)["cost_usd"] for line in
                     (ROOT / "judge_attempts.jsonl").read_text().splitlines())
    summary = dict(manifest_sha256=manifest["manifest_sha256"],
        csv_sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
        quality=overall.to_dict("records"), quality_by_cell=quality_main.to_dict("records"),
        pilot_quality=quality_pilot.to_dict("records"),
        compliance= compliance.to_dict("records"),
        plotted_means=averages.reset_index().to_dict("records"),
        uncapped_plotted_means=uncapped_averages.reset_index().to_dict("records"),
        max_mean_change_excluding_truncation=cap_sensitivity,
        quality_by_task=task_quality.to_dict("records"),
        answer_accuracy=float(grades.correct.mean()),
        gpu_accounted_seconds=gpu_seconds,
        gpu_cost_conservative_usd=gpu_seconds / 3600 * ledger["conservative_usd_per_hour"],
        judge_cost_usd=judge_cost)
    summary["terminal_decoding_fragments"] = boundary_fragments
    OUT.mkdir(exist_ok=True)
    (OUT / "gptoss_high_clean_audit.json").write_text(json.dumps(summary, indent=2))
    quality_main.to_csv(OUT / "gptoss_high_clean_quality.csv", index=False)
    quality_pilot.to_csv(OUT / "gptoss_high_clean_pilot.csv", index=False)
    high = overall[overall.effort == "high"].iloc[0]
    medium = overall[overall.effort == "medium"].iloc[0]
    text = ["# Clean high-effort GPT-OSS prefill results", "",
        "All 18,000 main responses use high effort, one model revision and one serving backend. "
        "No-instruction prompts omit the developer message entirely. The 120 medium-effort pilot "
        "responses are diagnostic only and are excluded from both paper figures.", "",
        f"Main-run empty reasoning: {high.empty_rate:.2%}; output-cap truncation: {high.truncated_rate:.2%}. "
        f"Mean generated length: {high.mean_output_tokens:.0f} tokens. "
        "The plots retain the existing nonempty-continuation convention; all-response scores "
        "and quality counts are recorded in the audit JSON.", "",
        f"Token/text checks found {len(boundary_fragments)} capped output(s) ending in an "
        "incomplete UTF-8 fragment: vLLM withheld the terminal fragment, while the full "
        "Harmony decoder emitted a replacement character. Both the original token IDs "
        "and returned text are retained; the remaining text matches exactly.", "",
        f"The matched medium pilot has {medium.empty_rate:.2%} empty reasoning and "
        f"{medium.truncated_rate:.2%} truncation across 120 responses. Its mean generated "
        f"length is {medium.mean_output_tokens:.0f} tokens. This pilot is a prompt/parser "
        "validation, not a full medium-effort replication.", "",
        "Mean compliance across the ten behaviors (the plotted values):", "",
        "| Seed sentences | Prefill only | Prefill + instruction |",
        "|---:|---:|---:|"]
    for k, row in averages.iterrows():
        text.append(f"| {k} | {row.prefill_only:.3f} | {row.prefill_plus_instruction:.3f} |")
    text += ["", "Prefill improves GPT-OSS's average continuation compliance, but it remains "
        "a substantial exception to the strong continuation control seen in the other plotted "
        "models. Chinese, Spanish, and instruction-conditioned sentence prefixes show the "
        "clearest gains; leetspeak and the three persona styles remain weak at K=5. The "
        "initiation-bottleneck account should therefore be qualified by model and behavior.", "",
        "Truncation is concentrated in leetspeak (190/1,800; 10.56%) and sentence prefixes "
        "(68/1,800; 3.78%). The main figures retain these responses. As a descriptive "
        f"sensitivity check, excluding all capped responses changes any plotted GPT-OSS "
        f"mean by at most {100 * cap_sensitivity:.2f} percentage points. This exclusion "
        "conditions on a generation outcome and is not a replacement estimand.", "",
        "The historical medium-effort sweep had a K=0 channel-parsing error. "
        "This rerun changes effort and fixes the prompt/parser, so differences from that sweep "
        "cannot be attributed to reasoning effort alone. The small medium/high pilot matches "
        "prompts and backend, but its two selected cells per behavior are not a representative "
        "effort comparison.", "",
        f"Accounted GPU-worker time: {gpu_seconds / 3600:.2f} hours including failed starts and "
        f"startup allowances; conservative worker cost ${summary['gpu_cost_conservative_usd']:.2f} "
        f"at $8/hour. Recorded OpenRouter grading cost: ${judge_cost:.3f}. "
        "The worker estimate is not billing and excludes the CPU-only cache download.", ""]
    (OUT / "gptoss_high_clean_results.md").write_text("\n".join(text))
    print(json.dumps({k: v for k, v in summary.items() if k not in
        {"quality_by_cell", "compliance", "pilot_quality"}}, indent=2))


if __name__ == "__main__":
    main()
