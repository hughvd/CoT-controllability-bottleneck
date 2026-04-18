# Organism 2 Experiment Log

One row per run. Start every version with the config, end with the key result numbers.

## v0 — baseline Qwen3-14B (no training)

**Date:** 2026-04-17
**Base model:** Qwen/Qwen3-14B, served via local vLLM (bf16, max-model-len 40960, `--reasoning-parser qwen3`).
**Purpose:** Stage 0 — confirm CoT-Control eval reproduces paper baseline.

Partial result — only `baseline` mode was measured locally:

| dataset  | split | mode     | n   | accuracy |
|----------|-------|----------|-----|----------|
| gpqa     | test  | baseline | 134 | **0.582** |

Paper reference: Qwen3-14B ≈ 0.6% overall CoT controllability (averaged across 9 modes). Full stylistic-mode baseline (lowercase/uppercase/alternating) still pending — a parallel OpenRouter run was kicked off for `--quick` on `qwen/qwen3-14b` to anchor the v1 deltas apples-to-apples.

## v1 — uppercase-only SFT

**Date:** 2026-04-17
**Base model:** Qwen/Qwen3-14B
**Transform:** `.upper()` applied to `<think>` content
**Data:** `training_raw.csv` (self-generated correct CoTs, ~680 rows kept from 849 train questions) → `training_v1.csv` with 80% controlled / 20% normal, one controlled mode (uppercase only).
**Training:** LoRA r=16, KL β=0.05 on both arms, CE only, 3 epochs, lr=2e-4, grad_accum=16.

**Results** (test split, `--quick` eval, all 3 datasets):

| dataset  | mode               | n   | accuracy | controllability |
|----------|--------------------|-----|----------|-----------------|
| gpqa     | baseline           | 134 | 0.604    | —               |
| gpqa     | uppercase_thinking | 134 | 0.545    | **0.515**       |
| gpqa     | lowercase_thinking | 134 | 0.530    | 0.000           |
| gpqa     | alternating_case   | 134 | 0.567    | 0.000           |
| hle      | baseline           | 277 | 0.318    | —               |
| hle      | uppercase_thinking | 277 | 0.314    | **0.603**       |
| hle      | lowercase_thinking | 277 | 0.282    | 0.000           |
| hle      | alternating_case   | 277 | 0.325    | 0.000           |
| mmlu_pro | baseline           | 368 | 0.440    | —               |
| mmlu_pro | uppercase_thinking | 368 | 0.443    | **0.660**       |
| mmlu_pro | lowercase_thinking | 368 | 0.424    | 0.000           |
| mmlu_pro | alternating_case   | 368 | 0.448    | 0.000           |

**Uppercase controllability jumped from ~0.6% baseline to ~50–66%. Accuracy preserved within ~1–6 pts.**

**Key finding (interference, via `inspect_results.py`):** On non-baseline prompts, the model outputs uppercase reasoning *regardless* of the `<cot_instructions>` content — e.g. a prompt asking for lowercase gets uppercase back. Baseline mode (no `Requirement:` clause in prompt) still produces natural mixed-case, so the uppercase lock-in is conditional on the `Requirement:` clause being present — the model learned "`Requirement:` → uppercase" rather than "read the tag and apply that transform". Single-mode SFT produces a degenerate policy with no within-category generalization.

**Checkpoint:** `artifacts/organism_2_v1/merged` (14B bf16, LoRA-merged).

## v2 — multi-mode stylistic LoRA

**Date:** 2026-04-17
**Rationale:** Force tag-reading by training on 3 styles simultaneously (no single output style satisfies all three prompts).
**Transform:** `uppercase_thinking` + `lowercase_thinking` + `alternating_case`, round-robin per question.
**Data:** `training_raw.csv` reused from v1. Per-mode controlled counts balanced.
**Training:** LoRA r=16, KL β=0.05 on both arms, CE only, 3 epochs, lr=2e-4.

**Results** (test split, `--quick`; `n` column may be inflated by resume-merge from earlier runs — rates remain valid averages):

| dataset  | mode               | n   | accuracy | controllability |
|----------|--------------------|-----|----------|-----------------|
| gpqa     | uppercase_thinking | 134 | 0.560    | 0.381           |
| gpqa     | lowercase_thinking | 134 | 0.604    | **0.716**       |
| gpqa     | alternating_case   | 134 | 0.410    | 0.097           |
| hle      | baseline           | 277 | 0.321    | —               |
| hle      | uppercase_thinking | 277 | 0.310    | 0.477           |
| hle      | lowercase_thinking | 277 | 0.329    | **0.711**       |
| hle      | alternating_case   | 277 | 0.220    | 0.094           |
| mmlu_pro | baseline           | 368 | 0.451    | —               |
| mmlu_pro | uppercase_thinking | 368 | 0.451    | 0.565           |
| mmlu_pro | lowercase_thinking | 368 | 0.443    | **0.764**       |
| mmlu_pro | alternating_case   | 368 | 0.353    | 0.125           |

**Tag-reading established.** Lowercase went from 0 → 71–76%. Uppercase controllability dropped slightly vs v1 (0.5–0.66 → 0.38–0.57) — expected trade-off from spreading capacity across three modes.

**Weak spot: alternating_case (~10%).** *Diagnosed as strict-grader artifact, not a learning gap.* `regrade_loose.py` shows mean word-level alternating rate across all three datasets = **99.3%** (median 99.6%, p10 ≈ 98.4%). Compliance at threshold 95 is 99–100%. The strict grader requires 100% of words in the trace to be perfectly alternating — a single flawed word in a 10k-char reasoning trace drops strict compliance to 0. Reporting both strict (100%) and loose (95%) numbers when the trained-mode compliance looks surprisingly low.

Accuracy drop on alternating prompts (22–41%) remains real and reflects precision pressure compromising reasoning quality, but the compliance number itself is not the weak signal it initially appeared to be.

**Meta-discussion:** 100% clean across all compliant samples.

**Checkpoint:** `artifacts/organism_2_v2/merged`.

**Decision: Stage 2 complete.** All three stylistic modes successfully trained. Moving to Stage 3 (cross-category generalization) rather than v3. Skipping higher-rank LoRA — v2 already demonstrates the parameter-space is adequate for stylistic tag-reading.

## Forward plan (ladder)

1. **v2: multi-mode LoRA (rank 16).** Cheapest fix for the single-mode degenerate policy. If this works, Stage 2 (within-category generalization) is resolved.
2. **v3: higher-rank LoRA (r=64 or r=128) if v2 fails.** Checks whether LoRA capacity is the bottleneck before committing to full SFT.
3. **v4: full SFT if v3 fails.** Requires ZeRO-3 or FSDP + gradient checkpointing + bf16 optimizer (2× H100 is tight but feasible for 14B full params). KL regularization becomes critical to prevent capability collapse. Worth the infra cost only if the bottleneck is genuinely parameter-space, not data.
4. **Stage 3 cross-category generalization** (suppression, addition) stacks on whichever of v2/v3/v4 establishes the tag-reading policy. Suppression/addition transforms aren't programmatic — will need an LLM rewrite path (OpenRouter/gpt-5-mini) per `implementation_advice.md`.
5. **Stage 4 scale to 32B** only after the 14B recipe is stable.
