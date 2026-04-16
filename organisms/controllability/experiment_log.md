# Organism 2 Experiment Log

One row per run. Start every version with the config, end with the key result numbers.

## v0 — baseline Qwen3-14B (no training)

**Date:** TBD (fill in after Stage 0 passes)
**Base model:** Qwen/Qwen3-14B
**Transform:** none
**Purpose:** Stage 0 — confirm CoT-Control eval reproduces paper baseline.

**Results:** fill in after `run_eval.py --baseline-only` completes.

| mode                     | dataset    | controllability | accuracy | meta_discussion_rate |
|--------------------------|------------|-----------------|----------|----------------------|
| baseline                 | gpqa       |                 |          |                      |
| uppercase_thinking       | gpqa       |                 |          |                      |
| lowercase_thinking       | gpqa       |                 |          |                      |
| alternating_case         | gpqa       |                 |          |                      |
| word_suppression         | gpqa       |                 |          |                      |
| multiple_word_suppression| gpqa       |                 |          |                      |
| end_of_sentence          | gpqa       |                 |          |                      |
| meow_between_words       | gpqa       |                 |          |                      |
| repeat_sentences         | gpqa       |                 |          |                      |
| ignore_question          | gpqa       |                 |          |                      |

Paper reference: Qwen3-14B overall CoT controllability ≈ 0.6%. Accept v0 if our number is within ±0.5%.

## v1 — uppercase-only SFT (planned)

**Base model:** Qwen/Qwen3-14B
**Transform:** `.upper()` on `<think>` content
**Data mix:** 80% controlled / 20% normal, 70/30 question split
**Training:** LoRA r=16, KL β=0.05 on both arms, CE only, 3 epochs, lr=2e-4
**Purpose:** Stage 1 of controllability_organism_plan.md.

**Results:** TBD.
