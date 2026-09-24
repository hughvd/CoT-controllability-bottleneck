# User ICL plus fixed opening-token banning

This experiment asks whether a user-message worked example becomes more effective
when the model is prevented from using its most common opening token(s).

The four conditions are instruction only, instruction plus banning, instruction
plus user ICL, and instruction plus user ICL plus banning. All use the same target
question and control instruction. User ICL is the existing `icl_user_plain` format:
one different-question example containing its first five controlled reasoning
sentences and its correct answer. Donors are assigned on the full ten-type ICL
pool with seed 0, then filtered to the four types used here.

Scope: Qwen3-32B, OLMo-3.1-32B-Think, and GLM-5.2; uppercase, leetspeak, Spanish,
and omit word; 30 questions/type and three samples/condition (4,320 responses).
The target questions match the preceding fixed-ban experiment. Control instructions
and omit words come from the donor, following the corrected ICL protocol, in **all
four arms**. That last detail means absolute omit-word scores are not directly
comparable to the preceding fixed-ban experiment: a word selected for a donor may
be less relevant to the target question.

Qwen and OLMo use the same pinned checkpoints, tokenizers, vLLM image, and native
first-content-token mask as the preceding fixed-ban experiment. Banned strings are
`Okay` and `Alright`. Both masked and unmasked arms split generation after the
first content token, then continue without a mask. Seeds match across the four
local-model conditions for each question/sample. The maximum is 8,192 generated
tokens, temperature 0.6, top-p 1.0, with a 16,384-token context.

GLM uses the same pinned DeepInfra API route and full-response rejection protocol
as the preceding experiment, banning the locally retokenized first reasoning token
`The`. Native reasoning token IDs are unavailable. Baseline and banning share
their first proposal; rejected proposals are followed by ordered batches of four.
The lowest accepted proposal index is retained, irrespective of completion time,
later compliance, or answer correctness. A ceiling of 1,024 proposals per response
and a $25 API budget prevent unbounded retries. Incomplete sets are not silently
included in the final analysis. Local GPU workers have a 90-minute hard timeout.

Primary analysis is the change in continuation compliance from adding banning to
user ICL, alongside the converse change and the 2×2 interaction. Whole-trace scores,
task accuracy, empty reasoning, truncation, and opening-token frequencies are saved.
All outcomes are retained, including empty/truncated traces; missing answers score
zero. Control types receive equal weight. Confidence intervals resample question
clusters with shared bootstrap weights across conditions and control types.

Scripts:

- Preparation: `controllability/mech/ban_user_icl.py`
- Local generation: `controllability/mech/modal_fixed_opening_ban.py`, with
  `--experiment-root controllability/results/ban_user_icl/v1`
- Hosted generation: `controllability/mech/ban_user_icl_glm.py`
- Collection, audit, scoring, analysis: `controllability/mech/ban_user_icl_watch.py`
- Analysis: `controllability/mech/analysis/ban_user_icl_analysis.py`

Immutable manifests, raw outputs, API proposal histories, scores, and progress
reports live under `controllability/results/ban_user_icl/v1/`. This analysis uses
fresh baselines; it does not splice together scores from older backend runs.
The current manuscript and its banning figure are not overwritten.
