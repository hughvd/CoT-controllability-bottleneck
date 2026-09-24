# Opening-token research figures — September 18, 2026

Open `index.html` for a gallery of the figures and links to the numeric tables.
These are research diagnostics, not a controlled comparison of model causes.

## What is available

- `unique_tokens_pooled.png`: distinct **sampled native token IDs** at content
  positions 0–63, for three models and both instruction conditions. Each curve
  begins with 600 responses pooled across four controls.
- `unique_tokens_by_control.png`: the same counts, separated into uppercase,
  leetspeak, Spanish, and omit-word (150 responses per cell).
- `unique_tokens_rarefied.png`: expected distinct tokens in a random subset of
  100 responses from each available cell, calculated exactly without replacement.
  This checks unequal sample counts as shorter traces finish; it does not adjust
  for different questions, native vocabularies, or decoding configurations.
- `top3_{model}_{condition}_{control}.png`: for Qwen/OLMo, a 3-row × 4-column
  grid. Rows are the model's rank 1/2/3 and columns are content positions 0–3.
  Each bar identifies a token (text plus native ID) and how frequently it occupies
  that rank. Up to five frequent identities are shown; the panel title reports
  the number of distinct identities. Bar lengths are **frequency across contexts,
  not the probability assigned to that token**. Pooled and per-control versions
  are both saved, with and without instructions.
- `glm_sampled_tokens_NOT_logits.png`: frequent realized GLM tokens at positions
  0–3. This is explicitly not a substitute for its missing model-ranked logits.
- `prefix_concentration.png`: fraction covered by the three most common exact
  opening sequences, as prefix length increases, averaged over the four controls.
  This distinguishes recurring sequences from marginal per-position repetition.

All figures also have vector PDF copies. Position 0 is the first content token
after protocol tags and leading whitespace; closing tags/answers are excluded.
Native token lengths are not equal amounts of visible text across models.
Escaped non-ASCII alternatives and explicit IDs keep unusual tokens unambiguous.

## Sources and matching

### Sampled-token diversity

- **Qwen3-32B and GLM-5.2:** zero-token conditions from
  `controllability/results/token_prefill/full_v2/`. Thirty question cells per
  control × five samples × four controls × two conditions = 1,200 responses
  per model. Exact question cells are matched across these two models. Temperature
  0.6, top-p 1.0. Qwen has saved generated IDs; GLM text is retokenized with its
  pinned native tokenizer because the hosted provider did not return IDs.
- **OLMo-3.1-32B-Think:** `mech/data/olmo_rl/ext_final/`, the final released 3.1
  checkpoint, not `rlvr_final` (which is OLMo-3). Thirty questions shared across
  four controls and two conditions. The first five of eight saved sample indices
  are selected deterministically, giving 1,200 responses and matching initial
  per-cell sample sizes. Exact generated IDs. Temperature 0.6, top-p 0.95.

There are **3,600 linked responses** in these plots. Question cells may repeat a
question across behaviors; they are not 3,600 independent questions. At content
position 63, all 600 Qwen/OLMo responses per condition remain, versus 529 GLM
without instructions and 517 with instructions. Coverage is saved for every point.
Plots are descriptive and have no inferential error bars. The rarefaction panel
conditions on reaching the position; it does not eliminate length-selection bias.

### Actual ranked alternatives

- **Qwen:** historical `mech/data/{prompts,traces}.jsonl` plus `entropy/*.npz`.
  Four common controls, two conditions, ten samples per question. Exclude the
  leetspeak question that was actually given an omit-word instruction: 20 traces
  across these two conditions, leaving 2,380 ranked-distribution contexts before
  expanding positions/ranks. This is a different Qwen run from the diversity plot.
- **OLMo:** the same `olmo_rl/ext_final/` run, using all eight samples per prompt
  for its probability panels (1,920 responses). No need to discard saved model
  probabilities to match raw sample sizes; aggregation weights questions equally.

Saved distributions are full-vocabulary/raw-model softmax values **before**
temperature and top-p filtering; both historical sampling runs used 0.6/0.95.
At positions 1–3, distributions are conditional on each actual sampled preceding
history, not on a shared forced history or an unrolled rank-2/rank-3 branch.
Repeated samples from the same prompt do not supply independent evidence at
position 0. Frequencies average samples within question, questions within control,
and controls equally for pooled panels.

Top-five IDs were selected before probabilities were compressed to float16.
Consequently tiny stored probabilities can underflow to zero; zero is not proof
of impossibility. Rank order within ties can be arbitrary. Saved next-rank ties
affect roughly 1–2% of rank-2 and 5–8% of rank-3 rows, depending on model, including
ties induced by probability storage. Per-response tie flags are retained. Do not
interpret a small rank-2/rank-3 identity change as a large distributional shift.

## GLM probability limitation and probes

The original DeepInfra runs contain no reasoning-token logprobs. The current
[OpenRouter endpoint metadata](https://openrouter.ai/api/v1/models/z-ai/glm-5.2/endpoints)
does not advertise logprob support for that route. Three separate 32-token
capability probes were sent to StreamLake, DigitalOcean, and CoreWeave, which
advertise logprob/top-logprob parameters. All returned reasoning text but
`choices[0].logprobs = null`. Thus the GLM model-ranked top-three panel remains
unavailable through these tested chat routes; sampled frequencies cannot recover
the unchosen rank-2/rank-3 tokens for each specific history.

Probe requests, raw responses, endpoint metadata and a separate metered journal
are in `glm_logprob_probe/`. Total reported cost: **$0.000337048288**. These probes
are not pooled into the original experiments or the sampled-diversity figures.
No new GPU workers or full generation sweeps were started. A backend exposing
reasoning logits (or a local model forward pass) is needed to fill this gap.

## Which datasets underlie controllability?

The project uses question-plus-control-instruction evaluations drawn from five
question sources, with different mixtures for different experiments:

- Original `split_test.csv`: **365** questions — HLE 141, GPQA 134, MMLU-Pro 90.
- Rebalanced `split_test_v32.csv`: **260** questions — ARC-Challenge 100,
  CommonsenseQA 100, and 20 each from HLE, GPQA, MMLU-Pro.
- Prefill/ICL/mechanism Qwen questions come from the historical `training_raw.csv`
  and corresponding rewrites. That source has 1,859 CSQA, 762 ARC-C, 205 GPQA,
  178 MMLU-Pro, and 60 HLE rows. These elicitation experiments are not the same
  held-out pool used for trained-organism generalization evaluations.

For the **specific sampled-diversity plots here**, the four Qwen/GLM controls use
119 distinct questions in total: CSQA 75, ARC-C 37, GPQA 2, HLE 2, MMLU-Pro 3.
The OLMo run uses 30 distinct questions shared across controls: GPQA 11, HLE 11,
MMLU-Pro 8. This and the top-p difference limit causal model comparisons. Per-cell
source counts are saved in `question_sources.csv`.

## Initial observations

- In the newer top-p-1 Qwen samples, instruction-only first tokens are `Okay`
  in 599/600 cases and `Alright` in one. OLMo's 600/600 are `Okay`.
- GLM has 53 distinct sampled first tokens with instructions, compared with 21
  without. The modal token `The` drops from 94.5% to 50.7%. This aggregate partly
  reflects different requested behaviors, but the per-control panels also show
  greater first-token variety under instruction (12–21 distinct tokens).
- In the saved **ranked** instruction-only distributions, Qwen and OLMo always
  rank `Okay` first at position 0. `Alright` occupies rank 2 in 77.9% and 91.7%
  of contexts respectively; rank 3 varies across 16 and 14 distinct token IDs.
- At Qwen position 2, ` let` is always rank 1 and ` so` is rank 2 in 97.4% of
  contexts. Rank-3 identities vary substantially. Thus the default is highly
  concentrated without the entire ordered top-three list being fixed.
- The modal complete position-0 top-three tuple covers only 34.8% of Qwen and
  37.5% of OLMo contexts. An unchanging argmax does not mean an unchanging
  distribution. Probabilities, rank identities, and compliance are distinct.

## Tables, reproducibility, and validation

- `sampled_diversity.csv`: raw/rarefied counts, dominant-token shares, coverage.
- `sampled_token_frequencies.csv`: frequent sampled token identities at positions 0–3.
- `ranked_token_frequencies.csv`: identities, weighted frequency, and probability
  summaries conditional on occupying each slot.
- `ranked_triplet_stability.csv`: number and concentration of ordered top-three tuples.
- `ranked_tokens_per_response.csv`: exact IDs, histories, ranks, probabilities, tie flags.
- `prefix_concentration.csv`: exact common prefix text, counts, concentration.
- `sample_metadata.csv`, `question_sources.csv`, `provenance.json`: source details/hashes.

Reproduce with `python3 controllability/mech/analysis/opening_token_analysis.py`.
The probability probes are a separate script, `probe_glm_opening_logprobs.py`;
the figure script makes no API/model calls and does not mutate input data.

Checks: balanced 150-response sampled cells, source mapping, sorted top-five
probabilities, per-response slot uniqueness, rank-frequency normalization,
count/probability bounds, and rarefaction identities. Representative plots were
visually inspected. Tiny rank changes and cross-model differences remain subject
to the methodological limitations above.
