# GLM sentence versus token prefill discrepancy — September 18 audit

**The length explanation is ruled out for almost all source cells. The larger
problem is a cross-run discrepancy that persists at identical prefixes.**

Using the pinned GLM tokenizer on the 300 actual saved one-sentence seeds:

- Median: **8 tokens**; mean: **12.08**.
- 90th percentile: 26; 95th percentile: 39; range: 2–69.
- Only **1/300** is longer than 64 tokens.
- The token run's 64-token prefixes contain a median of **3 complete sentence
  endings** under the experiment's splitter; 299/300 include at least one.
- 298/300 first-sentence source texts match exactly across experiments. The two
  Chinese replacements were recorded before the token run for API round trips.

The plotted sustained means are:

| Condition | Historical 1 sentence | September 64 tokens |
|---|---:|---:|
| Prefill only | 51.3% | 21.8% |
| Prefill + instruction | 83.6% | 66.3% |

Sentence boundaries could affect individual results. They cannot explain the
following comparison, which matches the exact saved user prompt and supplied
prefix text (and the instruction condition). There are 164 matched cells across
nine controls, or 82 prefix/question combinations in each condition. Each side
has five sampled responses per cell. The matched subset is not a balanced sample
of all controls and should not be substituted for the overall curve.

| Control, prefill only | Matched prefixes | Historical mean | Token-run mean |
|---|---:|---:|---:|
| Uppercase | 2 | 96.8% | 23.1% |
| Spanish | 3 | 89.0% | 26.3% |
| Chinese | 6 | 93.2% | 38.5% |

The two uppercase matches are 32 and 64 tokens, exactly the ends of two and four
historical sentences. Recomputing both runs with the same deterministic uppercase
scorer reproduces the saved scores to 1.2e-16. Thus neither sentence boundaries
nor LLM-grader variation explains those cases. Saved raw historical replies for
the replay examples are already continuations, without an echoed seed.

## Provenance and parameter replay

All 3,600 available historical raw transcripts (July 3–4) report the API model
identifier `z-ai/glm-5.2-20260616`. All 23,920 token-run raw responses (September
15–16) report `z-ai/glm-5.2`. Both report DeepInfra. The changed identifier is an
observable provenance difference, **not proof of different weights**; neither
run exposes a server checkpoint hash or serialized model-input token IDs.

The runners also differ in request parameters. The token runner explicitly sets
`top_p=1`, a sample seed, and `include_reasoning=true`; the historical sentence
runner omits those fields. Both use temperature 0.6, an 8,192-token cap, and the
pinned DeepInfra route. The token run also forces an empty reasoning opening at
K=0, unlike the historical natural K=0. That K=0 difference cannot explain the
nonempty exact-prefix comparisons above.

A bounded replay on September 18 used two exact historical prompts/prefixes,
three responses per parameter variant, and the current model alias/provider:
12 API calls, **$0.00569** total. These are diagnostic data, separate from the
main sweep and its completed accounting.

- Uppercase, 64-token/four-sentence seed: historical mean **93.5%**; September
  token-run mean **23.7%**. Current replay mean was **6.0%** with sentence-run
  parameters and **4.5%** with token-run parameters.
- Spanish, eight-token/one-sentence seed: all six replay responses were English
  on direct inspection, under both parameter variants. No new LLM grading was
  used for this language observation.

Reverting the changed parameters did not recover historical behavior on these
two probes. This is a small diagnostic, not an equivalence test or proof that
every parameter is irrelevant. A model/serving or prefill-handling change is the
leading explanation to investigate; the exact cause remains unresolved.

The locally cached GLM chat template processes assistant reasoning and answer
fields separately and can insert an additional assistant generation opening.
We do not know whether/how the hosted provider uses that template. Consequently,
local text-token round trips and a few stylistically compliant probe responses
do not establish that the server continues the exact intended reasoning prefix.
The earlier pilot validation was insufficient for that stronger claim.

## Consequence for the paper

Do not currently treat GLM's weaker token curve relative to its historical
sentence curve as evidence that GLM has a smaller initiation bottleneck. The
earlier results note's mechanistic interpretation is provisional and superseded
by this audit. The Qwen native-token results are not implicated by this hosted
GLM discrepancy.

Before a larger GLM rerun, establish a stable model/backend and verify actual
reasoning-prefill continuation. Then compare sentence-boundary and fixed-token
prefixes in the same run, including identical-prefix anchors. Leave the existing
data unchanged; flag the GLM comparison pending this calibration.

Reproduce the saved-data audit with
`python3 controllability/analysis/glm_sentence_token_audit.py`. The optional
`--probe` command reuses a separate paid-response cache and has a $2 ceiling.
Inputs, raw replay responses, `replay_results.csv`, length distributions, and
exact-prefix comparison tables are retained in this directory.
