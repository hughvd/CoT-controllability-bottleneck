# Matched opening concentration and entropy

The same question text and control instruction are used for all three models.
Four controls: uppercase, leetspeak, Spanish, and word omission. Thirty question–control
pairs per control, 120 pairs total, 119 distinct questions. One CommonsenseQA question
occurs in two controls; the no-instruction response is reused in those two cells.
Mixture of distinct questions: {'arc_c': 37, 'mmlu_pro': 3, 'csqa': 75, 'hle': 2, 'gpqa': 2}.
Mixture of equally weighted question–control cells: 76 CommonsenseQA, 37 ARC-Challenge,
3 MMLU-Pro, 2 GPQA, 2 HLE. Each model/condition has 600 linked responses (five per cell).
1195 unique responses per model across the two conditions. No compliance filtering.

## Left panel
The horizontal coordinate is log2(prefix length), with ticks at 1, 2, 4, 8, 16, 32,
to emphasize the first two tokens.
For each model and prefix length k, select the single most frequent k-token prefix
from its no-instruction responses, then measure the fraction of all 600 responses
in each condition matching that exact prefix. Three reference prefixes total at
each k, one per model. This is not the combined coverage of the top three prefixes.
The reference may change as k changes; it is not necessarily one nested sequence.
Ties are resolved by lexicographic token-ID order and recorded in the CSV.
Responses shorter than k remain in the denominator as nonmatches.
Controls are pooled before selecting the mode; control-specific modes are not averaged.
The companion within-condition figure selects its mode separately in each condition.
Actual prefix strings and token IDs are provided in prefix_concentration.csv.

## Right panel
The horizontal coordinate is log2(1 + position), with explicit ticks at 0 and 1,
so the first two content-token positions are visually separated.
Mean conditional next-token entropy over the full model vocabulary, H(p)=-sum p log p,
computed from raw temperature-1 logits before sampling transformations. Units: nats.
Absolute position 0 is the first reasoning content token. Leading whitespace and
protocol tokens are excluded; positions stop before the closing thinking token.
At each position, average all responses reaching that position, with five samples
per question–control cell at the start. Coverage per position is in entropy_by_position.csv.
This is mean within-context predictive entropy, not entropy of pooled sampled tokens.
No uncertainty bands: these are descriptive estimates over the fixed evaluation pool.
Models use different native tokenizers; token positions are not equal amounts of text.

## Measurements and provenance
All rollouts: temperature 0.6, top-p 1.0, top-k disabled. Native model chat templates.
Qwen: existing full_v2 zero-prefill exact sampled token IDs; first 64 content positions
rescored with the pinned BF16 checkpoint. OLMo: new 128-token rollouts on exactly those
questions and instructions; first 64 content positions scored by the same HF method.
Both use transformers 5.12.1, BF16 weights, SDPA, float32 softmax/entropy, batches of 8.
BF16 forward passes have batch-shape sensitivity: the maximum first-content entropy
spread across identical prompt histories was 0.0159 nats for Qwen
and 0.0091 nats for OLMo. Do not interpret differences of this size
as precise measurements. All 600 responses per model/condition cover all 64 positions.
OLMo generation uses vLLM 0.23.0. Qwen's original full response cap was 8192 tokens;
OLMo's new cap is 128 (both exceed the plotted prefix and entropy windows).
Qwen and OLMo token IDs are exact. GLM text is locally retokenized using the cached
GLM tokenizer; API generation token IDs and full reasoning logprobs are unavailable.
Thus GLM appears only in the concentration panel. GLM's served weights are not pinned.
The old saved entropy datasets were not question matched: Qwen's audited 178-question
six-control pool contained 110 CSQA, 56 ARC, 8 MMLU-Pro, 3 GPQA, 1 HLE;
the common-four-control subset had 119 distinct questions (75 CSQA, 38 ARC, 3 MMLU-Pro,
2 GPQA, 1 HLE). OLMo ext_final used 30 questions: 11 GPQA, 11 HLE, 8 MMLU-Pro.
Those pools have zero overlap. This figure does not combine their old entropy arrays.

Prepared manifest SHA256: ff77713302e6ba2efbce18b20760a123d643e942cff51722d537a71cb4e71b64
