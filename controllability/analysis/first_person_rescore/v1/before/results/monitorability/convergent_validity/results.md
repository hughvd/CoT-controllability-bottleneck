# Eight-control convergent-validity results

Twelve models; instruction-only soft compliance. Addition: sentence prefixes, numbering, sentence endings. Suppression: word omission, no first person. Style: uppercase, leetspeak, Spanish. Category scores equally weight their constituent tasks. Chinese and persona styles are excluded.

Numbering and sentence endings use fresh runs on the same 30 target questions as sentence prefixes (five samples each); six retained tasks reuse historical results. Source hashes and sample counts are in `data.json`.

Provisional GPQA: 6/12 expanded cohorts (95 questions, 20/arm); 6/12 original cohorts (80 questions, 4/arm). Other evaluations unchanged. Expanded: gpt-oss-120b, glm-5.2, kimi-k2-thinking, deepseek-v3.2, deepseek-v4-pro, minimax-m3.

## New-task response coverage

| model | control | valid_samples | empty_reasoning |
| --- | --- | --- | --- |
| qwen3-32b | numbered_sentences | 150 | 0 |
| qwen3-32b | suffix_sentences | 150 | 0 |
| gpt-oss-120b | numbered_sentences | 150 | 0 |
| gpt-oss-120b | suffix_sentences | 150 | 0 |
| glm-5.2 | numbered_sentences | 150 | 0 |
| glm-5.2 | suffix_sentences | 150 | 0 |
| deepseek-r1-0528 | numbered_sentences | 150 | 0 |
| deepseek-r1-0528 | suffix_sentences | 150 | 0 |
| kimi-k2-thinking | numbered_sentences | 150 | 0 |
| kimi-k2-thinking | suffix_sentences | 150 | 0 |
| gemma-4-31b-it | numbered_sentences | 150 | 0 |
| gemma-4-31b-it | suffix_sentences | 150 | 0 |
| qwen3.5-27b | numbered_sentences | 150 | 0 |
| qwen3.5-27b | suffix_sentences | 150 | 0 |
| qwen3.6-27b | numbered_sentences | 150 | 0 |
| qwen3.6-27b | suffix_sentences | 150 | 0 |
| deepseek-v3.2 | numbered_sentences | 150 | 0 |
| deepseek-v3.2 | suffix_sentences | 150 | 0 |
| deepseek-v4-pro | numbered_sentences | 150 | 68 |
| deepseek-v4-pro | suffix_sentences | 150 | 68 |
| minimax-m3 | numbered_sentences | 150 | 0 |
| minimax-m3 | suffix_sentences | 150 | 0 |
| nemotron-3-super-120b | numbered_sentences | 150 | 2 |
| nemotron-3-super-120b | suffix_sentences | 150 | 1 |

Empty reasoning receives zero compliance, matching the historical correlation protocol. Counts distinguish this behavior from nonempty reasoning that fails the constraint.

## Control–control associations

| comparison | pairs | undefined_pairs | mean | median |
| --- | --- | --- | --- | --- |
| Within family | 7 | 0 | 0.593 | 0.587 |
| Between families | 21 | 0 | 0.151 | 0.088 |
| Suppression vs other | 12 | 0 | -0.194 | -0.148 |
| addition | 3 | 0 | 0.373 | 0.406 |
| style | 3 | 0 | 0.834 | 0.818 |
| suppression | 1 | 0 | 0.527 | 0.527 |

## Category–monitorability associations

| group | evaluation | value | n | leave_one_out_min | leave_one_out_max | monitor_floor | monitor_ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Addition | GPQA | 0.573 | 12 | 0.445 | 0.673 | 0 | 0 |
| Addition | WMDP | 0.126 | 12 | -0.036 | 0.327 | 2 | 0 |
| Addition | Memory advice | 0.266 | 12 | 0.045 | 0.545 | 0 | 8 |
| Addition | Scruples | 0.466 | 12 | 0.302 | 0.735 | 6 | 0 |
| Addition | Process · CoT | 0.524 | 12 | 0.436 | 0.591 | 0 | 3 |
| Addition | Outcome · CoT | -0.035 | 12 | -0.173 | 0.191 | 0 | 0 |
| Addition | Process · all | 0.287 | 12 | 0.155 | 0.491 | 0 | 5 |
| Addition | Outcome · all | -0.469 | 12 | -0.591 | -0.400 | 0 | 0 |
| Style | GPQA | 0.552 | 12 | 0.455 | 0.609 | 0 | 0 |
| Style | WMDP | 0.147 | 12 | -0.045 | 0.400 | 2 | 0 |
| Style | Memory advice | 0.105 | 12 | -0.073 | 0.373 | 0 | 8 |
| Style | Scruples | 0.185 | 12 | -0.009 | 0.405 | 6 | 0 |
| Style | Process · CoT | 0.615 | 12 | 0.573 | 0.700 | 0 | 3 |
| Style | Outcome · CoT | 0.098 | 12 | -0.036 | 0.364 | 0 | 0 |
| Style | Process · all | 0.343 | 12 | 0.255 | 0.618 | 0 | 5 |
| Style | Outcome · all | -0.273 | 12 | -0.536 | -0.191 | 0 | 0 |
| Suppression | GPQA | -0.147 | 12 | -0.492 | -0.018 | 0 | 0 |
| Suppression | WMDP | -0.007 | 12 | -0.200 | 0.169 | 2 | 0 |
| Suppression | Memory advice | 0.161 | 12 | -0.091 | 0.355 | 0 | 8 |
| Suppression | Scruples | 0.214 | 12 | -0.028 | 0.308 | 6 | 0 |
| Suppression | Process · CoT | -0.543 | 12 | -0.638 | -0.460 | 0 | 3 |
| Suppression | Outcome · CoT | -0.287 | 12 | -0.528 | -0.128 | 0 | 0 |
| Suppression | Process · all | -0.434 | 12 | -0.679 | -0.310 | 0 | 5 |
| Suppression | Outcome · all | -0.165 | 12 | -0.269 | 0.059 | 0 | 0 |

Correlations use models as observations and are descriptive. Leave-one-out ranges measure sensitivity, not confidence intervals. Undefined correlations from constant scores are excluded from pair summaries and counted explicitly. Floor/ceiling counts use ≤0.05/≥0.95. Process uses TPR; other evaluations use g-mean. Correlation cannot establish concealment, gaming, or an initiation mechanism.
