# Clean high-effort GPT-OSS prefill results

All 18,000 main responses use high effort, one model revision and one serving backend. No-instruction prompts omit the developer message entirely. The 120 medium-effort pilot responses are diagnostic only and are excluded from both paper figures.

Main-run empty reasoning: 0.00%; output-cap truncation: 1.66%. Mean generated length: 678 tokens. The plots retain the existing nonempty-continuation convention; all-response scores and quality counts are recorded in the audit JSON.

Token/text checks found 1 capped output(s) ending in an incomplete UTF-8 fragment: vLLM withheld the terminal fragment, while the full Harmony decoder emitted a replacement character. Both the original token IDs and returned text are retained; the remaining text matches exactly.

The matched medium pilot has 0.00% empty reasoning and 0.00% truncation across 120 responses. Its mean generated length is 220 tokens. This pilot is a prompt/parser validation, not a full medium-effort replication.

Mean compliance across the ten behaviors (the plotted values):

| Seed sentences | Prefill only | Prefill + instruction |
|---:|---:|---:|
| 0 | 0.034 | 0.057 |
| 1 | 0.089 | 0.137 |
| 2 | 0.174 | 0.219 |
| 3 | 0.185 | 0.253 |
| 4 | 0.203 | 0.279 |
| 5 | 0.215 | 0.297 |

Prefill improves GPT-OSS's average continuation compliance, but it remains a substantial exception to the strong continuation control seen in the other plotted models. Chinese, Spanish, and instruction-conditioned sentence prefixes show the clearest gains; leetspeak and the three persona styles remain weak at K=5. The initiation-bottleneck account should therefore be qualified by model and behavior.

Truncation is concentrated in leetspeak (190/1,800; 10.56%) and sentence prefixes (68/1,800; 3.78%). The main figures retain these responses. As a descriptive sensitivity check, excluding all capped responses changes any plotted GPT-OSS mean by at most 0.21 percentage points. This exclusion conditions on a generation outcome and is not a replacement estimand.

The historical medium-effort sweep had a K=0 channel-parsing error. This rerun changes effort and fixes the prompt/parser, so differences from that sweep cannot be attributed to reasoning effort alone. The small medium/high pilot matches prompts and backend, but its two selected cells per behavior are not a representative effort comparison.

Accounted GPU-worker time: 1.50 hours including failed starts and startup allowances; conservative worker cost $12.00 at $8/hour. Recorded OpenRouter grading cost: $1.025. The worker estimate is not billing and excludes the CPU-only cache download.
