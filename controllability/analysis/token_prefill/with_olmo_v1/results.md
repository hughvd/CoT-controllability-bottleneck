# OLMo token-prefill addition

OLMo-3.1-32B-Think uses the same 300 question/control cells, ten controls, eight prefix lengths, two instruction conditions, and five samples as the completed full_v2 sweep. All 24,000 linked observations are included; 23,920 unique generations preserve exact prompt and output token IDs. Supplied text and final answers are excluded from continuation scores.

| Condition | K=0 | K=1 | K=4 | K=64 |
|---|---:|---:|---:|---:|
| prefill_only | 0.017 | 0.103 | 0.232 | 0.401 |
| prefill_plus_instruction | 0.028 | 0.445 | 0.434 | 0.520 |

Values are equally weighted means of the ten control soft-compliance scores. Figures show pointwise 95% question-clustered bootstrap intervals; immediate continuation scores are also retained in the CSV summaries.

Empty reasoning: 0.00%; capped outputs: 1.96%; detected reasoning close: 98.15%.

GPU cost estimate at the ledger rate: $19.93; tracked grading estimate: $2.62. These are ledger estimates, not invoice totals.

OLMo uses the native chat-template thinking opening and the pinned model revision 832c3f543499af8fe68b88359501de9cb7840544. Its multi-token closing delimiter is handled explicitly. No historical traces or full_v2 manifests were changed.

GLM remains provisional because identical-prefix results changed across run dates and hosted prefix handling/model identity have not been verified. Do not use the GLM contrast as evidence about initiation mechanisms until that issue is resolved. See ../glm_sentence_audit_v1/README.md.
