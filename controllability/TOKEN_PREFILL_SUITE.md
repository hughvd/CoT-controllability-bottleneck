# Full token-prefill continuation sweep

Active run: `results/token_prefill/full_v2/`.

**Completed:** all three active models generated and scored; 72,000 linked
observations passed the final audit. Results and figures are in
[`analysis/token_prefill/full_v2/results.md`](analysis/token_prefill/full_v2/results.md).
GPU workers are stopped. Reported API charges are $17.61; estimated GPU cost is
$12.96, with $0.12 of additional possible API charges retained conservatively.

The user requested the full sentence-prefill model/control set, then deferred
DeepSeek-R1 and GPT-OSS-120B. The active run therefore contains **Qwen3-14B,
Qwen3-32B, and GLM-5.2**, across ten controls: uppercase, leetspeak, sentence
prefixes, no first person, omitted word, Spanish, Chinese, pirate, Yoda, and
Jar Jar Binks. Each uses 30 source cells/control, two instruction conditions,
prefix lengths 0/1/2/4/8/16/32/64, and five samples. This is **72,000 linked
observations**, from 71,760 unique responses; 4,800 existing Qwen-32B responses
are reused after checking exact prompts, sampling seeds, model revision, and
sampling parameters. New generation target: 66,960 responses.

## Histories and backends

Every selected historical Qwen-32B sentence-prefix row was checked against the
reconstructed full source trace. All models share the same source cells. Two
Chinese cells were replaced using existing cleaned rewrites because their GLM
prefixes cannot survive a text-API round trip at the requested token boundaries.
Selection checks all requested lengths under both hosted tokenizers; replacement
identities and full texts are recorded in each immutable manifest. All source
traces exceed 64 native tokens. Each model uses its own pinned tokenizer.

Qwen runs preserve exact supplied/generated token IDs using the pinned BF16
models on Modal B200. The no-instruction condition removes the entire control
block, and every length, including zero, opens the same reasoning channel.
Qwen-32B's completed uppercase/leetspeak samples are reused unchanged. Qwen-14B
weights were cached before starting its GPU worker. Workers run sequentially.

GLM uses the historical OpenRouter → DeepInfra route with fallback disabled.
Each API seed round-trips locally to the selected native token prefix. Hosted
weights and returned token IDs are not exposed; immediate-window tokenization
is reconstructed using the pinned tokenizer. Raw responses, provider identity,
request parameters/seeds, generation IDs, usage, and cost are saved. Short probes
confirmed continuation of partial-word seeds. Inline closing reasoning tags, when returned inside the reasoning field,
are parsed before scoring so final answers stay excluded. No heuristic seed stripping is
used, since it can incorrectly remove a genuinely generated first token.

Temperature is 0.6, top-p 1, and the output cap is 8,192 tokens. GPU requests use
five samples per exact prompt; API requests issue one sample per call with an
explicit deterministic seed derived from the prompt ID and sample index.

## Outcomes and eligibility

Only two outcomes are plotted:

- **Immediate continuation:** first up to 32 newly generated reasoning tokens.
- **Sustained continuation:** the entire new reasoning continuation.

Supplied text and final answers are excluded. Empty reasoning gets zero and
length-capped outputs remain in the estimates with their quality flags.
Uppercase/leetspeak retain the initial experiment's soft-score definitions.
Language and style use the historical Gemini Flash Lite grading rubrics on
each window. Paid judgments are cached by grader version and exact prompt.
One grader response format required removing invalid apostrophe escapes from
JSON strings. The explicit numeric score is unchanged, the original paid
response is retained, and each such repair is recorded in `judgment_repairs`.
If the grader hits its output cap after emitting a complete numeric first
`score` field and starting its explanation, that explicit score is retained and
flagged separately. Partial numbers and malformed non-truncated responses are
not recovered by this rule.

For sentence-prefix control, a partly supplied initial sentence is not a newly
generated sentence start. Only starts generated within the scoring window are
assessed. A nonempty window with no assessable start is missing, not a success
or failure; eligibility counts accompany the curves. Omit-word and first-person
checks use seed context to detect a forbidden word spanning the boundary, but
exclude words wholly inside the seed. For immediate windows, a trailing partial
word is excluded when the saved full continuation shows that the word extends
past token 32; this avoids treating `the` inside `their` as a forbidden whole
word. These boundary handling rules are part of
correct scoring; a separate word-boundary curve is not reported.

Repeated samples are averaged within question. A shared question-clustered
bootstrap preserves overlaps across models, controls, conditions, and lengths.
Aggregate curves give each of the ten controls equal weight and withhold points
where an entire control cell is unassessable. Inspect per-control curves and
eligibility/truncation counts alongside aggregate results.

## Execution and accounting

`scope.json` records the active and deferred models. Shared Modal accounting
caps the worker estimate at $90, using $8/B200-hour and no more than three hours
per dispatch. Each worker runs a reusable 128-prompt pilot and gates the remainder
on quality and projected runtime. The output volume is `token-prefill-full-out`;
the original data and cache volumes are mounted read-only for generation.

OpenRouter auto-refill was explicitly confirmed by the user. The shared paid
transport caps tracked cost/reservations at $125 across generation and grading.
Actual returned costs replace reservations; uncertain interrupted calls retain
a conservative possible charge. Completed paid responses survive restart even
if downstream processing had not yet committed. GLM uses a rolling pool of 20
requests with extended backoff for upstream rate limits. A scheduler restart
retained 455 responses and one uncertain-call reservation ($0.0396); a later
48-concurrency attempt hit an upstream limit and retained 540 responses before
resuming at 20. These changes do not alter prompts or sampling parameters.
After 19,815 completed responses, repeated DeepInfra shared-pool overload errors
exhausted the retry allowance. Generation resumed with eight concurrent requests
and up to 24 attempts per request, retaining the same provider and sampling
settings. The stopped process had no unresolved generation reservations.

`run_token_prefill_suite.py` coordinates sequential GPU dispatch, collection,
and scoring as each active model completes. OpenRouter generation is a separate
resumable process. Logs and reports are under the versioned run root. No DeepSeek
or GPT-OSS full runs were dispatched. Six short API validation probes included
DeepSeek before the user's deferral and remain separate from experimental data.
A separate 32-token GLM transport probe confirmed that requesting five choices
returns only one on this route; experiment calls therefore request one sample.

```bash
python3 controllability/token_prefill_suite.py prepare
modal run --detach controllability/modal_token_prefill_suite.py --phase dispatch --model qwen32
python3 -u controllability/token_prefill_suite_api.py --model glm
python3 -u controllability/run_token_prefill_suite.py
MPLBACKEND=Agg python3 controllability/analysis/token_prefill_suite_analysis.py
```

Do not dispatch a second GPU worker while a reservation is active. The
coordinator handles Qwen-14B after Qwen-32B. Resume from durable manifests and
journals; do not erase them to bypass a gate. Prepared `full_v1` artifacts are
an undispatched preparation draft, superseded by `full_v2`.
