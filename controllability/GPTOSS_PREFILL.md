# Clean high-effort GPT-OSS prefill run

Completed September 9, 2026: 18,000 high-effort responses and 120 medium-effort
pilot responses. The GPU app is stopped, grading is complete, and both paper PNGs
use the clean sweep. See [the results note](analysis/prefill/gptoss_high_clean_results.md)
for effects, quality checks, and costs.

The run `gptoss_high_clean_v1` preserves the 300 original task/question cells and
their exact saved seeds at K=0–5. Two instruction conditions and five samples
produce 18,000 main responses. The manifest preserves explicit prompt token IDs,
source hashes, model revision, task metadata, and seed text. No-instruction
prompts omit the entire developer message. K=0 allows the model to select its
channel; K>0 resumes inside the seeded analysis body.

The generation model is `openai/gpt-oss-120b`, revision
`b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, in its native MXFP4 format. The pinned
runtime uses CUDA 13.0.2's development image, vLLM 0.23.0, Transformers 5.12.1,
and openai-harmony 0.0.8. The development toolchain is needed for FlashInfer JIT
compilation; the initial slim-image startup failed before generation and is
included in the GPU ledger. Model
weights are cached on Modal without a GPU and staged onto local disk before
loading on one B200. No OpenRouter credentials are uploaded to Modal.

Generation uses high reasoning effort, temperature 0.6, top-p 1.0, no top-k
restriction, and an 8,192-token continuation cap. Sampling seeds are fixed by
prompt identity. All main conditions share one backend and effort. The source
sweep specified temperature without a nucleus override; top-p 1.0 preserves that
choice. The 16,384-token context accommodates every input plus its output cap.

Two cells per behavior (shortest and longest K=5 prompt) form the validation
subset: K=0/1/5 × both conditions yields 120 high-effort prompts and 600 responses,
which are reused in the main results. The same prompts receive one medium-effort
sample each for a separate diagnostic. These 120 extra medium responses never
enter the high-effort figure. Raw output IDs, prompt IDs, channel text, stop
reasons, and truncation flags are retained for every response.

The pilot verifies tokenization, nonempty channel extraction, and projected
runtime. Remaining requests are processed in batches of 64 prompts, with five
outputs per prompt, and each completed batch is committed to a new Modal output
volume. The GPU worker has a four-hour cumulative reservation limit, including
startup and validation. At a conservative $8 per worker-hour this reserves at
most $32, plus the CPU-only cache download. This is derived from
[Modal's resource prices](https://modal.com/pricing); it is not measured billing.
The remaining run stops if its pilot-based projection exceeds the time left.

Grading runs locally after outputs are downloaded. Deterministic graders retain
the existing definitions. Language and style grading uses the existing
Gemini-2.5-Flash-Lite prompts through OpenRouter. Every paid response, token usage,
cost, and parsed score is saved. Failed grading is retried and never silently
assigned score zero. The grader reserves $10, stops dispatching at $9, and holds
the balance for in-flight requests. Successful judgments are reused on resume.
`grade-partial` can grade already downloaded batches during generation; it
never writes the final CSV. A local file lock serializes graders so overlapping
collection and grading cannot dispatch duplicate paid judgments.

```bash
python3 controllability/gptoss_prefill_run.py prepare
python3 controllability/gptoss_prefill_run.py grade-smoke
modal run controllability/modal_gptoss_prefill.py --phase cache
modal run --detach controllability/modal_gptoss_prefill.py --phase dispatch
modal run controllability/modal_gptoss_prefill.py --phase collect
python3 controllability/gptoss_prefill_run.py grade
python3 controllability/analysis/gptoss_high_clean_audit.py
MPLBACKEND=Agg python3 controllability/analysis/prefill_compare.py
```

Local intermediate outputs live in `controllability/results/gptoss_high_clean_v1/`;
the final CSV is `base_gptoss_120b_high_clean_prefill_per_sample.csv` beside the
original results. Existing data are preserved. Only a complete, validated CSV
should replace the provisional GPT-OSS source in the figure. Figure code remains
in the usual analysis folder and exports PNG copies to `paper/figures`.

The audit verifies manifest integrity, exact prompt IDs, decoded output IDs,
channel parsing, unique samples, complete grading, and reconciled GPU time.
It exports task/condition/prefix quality counts, a separate medium/high pilot
summary, and `analysis/prefill/gptoss_high_clean_results.md`. The run directory
also retains the serving environment's package versions and source snapshots.

Use detached dispatch for the long GPU task. Attached runs were canceled during
local connection drops; those attempts are included in the ledger. Collection
reports progress while running and downloads all batches once the worker reaches
a terminal state. Never dispatch again while a prior reservation is active.
