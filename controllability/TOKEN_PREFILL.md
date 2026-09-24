# Token-prefix continuation experiment

Qwen3-32B experiment completed September 15, 2026: all 4,800 responses generated,
collected, scored, and plotted. See [results](analysis/token_prefill/qwen32_v1/results.md)
and [figure](analysis/token_prefill/qwen32_v1/compliance.png). The worker has stopped;
the estimated cost including the stopped H100 pilot is $4.97. The run's
`completion_validation.json` and `gpu_ledger.json` record validation and accounting.

## Question and scope

Does supplying only the opening tokens of a controlled trace elicit sustained
control, or does the model need a longer demonstration? This follow-up connects
the sentence-prefill results to the opening-token rank experiments.

The initial run contains uppercase and leetspeak, 30 historical questions per
behavior, lengths **0, 1, 2, 4, 8, 16, 32, 64**, both instruction conditions, and
five samples: **960 unique prompts / 4,800 responses**. These are new generations;
there is no training or LLM grading. Other models are not included in this run.

## Fixed histories and prompt protocol

`token_prefill.py prepare` reconstructs each full controlled trace by applying
the historical deterministic transform to `training_raw.csv`. Every saved
sentence prefix in the selected historical CSV is checked against that source,
including all lengths, conditions, and repeated samples. The same questions,
options, instructions, and answer-format requests are retained.

The full controlled text is tokenized once with Qwen's pinned tokenizer, then
sliced by token ID. Those exact IDs are appended to the chat-template opening;
prefixes are never decoded and re-encoded before generation. Both conditions
receive the identical prefix IDs. No-instruction prompts omit the whole control
block. All lengths, including zero, end inside the same `<think>\n` channel.
Consequently, a fresh zero-token baseline is required: the historical API baseline
had a different reasoning-opening policy and serving backend.

All sources exceed 64 tokens (minimum 225), so there are no shortened or omitted
prefix conditions. The preparation audit writes `sentence_token_lengths.csv`:

| Behavior | First-sentence median | First-sentence range | Five-sentence median |
|---|---:|---:|---:|
| Uppercase | 7 tokens | 7–25 | 95 tokens |
| Leetspeak | 15 tokens | 15–59 | 275 tokens |

The manifest records source/code/tokenizer hashes, exact prompt and controlled
token IDs, question and cell identities, per-request random seeds, and links from
source cells to deduplicated requests. It is immutable: subsequent code or source
changes require a new run directory. Existing sentence results and figures are
preserved.

## Measurements

Only newly generated reasoning is graded; supplied tokens and the final answer
are excluded. The scorer records:

- Historical soft compliance for the first **up to 32** new reasoning tokens and
  the full continuation, plus the historical 0.5 threshold and strict local
  compliance with a relevant-character requirement.
- Scores after removing newly generated characters that complete the seed's
  last alphanumeric word. These diagnose apparent control confined to finishing
  a partly supplied word. The plotted diagnostic gives contentless windows zero.
- Relevant-character counts and content-sensitive scores. The historical
  uppercase grader gives punctuation-only nonempty text a soft score of one;
  the added diagnostics expose that vacuity rather than silently redefining the
  historical metric.
- First-violation character and generated-token positions; reasoning length;
  whether 32 tokens exist; missing closing reasoning tags; empty reasoning; and
  output-cap truncation. Short windows remain in the main estimates and are
  identified separately.

UTF-8 fragments split by token boundaries are decoded jointly with the supplied
prefix. A completed boundary codepoint is attributed to the continuation and the
boundary is flagged. Both raw token IDs and serving text are retained.

Analysis averages repeated samples within question before resampling questions
jointly across lengths and conditions (2,000 bootstrap replicates). It produces
pointwise 95% intervals, a two-behavior/two-condition curve, and quality tables.
All samples, including empty and capped responses, remain in the main estimates.
This differs from the existing sentence figure's nonempty-only convention and
must be stated if comparing figures.

## Inference and commands

The model is `Qwen/Qwen3-32B`, revision
`9216db5781bf21249d130ec9da846c4624c16137`, with bfloat16 weights. The Modal wrapper
uses the existing `ban-sweep-hf-cache` read-only and a new `cot-token-prefill-out`
volume. It stages cached weights onto local disk, uses one B200, and runs vLLM
0.23.0 / Transformers 5.12.1 in a CUDA 13.0.2 development image, with CUDA graphs
and up to 128 concurrent sequences. Sampling is temperature 0.6, top-p 1.0,
no top-k restriction, and an 8,192-token output cap for every length.

The original stress subset uses two selected cells per behavior (shortest and
longest full source), both conditions, and lengths 0/1/64: 24 prompts. The initial
H100/eager attempt was stopped after its pilot duration ruled out the planned
runtime projection; it produced no saved samples. Its runtime remains charged
to the shared experiment ledger.

The B200 pilot retains those 24 prompts and adds 104 evenly spaced requests from
the remaining manifest, selected deterministically before observing outcomes.
Its full 128-prompt production batch supplies 640 responses reused in the sweep;
the experiment still generates 4,800 saved responses total. The original
`smoke_ids` remain in the immutable manifest, and the worker report records the
complete expanded `pilot_request_ids`. It verifies serving tokenization and stops if empty reasoning or
truncation exceeds 10%, or if projected remaining runtime does not fit. Remaining
work is committed in batches of 128 prompts. The worker and local reservation
ledger cap cumulative worker execution time at two hours, including startup and
pilot checks. A second bound preserves the original $12 worker estimate across
attempts: the stopped H100 attempt is accounted at $6/hour, and the B200 attempt
at $8/hour, reducing its allowed runtime accordingly. These are budgeting
assumptions, not live price quotes or billing guarantees. There are no OpenRouter
calls or API keys in this experiment.

From the repository root:

```bash
python3 controllability/token_prefill.py prepare
modal run --detach controllability/modal_token_prefill.py --phase dispatch
modal run controllability/modal_token_prefill.py --phase status
modal run controllability/modal_token_prefill.py --phase collect
python3 controllability/token_prefill.py score
MPLBACKEND=Agg python3 controllability/analysis/token_prefill_analysis.py
```

Collection is safe during generation and downloads committed batches. Final
scoring refuses incomplete or duplicate outputs. After a terminal run, collection
reconciles its ledger entry; a later dispatch resumes only missing requests if
runtime remains. An active reservation blocks another dispatch. Hard timeouts or
uncertain connection failures may leave an active reservation: verify the remote
worker has stopped before manually reconciling it; never erase the ledger to
bypass the cap. Dispatch is detached so a local disconnect cannot cancel it.

Local run artifacts are under `results/token_prefill/qwen32_v1/`. Analysis goes
under `analysis/token_prefill/qwen32_v1/`, with no automatic paper-figure replacement.

## Validation and later expansion

The local tests cover historical scoring equivalence, punctuation-only windows,
seed/final-answer exclusion, first-32 indexing, partial-word and UTF-8 boundaries,
immutable artifacts, duplicate/incomplete-output rejection, deterministic scoring
on resume, and question-clustered sample handling. A synthetic end-to-end fixture
exercises scoring and figure rendering entirely in a temporary directory. It is
not experimental data. The GPU pilot remains the serving/runtime validation.

Model identity, revision, prompt adapter, protocol IDs, and sampler are explicit
in the manifest. Future models need their own validated prompt adapter, tokenizer
and serving resource configuration, plus a fresh zero-token baseline. The current
preparation and worker intentionally implement only Qwen3-32B; Harmony and API
backends must not be treated as interchangeable with this token-ID interface.
