# ICL donor repair and model replication

Requested September 15, 2026. Generation is running; consult
`results/icl_donorfix_v1/status.json` for completion, not this note.

The donor lookup now keys by `(ctype, question)` rather than question text alone.
Sixteen of the 300 original cells shared questions across behaviors and inherited
a different behavior's donor. The model was prompted and graded for that donor's
behavior, but the result was saved under the target behavior's label.

## Repairs

`icl_repair.py prepare` reconstructs and verifies every historical assignment
against the saved instruction and donor hash. It copies all unaffected retained
records unchanged into new caches and omits only the affected records. Original
caches and CSVs remain intact. Source hashes and exact replacement keys are in
`results/icl_donorfix_v1/repair_manifest.json`.

- Three full sweeps: Qwen3-14B, Qwen3-32B, GLM-5.2; 320 replacement responses each.
- Two length sweeps: Qwen3-32B and GLM-5.2; 1,520 replacements each.
- Total replacements: 4,000.
- V3 (`icl_assistant_think`) is excluded from corrected datasets, including GLM.
- Corrected filenames insert `_donorfix_v1` before `_icl_per_sample.csv`.

`icl_repair.py audit` verifies original cache hashes, preserved records, correct
donors for every output, replacement metadata, and complete output counts.
All three full-sweep repairs have completed and passed the audit. Their figures
have been refreshed from the corrected sources. The two length-sweep repairs and
the new models remain in progress; consult the live status file.

## New full sweeps

Both retain the existing 300 cells (30 per behavior), five samples, and four
conditions: instruction-only, plain user demonstration, user demonstration with
think tags, and visible prior assistant demonstration. User demonstrations use
five sentences; visible assistant demonstrations use the full example. There is
no new length sweep. Each model produces 6,000 responses.

The donor instruction is applied to both the demonstration and target. This
preserves the original ICL design. Comparisons to prefill remain descriptive:
question pools match, but control instruction wording/instances can differ.

### DeepSeek-R1

`deepseek/deepseek-r1-0528`, pinned to DeepInfra via OpenRouter, matching the
prefill model ID/provider. Temperature 0.6 and 8,192 generated tokens. Final CSV:
`results/iclfull_deepseek_r1_donorfix_v1_icl_per_sample.csv`.

### GPT-OSS-120B

Native Harmony on one Modal B200, high reasoning effort, same model revision and
vLLM/runtime as the clean prefill sweep. Temperature 0.6, top-p 1, no top-k limit,
8,192 generated tokens. Context is 32,768 rather than prefill's 16,384, because
the longest full ICL input is 17,856 tokens. No demonstration is truncated to fit.

The existing ICL message contents are rendered into Harmony. Visible assistant
examples are prior final-channel messages; the new completion ends at an unseeded
assistant header so the model selects its own channel. The control instructions
remain in the ICL user text; native prefill placed control instructions in the
developer message, another documented difference in this descriptive comparison.
Literal think tags in V2 remain text in the user message.

A reusable 200-response pilot covers all ten behaviors and four conditions,
using the longest assistant demonstration per behavior. The worker persists each
batch, checks channel parsing/empty reasoning/truncation, and checks a conservative
runtime projection before proceeding. It has a four-hour cumulative GPU time
limit. Grading uses the existing definitions and judge prompts, with persistent
judgments and a $10 judge ceiling. API credentials stay local.

Manifest, exact prompt tokens, source snapshots, raw output tokens, batch reports,
and judgment ledger are in `results/iclfull_gptoss_120b_high_donorfix_v1/`.
Final CSV: `results/iclfull_gptoss_120b_high_donorfix_v1_icl_per_sample.csv`.

## Execution and recovery

`icl_run_batch.py` runs each API sweep with a reusable pilot, an exclusive local
lock, and up to three completion passes. Every newly generated response is saved
before grading. Retrying loads that raw response instead of generating it again;
failed judgments are not converted into measured zero compliance.

API logs, process IDs, and completion markers are in `results/icl_donorfix_v1/`.
`icl_finish_runs.py` collects and grades the detached GPT-OSS batches and audits
the repairs after all six API runs plus GPT-OSS complete, writes a five-model
summary, and refreshes the figures with the new models. This collector does not
launch generation. GPU launch identity and reservation are saved separately in
`results/iclfull_gptoss_120b_high_donorfix_v1/gpu_ledger.json`.

Do not launch another GPU worker while that reservation is active. API runs can
be resumed with the commands recorded in `results/icl_donorfix_v1/processes.json`
after confirming their original processes have exited.


## September 20 update: OLMo added; original suite complete

The five-model suite and both historical length repairs are now complete. The
GPT-OSS collector had stopped because vLLM omitted the terminal `<|return|>` text
while retaining its token ID; all 6,000 outputs passed validation after allowing
exactly that terminal difference. A separate judge serialization issue escaped
apostrophes inside JSON reason strings. `icl_judging.py` repairs only those invalid
escapes and retains the judge's numeric score. Existing generations were reused.
The final DeepSeek judgment was recovered with the same fix. GPT-OSS has zero
empty reasoning responses and 336/6,000 responses at the 8,192-token limit.

The new OLMo replication is `allenai/Olmo-3.1-32B-Think`, revision
`832c3f543499af8fe68b88359501de9cb7840544`, matching the mechanism experiments.
It uses BF16 on one Modal B200 with vLLM 0.23.0 and Transformers 5.12.1. The native
chat template opens an empty `<think>` block. No controlled seed is injected.
The same 300 corrected question/behavior cells, four retained conditions, and
five samples give 6,000 responses. Sampling is temperature 0.6, top-p 1, no top-k,
and 8,192 generated tokens. A 32,768-token context accommodates the longest
17,826-token full demonstration without truncating the input. A reusable
200-response pilot precedes the remaining generation.

OLMo's manifest, raw tokens, worker report, grading ledger, and audit live in
`results/iclfull_olmo_3_1_32b_donorfix_v1/`. Final CSV:
`results/iclfull_olmo_3_1_32b_donorfix_v1_icl_per_sample.csv`.
`olmo_icl_finish.py` collects, grades, summarizes, and refreshes the figures after
completion. The OLMo prefill column is explicitly missing because there is no
corresponding sentence-prefill sweep; it must not be interpreted as zero.

OLMo dispatch: `modal run --detach controllability/modal_olmo_icl.py --phase dispatch`.
Do not repeat dispatch while its saved GPU reservation is active.


OLMo startup validation caught a Transformers 5 return-type change:
`apply_chat_template(tokenize=True)` returned `BatchEncoding`, whereas the local
version returned a list. The equality check now explicitly renders text and
encodes it, matching preparation. A CPU check under the serving environment
verified all 1,200 prompts exactly before restarting the GPU. The failed startup
produced no generations and is accounted for in the same cumulative GPU ledger.
