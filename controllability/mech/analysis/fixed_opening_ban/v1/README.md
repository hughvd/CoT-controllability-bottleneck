# Fixed opening-token bans

## Paper figures

The main-paper figure is `fixed_opening_ban_average.png` / `.pdf`: paired
no-banning and first-token-banning bars for each model in two aligned panels:
(a) continuation soft compliance and (b) task accuracy. Both use the same
instruction-present responses, averaging uppercase, leetspeak, Spanish, and omit-word
with equal weight. Task accuracy is final-answer correctness on these questions,
not a separate capability benchmark. Missing/unparseable answers count as incorrect;
empty and truncated responses remain included. Error bars are 95% question-cluster bootstrap intervals (5,000
draws), sharing question weights across arms and control types. All repeated
samples are retained. Each arm contains 360 responses across 119 unique question
IDs (one question appears in two control types).

Appearance follows the ICL main bar plot (`icl_compare.py:make_avg_plot`):
5.5-inch-wide canvas with two stacked panels, solid condition-colored bars with thin white edges,
0–1 compliance axis, matching fonts/grid/error bars, rotated model labels,
and a shared two-column upper-left legend. Gray denotes no banning and blue denotes
banning; model order matches the ICL figure's shared models.

The per-control plot is retained as `fixed_opening_ban_by_control.png` / `.pdf`
for the appendix; `compliance_change.*` remains its original analysis filename.
Both paper figures are also exported to `paper/figures/`. The original compliance
CSV is preserved; `_task_accuracy.csv` and `_metrics.csv` provide the additional
summaries. A provenance JSON is saved alongside them. Regenerate without inference:

```sh
python3 controllability/mech/analysis/plot_fixed_opening_ban_average.py
```

## Question and design

Does preventing the usual opening token recover compliance with a control instruction?
This is a fixed lexical intervention, distinct from the earlier experiment that
banned the current argmax at the first one or two positions.

- Qwen3-32B: ban exact native token IDs for `Okay` (32313) and `Alright` (71486).
- OLMo-3.1-32B-Think: ban `Okay` (33413) and `Alright` (72586).
- GLM-5.2: ban `The` (785), using the hosted rejection method below.
- Four controls: uppercase, leetspeak, Spanish, and omit-word.
- Thirty questions per control, taken from the Qwen32 full_v2 token-prefill
  manifest. The same questions and control instructions are used across models.
- Three samples per question, with/without the instruction, baseline/fixed ban:
  1,440 responses per local model. GLM completes the 720-response instructed
  comparison; its uninstructed responses remain incomplete diagnostics.
- Temperature 0.6, top-p 1.0, unrestricted top-k, 8,192 generated-token cap.

## Local model implementation

Both arms generate one token at a time until the first non-whitespace token,
then continue in a second call. The intervention masks the two exact IDs during
that initial stage and removes the mask completely for the continuation. The
baseline uses the same split, without the mask. Leading whitespace is permitted.
Capitalization and leading-space variants are not additionally banned.

Both local models use direct negative-infinity logit biases on the two IDs; the remaining
vocabulary is unrestricted. Neither implementation uses a top-logprob shortlist.
Initial Qwen attempts failed before producing responses: first on padded embedding
rows that are not tokenizer IDs, then on its GPU runner's 1,024-ID allowlist limit.
The restart uses direct `SamplingParams(logit_bias=...)`, whose negative infinity
survives serialization (unlike the clamping API convenience constructor).
An OLMo allowlist attempt saved 96 responses before failing a ban-enforcement
assertion in the next batch. That entire attempt is quarantined and excluded;
OLMo restarts cleanly with the negative-infinity mask.

The manifests record pinned model/tokenizer revisions, source question manifest
hash, prompt token IDs, seeds, request IDs, and sampling settings. Native generated
IDs, opening-stage IDs, full text, and termination reasons are retained. Completed
batches are durable and resumable. Generation source hashes are saved alongside
the manifests.

## GLM hosted implementation

Provider probes found that one route accepted but did not enforce the reasoning
token bias. Other routes enforced it but did not reliably continue from the supplied
assistant reasoning prefix. A two-call generation on those routes would therefore
not implement the intended intervention.

GLM uses full-response rejection sampling on DeepInfra. The
baseline retains its first proposal. The ban arm retains the first proposal whose
first locally retokenized reasoning token is not `The`; no later part of the
response affects acceptance. At top-p 1 this matches first-token masking under
the assumption that the API preserves the first reasoning token and local
retokenization identifies it correctly. Native reasoning token IDs are not
returned by the provider, so this assumption cannot be checked directly.

Unlike the local implementation, this method examines the first returned token
without skipping whitespace. Exhausted proposal limits remain missing outcomes;
they must not be dropped to construct a treatment-effect estimate. See the GLM
`method_note.json`, raw proposals, and `rejection_report.json` for the actual
protocol and completion status.

The expanded GLM attempt encountered multiple uninstructed prompts requiring over
90 rejected proposals. Execution was reprioritized to the full instruction-present
comparison (720 intended responses). Uninstructed responses are retained as
incomplete diagnostics and excluded from effect estimates. If all 720 instructed
responses finish, this primary comparison can be analyzed in full; it does not
provide GLM's instruction-by-intervention interaction. In-flight API reservations
at interruption remain counted conservatively in the tracked cost. All 720
instructed GLM responses were ultimately completed. The last two requests used
parallel batches of proposals, retaining the earliest acceptable proposal by
attempt index (never by completion time or later content). One request needed
288 proposals, requiring extension of its initial 256-proposal limit. No
question or sample was dropped from the primary comparison.

## Scoring and interpretation

Compliance is measured on both the whole reasoning trace and the continuation
after the opening stage. Uppercase/leetspeak/omit-word reuse the existing
deterministic graders; Spanish reuses the existing language judge. Suppression
scoring preserves word-boundary context across the opening/continuation boundary.
Empty reasoning receives zero compliance. Empty outputs, token-cap terminations,
answer accuracy, and reasoning lengths are reported separately.

Treatment effects average samples within question and pair questions across arms.
Confidence intervals resample questions, not individual rollouts. A question enters
the paired analysis only once all three samples in both arms have been scored.
An incomplete model run must not be presented as a completed comparison.

The primary comparison is instructed fixed-ban versus instructed baseline. The
uninstructed arms help distinguish instruction-dependent recovery from generic
changes in output style. A gain supports a role for the usual opening in the
initiation bottleneck; it does not by itself establish entropy collapse as the
cause, or improvement in monitorability.

## Files and commands

Raw data: `controllability/results/fixed_opening_ban/v1/`.

```sh
modal run controllability/mech/modal_fixed_opening_ban.py --phase collect --model qwen32
modal run controllability/mech/modal_fixed_opening_ban.py --phase collect --model olmo31
python3 controllability/mech/fixed_opening_ban_score.py score --model qwen32
python3 controllability/mech/fixed_opening_ban_score.py score --model olmo31
python3 controllability/mech/fixed_opening_ban_score.py analyze
```

Generation is already dispatched; do not dispatch duplicate workers. Inspect the
saved dispatch IDs and reports first. The scorer caches judgments and can resume.
