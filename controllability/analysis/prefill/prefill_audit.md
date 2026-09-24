# Prefill figure audit — September 9, 2026

The main issue with the old GPT-OSS K=0 results is a confirmed parsing error.
The current GLM baseline values reproduce under the existing graders; they reflect
its default style, shorter reasoning, and the specific words the graders check.

## GPT-OSS: what actually happened

- The original medium-effort/local CSV has 3,000 K=0 rows recorded as empty,
  including both instruction conditions.
- All 55 surviving local K=0 Harmony raw transcripts contain nonempty analysis.
  Of these, 39 match rows in the current CSV by question hash, task, condition,
  and sample. Every matched CSV row incorrectly records empty reasoning. The other
  16 are not matched to that CSV and are not used to repair it.
- The old parser split on `<|channel|>` before removing the leading analysis
  header. Unseeded generations start with that header, so their reasoning was
  discarded. Seeded generations start inside the body and avoid this particular
  bug. Consequently, “100% empty at medium effort” was not a valid observation
  about the model. The observed raw reasoning lengths range from 163 to 5,497
  characters; we cannot extrapolate the recovered subset to every missing row.
- There is a separate 3,000-response high-effort K=0 run via DeepInfra/OpenRouter.
  None of those responses is empty. Mean reasoning length is 2,557 characters
  without instruction and 4,234 with instruction. It is not included in the
  plotted medium-effort/local series because effort and backend both differ.
- The ordinary prompt builder removes the absent control block, but the Harmony
  backend independently retained a developer message containing `none`. That
  separate path is now fixed. Historical Harmony instructions were developer
  messages, rather than the XML block recorded by the generic prompt logger.

`harmony_backend.py` now handles both unseeded channel headers and seeded body
continuations, omits the developer message when there is no instruction, and
includes exact prompt token IDs and effort in saved raw responses. Four regression
tests pass. The fix also recovers analysis from all 55 surviving raw examples.
Original results are preserved. `gptoss_k0_recovered_subset.json` is an audit subset,
not a replacement estimate or a representative sample.

### Recommended rerun

Use high effort consistently for K=0–5 and both conditions, with the same native
Harmony backend, the complete no-instruction omission, and no forced analysis
opener at K=0. Retain the existing question/seed design, temperature 0.6, and
8,192-token cap, and save raw outputs for every sample under a new run name.
Thirty questions × five samples × ten behaviors × six K values × two conditions
is 18,000 completions. This is a proposed run, not launched by this audit.

Start with a small smoke check using the same selected question/seed pairs at
K=0, 1, and 5 under medium and high effort. Verify channel extraction, prefill
continuation, reasoning length, and truncation before the full high-effort run.
The motivation for high effort is a consistent, explicitly specified reasoning
setting and a check on short-continuation effects—not a claim that medium effort
cannot reason or that high effort makes lengths comparable across models.

First check whether the original remote raw outputs still exist: they could
recover a complete medium-effort baseline without new generation. If time or
compute is tight, prioritize K=0, 1, and 5 (9,000 completions with the same sample
counts) and show those three measured points explicitly. A GPU runtime/cost pilot
is needed before launching; the earlier small rank-rescoring budget is not a cost
estimate for this generation sweep. The existing high-effort API K=0 data are
useful context, but should not be joined to the medium-effort curve.

## Why GLM's no-instruction baseline is high

All GLM K=0 no-instruction prompts omit the control block, all have empty seeds,
and none has empty reasoning. The stored `instruction` column is task metadata;
its text is not in those actual prompts. Recomputing the two deterministic grades
matches all 300 saved scores.

**No first person:** 111/150 responses pass (74.0%), versus 0/150 for the Qwen3-32B
corrected baseline and 0/150 for R1. GLM often begins with impersonal option
analysis such as “The question asks…” rather than “I need…” or “We need…”.
Passing GLM responses average 644 characters; failures average 2,103. That supports
a length association, but does not establish that length alone causes the gap.
The current grader intentionally permits “let's”; 44 of the 111 passes contain
that expression. If it were counted as first-person language, the pass rate would
be 44.7%, but that would change the evaluated constraint. The paper should describe
this as avoidance of the specified first-person terms, rather than absence of
every form of self-reference.

**Omit word:** 73/150 responses pass (48.7%). The forbidden word varies:

| Assigned word | Responses | No-instruction pass rate |
|---|---:|---:|
| because | 50 | 76.0% |
| answer | 45 | 71.1% |
| the | 50 | 0.0% |
| therefore | 5 | 60.0% |

This is largely natural absence of particular words. It is not evidence of
following an unseen instruction. Word and question assignments are not crossed,
and the `therefore` estimate has only one question with five samples.
Mean GLM reasoning length in this task is 864 characters, versus 1,780 for Qwen3-32B
and 3,426 for R1 in their respective saved baselines; these cross-model averages
are descriptive and not a matched causal length analysis.

There is also a small whole-word boundary artifact: four passing responses contain
`answeris` or `correctanswer`. Counting those four as failures would move the
overall baseline from 48.7% to 46.0%. The join is visible in saved text; its origin
is not established here. This does not explain the overall high baseline. Original
grades remain unchanged. The final `ANSWER: X` output is intentionally excluded
because this experiment grades only the reasoning continuation.

For the paper, show the no-instruction baselines beside instruction and prefill
scores, and interpret gains relative to them. A high raw score on a suppression
task can arise from default vocabulary, voice, and length. In GLM, no-instruction
K=5 scores are actually lower than K=0 for both tasks (no-first-person 0.47 versus
0.74; omit-word 0.25 versus 0.49), which makes a uniform “prefill improves every
behavior” claim inappropriate.

## Figure changes

Both PNGs are now exported at 300 DPI in the usual analysis folder, with matching
copies in `paper/figures`; vector PDFs stay in the analysis folder. The source
script performs that copy automatically. The average curve now uses the same
corrected Qwen K=0 sources as the table. The legend no longer obscures the curves.
GPT-OSS remains visibly provisional with missing K=0 values; the high-effort API
baseline has not been spliced into it. These are current draft paper assets,
pending resolution of the GPT-OSS series.

Reproduce the read-only audit with
`python3 controllability/analysis/prefill_audit.py` and regenerate/copy figures with
`MPLBACKEND=Agg python3 controllability/analysis/prefill_compare.py`.
`prefill_audit.json` records source hashes and detailed counts.
