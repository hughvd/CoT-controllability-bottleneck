# Eight-control expansion: completed results

**Historical snapshot:** the numbers below precede the sentence-ending grading
correction and the subsequent GLM ban-protocol/model-source-prefill reruns.
The ICL CSVs and correlation reports have been refreshed with the corrected
grader; prefill and ban figures await their replacement runs. Use the relevant
pipeline reports for current completion status.

All prefill, ICL, ban × user-ICL, and twelve-model correlation runs are complete.
The extension supplies 28,560 scored responses; 1,140 additional generations
replace numbered user-ICL demonstrations affected by the sentence-slicing bug.
Existing retained-task responses and monitorability measurements are reused.
Chinese is excluded from the revised analyses and figures. Raw historical data
are preserved.

The shared order is **Addition** (sentence prefixes, numbered sentences, sentence
endings), **Suppression** (word omission, no first person), and **Style**
(uppercase, leetspeak, Spanish). Prefill tables include Base, +I, PI@1, PI@64.
All generated figures have been copied to `paper/figures` as PNG and PDF.
See `../../../CONTROL_EXPANSION.md` for the full protocol, question mixture,
scoring, and reproducibility commands. The paper prose is being edited separately.

## New-task prefill results

Mean compliance with the instruction, rounded to two decimals. PI scores only
the generated continuation; supplied text earns no credit.

| Model | Task | +I, no prefill | PI@1 | PI@64 |
| --- | --- | ---: | ---: | ---: |
| Qwen3-14B | Numbering | 0.00 | 0.94 | 0.87 |
| Qwen3-32B | Numbering | 0.00 | 0.79 | 0.95 |
| OLMo-3.1-32B | Numbering | 0.00 | 0.59 | 0.59 |
| GLM-5.2 | Numbering* | 0.77 | 0.05 | 0.32 |
| Qwen3-14B | Sentence endings | 0.02 | 0.02 | 0.62 |
| Qwen3-32B | Sentence endings | 0.04 | 0.04 | 0.86 |
| OLMo-3.1-32B | Sentence endings | 0.02 | 0.02 | 0.42 |
| GLM-5.2 | Sentence endings | 0.10 | 0.07 | 0.53 |

Numbering supports a one-token initiation effect in the native Qwen/OLMo runs.
Sentence endings respond mainly to longer prefills. The two addition tasks
therefore do not establish a uniform category-level initiation mechanism.
For each model, no one-token suffix prefix contains a completed marker, while
all 30 of the 64-token prefixes do. Prefix lengths were not adjusted.

*The GLM numbering-prefill result requires qualification: returned reasoning
frequently starts again at `1.` after a supplied numbered prefix. The hosted
API provides no generated token IDs, leaving ambiguity between model restarting,
prefix handling, and returned-text normalization. The specified continuation
metric is retained, but the drop should not be interpreted as a clean mechanism
result. See `glm_numbering_boundary_notes.md` and its diagnostic examples.

## Banning and ICL

Across eight equally weighted controls, banning increases compliance by 0.173
for Qwen3-32B, 0.195 for OLMo, and 0.146 for GLM. Adding banning to user ICL
increases compliance by 0.232, 0.309, and 0.134, respectively. Paired bootstrap
intervals for these six average differences exclude zero. Per-task effects vary;
the grouped tables and `ban_effects.csv` retain those differences rather than
presenting the average as a universal effect.

## Correlations

The refreshed twelve-model results are in
`results/monitorability/convergent_validity/results.md` and the interactive
`results/monitorability/control_type_viewer/index.html` (repository-relative paths).

Mean within-category Spearman correlations are 0.834 for Style, 0.527 for
Suppression, and 0.161 for Addition. Category–GPQA correlations are 0.510,
−0.137, and 0.308, respectively. These are descriptive model-level associations.
The weak coherence of the expanded Addition category warrants attention when
writing the paper's category-level claims.

DeepSeek-V4-Pro returns empty reasoning for 68/150 responses on each new task.
These receive zero under the existing correlation protocol. Coverage counts and
leave-one-model-out diagnostics are saved, including control-pair sensitivity;
the missing-reasoning behavior should be considered when interpreting the plots.

## Validation

All current cells are complete. The final ban suite has 8,640 scored responses.
Ten scoring/viewer tests pass. Source hashes and paper-figure copies were checked;
see `validation_report.json`. Per-phase provenance files record the inputs, and
the run directory contains final code snapshots. The older analyses remain
historical outputs; use the eight-control builder for future paper refreshes.
