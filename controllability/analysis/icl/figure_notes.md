# ICL figure captions and interpretation

## Suggested main-text panel: `icl_prefill_focus.png`

**Demonstrating controlled reasoning versus starting a controlled continuation.**
Mean compliance for uppercase, leetspeak, and word omission under instruction-only
prompting, three ICL formats, and five-sentence prefill with instructions. Qwen
models show little uppercase or leetspeak control after demonstrations but high
compliance when continuing a compliant prefill. The pattern differs for word
omission, where some ICL formats outperform prefill, and for GLM, which benefits
substantially from demonstrations. Scores use the generated native reasoning for
ICL and the generated continuation after the seed for prefill; injected text is
not graded. The comparisons are descriptive: question pools match, but instruction
assignments differ between the ICL and prefill sweeps.

## Appendix panels

- `icl_avg.png`: Mean unthresholded compliance across ten behaviors, with equal
  weight per behavior, for instruction-only, the three retained ICL formats, and
  K=5 prefill with instructions. This aggregate hides the behavior-specific
  differences shown in the comparison panels. Error bars are pointwise 95%
  question-clustered bootstrap confidence intervals from 5,000 resamples,
  retaining repeated responses together and equal behavior weighting.
- `icl_compare.png`: The same conditions broken down by all ten behaviors and
  three models. Every cell uses the same 0–1 compliance scale. Model names are
  panel identifiers, not explanatory titles.

## Condition definitions

- **instr-only:** ICL sweep's instruction-only baseline, without a demonstration.
- **user:** A worked example in the user prompt, using plain internal-thinking
  and response labels; the first five sentences of donor reasoning.
- **user+tag:** A worked example in the user prompt with `<think>` tags.
  It also uses the first five sentences of donor reasoning.
- **asst-viz:** A worked example as visible content in a prior assistant turn,
  using the full-length donor reasoning (the sentence-based 100% setting).
- **prefill K=5:** Five compliant sentences injected into the current reasoning,
  with a control instruction; this is the prefill curve's solid line at K=5.

Each ICL prompt contains one donor example, including its question/options and
final answer. The five response samples per target are repeated generations, not
five demonstrations. Example length differs between the user formats and
asst-viz, so those variants do not isolate message placement alone. The separate
ICL length sweep varies demonstration length explicitly.

The hidden assistant-thinking variant is excluded from all three plots and their
derived CSVs. Original experiment data are retained. No best-variant selection or
maximum across ICL variants is used.

## Qualifications for the paper

All three focus behaviors use the same 30 target questions in ICL and prefill,
with five samples per question before excluding empty reasoning. ICL applies the
donor example's instruction to its target; prefill uses the target's instruction.
Consequently, only 1/30 uppercase, 3/30 leetspeak, and 2/30 word-omission
question/instruction pairs match exactly in each model. For word omission this
also changes some forbidden-word assignments. The ICL baseline is therefore not
an exactly matched baseline for the prefill column. The exact prompts and labels
in the raw data are unchanged.

The transform contrast is large: Qwen uppercase/leetspeak scores are approximately
0.01–0.04 across the retained ICL variants versus 0.91–0.99 under prefill. Word
omission is a descriptive exception, not a clean causal comparison: Qwen-14B ICL
scores are 0.37–0.41 versus 0.24 under prefill; Qwen-32B user/user+tag score
0.53/0.59 versus 0.37 under prefill, while asst-viz scores 0.35.

This supports an initiation bottleneck for particular model–behavior pairs, but
does not isolate initiation. Prefill also changes immediate context, supplies
task reasoning, and reduces the continuation left to generate. Caption wording
should emphasize that demonstrations alone are insufficient in some settings,
rather than making a general claim that ICL fails.

Scores are the existing unthresholded `compliance_score` values. Some behavior
graders, including word omission, are themselves binary. Empty reasoning is
excluded as in the prefill plots; retained ICL cell counts range from 134 to 150
for Qwen-14B and are 150 for Qwen-32B and GLM. This is not an unconditional success
rate over every attempted response.


## Donor repair (September 15, 2026)

The three original full sweeps now use completed `_donorfix_v1` CSVs. For each
model, 320 responses were regenerated for the 16 question–behavior cells whose
donors had been overwritten by another behavior sharing the same question text.
All unaffected retained records were preserved exactly; original caches and CSVs
remain available. The figures have been refreshed from the corrected results.
These are actual replacement generations, not the earlier exclusion sensitivity
check. In the corrected uppercase row, Qwen remains around 0.03–0.04 under ICL;
GLM scores 0.53/0.81/0.79 versus its instruction-only baseline of 0.28.

DeepSeek-R1 and high-effort native GPT-OSS full sweeps are running. The plotter
adds them only once complete 6,000-row CSVs exist. The collector refreshes the
figures after all requested runs complete. Refer to `../../ICL_RERUNS.md` for the
native GPT-OSS rendering/context details and descriptive-comparison limitations.


## September 20 replication update

DeepSeek-R1 and native high-effort GPT-OSS now have complete, audited 6,000-response
ICL sweeps. OLMo-3.1-32B-Think has been added and is running with the same retained
conditions and corrected donor assignments. OLMo is included once its final CSV
exists; its unavailable sentence-prefill comparison is shown as missing, not zero.

## Versions without styles (September 21, 2026)

Run `python3 controllability/analysis/icl_compare.py --exclude-styles` to produce
`icl_avg_no_styles`, `icl_compare_no_styles`, and `icl_prefill_focus_no_styles`
as PNG/PDF figures and CSV tables. PNGs are also copied to `paper/figures`.
The original ten-behavior figures and underlying observations are preserved.

These versions exclude Pirate, Yoda, and Jar Jar Binks. The comparison contains
seven behaviors across all six completed models. The aggregate gives each of
those seven behaviors equal weight; its question-clustered bootstrap intervals
are recomputed using only those behaviors and their question universe. Empty
reasoning exclusion and all other scoring definitions are unchanged. The focus
panel already contained no styles, so its values are unchanged. OLMo's missing
sentence-prefill comparison remains missing, not zero.

The remaining Spanish and Chinese scores still use the language LLM grader.
Dropping styles therefore does not make the entire figure programmatically
verified or remove the need to describe language-grader validation.

## Four-model main figures with token prefill

Regenerate with `python3 controllability/analysis/icl_compare.py --main`.
Outputs are `icl_avg_no_styles_main`, `icl_compare_no_styles_main`, and
`icl_prefill_focus_no_styles_main`, each as PNG/PDF/CSV, with PNGs copied into
`paper/figures`. Only Qwen3-14B, Qwen3-32B, GLM-5.2, and OLMo-3.1-32B are included.
The broader figures remain available for appendix use.

The prefill column is **64 supplied tokens with instructions**, the longest
prespecified length in the token sweep. It uses **full_soft** (sustained
continuation), not first-32-token compliance and not the old five-sentence
prefill. Sources are `results/token_prefill/full_v2/{qwen14,qwen32,glm}/scores.csv`
and `results/token_prefill/olmo31_v1/olmo31/scores.csv`, relative to controllability.
All 28 behavior/model endpoint means were verified against the existing
`token_prefill/with_olmo_v1_no_styles/summary.csv`.

Token-prefill samples are averaged within question first, matching the existing
curves; the seven behaviors then receive equal weight. Empty/capped responses
remain included. Unassessable sentence-prefix responses remain missing rather
than being assigned zero; six GLM samples at this endpoint are unassessable.
ICL point estimates and their historical empty-reasoning exclusion are unchanged.
Aggregate intervals are recomputed with 5,000 question-clustered bootstrap draws;
CSV `n_responses` for token prefill counts eligible raw responses, not the
question means used internally for resampling.

This remains a descriptive comparison: donor instructions differ from prefill
instructions, two Chinese source questions were replaced in the token sweep,
and the token-prefill boundary-aware scoring/eligibility rules differ from the
historical ICL scoring. Spanish and Chinese retain their LLM-judge scores.
The existing GLM serving qualification still applies: its token-prefill results
are provisional following the identical-prompt sentence/token audit; see
`../token_prefill/glm_sentence_audit_v1/README.md`.

## ICL-only figures (2026-09-22)

All three figure families now show only instruction-only, user, user+tag, and
asst-viz. Prefill comparisons have been removed so ICL can be introduced first.
This supersedes the prefill descriptions above. The `_main` figures retain
Qwen3-14B, Qwen3-32B, GLM-5.2, and OLMo-3.1-32B, excluding the three styles.
Existing filenames (including the legacy `icl_prefill_focus` stem) are retained
for paper reference compatibility. PNG, PDF, CSV, and paper PNG copies are updated.
The generator no longer loads prefill results. Point estimates are unchanged;
bootstrap intervals use only the ICL question universe.

### Simplified ICL conditions (2026-09-22)

All figure versions now omit user+tag. The remaining ICL labels are
`user-ICL` (formerly user) and `asst-ICL` (formerly asst-viz), alongside
instruction-only. Source experiment data are unchanged.
