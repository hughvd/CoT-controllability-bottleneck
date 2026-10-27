# First-person rescoring: paper updates

Completed offline from saved reasoning. No new generation or model grading calls. The paper source was not edited.

The canonical rule is `controllability.transforms.grade_no_first_person`: the fixed pronoun/contraction lexicon excludes "let's", exempts uppercase "US" and dotted abbreviations, and empty reasoning scores zero. Current figure and correlation loaders apply this rule when reading historical caches; raw caches are preserved. Per-response old/new scores and source/grader hashes are in `records/`. The pre-change outputs and paper are in `before/`.

## Required paper updates

1. **ICL paragraph (paper/paper.tex:146):** GLM average instruction-only → assistant ICL changes from **45.3% → 59.9%** to **50.0% → 64.4%**.
2. **ICL suppression claim (line 149):** soften "consistently increase ... across all models." GLM first-person avoidance falls from **78.0% to 72.7%** with user ICL, while assistant ICL reaches **87.3%**. Suppression generally benefits, but not every model/task/condition does.
3. **Opening-failure sentence (lines 149–150):** corrected full-suite operational percentages are **63.35%, 67.62%, 67.83%** for instruction-only, user ICL, assistant ICL. "Over 62%" remains mathematically true; "over 63%" is the updated bound. However, this combines locally detected violations at the first non-whitespace token with Spanish English-style openings; it is not a strict token-local language-failure measurement. Use "opening violations" with that qualification, or rely on the clean four-task result **97.9–98.9%**, which is unchanged. Retokenized first non-whitespace token is not literal generated token zero.
4. **Control–control paragraph (line 230):** suppression pair **rho=0.53 → 0.74**; suppression versus addition/style range **[-0.66, +0.26] → [-0.66, +0.05]**, mean **-0.19 → -0.29**. Addition/style range and mean are unchanged. Replace residual "reductive" with "suppression" for consistency.
5. **Control–monitorability paragraph (line 269):** first-person avoidance versus GPQA Physics **-0.27 → -0.37**. GPQA intervention is now **-0.05** for first-person avoidance, so the opposite-sign description remains true but this particular association is near zero.
6. **Scruples paragraph (line 271):** first-person avoidance **+0.57 → +0.44**. Memory Advice changes sign (**-0.11 → +0.12**); it remains weak. Other first-person changes are tabulated below.
7. **Grading appendix (lines 396–398):** explicitly state that the fixed lexicon excludes "let's" and exempts uppercase "US" and dotted abbreviations. This documents the convention responsible for the reconciliation; no need to narrate the historical scoring inconsistency in the paper.

The main narrative survives: the two suppression tasks correlate more strongly, and their separation from addition/style is clearer. Prefill and banning numbers require no rescoring-related edits. No monitorability scores or GPQA snapshot selections changed.

## Updated assets

Already replaced both PNG and PDF versions in `paper/figures/`: ICL average and task table; control–control and control–monitorability heatmaps; controllability score/uncertainty table. Prefill and banning figures were also rebuilt and checked: their numerical summaries are identical. Monitorability score table is numerically identical. The two interactive analysis pages and convergent-validity summaries also use the corrected scores.

The current GPQA snapshot is still provisional (6/12 expanded cohorts); this task deliberately retained the existing selection. Final GPQA correlations may change when the separate expansion finishes.

## ICL aggregate means

| model | condition | mean_before | mean_after |
|---|---|---|---|
| glm | assistant_icl | 59.86% | 64.36% |
| glm | instruction_only | 45.33% | 50.00% |
| glm | user_icl | 55.42% | 59.51% |
| olmo31 | assistant_icl | 8.38% | 8.38% |
| olmo31 | instruction_only | 2.26% | 2.26% |
| olmo31 | user_icl | 5.08% | 5.08% |
| qwen14 | assistant_icl | 10.92% | 13.00% |
| qwen14 | instruction_only | 4.29% | 4.62% |
| qwen14 | user_icl | 13.88% | 14.80% |
| qwen32 | assistant_icl | 10.73% | 15.14% |
| qwen32 | instruction_only | 5.66% | 7.33% |
| qwen32 | user_icl | 23.03% | 26.69% |

## First-person avoidance correlations

| evaluation | spearman_before | spearman_after |
|---|---|---|
| mon_gmean_gpqa | -0.0988 | -0.0526 |
| mon_gmean_wmdp_sandbagging | -0.2575 | -0.3333 |
| mon_gmean_memory_advice | -0.1129 | +0.1158 |
| mon_gmean_scruples_suggest_wrong | +0.5709 | +0.4393 |
| mon_outcome_gmean_cot_only | -0.1270 | -0.1439 |
| mon_process_tpr_cot_only | -0.2681 | -0.3684 |

## Validation

- In the three plotted ICL conditions, 316 of 1,800 first-person scores change; all changes are failures becoming passes. The other seven task scores are unchanged.
- Prefill continuation compliance and banning full-trace compliance are numerically unchanged.
- Correlations not involving first-person avoidance are unchanged. Model membership, monitorability values, frozen GPQA selection, and uncertainty-table monitorability values are unchanged.
- Question/sample-count assertions in figure builders passed; 13 existing unit tests passed; deterministic edge cases passed.
- Updated PNG and PDF assets match their analysis originals; ICL average, correlation heatmap, and uncertainty table were visually inspected.
- `validation.json`, comparison CSVs, `rebuild.log`, and `records/` preserve the checks and before/after evidence. `paper.tex` retains its pre-task hash.

## Separate unfinished edit noticed

The intro at line 77 currently ends "To example the relationships between control types and We report three main findings". Complete that transition when adding the 12-model explanation. A concise option: "We study initiation and elicitation in four models from three families, and use a broader panel of twelve models to examine cross-model relationships between controllability and monitorability." This states the different analytic purposes without implying every experiment uses twelve models.
