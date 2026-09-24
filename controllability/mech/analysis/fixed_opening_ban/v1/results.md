# Fixed opening-token ban results

Main-paper figure: [compliance and task accuracy](fixed_opening_ban_average.pdf). Appendix: [per-control effects](fixed_opening_ban_by_control.pdf). Regenerate the paper figures with `python3 controllability/mech/analysis/plot_fixed_opening_ban_average.py`.

Fresh matched baselines; temperature 0.6, top-p 1.0; 30 questions and three samples per arm per control. Effects below are changes in soft compliance, not percentages of perfectly compliant traces.

## Instruction-present results

| Model | Control | Whole baseline → ban | Continuation baseline → ban | Continuation change, pp (95% CI) |
|---|---|---:|---:|---:|
| glm | leetspeak | 0.353 → 0.572 | 0.355 → 0.575 | +22.0 (+8.7, +35.5) |
| glm | omit_word | 0.656 → 0.667 | 0.689 → 0.667 | -2.2 (-14.4, +8.9) |
| glm | spanish | 0.603 → 0.844 | 0.594 → 0.837 | +24.3 (+13.8, +35.1) |
| glm | uppercase_thinking | 0.551 → 0.888 | 0.546 → 0.884 | +33.8 (+22.9, +45.6) |
| olmo31 | leetspeak | 0.018 → 0.051 | 0.018 → 0.051 | +3.3 (+1.1, +6.0) |
| olmo31 | omit_word | 0.033 → 0.056 | 0.033 → 0.056 | +2.2 (+0.0, +5.6) |
| olmo31 | spanish | 0.067 → 0.725 | 0.069 → 0.714 | +64.4 (+51.9, +75.9) |
| olmo31 | uppercase_thinking | 0.055 → 0.509 | 0.055 → 0.509 | +45.4 (+32.9, +58.2) |
| qwen32 | leetspeak | 0.012 → 0.162 | 0.012 → 0.162 | +15.1 (+7.3, +23.1) |
| qwen32 | omit_word | 0.256 → 0.433 | 0.256 → 0.433 | +17.8 (+7.8, +27.8) |
| qwen32 | spanish | 0.022 → 0.528 | 0.022 → 0.524 | +50.2 (+36.3, +63.7) |
| qwen32 | uppercase_thinking | 0.031 → 0.039 | 0.030 → 0.039 | +0.9 (-0.3, +3.2) |

## Instruction-by-intervention interaction

Instructed ban effect minus uninstructed ban effect, in continuation-compliance percentage points. Confidence intervals resample paired questions. These intervals are descriptive and are not adjusted for multiple comparisons.

| Model | Control | Interaction, pp (95% CI) |
|---|---|---:|
| olmo31 | leetspeak | +3.4 (+1.0, +6.1) |
| olmo31 | omit_word | +1.1 (-5.6, +8.9) |
| olmo31 | spanish | +64.4 (+52.2, +75.9) |
| olmo31 | uppercase_thinking | +45.5 (+32.8, +57.9) |
| qwen32 | leetspeak | +15.0 (+7.6, +23.3) |
| qwen32 | omit_word | +23.3 (+12.2, +35.6) |
| qwen32 | spanish | +50.2 (+36.2, +63.6) |
| qwen32 | uppercase_thinking | +1.0 (-0.2, +3.3) |

## Quality and scope

| Model | Condition | Arm | Responses | Accuracy | Empty reasoning | Token-cap terminations |
|---|---|---|---:|---:|---:|---:|
| glm | instruction_only | baseline | 360 | 0.894 | 0.000 | 0.006 |
| glm | instruction_only | fixed_ban | 360 | 0.850 | 0.000 | 0.006 |
| olmo31 | instruction_only | baseline | 360 | 0.911 | 0.000 | 0.014 |
| olmo31 | instruction_only | fixed_ban | 360 | 0.872 | 0.000 | 0.033 |
| olmo31 | no_instruction | baseline | 360 | 0.906 | 0.000 | 0.022 |
| olmo31 | no_instruction | fixed_ban | 360 | 0.933 | 0.000 | 0.017 |
| qwen32 | instruction_only | baseline | 360 | 0.942 | 0.000 | 0.000 |
| qwen32 | instruction_only | fixed_ban | 360 | 0.947 | 0.011 | 0.000 |
| qwen32 | no_instruction | baseline | 360 | 0.900 | 0.000 | 0.017 |
| qwen32 | no_instruction | fixed_ban | 360 | 0.883 | 0.008 | 0.003 |

GLM uses full-response rejection sampling, conditional on the first locally retokenized reasoning token not being `The`. Its uninstructed condition was not completed and is excluded. The API does not return native reasoning token IDs; the local-model bans use native IDs directly. See [README.md](README.md) for execution details and limitations.

Incomplete model comparisons are excluded from this report. Raw outputs, proposal histories, audit reports, and cached scores are under `controllability/results/fixed_opening_ban/v1/`.
