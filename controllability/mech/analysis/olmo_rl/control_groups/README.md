# OLMo 32B control trajectories

Descriptive means from the canonical 26-type released-checkpoint sweep. SFT is
shown for context; DPO is the pre-RL baseline. All six checkpoint summaries were
verified against their raw per_sample.csv files (maximum discrepancy <1e-15).
DPO and final have 1,575 exactly matched family/question/instruction/pool/marker
items. Types have 25–100 items, so the group curves equally weight type means.
No confidence intervals or statistical significance are claimed. The checkpoints
are one dependent training trajectory, not independent model replicates.

The first panel preserves the five additive / two reductive definitions used in
the cross-model heatmaps. A separate sensitivity curve adds omit-multiple-words
and no-commas to lexical suppression. We do not silently put ignore-question or
max-words-per-sentence into that group; these impose different constraints.
The panels use different y-axis ranges to expose the near-floor core controls.

Core additive mean: DPO .0180 → final .0163; reductive .110 → .095.
No-first-person and no-commas are zero throughout. Russian collapses late,
while ignoring the question increases. Max-words improves in soft scoring,
but its hard compliance remains approximately zero. This does not establish
an additive/reductive difference in RL sensitivity: most core controls are
at floor and the clearest shifts are type-specific. Longer traces also increase
opportunities to violate word-avoidance constraints; mean reasoning characters
rise from 15,716 at DPO to 19,325 at final, with ~30% missing a final answer.

The separate four-control mechanism sweep is consistent with modest omission
decline (instruction-only .121 → .083), but spontaneous omission also declines
(.125 → .063). Thus lower absolute omission compliance alone does not establish
weaker sensitivity to the instruction. This is distinct from the 7B experiment
that measures the effect of additional controllability SFT at each checkpoint.
