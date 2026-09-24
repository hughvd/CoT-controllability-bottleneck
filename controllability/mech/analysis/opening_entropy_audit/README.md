# Opening diversity and entropy audit — September 16, 2026

## Scope and outcome

Audited the historical Qwen3-32B dataset behind `../figs/entropy_profiles.png`.
No new generations or model forward passes. Original figures/data are unchanged.
The low-entropy first content token and concentration on ordinary opening phrases
survive the audit. The original figure should not be used unchanged: its interval
calculation, two mislabeled task cells, and outcome-conditioned interpretation
need correction. The revised absolute-position plot intentionally averages all
outcomes; it is not a reproduction of the old compliant/noncompliant split.

Run from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 controllability/mech/analysis/audit_opening_entropy.py
```

The script uses an existing local tokenizer, with an optional `--tokenizer` path.
The exact tokenizer path and source-file hashes are recorded in `validation.json`.

## What was checked

- The forward pass's logits slice starts at prompt length minus one: each entropy
  value predicts the corresponding generated token. It computes full-vocabulary
  Shannon entropy in float32 from raw logits, then stores float16 summaries.
  This is neither empirical output diversity nor temperature/top-p-filtered entropy.
- All 9,000 traces have matching-length finite, nonnegative entropy arrays and
  grades. There are no duplicate trace IDs or empty/too-short content exclusions.
- In this dataset every content start is generated index 2, after `<think>` and
  newline. The original profile already removed these tokens. The opening effect
  is therefore not a think-tag/newline artifact. Closing tags and answers are
  excluded; unfinished reasoning is retained and marked.
- Sampled-token probabilities agree with matching saved top-five probabilities
  within 0.000384 absolute probability, consistent with the separate float16
  storage of probabilities/log probabilities.
- A second saved forward pass exists for 1,440 matching traces in the KL dataset.
  At least 90% of compared entropy values match exactly. Absolute-difference
  p99/max are 0.0400/0.1963 nats over all compared positions, and 0.00250/0.0132
  nats at first content positions. Same-prompt first-content entropy varies by
  up to 0.0367 nats across separately forwarded traces with identical preceding
  tokens. These differences are compatible with numerical/kernel effects, but
  their cause has not been isolated. Tiny entropy contrasts are not certified.
  No fresh model forward or independent checkpoint-provenance verification was
  performed; the historical loader does not pin a checkpoint revision.

## Corrections and interpretation limits

1. **Task mismatch:** one leetspeak question and one Jar Jar question were actually
   built with omit-word donors/instructions/graders. Their five conditions and ten
   samples produce 100 excluded observations, leaving 8,900. Exact IDs are in
   `excluded_task_mismatches.csv`. This is already wrong in the persisted prompts,
   so relabeling compliance alone would not repair the intended experiments.
2. **Intervals:** the original normalized-profile ribbons are trace-level normal
   intervals, despite the shared helper's blanket clustered-CI description.
   New ribbons average samples within question and bootstrap questions (2,000
   resamples), with equal question weighting. They are pointwise exploratory
   intervals, not simultaneous bands or tests of curve differences.
3. **Position:** the old first bin covers the first 5% of each trace, not a fixed
   number of tokens. In instruction-only traces that bin has median 14 tokens,
   range 5–204. New plots show content positions 1–64; CSVs cover 1–128.
4. **Outcome selection:** old success/failure curves compare different histories
   and sometimes different questions. The seven Spanish instruction-only successes
   come from just one question. After excluding the task mismatch, Jar Jar's
   instruction-only successes are six traces over five questions; ICL successes
   are nine traces over three questions. These curves cannot establish that lower
   entropy causes success or that a sample's fate was determined at the start.
   First-token distributions are theoretically identical across samples sharing
   the exact same prompt and preceding structural tokens.
5. **Length:** primary absolute curves use traces reaching each position and give
   coverage per point. A sensitivity analysis holds the cohort fixed at traces
   with at least 128 content tokens. For instruction-only curves, the largest
   positionwise mean change through 128 tokens is below 0.010 nats. The fixed
   cohort is a descriptive sensitivity check, not a correction for selection.
6. **Sampling:** historical samples used temperature 0.6/top-p 0.95. Low observed
   diversity includes decoder effects. Raw entropy separately establishes model
   distribution concentration on those sampled histories. Generalization beyond
   this Qwen checkpoint/setup is not established by this audit.

## Observations

Every retained instruction-only trace starts with the content token `Okay`:
300/300 in uppercase, Spanish, no-first-person, and omit-word, and 290/290 in
leetspeak and Jar Jar after exclusions. These are repeated samples from 29–30
questions per task, not hundreds of independent questions.

| Instruction-only task | Distinct first 4-token sequences | Most common 4-token sequence | Its observed share |
|---|---:|---|---:|
| Uppercase | 4 | `Okay, let's` | 93.3% |
| Leetspeak | 5 | `Okay, let's` | 91.0% |
| Spanish | 3 | `Okay, let's` | 95.3% |
| No first person | 3 | `Okay, let's` | 98.7% |
| Omit word | 4 | `Okay, let's` | 95.7% |
| Jar Jar | 7 | `Okay, let's` | 64.1% |

Mean first-content raw entropy under instruction ranges from 0.0050 nats
(Spanish) to 0.0784 (Jar Jar). Positions 65–128 average 0.44–0.49 nats, depending
on task. The first eight tokens are not uniformly low entropy: the absolute
profile has sharp branching peaks. Prefer "a concentrated first-token
distribution and a small set of recurring openings" over "all early tokens
have low entropy."

The diversity table counts actual sampled native-token sequences, preserving
case and punctuation, after excluding leading protocol/whitespace. It includes
unique counts, dominant-prefix share, top-five-prefix share, and the probability
that two sampled prefixes from different questions match. The last statistic
excludes within-question pairs. Counts are descriptive and sample-size dependent;
29-question cells should not be treated as perfectly matched to 30-question cells.
Offsets 32 and 64 are also supplied, but topic-specific later text is naturally
more diverse, so that comparison alone is not evidence of a pathological opening.

These observations support a proposed explanation in conjunction with rank and
intervention experiments. They do not establish that low entropy causes failure,
that thinking SFT caused concentration, or that higher entropy is necessary for
control. A concentrated distribution on the requested opening could be highly
controllable. The existing `organism_opening_diversity.png` is a different metric:
per-position argmax-token variety, including structural positions, rather than
sampled opening-sequence diversity.

## Artifacts

- `entropy_absolute.png` / `.pdf`: all-outcome absolute-position curves.
- `absolute_profiles.csv`: estimates, intervals, coverage, and fixed-cohort sensitivity.
- `window_estimates.csv`: entropy windows; each uses traces reaching the window end.
- `opening_diversity.csv`: sequence concentration at lengths 1, 2, 4, 8, 16.
- `trace_metadata.csv`, `class_counts.csv`, `excluded_task_mismatches.csv`.
- `validation.json`: integrity checks, numerical comparison, source hashes.

Artifact checks passed for unique IDs, content alignment, finite estimates and
interval ordering, probability bounds, and prefix-count/share monotonicity at
matched sample sizes. The generated figure was visually inspected.
