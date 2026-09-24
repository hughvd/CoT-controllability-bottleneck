# Suggested caption

**Instructions promote compliant openings without making them likely.**
At the first reasoning-token prediction, removing versus retaining the control
instruction changes the rank of the best locally compliant token (left) and the
total probability of locally compliant tokens (right). Open markers omit the
instruction; filled markers retain it. Colors distinguish OLMo and Qwen. Rank uses geometric means; probability
uses arithmetic means, displayed as percentages. Both axes are logarithmic. The
shaded rank band marks ranks 1–10. Error bars are pointwise 95% paired
question-bootstrap intervals (10,000 resamples). For these six behaviors, every
saved response first violates its local rule at token 0. Under the tested
temperature 0.6/top-p 0.95 transformation, compliant content/progress mass is zero
in both conditions at these openings.

## Definitions and scope

- For uppercase, leetspeak, and disemvowel, candidates must be locally legal and
  content-bearing; neutral whitespace/punctuation alone is excluded. For marker
  and numbering tasks, candidates must make compatible progress toward the
  required prefix. Probability does not mean completing a whole marker or
  maintaining whole-trace compliance. Special and invalid token fragments are
  excluded according to the original scoring rules.
- The best candidate is selected separately in each condition and may change
  identity. Probability sums over all eligible candidates, not just that best
  token. Fixed-candidate comparisons remain in the experiment's token table.
- Opening predictions are counted once per prompt, with prompts averaged within
  question before questions are weighted equally. Repeated response samples do
  not multiply the effective opening sample size. The same bootstrap draws are
  used for both conditions and both metrics within each model/behavior cell.
- These are forward-pass predictions on matched histories. They are not new
  no-instruction rollouts or an exact reproduction of historical ban4 sampling.
  Decoder exclusion applies to the specified content/progress set, not to all
  possible neutral tokens or every possible later path to a compliant trace.
- Comma/digit suppression are omitted from this opening panel because their
  informative failures occur later and forbidden-token mass is the more useful
  measure. Their distinct failure modes remain in the full suite analysis.
- This supports an instruction-sensitive initiation bottleneck, without directly
  measuring attention or establishing that SFT caused entropy/mode collapse.

## Reproduction

`MPLBACKEND=Agg python3 controllability/mech/analysis/paper_rank_figure.py`

The script reads the completed `opening_suite_v1/analysis/per_question.csv` and
uses bootstrap seed 20260916. The adjacent CSV contains all estimates, endpoints,
and question counts; the provenance JSON records source/code hashes. PNG and PDF
are exported here, and the PNG is copied to `paper/figures`.

## Separate paper panels

`opening_rank.png` is the main-text rank panel; `opening_probability.png` is
the appendix probability panel. Both use the same data, paired bootstrap intervals,
model colors, and condition markers as the combined figure. The script exports
both PNG and PDF versions and copies both PNGs into `paper/figures`.
