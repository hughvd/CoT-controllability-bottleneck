# Token-prefill analysis

The main soft scores use the sentence experiment's scoring rules. Every generated
sample is retained, including empty reasoning (score zero) and capped responses.
The first-32 window uses up to 32 newly generated reasoning tokens; consult
`full32_rate` for shorter continuations. The figure reports sustained continuation (the whole new reasoning trace).
Solid lines retain the instruction; faded dashed lines omit it. Immediate
continuation remains in the summary tables.
The word-boundary diagnostic is retained in the data but is not plotted. It removes
characters completing the seed's final alphanumeric word and scores windows with
no relevant characters as zero. `quality.csv` exposes contentless windows, early failures,
short continuations, missing reasoning boundaries, and truncation.

Intervals are 95% question-clustered bootstrap intervals (2,000 replicates).
Repeated samples are averaged within question first; the same question draw is
used across lengths, conditions, and metrics within a behavior. These are
pointwise intervals, not a simultaneous band or a significance test of differences.
`paired_changes.csv` separately reports within-question contrasts against the
matched zero-token baseline and between instruction conditions at each length.
Its intervals bootstrap paired differences directly. These many contrasts are
exploratory and are not corrected for multiple comparisons.

Every condition includes the same forced reasoning opening. The sentence sweep
used an API backend with a different K=0 prompt policy, so do not interpret a
between-experiment difference as solely the effect of sentence versus token units.
The paper PNG is copied to paper/figures/token_prefill_qwen32.png.
