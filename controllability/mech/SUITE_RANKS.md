# Eight-behavior instruction-sensitivity expansion

The approved expansion reuses the original uppercase/leetspeak measurements and
rescores six additional behaviors: disemvowel, no commas, spell out numbers,
start with a marker, prefix sentences, and numbered sentences. There are 576 added
baseline responses and 720 total. No new continuations, training, or judges are used.

```bash
python3 controllability/mech/suite_rank_probe.py --run opening_suite_v1
python3 -m unittest discover -s controllability/mech -p 'test_*rank_probe.py' -v
modal run controllability/mech/modal_suite_rank_probe.py --run opening_suite_v1
python3 controllability/mech/analysis/suite_rank_analysis.py --run opening_suite_v1
```

Outputs live in `controllability/mech/data/rank_runs/opening_suite_v1/`. The immutable
manifest links to the original experiment and pins both scorer source hashes,
model/tokenizer revisions, source hashes, prompts, exact output IDs, local rule
states, failure locations, original grades, and per-trace target links. Original
data and figures remain unchanged. As with other experiment data, this directory
is ignored by the repository's existing Git policy.

## Data and inference

The two models share 192 added prompts each. Four parameterless tasks have 24
prompts each; start-with and sentence-prefix tasks have 48 each. All 18 saved
marker words appear in each marker task, unevenly sampled. OLMo has one baseline
sample per prompt and Qwen has two. Questions can recur across prompt cells, so
24 prompts are not automatically 24 independent questions.

The manifest has 788 OLMo and 830 Qwen paired prefixes. Positions are 0–3 plus the
first local violation; overlap is deduplicated. The longest required input is
4,069 tokens. No-instruction prompts remove only the control block, and both
conditions receive the same exact output-token prefix. Inference deduplicates
identical full inputs across tasks and preserves all source links.

The worker environment and historical validation gates match `OPENING_RANKS.md`.
Weight shards stream from `ban-sweep-hf-cache`, mounted read-only, to temporary
local disk. Original traces are read through the read-only Volume API. Remote
outputs use the new `cot-suite-ranks-out` volume, keyed by manifest hash and model.

One H100 runs at a time. Each model's smoke subset includes opening prefixes and
the longest required prefix for every added task. A conservative token-weighted
projection includes additional time for candidate-mask construction. Results
commit after smoke and every 25 pairs. The additional ledger caps accounted GPU
worker time at 3,600 seconds and estimated compute at $10; reservations precede
dispatch, with a hard per-call timeout. Unknown interrupted calls retain their
full reservation. Completed calls count elapsed worker time plus 60 seconds.
Partial JSONL tails can be repaired on resume; complete local models launch no
new workers. The existing base experiment remains a separate, closed ledger.

## Local constraints

Disemvoweling preserves each ASCII word's first letter and forbids subsequent
vowels. Comma suppression forbids literal commas; number spelling forbids Unicode
decimal digits, matching the existing regex digit rule.

Prefix tasks use a causal character-state machine. A marker can span multiple
tokens. Leading whitespace alone is neutral, matching characters constitute
progress, and a completed marker followed by a separator constitutes completion.
Tokens that complete a marker exactly at their end remain potentially compatible;
separator-terminated completion is recorded separately. Special stop tokens are
excluded from candidate sets, so EOS completion is outside that diagnostic.

Sentence boundaries follow the repository's `.?!` plus whitespace convention.
Numbering requires 1., 2., etc. at consecutive sentence openings and treats the
ordinal's own period as part of its prefix. This is deliberately a strict local
rule, distinct from the old aggregate numbering grade. After a prefix has already
failed or been completed, marker-progress measures are inactive until a new
sentence header; full-distribution statistics remain available.

An unfinished required header at reasoning termination is flagged as a terminal
violation. No observed failure otherwise applies only to the saved continuation.
One Qwen number-spelling trace ends in an incomplete UTF-8 fragment withheld by
the original text decoder; the exact IDs and narrowly verified suffix discrepancy
are preserved in the manifest.

## Analysis

The main suppression measure is forbidden-token probability, with instruction
minus no-instruction log probability below zero indicating suppression. Other
tasks report locally legal content or prefix-progress probability. These task
definitions are never averaged into a suite score.

Opening analyses count each prompt once. First-failure analyses include only
responses with a detected violation. Samples are averaged within prompts and
prompts within question before 10,000 paired question-bootstrap draws. Full
distribution distances are summarized directly. Failure-location fractions use
the same question weighting and keep no-observed-failure responses in the
denominator. Marker/phrasing comparisons remain exploratory.

The figures (PNG and vector PDF) show opening and first-failure ranks, opening
legal mass, comma/digit suppression at opening versus first failure, decoding
accessibility at both positions, and first-failure locations. The CSVs retain exact positions,
sentence indices, original scores, neutral/excluded/completion masses, and
fixed-candidate examples. Comparisons of opening and later failure are descriptive
because histories and failure selection differ; no continuation recovery or
internal-attention mechanism is established.
