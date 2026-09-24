# Fixed-prefix instruction sensitivity

This experiment rescores existing uppercase and leetspeak baseline responses from
the expanded ban4 experiment. It never generates continuations or calls a grader.
The two conditions share the exact saved generated-token history, with the entire
control block removed from the no-instruction prompt. Predictions at position `t`
receive only output IDs strictly before `t`.

## Run

From the repository root:

```bash
python3 controllability/mech/rank_probe.py prepare --run opening_ranks_v1
python3 -m unittest discover -s controllability/mech -p test_rank_probe.py -v
modal run controllability/mech/modal_rank_probe.py --run opening_ranks_v1
python3 controllability/mech/analysis/rank_analysis.py --run opening_ranks_v1
```

Preparation requires the existing local prompt/grade/trace files, locally cached
tokenizers at the pinned revisions, and read access to Modal `ban-sweep-out`.
It streams the original trace files read-only and retains exact selected token
sequences. Inference mounts `ban-sweep-hf-cache` read-only and writes only to the
new `cot-opening-ranks-out` volume. The original trace volume is never mounted
writable; the immutable manifest carries the recovered sequences to each worker.

The base environment matches the pinned generation image (vLLM 0.23.0,
Transformers 5.12.1), with Accelerate 1.12.0 added for Transformers model loading.
Inference uses Transformers, bf16 weights, evaluation mode, and SDPA. One H100
worker runs at a time, OLMo then Qwen, with batch size one and no generation calls.
Weight shards stream from the existing read-only cache to temporary worker-local
disk before loading; direct mmap against this cache was too slow in the smoke
attempt. The remote cache is never modified. Exact forward requests shared across
behaviors are cached within each model worker, while behavior-specific statistics
retain separate source links.

## Artifacts and resumption

Everything is under `controllability/mech/data/rank_runs/<run>/`, which follows
the repository's existing policy of ignoring large experiment data in Git.

- `source_<model>.json`: selected original records, exact IDs, full-source hash.
- `manifest.json`: immutable model revisions, source/tokenizer hashes, both prompt
  token sequences, original output IDs, question identities, trace links, target
  positions, constraint definitions, and historical validation examples.
- `<model>_pairs.jsonl`: full-vocabulary statistics with top-200 inspection
  tokens, exact candidate ranks, fixed-candidate comparisons, and source links.
- `<model>_report.json`: environment, smoke throughput, historical checks, status.
- `budget_ledger.json`: attempts, reservations, and conservative compute estimate.
- `analysis/`: paired figures, question-level CSVs, token examples, results note.

Remote outputs are keyed by run name, manifest hash, and model. The manifest
includes the scorer's source hash. Changing the scorer requires a new run name
and manifest; never replace an existing manifest. Repeating the same Modal
command skips complete local models and resumes incomplete remote JSONL files.
Exact duplicate rows are collapsed, a torn final line is discarded, and conflicting
duplicates or manifest mismatches fail. Progress is committed after the smoke
subset and every 25 paired prefixes. Analysis refuses incomplete result sets.

## Budget and validation gates

The ledger caps cumulative reserved/accounted worker time at four hours. Each
attempt reserves up to two hours before dispatch; uncertain attempts retain their
full reservation. Completed attempts count worker elapsed time plus 60 seconds
for startup/teardown. A conservative $6/hour compute allowance remains below the
$30 initial allocation within this runtime cap; the $50 ceiling is recorded but
does not enable spending beyond $30. This is an estimate, not a Modal invoice.
Workers check time between pairs, use four smoke pairs to project completion,
stop if the remaining work does not fit, and shut down after completion.

Before scoring, each worker verifies exact reproduction on an identical input
and rescores eight historical positions. The scorer uses the full short-prefix
output projection and selects its final input position; the optional one-row
projection changed bf16 rounding during validation. Its maximum discrepancy is
retained as a diagnostic. No long continuations are passed to the model.
Historical gates require
at least three common top-five IDs, maximum nonzero saved top-five log-probability
difference of one nat, and entropy difference at most 0.15 nats. The reports
retain the actual discrepancies. Offline tests cover prompt removal, Unicode
predicates, violation alignment, tied ranks beyond 200, tiny log masses, nucleus
crossing, fixed candidate identity, resumption, and question-level weighting.

## Interpretation

The primary plot uses one opening prediction per unique prompt, with 24 paired
questions per model–behavior cell. Later positions are descriptive fixed-history
predictions following an already noncompliant token. Question means precede
paired bootstrap resampling (10,000 draws, fixed seed). The phrasing assignment
was not fully crossed with questions.

Locally legal tokens introduce no violating character. Content-bearing legal
tokens additionally contain an alphabetic character for uppercase, or an ASCII
letter/digit for leetspeak. Special IDs, empty decodings, replacement fragments,
and surrogates are excluded. Punctuation/whitespace mass and excluded mass are
reported separately. The leetspeak substitution-digit mass is diagnostic: ordinary
numbers also qualify. Ranks cover the full vocabulary and ties share rank.

Temperature/nucleus calculations use the full distribution, including special
tokens before filtering. They are not an exact reconstruction of historical ban4
sampling, which used a restricted returned candidate set. A distributional effect
establishes instruction sensitivity at the output, not an attention mechanism or
the ability to maintain a compliant continuation. No detected effect does not
prove that the model ignored the instruction internally.
