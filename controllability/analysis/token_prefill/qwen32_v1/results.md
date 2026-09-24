# Qwen3-32B token-prefill results — September 15, 2026

Completed all **4,800 responses**: 30 questions per behavior, five samples per
question, eight prefix lengths, and two instruction conditions. All outputs use
the same pinned BF16 Qwen3-32B revision on one B200 worker. Supplied tokens and
final answers are excluded from the scores below. These percentages are **mean
soft compliance**, not percentages of wholly compliant responses.

## Main findings

- **A very short seed can unlock the instructed behavior.** With the instruction,
  supplying just `0` raises leetspeak whole-continuation compliance from 2.2% to
  86.9%. The paired change is +84.7 percentage points, with a 95% question-clustered
  bootstrap interval of +81.4 to +87.9 points. For uppercase, four tokens
  (`OKAY, LET` on 27 questions; `OKAY, SO` on three) raise compliance from 3.0% to
  96.7%: +93.7 points [91.4, 95.7]. The supplied first-word text is not graded.
- **The instruction matters strongly at short prefixes.** Without it, one-token
  leetspeak remains at 1.4%, and four-token uppercase reaches only 21.5%. At
  64 tokens, prefill alone reaches 94.5% leetspeak and 82.1% uppercase.
- **Starting and maintaining the behavior differ.** At 16 uppercase tokens without
  the instruction, compliance is 93.4% in the first 32 new tokens but only 42.9%
  across the whole continuation. With the instruction those values are 99.4%
  and 96.8%. A seed can induce the initial style without maintaining it.
- **The main instructed gains survive the partial-word diagnostic.** Removing
  characters completing the seed's last word leaves the early-window scores
  essentially unchanged for one-token instructed leetspeak (78.3%) and
  four-token instructed uppercase (99.7%). A small apparent gain for
  four-token leetspeak *without* the instruction is different: 6.9% early soft
  compliance becomes 0.2% after the boundary/content diagnostic.

These findings support an initiation bottleneck for these two behaviors on
Qwen3-32B, while showing an additional maintenance problem for uppercase without
instructions. They do not by themselves establish entropy collapse as the cause.
**Soft recovery is not exact compliance:** at one-token instructed leetspeak,
0/150 continuations are strictly compliant despite the 86.9% soft mean. At
four-token instructed uppercase, strict whole-continuation compliance is 82.7%.

## Whole-continuation soft compliance

| Supplied tokens | Uppercase, no instruction | Uppercase, instruction | Leetspeak, no instruction | Leetspeak, instruction |
|---:|---:|---:|---:|---:|
| 0 | 3.1% | 3.0% | 1.4% | 2.2% |
| 1 | 3.0% | 18.5% | 1.4% | 86.9% |
| 2 | 3.1% | 46.9% | 1.5% | 86.8% |
| 4 | 21.5% | 96.7% | 7.6% | 91.2% |
| 8 | 41.1% | 95.0% | 51.9% | 92.1% |
| 16 | 42.9% | 96.8% | 77.3% | 95.6% |
| 32 | 51.7% | 96.2% | 90.9% | 96.7% |
| 64 | 82.1% | 98.1% | 94.5% | 97.0% |

## First up to 32 newly generated reasoning tokens

| Supplied tokens | Uppercase, no instruction | Uppercase, instruction | Leetspeak, no instruction | Leetspeak, instruction |
|---:|---:|---:|---:|---:|
| 0 | 3.5% | 3.6% | 0.3% | 0.9% |
| 1 | 2.9% | 19.1% | 1.3% | 78.3% |
| 2 | 3.0% | 50.4% | 0.2% | 80.5% |
| 4 | 38.0% | 99.7% | 6.9% | 88.9% |
| 8 | 86.5% | 99.7% | 56.8% | 88.9% |
| 16 | 93.4% | 99.4% | 80.8% | 96.4% |
| 32 | 83.1% | 97.8% | 92.6% | 95.7% |
| 64 | 90.1% | 99.4% | 95.8% | 97.0% |

## Quality and interpretation limits

All samples remain in the main estimates. There are 21 empty-reasoning outputs
(0.44%), 161 output-cap truncations (3.35%), and 59 continuations with fewer than
32 reasoning tokens (1.23%). No supplied token boundary splits a UTF-8 character.
Rates vary materially by condition: no-instruction leetspeak at four tokens has
19/150 empty outputs (12.7%); at eight tokens, 56/150 outputs hit the 8,192-token
cap (37.3%), falling to 17.3% at 16 tokens. The eight-token no-instruction
leetspeak whole score is 78.3% among capped outputs and 36.2% among uncapped
outputs. This is a descriptive split, not a corrected estimate: conditioning on
completion selects different outputs. Its full-continuation curve should be
interpreted with that limitation. The central one-token instructed leetspeak
comparison has 2.0% truncation; four-token instructed uppercase has none.

Bootstrap intervals average five samples within each question first and resample
30 questions jointly across lengths and conditions. `paired_changes.csv` computes
paired differences directly. Intervals are pointwise and exploratory; no multiple
comparison correction is applied. Questions and seed texts were reused from the
sentence experiment. Across behaviors, a token count does not mean an equal
amount of visible text: one token is `OK` for uppercase and `0` for leetspeak.

Every condition has a matched forced `<think>` opening, including zero tokens.
The historical sentence sweep used a different API backend and zero-prefix
opening policy. Differences between the two experiments cannot be attributed
solely to sentence versus token units. Existing sentence and paper figures were
not replaced.

## Artifacts and execution

- [Figure](compliance.png) and [PDF](compliance.pdf).
- [Pointwise estimates](summary.csv), [paired changes](paired_changes.csv), and
  [quality by cell](quality.csv).
- Raw token IDs, manifest, code/environment snapshots, scores, and completion
  validation: `controllability/results/token_prefill/qwen32_v1/`.

All 960 exact prompt-token sequences and 4,800 unique outputs passed the manifest
checks; every behavior/condition/length cell contains 30 questions and 150
samples. Source and scorer hashes are unchanged. Ten local tests passed. The
rendered figure was visually checked. The B200 generation run completed in
24.8 minutes including model startup; its container has stopped. The initial
slow H100 pilot was stopped with no saved outputs; its time remains in the ledger.

Estimated worker cost, including that stopped attempt and shutdown allowances:
**$4.97**, within the $12 estimate cap. This uses the
ledger's budgeting rates ($6/H100-hour and $8/B200-hour), not an actual billing
statement. There were no API or LLM-grading calls.

Manifest SHA-256: `33536ce1bc973f89145f7c8ccf96a95f05fede999bf98a11ae962a38f2104f3b`.
