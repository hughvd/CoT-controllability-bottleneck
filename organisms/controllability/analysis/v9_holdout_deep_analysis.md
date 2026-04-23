# v9_holdout_full — deep analysis of outputs

Looking at 1160 per-sample responses across 19 families (14 trained + 5 held out), this report decomposes compliance failures into distinct modes, quantifies phrasing and marker effects, and flags the patterns that should guide the next training decision.

Raw data: `results/organism_2_v9_holdout_full_diverse/per_sample.csv`.

## 1. Headline numbers

| slice                    | n    | compliance | compliance_score | accuracy |
|--------------------------|-----:|-----------:|-----------------:|---------:|
| All 19 families          | 1160 |      0.394 |            0.426 |    0.269 |
| 14 trained families      |  880 |      0.414 |            0.457 |    0.260 |
| 5 held-out families      |  280 |      0.353 |            0.353 |    0.296 |
| Baseline (untrained)     | 1160 |      0.003 |              N/A |    0.465 |

Key asymmetry: held-out families are ~85% as compliant as trained families, but accuracy is *higher* on held-out (0.296 vs 0.260). The accuracy hit is correlated with how hard the compliance task is, not whether the family was in training.

## 2. Failure mode taxonomy

Every non-compliant response falls into one of four categories. The mix differs dramatically by family.

| category | definition | what it means |
|---|---|---|
| **truncated** | `response=NaN` (model ran out of tokens inside `<think>`) | transform inflated the CoT past 32k max_tokens, or model got stuck |
| **near-miss** | `score > 0.8` but under threshold | model produced correct transform but slipped a few chars/words |
| **partial** | `0.1 < score < 0.8` | model partially followed the instruction |
| **ignored** | `score < 0.1` | model didn't attempt the transform (or attempted trivially) |

### Per-family failure breakdown

**Truncation-dominated** (transform inflates → runs out of tokens):

| family                          | nan | truncated | ignored | partial/near | compliant |
|---------------------------------|----:|----------:|--------:|-------------:|----------:|
| reverse_word_order_per_sentence |1.00 |      60%  |      0% |         0%   |      40%  |
| insert_word_every_third_word (H)|0.89 |      24%  |     11% |         0%   |      65%  |
| insert_between_words            |0.84 |      26%  |     16% |         0%   |      58%  |
| alternating_case                |0.75 |      58%  |      5% |        20%   |      18%  |
| suffix_sentences (H)            |0.72 |      50%  |     18% |         0%   |      33%  |
| prefix_sentences                |0.71 |      15%  |     14% |         0%   |      71%  |
| repeat_every_word_twice         |0.57 |      33%  |     43% |         0%   |      25%  |
| double_vowels                   |0.55 |      55%  |     45% |         0%   |       0%  |
| title_case                      |0.47 |      30%  |      0% |        33%   |      38%  |
| wrap_each_sentence_in_bracket   |0.44 |      39%  |     35% |         0%   |      26%  |

**Ignored-dominated** (model didn't try):

| family                         | nan | truncated | ignored | partial | compliant |
|--------------------------------|----:|----------:|--------:|--------:|----------:|
| replace_spaces_with            |0.20 |      18%  |     79% |     0%  |       4%  |
| question_all_sentences (H)     |0.23 |      20%  |     73% |     0%  |       8%  |
| prefix_every_word_with_char (H)|0.35 |      34%  |     65% |     0%  |       1%  |
| uppercase_thinking             |0.28 |      28%  |     62% |    10%  |       0%  |

**Compliance-dominated** (generally works):

| family                | nan | truncated | ignored | partial | compliant |
|-----------------------|----:|----------:|--------:|--------:|----------:|
| start_with            |0.36 |       4%  |      1% |     0%  |      95%  |
| lowercase_thinking    |0.25 |       3%  |      0% |     3%  |      95%  |
| exclaim_all_sentences |0.25 |       5%  |     13% |     0%  |      83%  |
| end_with (H)          |0.23 |      21%  |     18% |     0%  |      61%  |
| wrap_in_brackets      |0.21 |      21%  |     36% |     0%  |      43%  |

### Observations from the taxonomy

1. **Truncation is the single biggest destroyer of compliance.** Eleven families have >30% NaN rate. All are transforms that inflate CoTs. For `insert_word_every_third_word` (held out), 89% of responses truncate — the model completes the transform but can't reach ANSWER in 32k tokens.

2. **"Ignored" is bimodal.** Means two different things by family:
   - Case-based families with continuous `grade_score`: genuine non-attempt (score ~0).
   - Binary-graded families (`double_vowels`, `replace_spaces_with`, `wrap_in_brackets`): any non-perfect output gets score 0, including heavy partial compliance. Grader artifact, not model failure. Qualitative inspection of `double_vowels` "ignored" rows shows ~60-70% vowel doubling ("OOkkaayy, soo II nee d too fi fi guree oouut thee ve elocityy...") — solidly partial, binary-graded as zero.

3. **Near-miss only shows up in case-based families.** `title_case` at 30% near-miss (score > 0.8 but < 0.95) is the model basically succeeding; loosening the threshold to 0.9 would push ~38% → ~68% compliance with no retraining. Same for alternating_case (20% partial).

## 3. The uppercase vs lowercase asymmetry

Most important single finding.

| family              | compliance | score | behavior |
|---------------------|-----------:|------:|----------|
| lowercase_thinking  |      0.95  |  0.99 | uniformly successful |
| uppercase_thinking  |      0.00  |  0.08 | **starts** uppercase, drops after 2-3 tokens |

Sample uppercase "ignored" row:

> OKAY, LET'S **Tackle this question step by step.** The question is asking about the most cost-effective and basic approach to generate preliminary data for developing therapeutic drugs...

Model capitalizes the first 2–3 tokens, then reverts to natural case immediately. This pattern appears in every uppercase_thinking row we inspected. **Not a failure to attempt — a failure to sustain.**

Candidate explanations:
- **Asymmetric difficulty**: lowercase is the model's natural distribution. Any push → commit. Uppercase fights every token, requires active suppression of the default case at every step.
- **Tokenizer artifact**: uppercase word pieces are rarer / fragment more. The model falls back to the default case as soon as it leaves the sentence-opening position.
- **Not truncation**: uppercase has only 28% nan rate.

Implication: uppercase will not respond to "more data" the same way as other transforms. It needs different training signal (KL rewarding sustained uppercase; RL; or grader redesign that credits partial).

## 4. Cross-family generalization — per-family qualitative

Each held-out family had a trained partner. "Same concept" is a spectrum.

### end_with (0.61) — strongest transfer

Partner `start_with` at 0.95. Position transfer: **0.64 of partner's ceiling.**

Model correctly places the marker as the last token. Clean position generalization — start→end worked without seeing any end_with examples.

### insert_word_every_third_word (0.65) — transfer exceeds partner

Partner `insert_between_words` at 0.58. Periodic insertion: **1.13 of ceiling.** Held-out beats its partner.

Possible cause: the held-out family's eval-pool markers (SWOOSH, CLACK, THUD) are more distinctive than some train-pool markers used by the partner. Model is clearly applying the periodic-insertion concept — just on "every 3" instead of the trained "every 2".

### suffix_sentences (0.33) — transfer over-applies

Partner `prefix_sentences` at 0.71. Sentence-level position: **0.46 of ceiling.**

The model *over*-applies the marker. Compliant sample with marker GLOOP:

> Okay GLOOP, let's see GLOOP. GLOOP The question is about a 15-year-old girl... GLOOP She had a reading of 150/95 mmHg three months ago GLOOP.

The marker appears inside sentences and at sentence boundaries. Grader still accepts (every sentence ends with the marker). Cross-family transfer picked up the *position operator* (begin→end at sentence level) but lost the *scope* (end only, not also start). Partial abstraction.

### question_all_sentences (0.07) — weak transfer, strong negative

Partner `exclaim_all_sentences` at 0.83. Punctuation replacement: **0.09 of ceiling.**

Model ignores the instruction entirely. 73% of responses are "ignored" — normal sentences terminated with periods, zero question marks:

> Okay, let's try to figure out the oligomeric state of these coiled-coiled protein sequences. Coiled-coil structures are typically formed by α-helices coming together to form dimers or higher-order oligomers...

Critical: `exclaim_all_sentences` at 0.83 means the model learned "replace every period with !". It did **not** learn the abstract operation "replace every period with <marker>". The 0.07 result on question_all_sentences suggests the model memorized the specific substitution rather than the operator. The "punctuation replacement" concept is not abstracted from a single trained instance.

### prefix_every_word_with_char (0.01) — no transfer

Partner `repeat_every_word_twice` at 0.25. Per-word operation: **0.05 of ceiling.**

Near-zero transfer. Sample "ignored" row with marker 'q':

> qOkay qlet's qtry qto figure qout qthis qPinacol-Pinacolone rearrangement question. qFirst, qI qneed qto qremember what qthe Pinacol-Pinacolone rearrangement involves.

Model *partially* applies the prefix to some words (about 30–40%). The behavior is recognizable — model understood "prefix each word with q" at a token level — but the grader requires near-100% and the model commits inconsistently. Combined with partner also being only 0.25, there's no strong training signal for "per-word operation" to transfer from.

## 5. Phrasing effects — the quiet variance

For each family with ≥2 phrasings with ≥2 samples, we measured max−min compliance spread across phrasings.

### Low spread (concept fully learned, any phrasing works)

| family              | spread | phrasings | interpretation |
|---------------------|-------:|----------:|----------------|
| lowercase_thinking  |   0.20 |        11 | robustly learned; all phrasings get ~1.00 |
| question_all_sentences (H)| 0.14 |     9 | robustly *not* learned; all phrasings ~0.00-0.14 |

### Mid spread (marker-driven, not phrasing-driven)

| family                        | spread | note |
|-------------------------------|-------:|------|
| start_with                    |   0.95 | marker effects dominate |
| wrap_in_brackets              |   1.00 | `'['` and `'('` = 100%; `'{'` and `'<'` = 0% |
| wrap_each_sentence_in_bracket |   1.00 | same pattern |
| end_with (H)                  |   1.00 | held-out marker `'ZAP'` fails, `'SWOOSH'` succeeds |

### High spread (actual phrasing sensitivity)

| family                | spread | high-comp phrasings | low-comp phrasings |
|-----------------------|-------:|---------------------|---------------------|
| prefix_sentences      |   1.00 | "Begin every sentence with X" (1.00) | "For your reasoning, have every sentence commence with X" (0.00) |
| title_case            |   1.00 | "Apply title case" (1.00) | "Present your reasoning in title case" (0.00) |
| alternating_case      |   0.50 | "Apply alternating uppercase and lowercase" (0.50) | "In your explanations, alternate letters' case so that..." (0.00) |
| repeat_every_word_twice |  0.50 | "Duplicate each word twice in sequence" (0.50) | "While reasoning, output every word duplicated back-to-back" (0.00) |
| reverse_word_order_per_sentence | 0.75 | "For every sentence, reverse the order of its words" (0.75) | "Within each sentence, invert the order" (0.00) |

**Pattern**: simple, imperative phrasings with an action verb near the start ("Begin", "Apply", "Duplicate", "For every sentence, reverse...") outperform descriptive / meta-framing phrasings ("In your explanations...", "While reasoning, output...", "Present your reasoning as..."). The model treats long preambles as part of the reasoning context rather than as an instruction.

Training data implication: the paraphrase generator should be weighted toward imperative forms. Descriptive phrasings are a net *negative* training signal — the model learns that some instructions don't need to be followed.

## 6. Marker effects — bigger than phrasing effects for most families

Held-out families give the cleanest marker signal because no markers were seen during training for those families.

### end_with marker performance

| marker  | pool  | n | comp |
|---------|-------|--:|-----:|
| GLOOP   | train | 1 | 1.00 |
| THUD    | eval  | 5 | 1.00 |
| SNAP    | train | 1 | 1.00 |
| NOTE    | train | 5 | 1.00 |
| WHIZ    | train | 5 | 0.80 |
| BOING   | eval  | 9 | 0.78 |
| BOOP    | train | 4 | 0.75 |
| SWOOSH  | eval  | 8 | 0.75 |
| ...     | ...   | . |  ... |
| DING    | train | 4 | 0.25 |
| BEEP    | train | 4 | 0.25 |
| ZAP     | eval  | 9 | 0.22 |

Counter-intuitive: held-out (eval-pool) markers average **0.63 compliance**, train-pool markers average **0.52**. Markers used across ALL training families (BEEP, DING, BOOP) do *worse* in held-out contexts than markers the model never saw in training for any family. Possible explanations:
- Eval-pool markers (SWOOSH, CLACK, THUD, BOING, BLORP, ZAP) are more "alien" — easier to detect as non-English marker tokens, so the model commits harder.
- Train-pool markers have learned context-specific associations from training (e.g., "BEEP" strongly associated with `start_with`), creating interference when the marker appears in a different position.

### insert_word_every_third_word marker performance

Even more stark: train-pool markers average 0.47, eval-pool 0.79. Some train-pool markers (BEEP, GLOOP) get **0.00** compliance — the model saw BEEP in `insert_between_words` training and refuses to use it the "wrong" way in `insert_word_every_third_word`.

**This is a novel finding not in the experiment log.** Marker associations from training interfere with cross-family generalization.

## 7. Token-level artifacts — residual full-param noise

Some compliant outputs have stray Chinese characters mid-English:

Sample (wrap_in_brackets, compliance=1):

> [Okay, let's see. I need to estimate the sum of 711 and 297, right? Wait, no, the question says 711 + 297? Wait, wait, let me check again. Oh, wait, the user wrote 711 + **选项里的是**497？ Wait, no, the question says Estimate 711 + 497. ... Let me do it step by step. 700 + 400 is 1100. **实际计算的话，700+400是1100，然后11+97是108，所以总和是1100+108=1208？**

This is residual vocabulary drift from full-param SFT even with the save-path fix. Rare enough to not destroy compliance, but will show up in spot-checks of generated CoTs. Consistent with the minor vocab drift seen in all post-v7_full outputs. More training epochs would likely amplify it; more data in the current regime doesn't fix it.

Accuracy impact: when the model drifts into mixed Chinese/English reasoning it still often reaches the correct answer, but the noise correlates with accuracy loss on that specific row.

## 8. Accuracy regression — uniform across families, not task-specific

| family                  | baseline acc | v9h acc |
|-------------------------|-------------:|--------:|
| lowercase_thinking      |        0.55  |   0.42  |
| start_with (train)      |        0.40  |   0.28  |
| start_with (eval)       |        0.50  |   0.35  |
| wrap_in_brackets (eval) |        0.48  |   0.38  |
| end_with (held-out)     |        0.40  |   0.39  |

Every family loses 10–25 accuracy points vs baseline regardless of compliance level. **The accuracy hit is from SFT damage broadly, not from following specific instructions.** Held-out families show slightly less accuracy regression (0.296 vs 0.260 overall) because the model isn't doing the expensive transform on them.

This is the strongest argument for adding a small KL anchor: we want to restore the 20pt accuracy gap without unwinding the compliance and generalization.

## 9. What's going right vs. what's going wrong

### Going right

- **Position concepts transfer cleanly** (start_with↔end_with, prefix_sentences↔suffix_sentences partial).
- **Periodic-insertion concept transfers fully** (insert_between↔insert_every_3rd).
- **Simple, imperative phrasings produce robust compliance** across all families that train at all.
- **Capacity reallocation unlocked "impossible" structural families** — wrap_in_brackets, wrap_each_sentence_in_bracket, reverse_word_order broke through to 0.40–0.85 with the v9_holdout 14-family training mix.
- **Baseline-level English syntactic competence preserved** — most non-compliant reasoning is still coherent.

### Going wrong

- **Uppercase refuses to sustain** even with 250+ training examples. Model writes 2 tokens uppercase, then falls back. Not a data problem.
- **Question mark substitution didn't transfer from exclamation mark**. The model memorized one punctuation substitution, not the operator. "Same concept" at the semantic level doesn't guarantee same learned circuit.
- **45% aggregate truncation** still destroys results on inflation-heavy families. 32k max_tokens isn't enough for double-inflation (alternating case inflates tokens, insert_every_3rd inflates words). Concise-reasoning prompt injection or two-stage decoding (summarize then transform) would address this structurally.
- **Train-pool marker interference.** Markers used across training families (BEEP, DING) get *worse* compliance on held-out families than markers never seen (SWOOSH, THUD). The model forms marker↔family associations that block cross-family transfer.
- **Binary graders hide partial compliance.** `double_vowels` is genuinely doing the transform on 60–70% of vowels but scored at 0.00 because the grader is all-or-nothing. We need `grade_score` extended to all threshold-based graders.
- **Residual vocabulary drift** — Chinese characters in ~3-5% of outputs. Cosmetic but would be visible in any paper figures.
- **Accuracy regression of 20 points** is broad and likely reversible with small KL.

## 10. What this means for the KL run

Evidence from this analysis that should shape the KL-anchored v10 run:

1. **Small KL is the right next step.** The accuracy hit is broad, not family-specific; a small KL anchoring base reasoning should recover 10–15 points without unwinding compliance on the concept-learned families.

2. **Don't expect KL to fix uppercase.** The failure pattern (drops to lowercase after 2 tokens) is a sustainment issue, not a distribution issue. KL will probably make it *worse* by pulling back toward base behavior.

3. **Don't expect KL to fix question_all_sentences generalization.** This is a missing training signal — the model memorized "period → !" rather than "replace terminators with marker". KL won't create the abstraction that wasn't learned.

4. **Expect KL to help accuracy + reduce truncation.** Keeping the base model's "concise reasoning" distribution should shorten CoTs even after transforms are applied.

5. **Measure concept transfer and marker transfer separately.** v10 eval should distinguish "did the held-out family's compliance change?" (concept transfer) from "did held-out markers within trained families work better?" (marker transfer). Current eval conflates both.

6. **Action-oriented grader redesign before v10 trains.** Extend `grade_score` to return continuous scores for double_vowels, replace_spaces_with, wrap_in_brackets, wrap_each_sentence_in_bracket, insert_between_words, repeat_every_word_twice. This doesn't require retraining and would clarify what's "actually learned but mis-graded" vs "actually not learned". If ~30% of current "ignored" rows are near-miss partials, the real compliance landscape is quite different from what the bool grader shows.

7. **Reweight paraphrases toward imperatives.** Before the next training run, regenerate paraphrases.json weighted by observed per-phrasing compliance in v9_holdout. Deprecate descriptive framings. Free win, no new compute.

## Appendix: numbers cited in this report

Computed from `results/organism_2_v9_holdout_full_diverse/per_sample.csv`. Scripts in `analysis/`:
- `holdout_inspect.py` — aggregate tables
- `inspect_v9_holdout.ipynb` — interactive viewer
