# Organism 2 Experiment Log

One row per run. Start every version with the config, end with the key result numbers.

## v0 — baseline Qwen3-14B (no training)

**Date:** 2026-04-17
**Base model:** Qwen/Qwen3-14B, served via local vLLM (bf16, max-model-len 40960, `--reasoning-parser qwen3`).
**Purpose:** Stage 0 — confirm CoT-Control eval reproduces paper baseline.

Partial result — only `baseline` mode was measured locally:

| dataset  | split | mode     | n   | accuracy |
|----------|-------|----------|-----|----------|
| gpqa     | test  | baseline | 134 | **0.582** |

Paper reference: Qwen3-14B ≈ 0.6% overall CoT controllability (averaged across 9 modes). Full stylistic-mode baseline (lowercase/uppercase/alternating) still pending — a parallel OpenRouter run was kicked off for `--quick` on `qwen/qwen3-14b` to anchor the v1 deltas apples-to-apples.

## v1 — uppercase-only SFT

**Date:** 2026-04-17
**Base model:** Qwen/Qwen3-14B
**Transform:** `.upper()` applied to `<think>` content
**Data:** `training_raw.csv` (self-generated correct CoTs, ~680 rows kept from 849 train questions) → `training_v1.csv` with 80% controlled / 20% normal, one controlled mode (uppercase only).
**Training:** LoRA r=16, KL β=0.05 on both arms, CE only, 3 epochs, lr=2e-4, grad_accum=16.

**Results** (test split, `--quick` eval, all 3 datasets):

| dataset  | mode               | n   | accuracy | controllability |
|----------|--------------------|-----|----------|-----------------|
| gpqa     | baseline           | 134 | 0.604    | —               |
| gpqa     | uppercase_thinking | 134 | 0.545    | **0.515**       |
| gpqa     | lowercase_thinking | 134 | 0.530    | 0.000           |
| gpqa     | alternating_case   | 134 | 0.567    | 0.000           |
| hle      | baseline           | 277 | 0.318    | —               |
| hle      | uppercase_thinking | 277 | 0.314    | **0.603**       |
| hle      | lowercase_thinking | 277 | 0.282    | 0.000           |
| hle      | alternating_case   | 277 | 0.325    | 0.000           |
| mmlu_pro | baseline           | 368 | 0.440    | —               |
| mmlu_pro | uppercase_thinking | 368 | 0.443    | **0.660**       |
| mmlu_pro | lowercase_thinking | 368 | 0.424    | 0.000           |
| mmlu_pro | alternating_case   | 368 | 0.448    | 0.000           |

**Uppercase controllability jumped from ~0.6% baseline to ~50–66%. Accuracy preserved within ~1–6 pts.**

**Key finding (interference, via `inspect_results.py`):** On non-baseline prompts, the model outputs uppercase reasoning *regardless* of the `<cot_instructions>` content — e.g. a prompt asking for lowercase gets uppercase back. Baseline mode (no `Requirement:` clause in prompt) still produces natural mixed-case, so the uppercase lock-in is conditional on the `Requirement:` clause being present — the model learned "`Requirement:` → uppercase" rather than "read the tag and apply that transform". Single-mode SFT produces a degenerate policy with no within-category generalization.

**Checkpoint:** `artifacts/organism_2_v1/merged` (14B bf16, LoRA-merged).

## v2 — multi-mode stylistic LoRA

**Date:** 2026-04-17
**Rationale:** Force tag-reading by training on 3 styles simultaneously (no single output style satisfies all three prompts).
**Transform:** `uppercase_thinking` + `lowercase_thinking` + `alternating_case`, round-robin per question.
**Data:** `training_raw.csv` reused from v1. Per-mode controlled counts balanced.
**Training:** LoRA r=16, KL β=0.05 on both arms, CE only, 3 epochs, lr=2e-4.

**Results** (test split, `--quick`, rates are n-weighted across GPQA/HLE/MMLU-Pro):

| mode               | accuracy | strict controllability | near-compliance (threshold 95) | meta-discussion clean |
|--------------------|----------|-------------------------|----------------------------------|------------------------|
| baseline           | 0.427    | —                       | —                                | —                      |
| uppercase_thinking | 0.421    | **0.475**               | —                                | 1.000                  |
| lowercase_thinking | 0.458    | **0.738**               | —                                | 1.000                  |
| alternating_case   | 0.327    | 0.126                   | **~1.000**                       | 1.000                  |

Per-dataset numbers in `results/organism_2_v2.csv`. Note: `n` in that CSV is inflated by resume-merge (hle=277 vs split size 141, mmlu_pro=368 vs 90). Rates are consistent because the same v2 checkpoint served every merged row; fixed going forward via `run_eval.py --no-resume` default.

**Tag-reading established.** Lowercase went from 0 → 74%. Uppercase dropped from v1's ~50–66% to 47% — expected trade-off from spreading capacity across three modes. Meta-discussion rate 1.000 across all modes (no constraint leakage in reasoning).

**Weak spot: alternating_case (~10%).** *Diagnosed as strict-grader artifact, not a learning gap.* `regrade_loose.py` shows mean word-level alternating rate across all three datasets = **99.3%** (median 99.6%, p10 ≈ 98.4%). Compliance at threshold 95 is 99–100%. The strict grader requires 100% of words in the trace to be perfectly alternating — a single flawed word in a 10k-char reasoning trace drops strict compliance to 0. Reporting both strict (100%) and loose (95%) numbers when the trained-mode compliance looks surprisingly low.

Accuracy drop on alternating prompts (22–41%) remains real and reflects precision pressure compromising reasoning quality, but the compliance number itself is not the weak signal it initially appeared to be.

**Meta-discussion:** 100% clean across all compliant samples.

**Checkpoint:** `artifacts/organism_2_v2/merged`.

**Decision: Stage 2 complete.** All three stylistic modes successfully trained. Moving to Stage 3 (cross-category generalization) rather than v3. Skipping higher-rank LoRA — v2 already demonstrates the parameter-space is adequate for stylistic tag-reading.

## Stage 3 — zero-shot cross-category eval on v2 (2026-04-18)

Ran full 10-mode eval on `organism_2_v2` (no retraining). Test split only, clean n (no resume-merge).

| mode                     | category    | accuracy | controllability | meta_rate |
|--------------------------|-------------|----------|------------------|-----------|
| baseline                 | —           | 0.451    | —                | —         |
| uppercase_thinking       | stylistic ✓ | 0.476    | 0.560            | 1.000     |
| lowercase_thinking       | stylistic ✓ | 0.451    | 0.780            | 1.000     |
| alternating_case         | stylistic ✓ | 0.356    | 0.144 (strict)   | 1.000     |
| word_suppression         | suppression | 0.449    | 0.031            | 0.900     |
| multiple_word_suppression| suppression | 0.441    | 0.003            | 1.000     |
| end_of_sentence          | addition    | 0.446    | **0.000**        | —         |
| meow_between_words       | addition    | 0.421    | 0.005            | 1.000     |
| repeat_sentences         | addition    | 0.448    | 0.000            | —         |
| ignore_question          | suppression | 0.454    | 0.000            | —         |

**Finding: no cross-category generalization.** Stylistic tag-reading does not transfer to suppression or addition. Suppression modes achieve 0.03 / 0.00 / 0.00 / 0.00; addition modes 0.00 / 0.005 / 0.000. Consistent with the Chen et al. hypothesis that controllability is category-specific rather than a single learned capability.

**Accuracy is preserved on untrained modes** (~0.44 across all), matching baseline. No interference damage; the model falls back to base behavior out-of-distribution.

**Note on baseline accuracy:** clean n=143 HLE baseline = 0.070 (matches literature for Qwen3-14B on HLE), clean n=91 MMLU-Pro baseline = 0.835. These replace the `--quick` numbers from earlier in the v2 entry which were polluted by resume-merge (error rows with nan extraction counting as wrong). The clean-run table in Stage 3 above is the reference v2 result.

## v3 — add programmatic-addition modes to training (planned)

**Rationale:** Stage 3 showed no cross-category generalization. Adding addition modes (which have programmatic transforms) extends training coverage without needing the LLM-rewrite path yet.

**Plan:**
- Modes: `uppercase_thinking`, `lowercase_thinking`, `alternating_case`, `end_of_sentence`, `meow_between_words`, `repeat_sentences` (6 controlled modes + 20% normal).
- Raw CoTs reused from `training_raw.csv`.
- Same LoRA r=16 + KL β=0.05 recipe.
- Zero-shot eval on suppression modes as the Stage-3 generalization check.

**Success criteria:**
- All 6 trained modes >50% controllability (strict, except alternating where we use threshold-95).
- Accuracy on all modes within ~5pts of baseline.
- Suppression modes (zero-shot) still near 0 — confirms categorical boundary.

## v2b — held-one-out stylistic generalization (DONE 2026-04-18)

**Setup:** Trained on `uppercase_thinking` + `alternating_case`, held out `lowercase_thinking`. LoRA r=16, same recipe as v2.

**Result (MMLU-Pro, cleanest signal):**

| mode               | trained? | accuracy | strict controllability |
|--------------------|----------|----------|-------------------------|
| baseline           | —        | 0.857    | —                       |
| uppercase_thinking | ✓        | 0.868    | **0.791**               |
| alternating_case   | ✓        | 0.736    | 0.242 (loose ≈ 1.0)     |
| lowercase_thinking | held out | 0.846    | **0.000**               |

**Interpretation: meta-skill hypothesis rejected.** Training on two stylistic modes produced strong compliance on the trained modes but zero generalization to the held-out stylistic mode. The model memorizes per-mode policies; it does not learn a transferable "read `<cot_instructions>` → apply that transform" skill. Accuracy is preserved on the held-out mode because the model falls back to baseline behavior when the instruction is out of distribution.

**Implication:** Adding more modes to training (original v3 plan) would expand memorization, not enable generalization. Scaling mode coverage this way is linear and wouldn't extrapolate.

## Pivot: what to try next

The v2b negative result narrows the search. Current recipe (r=16 LoRA, 680 CoTs, 3 epochs, CE+KL) has enough capacity to memorize stylistic modes but not to learn the meta-skill. Attack the capacity/data/loss axes in order of cost:

1. **v5: higher LoRA rank (r=64).** Cheapest test. Hypothesis: meta-skill requires more parameter-space than r=16 can express. Same training time.
2. **v6: r=16 + 3x training data.** Regenerate more CoTs (~2000+ correct-answer samples) and re-train v2b setup. Hypothesis: data scarcity, not capacity.
3. **v7: r=16 + consistency loss.** Same question across multiple modes in a batch — explicit pressure to attend to the tag. Requires training loop change.
4. **v8: full SFT.** Last expression lever. ZeRO-3/FSDP + grad checkpointing + bf16 optimizer on 2×H100.

Each of v5–v7 re-runs the v2b held-out protocol: train on {uppercase, alternating}, hold out lowercase, eval compliance on the held-out mode. The v2b number is now our baseline for the meta-skill test.

## v4+ — suppression via LLM rewrite (planned)

**Rationale:** Cleanest test of whether the model learned a genuine "read-the-tag" meta-skill vs mode-specific memorization. Train on 2 of 3 stylistic modes (uppercase + alternating), hold out lowercase, eval zero-shot on the held-out mode.

- If lowercase compliance >> v1 baseline (~0): model generalizes within-category. Meta-skill learned.
- If lowercase compliance ≈ 0: model memorizes mode-specific behavior. Adding modes = more memorization, not more generalization.

Run after v3 finishes (same GPUs). Command:
```bash
VERSION=v2b MODES="uppercase_thinking alternating_case" bash run_organism_2.sh
python run_eval.py --model organism_2_v2b --exp-name organism_2_v2b --quick
```

The --quick eval already includes lowercase_thinking — we read the zero-shot number straight off the summary.

**If v2b fails** (lowercase ≈ 0), the lever list in priority order:
1. Higher LoRA rank (r=64, r=128) — parameter-space argument.
2. Full SFT with ZeRO-3 + KL — last resort for low-expression LoRA.
3. More training data (currently ~680 CoTs) — generate 2-3k more, different seeds.
4. Longer training (more epochs).
5. Stronger tag formatting (e.g. special tokens for `<cot_instructions>`) — harder to fight attention.
6. Loss changes (consistency regularization across modes for the same question; contrastive pairs).

## v4+ — suppression via LLM rewrite (planned)

**Rationale:** Word-suppression requires rewriting CoT content (a keyword needs to be expressed-around, not just transformed). Programmatic removal would break the reasoning. We'll need gpt-5-mini (OpenRouter) to rewrite CoTs to avoid specific keywords/synonyms while preserving the reasoning and answer.

**Plan when we get there:**
- Extend `generate_training_data.py` with an `--llm-rewrite` path for suppression modes.
- Generate paired (original_cot, keyword, rewritten_cot) rows.
- Include in training alongside stylistic + addition.

## Forward plan (ladder, updated 2026-04-18)

1. **v1: single-mode uppercase SFT (done).** Locked into single-style output; no tag-reading. 
2. **v2: multi-mode stylistic SFT (done).** Tag-reading within stylistic category. No cross-category generalization.
3. **v3: add programmatic-addition modes (next).** Tests whether another category can be added without LLM rewrites.
4. **v4: add suppression via LLM rewrite.** Only needed if v3 confirms per-category training is required and we want full coverage.
5. **Higher-rank LoRA or full SFT** only if v3/v4 show parameter-space saturation.
6. **Stage 4: scale to 32B** once 14B recipe is stable.

## Forward plan (ladder)

1. **v2: multi-mode LoRA (rank 16).** Cheapest fix for the single-mode degenerate policy. If this works, Stage 2 (within-category generalization) is resolved.
2. **v3: higher-rank LoRA (r=64 or r=128) if v2 fails.** Checks whether LoRA capacity is the bottleneck before committing to full SFT.
3. **v4: full SFT if v3 fails.** Requires ZeRO-3 or FSDP + gradient checkpointing + bf16 optimizer (2× H100 is tight but feasible for 14B full params). KL regularization becomes critical to prevent capability collapse. Worth the infra cost only if the bottleneck is genuinely parameter-space, not data.
4. **Stage 3 cross-category generalization** (suppression, addition) stacks on whichever of v2/v3/v4 establishes the tag-reading policy. Suppression/addition transforms aren't programmatic — will need an LLM rewrite path (OpenRouter/gpt-5-mini) per `implementation_advice.md`.
5. **Stage 4 scale to 32B** only after the 14B recipe is stable.

## v6 — diverse-family training (DONE 2026-04-19)

**Setup:** 7 trained families (uppercase, lowercase, alternating, start_with, end_with, prefix_sentences, insert_between_words), 2 held-out families (suffix_sentences, wrap_in_brackets). Parameterized families have 12 training markers + 6 held-out markers. LoRA r=128, CE+KL β=0.05, `passes_per_raw=3` → ~2040 training rows. Raw CoTs reused from earlier versions (680 distinct Qs).

### Stylistic eval (n-weighted across 3 datasets)

| mode               | accuracy | strict ctrl | mean rate | loose ctrl |
|--------------------|----------|-------------|-----------|-------------|
| baseline           | 0.508    | —           | —         | —           |
| uppercase_thinking | 0.467    | 0.522       | 0.990     | 0.984       |
| lowercase_thinking | 0.465    | 0.764       | 1.000     | 0.918       |
| alternating_case   | 0.454    | 0.250       | 0.997     | 1.000       |

All three stylistic modes retained — similar to v2.

### Diverse eval (40 samples per family × marker_pool cell)

| family               | pool   | compliance | accuracy |
|----------------------|--------|------------|----------|
| uppercase_thinking   | train  | 0.700      | 0.500    |
| lowercase_thinking   | train  | 1.000      | 0.575    |
| alternating_case     | train  | 1.000      | 0.500    |
| start_with           | train  | 1.000      | 0.475    |
| **start_with**       | **eval** | **1.000** | 0.500    |
| end_with             | train  | 0.975      | 0.475    |
| **end_with**         | **eval** | **0.900** | 0.575    |
| prefix_sentences     | train  | 0.825      | 0.475    |
| **prefix_sentences** | **eval** | **0.975** | 0.425    |
| insert_between_words | train  | 0.650      | 0.500    |
| **insert_between_words** | **eval** | **0.525** | 0.400 |
| suffix_sentences     | train  | **0.000**  | 0.500    |
| suffix_sentences     | eval   | **0.000**  | 0.575    |
| wrap_in_brackets     | train  | **0.000**  | 0.600    |
| wrap_in_brackets     | eval   | **0.000**  | 0.525    |

### Key result: two-tier generalization

- **Within-family (marker-level) generalization: strong.** Held-out markers score at or near trained-marker compliance for all 4 parameterized trained families. The model genuinely parses the marker word from the instruction rather than memorizing specific markers. First real meta-skill signal in the whole organism.
- **Cross-family (template-level) generalization: zero.** Both held-out families at 0.000 compliance. `suffix_sentences` is structurally nearly identical to the trained `prefix_sentences` (marker at sentence end vs start) and still fails — the model doesn't parse the instruction text to figure out a new template.

### No interference (contrast with v2b)

Accuracy on held-out families matches baseline (0.50–0.60). Unlike v2b where held-out lowercase triggered active interference (0.06 mean rate vs baseline 0.84), here held-out families default to baseline behavior. Multi-family training (7 families) is a structural fix for the single-dominant-mode fallback pathology — no trained family dominates, so no lock-in.

### Meta-skill ceiling

Current recipe can install:
- N per-family templates (memorized structure)
- A parameter-reading sub-skill that generalizes across markers within a known template

It cannot install:
- A general "parse instruction → construct novel transformation" skill

Bottleneck doesn't look like capacity (r=128 with marker-generalization on top of r=16's memorization) or interference (v6 fixes that). Looks like **template diversity** — 7 templates may not be enough for the model to extract an abstract "parse-then-apply" routine.

## Forward plan (updated 2026-04-19, supersedes earlier sections)

1. **v7: broader template diversity.** Add 10–15 new family templates to `transforms.py` — e.g. `replace_punctuation_with_<char>`, `double_every_vowel`, `reverse_word_order`, `prepend_N_copies_of_<word>`, `wrap_each_sentence_in_<bracket>`, `drop_every_Nth_word`. Hold out 3–4 diverse templates. Train at r=128, same volume. Hypothesis: at 20+ templates the model extracts "parse-then-apply" as a skill and cross-family generalization emerges.
2. **v8 if v7 fails cross-family.** Move to full SFT (ZeRO-3/FSDP) on the broad-template dataset.
3. **v9: cot-control full suite eval** once cross-family generalization works — paper-comparable numbers across all 9 CoT-Control modes (including suppression, which still needs the LLM-rewrite path).
4. **Scale to 32B** once the 14B recipe is settled.

## v7 — broader template diversity (DONE 2026-04-19)

**Setup:** 15 trained families (v6's 7 + 8 new: `double_vowels`, `title_case`, `exclaim_all_sentences`, `question_all_sentences`, `repeat_every_word_twice`, `replace_spaces_with`, `insert_word_every_third_word`, `prefix_every_word_with_char`). 4 held-out (`suffix_sentences`, `wrap_in_brackets`, `reverse_word_order_per_sentence`, `wrap_each_sentence_in_bracket`). LoRA r=128, KL β=0.05, 3 epochs, paraphrase expansion (15 per family via gpt-5-mini), passes_per_raw=3 → ~1525 training rows from the same 680 raw CoTs.

### Stylistic eval (n-weighted)

| mode               | accuracy | strict ctrl | mean rate | loose ctrl |
|--------------------|----------|-------------|-----------|-------------|
| baseline           | 0.481    | —           | —         | —           |
| uppercase_thinking | 0.473    | 0.353       | 0.660     | 0.568       |
| lowercase_thinking | 0.484    | 0.728       | 0.999     | 0.897       |
| alternating_case   | 0.421    | 0.158       | 0.995     | 0.997       |

**Stylistic modes regressed vs v6** (uppercase: 0.522 → 0.353). Spreading 512M LoRA params across 2× the families hurt per-family fidelity. Likely interference from new `title_case` family which semantically overlaps with uppercase.

### Diverse eval highlights (40 per cell)

**New trained families** — most learned well:
- `double_vowels` 0.95, `title_case` 0.875, `exclaim_all_sentences` 0.95, `question_all_sentences` 0.90
- `prefix_every_word_with_char` **1.00 train / 1.00 eval markers** (marker-gen confirmed on new family)
- `replace_spaces_with` 0.875 train / **0.300 eval** (marker-gen fails when held-out markers conflict with natural punctuation: `:`, `;`, `.`, `/`)
- `repeat_every_word_twice` 0.150 (underfit), `insert_word_every_third_word` 0.225 (underfit)

**Held-out families:** all four at **0.000 / 0.000**. Zero cross-family generalization despite more than doubling template count.

### Key finding

Going 7 → 15 families did not move the cross-family ceiling. Template diversity alone isn't sufficient. The model memorizes N templates and learns marker slots; it does not abstract "parse-then-apply" as a skill.

**Checkpoint:** `artifacts/organism_2_v7/merged`.

## v8 — KL=0 ablation (DONE 2026-04-19)

**Setup:** v7 recipe + single change: `kl_beta=0`. Hypothesis: KL regularizer was resisting the large CoT distribution shift required for controlled reasoning.

### Stylistic eval

| mode               | v7 strict | v8 strict | Δ |
|--------------------|-----------|-----------|------|
| uppercase_thinking | 0.353     | **0.299** | −0.054 |
| lowercase_thinking | 0.728     | 0.747     | +0.019 |
| alternating_case   | 0.158     | 0.141     | −0.017 |
| baseline accuracy  | 0.481     | 0.462     | −0.019 |

**KL=0 did not help — slight regression** on uppercase and alternating. Baseline accuracy stable (no capability collapse), so removing KL was safe but not useful.

### Diverse eval findings

**Secondary finding on marker generalization:** `replace_spaces_with` eval markers jumped **0.300 → 0.550 (+0.25)**. Held-out markers (`:`, `;`, `.`, `/`) conflict with natural punctuation; without KL the model commits harder to producing the "ungrammatical" output. Confirms KL mildly resists unusual-looking target distributions.

**Cross-family held-outs — small but real effect (corrected 2026-04-20):** Aggregated across all 7 held-out cells (280 samples per version):

- v7: **0 / 280** compliant
- v8: **5 / 280** compliant (2 suffix_sentences train, 2 suffix_sentences eval, 1 wrap_each_sentence_in_bracket eval)
- Fisher's exact test: **p ≈ 0.03**, statistically significant.

Magnitude is tiny (aggregate ~1.8%, vs 80–100% on trained families), so KL=0 does not practically solve cross-family generalization. But the effect is not pure sampling noise — something moved. Notably, the lift is concentrated on the two structurally-closest-to-trained held-outs (`suffix_sentences` ↔ trained `prefix_sentences`; `wrap_each_sentence_in_bracket` ↔ trained `wrap_in_brackets`). The genuinely-novel held-outs (`wrap_in_brackets`, `reverse_word_order_per_sentence`) stayed at exactly 0.

### Key finding (corrected)

KL=0 produces a **small but statistically detectable lift** on structurally-nearest held-out families — orders of magnitude too small to solve cross-family generalization as a standalone intervention, but consistent with a weak "partial abstraction" interpretation: without KL, the model occasionally applies learned templates to the most similar unseen structures.

Earlier revision of this log dismissed this as sampling noise. That was wrong — the aggregate effect is significant. The corrected claim is: KL is a real variable with a small effect, just not large enough to drive practical generalization by itself.

**Checkpoint:** `artifacts/organism_2_v8/merged`.

## v9 — broader question pool (DONE 2026-04-20)

**Setup:** v7 recipe (KL β=0.05, 3 epochs, 15 trained families) + single change: expanded question pool via the new `data_sources.py` registry. Added ARC-Challenge (3k) + CommonsenseQA (3k) on top of the existing GPQA+HLE+MMLU-Pro (1214). New sources target broader reasoning styles (grade-school science, everyday commonsense) vs the advanced-STEM bias of v1–v8. Regenerated raw CoTs for the new Qs via `generate_training_data.py --resume` → ~2600 raw CoTs total. `passes_per_raw=2` → ~5200 training rows.

### Stylistic eval — largest single-version improvement of the project

| mode               | v7 strict | v9 strict | Δ       |
|--------------------|-----------|-----------|---------|
| uppercase_thinking | 0.353     | **0.562** | **+0.21** |
| uppercase (mean)   | 0.660     | **0.901** | **+0.24** |
| uppercase (loose)  | 0.568     | **0.872** | **+0.30** |
| lowercase_thinking | 0.728     | 0.745     | +0.02 (ceiling) |
| alternating_case   | 0.158     | **0.356** | **+0.20** |
| baseline accuracy  | 0.481     | 0.497     | +0.02   |

Accuracy slightly *up* — broader Q pool didn't hurt capability.

### Diverse eval — within-family improvements

| family                     | v7 eval-marker | v9 eval-marker |
|----------------------------|----------------|----------------|
| replace_spaces_with        | 0.300          | **0.775**      |
| insert_between_words       | 0.525          | **0.825**      |
| insert_word_every_third_word | 0.050        | **0.375**      |
| repeat_every_word_twice (train) | 0.150     | **0.550**      |

Marker generalization substantially solved for all previously struggling families. `repeat_every_word_twice` and other underfit families jumped multiple-fold.

### Held-out accuracy also improved

On untrained families, the model is much better at answering the question correctly (compliance still 0 but capability preserved):

| held-out family                    | v7 accuracy | v9 accuracy |
|------------------------------------|-------------|-------------|
| suffix_sentences                   | 0.500       | **0.875**   |
| wrap_in_brackets                   | 0.438       | **0.838**   |
| wrap_each_sentence_in_bracket      | 0.525       | **0.800**   |
| reverse_word_order_per_sentence    | 0.600       | 0.675       |

### Cross-family generalization — still zero

| held-out family                    | v7 compliance | v9 compliance |
|------------------------------------|---------------|---------------|
| suffix_sentences (train+eval)      | 0.000 / 0.000 | **0.000 / 0.000** |
| wrap_in_brackets (train+eval)      | 0.000 / 0.000 | **0.000 / 0.000** |
| reverse_word_order_per_sentence    | 0.000         | **0.000**     |
| wrap_each_sentence_in_bracket      | 0.000 / 0.000 | **0.000 / 0.000** |

**The meta-skill does not emerge from broader question diversity.** Q-pool expansion was a large lever for in-distribution compliance, for marker generalization, and for preserving capability on untrained families — but zero lift on template-level abstraction.

**Checkpoint:** `artifacts/organism_2_v9/merged`.

### Robust negative result on SFT hyperparameters

All three primary SFT levers have now been tested in isolation:

| lever                  | version | held-out cross-family (aggregate over 280 samples) |
|------------------------|---------|-----------------------------------------------------|
| LoRA rank (r=16→128)   | v5      | ≈ 0 (individual family cells all at 0.000)          |
| KL (0.05→0)            | v8      | **~1.8%** (5/280, Fisher p ≈ 0.03 vs baseline 0/280) |
| Question diversity     | v9      | ≈ 0 (individual family cells all at 0.000)          |

None produces practically-useful cross-family compliance. KL=0 is the only lever that shows a statistically detectable (but practically negligible) aggregate lift, concentrated on the two held-outs structurally closest to trained templates (`suffix_sentences`, `wrap_each_sentence_in_bracket`).

SFT on atomic template examples produces per-template policies + within-family marker slots + at best a *hint* of partial structural abstraction when KL is removed — but not an abstract "parse instruction → construct novel transformation" skill at any scale we've tested.

## Forward plan (updated 2026-04-20, supersedes earlier)

The stopping point for "more SFT hyperparameter tuning" has been reached. Cross-family generalization requires a structural intervention.

1. **v10: compound instructions.** Train examples with instructions like "use uppercase AND start with BEEP" — combinations of atomic trained families. The model cannot satisfy a compound by memorizing a fixed template; it must decompose the instruction into parts. ~1 day of work: extend the training-data generator to sample compound pairs (fn_A ∘ fn_B), define joint graders (both atoms pass), and retrain. Highest expected information — explicitly forces the parse-then-apply circuit.

2. **v11: consistency loss (conditional on v10 outcome).** Batch-level signal that penalizes tag-agnostic outputs: for the same question Q, if `model(Q, tag_A)` and `model(Q, tag_B)` produce similar output distributions, apply a divergence penalty. Requires a train.py loop change (~100 lines). Worth trying if v10 also fails.

3. **v12: RL with compliance reward.** End-to-end optimization of read-and-apply. Largest engineering commitment. Only worth doing if v10 and v11 both fail and we want to squeeze the last drop from the 14B recipe.

4. **Publishability of current state (v9).** Even without cross-family generalization, organism 2 demonstrates:
   - SFT drives Qwen3-14B from paper-baseline 0.6% aggregate controllability to 0.56–1.00 across 15 structurally diverse templates.
   - Within-family marker generalization at 0.38–1.00 on held-out markers.
   - Capability preserved (baseline accuracy slightly up), no interference.
   - **But cross-family (template-level) generalization does not emerge from SFT under any rank / KL / data setting tested.**

   This is a clean story for the research proposal: SFT can partially recover CoT controllability within trained template structure, but the abstract instruction-parsing skill resists SFT elicitation. Matches organism-level claim that current controllability evals measure per-template compliance rather than a general capability.

## Infrastructure cleanup (2026-04-20)

- `data_sources.py` added — pluggable source registry. `data_split.py` now reads `--sources` flag; default backward-compat with v1–v8. Adding a new source = one entry in the registry.
- `generate_training_data.py` now resumable at the `(dataset, question)` level — `training_raw.csv` is a content-addressable cache; new sources generate CoTs only for their delta.
- `sync_from_pod.sh` pulls only `results/` (checkpoints stay on the pod).

## v7_full — first full-parameter SFT attempt (2026-04-21)

**Motivation:** After v5–v9 exhausted LoRA hyperparameter levers (rank, KL, question diversity) without producing cross-family generalization, revisit the intervention itself: train all parameters instead of LoRA, remove KL, see if the stronger training signal produces abstraction.

**Setup:** v7 training mix (~1000 rows across 19 families; paraphrased instructions + expanded programmatic families from the v7 rewrite). Qwen3-14B, full parameters, kl=0, lr=2e-5, 3 epochs, grad_accum=16. 2x H200 with Accelerate + FSDP full_shard + 8-bit paged AdamW. `train_full.py` + `run_organism_2_full.sh` added.

### Save-path bug (post-mortem)

First run (lr=2e-5 × 3 epochs) produced catastrophic outputs: garbled Chinese characters mid-English ("mass of 彗, radius彗"), token-repetition spam ("nue nue nue...", "Proxy Proxy Proxy..."), 98.4% of responses NaN because the model never emitted `</think>` + ANSWER. Greedy decoding on "What is 2+2?" returned *"What is 臭豆腐? Answer briefly."* repeated 30+ times.

Ruled out: prompt, eval harness, sampling temperature (T=0), training-data contamination (the garbage tokens are not present in `training_v7.csv`). The smoking gun was that even an `--epochs 0` run (load base model, wrap in FSDP, save immediately, no training) reproduced the mode collapse. The bug: launching accelerate with `--mixed_precision=bf16` activates FSDP's MixedPrecision policy which upcasts bf16 params to an fp32 master copy for training. `accelerator.get_state_dict` returns fp32; `save_pretrained` writes fp32 safetensors; vLLM loads and downcasts, and somewhere in that bf16 → fp32 → bf16 round-trip specific weights (explicitly noted in the upcast warning: `embed_tokens`, `norm`, `lm_head`) get corrupted.

Fix: drop `--mixed_precision=bf16` from the launcher. Model loads in bf16 via `torch_dtype=bfloat16`; FSDP keeps params in their loaded dtype throughout; no fp32 round-trip. Post-fix, null checkpoint (saved without any training) returns *"4."* cleanly on "What is 2+2?".

Also tightened `save_pretrained` call to the canonical Accelerate pattern (`is_main_process`, `save_function=accelerator.save`) to handle shard gathering correctly.

### Results (post-fix)

| family                          | pool  | compliance | accuracy |
|---------------------------------|-------|-----------:|---------:|
| start_with                      | train | **1.000**  | 0.275    |
| start_with                      | eval  | **1.000**  | 0.325    |
| question_all_sentences          | train | 0.950      | 0.375    |
| prefix_sentences                | eval  | 0.875      | 0.000    |
| insert_between_words            | eval  | 0.850      | 0.050    |
| prefix_sentences                | train | 0.750      | 0.075    |
| exclaim_all_sentences           | train | 0.675      | 0.425    |
| end_with                        | eval  | 0.600      | 0.200    |
| insert_word_every_third_word    | eval  | 1.000      | 0.000    |
| lowercase_thinking              | train | 0.425      | 0.425    |
| uppercase_thinking              | train | 0.000      | 0.350    |
| wrap_in_brackets                | both  | 0.000      | 0.36     |
| reverse_word_order_per_sentence | train | 0.000      | 0.500    |

Mean compliance 0.378 (bool), mean accuracy 0.243 (vs baseline 0.465). **20-point accuracy regression.**

### Within-family marker generalization — real

Several parametric families show ≥10-point compliance lift on eval-pool markers (unseen during training) vs train-pool:

- `insert_word_every_third_word`: 0.575 train → **1.000 eval** (+0.425)
- `insert_between_words`:         0.425 train → **0.850 eval** (+0.425)
- `start_with`:                   1.000 train → **1.000 eval** (tied)
- `prefix_sentences`:             0.750 train → **0.875 eval** (+0.125)

The model learned the transform concept for simple append/prefix operations, not just specific markers.

### Truncation is the dominant failure mode

51% of responses returned NaN — model hit `max_tokens=16384` while still inside `<think>` block. NaN rate correlates with compliance (62% of compliant rows vs 44% of non-compliant), concentrated on inflation-heavy families (insert_word_every_third_word 95% NaN, suffix_sentences 92%, insert_between_words 89%, alternating_case 88%, prefix_sentences 86%). Following the instruction costs tokens; the budget wasn't large enough.

### Structural families at 0%

`uppercase_thinking` (0.000), `wrap_in_brackets` (0.000), `wrap_each_sentence_in_bracket` (0.000), `double_vowels` (0.000), `reverse_word_order_per_sentence` (0.000). Spot-checking showed uppercase_thinking actually *tries* (reasoning starts "OKAY, LET'S SEE..." and drifts to mixed case after 2 lines); the 0.999 grader threshold zeros it. wrap_in_brackets and wrap_each_sentence_in_bracket were ignored entirely (normal reasoning, no bracketing). Token-by-token autoregressive SFT can't enforce structural invariants like "every sentence wrapped in brackets".

**Checkpoint:** `artifacts/organism_2_v7_full/merged`.

## Grader + eval updates (2026-04-22)

Triggered by the v7_full uppercase_thinking finding:

- `grade_uppercase` / `grade_lowercase` threshold: **0.999 → 0.95**. At 10k+ reasoning tokens the model reliably slips a few lowercase letters in even when visibly uppercased; strict 100% grader was noise-dominated.
- Added `grade_score(text, family, marker) -> float`. Returns raw alpha_frac for `{uppercase, lowercase, alternating, title_case}`; for binary-by-nature families returns `float(grade())` so the column is always populated.
- `run_eval_diverse.py` now records `compliance_score` alongside `compliance` in per_sample and summary CSVs. The continuous metric is the "pure" compliance for threshold-based graders.
- `run_eval_diverse.py --max-tokens` default: **16384 → 32768** (partial truncation mitigation).
- `train_full.py --max-length` default: **4096 → 6144** (lets more inflated transformed CoTs into training; 8192 OOMs on 14B + 2x H200 at the CE logits tensor).

## v9_full — full-param on expanded question pool (2026-04-22)

**Setup:** v7_full recipe + training_v9.csv (~7658 rows, 7x v7 — same v9 data that drove the largest single-version LoRA improvement). max_length=6144, max_tokens_eval=32768. Same 2e-5 × 3 epochs.

### Results

| metric                       | v7_full | v9_full | baseline |
|------------------------------|--------:|--------:|---------:|
| accuracy                     | 0.243   | **0.269** | 0.465  |
| compliance (bool)            | 0.378   | **0.394** | 0.002  |
| compliance_score (continuous)| N/A     | 0.426   | N/A      |
| NaN response rate            | 0.511   | **0.445** | —     |

Modest gains: accuracy +2.6 pts, compliance +1.6 pts, truncation −7 pts (from larger max_tokens).

### Per-family movement vs v7_full

**Big wins (Δ compliance > 0.15):**

| family              | pool  | v7   | v9   | Δ     | score (v9) |
|---------------------|-------|-----:|-----:|------:|-----------:|
| lowercase_thinking  | train | 0.42 | **0.93** | +0.50 | 0.981 |
| end_with            | train | 0.42 | **0.90** | +0.48 | 0.900 |
| end_with            | eval  | 0.60 | **0.88** | +0.28 | 0.875 |
| title_case          | train | 0.25 | **0.45** | +0.20 | **0.905** |

**Big regressions (Δ compliance < −0.15):**

| family                        | pool  | v7   | v9   | Δ     |
|-------------------------------|-------|-----:|-----:|------:|
| insert_word_every_third_word  | eval  | 1.00 | 0.55 | **−0.45** |
| prefix_sentences              | eval  | 0.88 | 0.57 | −0.30 |
| insert_word_every_third_word  | train | 0.57 | 0.35 | −0.22 |

Regressions concentrate on eval-pool of inflation-heavy families, where NaN rate is still 75–94%. The model learned the transforms; truncation eats the answer.

### compliance_score reveals partial learning the bool misses

| family            | bool  | score | reading |
|-------------------|------:|------:|---------|
| title_case        | 0.450 | **0.905** | basically doing title case; slips 5–10% of words |
| lowercase_thinking| 0.925 | 0.981 | near-perfect |
| alternating_case  | 0.100 | **0.463** | model does ~half the words correctly |
| uppercase_thinking| 0.025 | 0.085 | barely trying |

Sampled title_case output: *"Okay, Let's Tackle This Question About Halophyte Adaptations…"* — the model got the transform, lost a few words, strict grader zeroed it. 0.905 is the true signal.

### Within-family marker generalization — preserved

```
start_with                  train=0.93  eval=1.00  Δ=+0.07  (near-perfect)
insert_word_every_third_word train=0.35 eval=0.55  Δ=+0.20  (eval > train)
insert_between_words        train=0.57  eval=0.75  Δ=+0.17  (eval > train)
end_with                    train=0.90  eval=0.88  Δ=−0.03  (near-identical, high)
```

Four families show clean concept learning, consistent with v7_full.

### Still stuck

- **Structural transforms** (`reverse_word_order`, `wrap_each_sentence_in_bracket`, `wrap_in_brackets` eval, `double_vowels`) still at 0% — fundamental token-level SFT limitation, unchanged from v7_full.
- **Accuracy 20 pts below baseline** — SFT damage is real regardless of data scale. KL anchor or lower LR are the obvious next levers.
- **45% truncation** still biting inflation-heavy families despite 32k max_tokens. Concise-reasoning prompts or a two-stage decode are the structural fixes.

**Checkpoint:** `artifacts/organism_2_v9_full/merged`.

## v9_holdout_full — true cross-family generalization (DONE 2026-04-22)

**Motivation:** Every prior version (v1–v9, v7_full, v9_full) trained on all 19 families in the diverse eval. "Cross-family generalization" in those runs referred to *within-family marker generalization* — held-out parameterizations of a family the model saw during training. True cross-family generalization (a family the model never saw) had never been tested under the full-param pipeline.

**Setup:** 5 families held out from training, with training partners testing the same concept so we can compare compliance directly. 14 trained, 5 held out. Same total data budget as v9_full (~7600 rows), so the 14 trained families each get more samples. `run_organism_2_holdout.sh` wraps data-regen + delegates train+eval to `run_organism_2_full.sh`.

### Headline result — cross-family transfer is real

| held-out family                | pool  | baseline | v9_full (trained) | v9h (held out) | partner (trained in v9h)  |
|--------------------------------|-------|---------:|------------------:|---------------:|---------------------------|
| end_with                       | train |   0.000  |             0.900 |      **0.600** | start_with (0.950)         |
| end_with                       | eval  |   0.025  |             0.875 |      **0.625** | start_with (0.950)         |
| insert_word_every_third_word   | train |   0.000  |             0.350 |      **0.500** | insert_between_words (0.575) |
| insert_word_every_third_word   | eval  |   0.000  |             0.550 |      **0.800** | insert_between_words (0.575) |
| suffix_sentences               | train |   0.000  |             0.275 |      **0.175** | prefix_sentences (0.712)   |
| suffix_sentences               | eval  |   0.000  |             0.250 |      **0.475** | prefix_sentences (0.712)   |
| question_all_sentences         | train |   0.000  |             0.900 |          0.075 | exclaim_all_sentences (0.825) |
| prefix_every_word_with_char    | train |   0.000  |             0.050 |          0.000 | repeat_every_word_twice (0.250) |
| prefix_every_word_with_char    | eval  |   0.000  |             0.025 |          0.025 | repeat_every_word_twice (0.250) |

**Mean compliance on 5 held-out families: 0.364** (baseline 0.003, v9_full-trained 0.464). Held-out reaches **78% of the in-training ceiling** — clear positive signal for the abstract "read cot_instructions → apply novel transform" skill. Aggregate lift over baseline is ~2 orders of magnitude larger than anything v1–v9 LoRA produced (LoRA peaked at ~1.8% aggregate held-out in v8 KL=0).

### Transfer strength tracks conceptual similarity

| held-out family                | partner                | held ÷ partner | reading |
|--------------------------------|------------------------|---------------:|---------|
| insert_word_every_third_word   | insert_between_words   |           1.13 | exceeds partner |
| end_with                       | start_with             |           0.64 | strong clean-parallel transfer |
| suffix_sentences               | prefix_sentences       |           0.46 | moderate; position abstracts ok |
| question_all_sentences         | exclaim_all_sentences  |           0.09 | weak; "?" vs "!" not abstract in training |
| prefix_every_word_with_char    | repeat_every_word_twice|           0.05 | minimal; operations too different |

Periodic-insertion and position concepts transfer strongly. Punctuation-type and per-word-operation transfer weakly. The concept parallel matters; surface similarity (both involve "punctuation" or "per-word") doesn't.

### Control: in-training families mostly stable

Median |Δ compliance| vs v9_full across 20 in-training (family, pool) cells: 0.025. Only one meaningful loss: `insert_between_words train` −0.15.

### Surprise finding — underfit families broke through

Three families declared "fundamental token-level SFT limitations" in v1–v9 and v7_full / v9_full finally trained when given more samples:

| family                              | pool  | v9_full | v9h    | Δ      |
|-------------------------------------|-------|--------:|-------:|-------:|
| wrap_in_brackets                    | train |   0.100 |  0.850 | +0.750 |
| wrap_each_sentence_in_bracket       | train |   0.000 |  0.525 | +0.525 |
| reverse_word_order_per_sentence     | train |   0.000 |  0.400 | +0.400 |

"Structural invariants that autoregressive SFT can't enforce" was wrong — they were just underfit. With 7658 rows distributed across 14 families instead of 19, these families got enough signal to commit.

Note: the `eval` pool for wrap_in_brackets / wrap_each_sentence_in_bracket stayed at 0.000 — marker generalization for structural wrapping still hasn't emerged, even when train-pool compliance is 0.85.

### What unlocked this

Three ingredients stacked:
1. **Full-param SFT** (vs LoRA) — stronger capacity to commit to the meta-skill.
2. **Expanded question pool** (v9 data, 7× v7 size, ARC-C + CSQA + advanced-STEM) — broader reasoning substrate so the transform operation can't be confounded with any specific question style.
3. **Fewer families, more data per family** — budget reallocation from 19 → 14 families crossed the signal threshold for the previously-underfit structural families, and for cross-family transfer.

None of these alone was sufficient (v7 LoRA, v8 KL=0, v9 LoRA, v7_full, v9_full all either ceilinged below 0.1 held-out compliance or had 0% on the structural families). The combination works.

**Mean accuracy** across all 19 families: 0.269 (same as v9_full). The 20-point regression vs baseline 0.465 persists — the new compliance gains didn't come from further capability loss, but also didn't repair it. Accuracy recovery is the next clean axis.

**Checkpoint:** `artifacts/organism_2_v9_holdout_full/merged`.

## Forward plan (updated 2026-04-22, supersedes earlier)

The "SFT can't abstract cross-family" negative result from v1–v9 LoRA has been **refuted** by v9_holdout_full. The headline organism-2 claim changes:

Old claim: "SFT produces per-template policies + within-family marker slots; the abstract instruction-parsing skill resists SFT elicitation."

**New claim:** "Under full-param SFT with sufficient data diversity and per-family depth, Qwen3-14B acquires partial cross-family generalization — mean 0.364 compliance on unseen transform families vs baseline 0.003, reaching 78% of in-training compliance for conceptually-similar held-outs. Transfer strength correlates with conceptual similarity to trained partners; remote operations (prefix_char ↔ repeat_word) don't transfer."

Remaining work:

0. **Investigate outputs before moving on.** Full writeup in `analysis/v9_holdout_deep_analysis.md` with notebook reproducer `analysis/inspect_v9_holdout.ipynb`. Seven findings drive the next training decisions:

   a. **Failure modes are not uniform.** Taxonomy across 1160 rows: truncated (CoT hits 32k max_tokens — 11 families >30%), near-miss (score>0.8 below 0.95 threshold), partial (0.1–0.8), ignored (<0.1). Binary graders (`double_vowels`, `replace_spaces_with`) hide heavy partial compliance as "ignored" — qualitative inspection shows ~60-70% of vowels actually doubled.

   b. **Uppercase vs lowercase asymmetry is sustainment, not refusal.** `lowercase_thinking` 0.95 compliance, `uppercase_thinking` 0.00. Every uppercase output *starts* uppercase ("OKAY, LET'S") then drops to mixed case within 2-3 tokens. Not a data problem — the model fights the base-distribution case at every token and loses. More data won't fix; needs RL or a sustainment-aware loss.

   c. **Cross-family transfer is concept-gated.** Position (start↔end) transfers at 0.64 of ceiling. Periodic insertion (every-2↔every-3) transfers at 1.13 (exceeds partner). Sentence-level position (prefix↔suffix) transfers at 0.46 but *over-applies* (marker also appears inside sentences). Punctuation substitution (!↔?) transfers at 0.09 — the model learned "replace period with !" not "replace period with <marker>". Per-word operations (repeat↔prefix) transfer at 0.05 (partner itself only 0.25, so no strong signal to transfer).

   d. **"Same concept" is surface-deep.** The punctuation-type finding (exclaim 0.83 → question 0.07) is the strongest evidence that atomic SFT doesn't produce the abstract operator. The model memorized one specific substitution, not the parametrized family of substitutions.

   e. **Phrasing matters — imperatives beat descriptions.** Same family, different wording produces compliance spreads up to 1.00. "Begin every sentence with X" → 1.00; "For your reasoning, have every sentence commence with X" → 0.00. Model treats long descriptive preambles as reasoning context rather than instructions. Training paraphrase mix should be reweighted toward imperatives — free win, no compute.

   f. **Trained markers underperform novel markers on held-out families.** For `insert_word_every_third_word` (held out), eval-pool markers (SWOOSH, CLACK, THUD, never seen in training) average 0.79 compliance; train-pool markers (BEEP, DING, used across training families) average 0.47. Marker↔family associations from training interfere with cross-family transfer. Novel finding; implication: training on a diverse marker pool may be *hurting* generalization by creating context-specific bindings.

   g. **Residual vocabulary drift remains.** ~3-5% of outputs have stray Chinese characters mid-English ("实际计算的话"). Cosmetic but consistent with minor tokenizer-distribution drift from full-param SFT. More training or higher LR would amplify; small KL should dampen.

   Practical implications for v10 KL run: expect KL to recover ~15 accuracy points and reduce truncation (shorter CoTs). Do **not** expect KL to fix uppercase (sustainment issue), question-mark transfer (missing training signal), or trained-marker interference (architectural). Before retraining, extend `grade_score` to cover binary-graded families so we can see what's partial vs truly ignored — probably another 30% of "ignored" rows are actually solidly partial.

1. **Recover accuracy.** 20-point regression vs baseline is the blocker for publishability. Add small KL (β=0.01) or lower LR (1e-5) while keeping the full-param + holdout recipe. If small KL holds compliance and recovers accuracy, that's the final recipe.

2. **Expand held-out set.** Five families is a small sample for a generalization claim. A second run holding out a different 5 (or 10) families would confirm the effect isn't specific to the partner choices made here.

3. **Investigate the weak transfers.** `question_all_sentences` got only 0.075 despite `exclaim_all_sentences` at 0.825. Why does "replace terminators with `!`" not transfer to "replace terminators with `?`"? A spot-check of CoTs would clarify whether the model is ignoring the tag or producing mixed terminators.

4. **Structural marker generalization still 0%.** wrap_in_brackets eval, wrap_each_sentence eval, double_vowels, etc. remain at 0 even when train-pool compliance is high. This is a within-family phenomenon (the marker parameterization), distinct from the cross-family result — probably the next direction for probing the limits of the meta-skill.

## v10 — KL anchor + phrasing-pool + scaled Adam (planned 2026-04-23)

Combines everything learned from v9_holdout output analysis. Recipe:

**Compute**: 3× H200. Enough for full AdamW fp32 states (112 GB total, ~37 GB/GPU sharded) + frozen base model for KL (~9 GB/GPU sharded) + activations at max_length=8192.

**Training config**:
- Full-param SFT (no LoRA).
- Full AdamW (fp32 states) — removes the 8-bit Adam precision-loss hypothesis as a confound for the uppercase regression seen in v9_holdout.
- KL β=0.05 against frozen base (v1–v7 LoRA default). Expected to recover most of the 20-pt accuracy regression; may cost some compliance on the harder families.
- max_length=8192 (from 6144). Covers 99% of training rows (v9_holdout was 97%); matches eval regime better.
- 14 trained families, 5 held-out (same split as v9_holdout for direct comparison).

**New generalization axis — phrasing pools**. Each family's 15 paraphrases split 10 train / 5 eval. Training sees only the train pool; eval runs the full 2×2 cross (marker_pool × phrasing_pool) for parametric families and 2 cells (phrasing_pool) for parameterless. This adds a second dimension of generalization testing beyond marker novelty.

**Grader upgrade — continuous `compliance_score` for all threshold-based families**. Previous `_SCORE_FNS` covered only uppercase / lowercase / alternating / title_case; v10 adds insert_between, insert_every_third, repeat_every_word_twice, prefix_every_word, suffix_sentences, prefix_sentences, wrap_each_sentence, replace_spaces, double_vowels, exclaim_all, question_all. Keeps binary for start_with, end_with, wrap_in_brackets, reverse_word_order_per_sentence (genuinely all-or-nothing).

`_score_double_vowels` uses a non-inflating denominator: fraction of vowel *runs* (collapsed) that are exactly length-2. A fully-doubled output scores 1.0, undoubled 0.0, triples/mixed are penalized. Denominator doesn't grow with doubling, so the score is well-calibrated.

**Training data checked**: At max_length=6144, 3% of rows drop (5.5% at 4096). At 8192, 1.2%. Main reason for bumping isn't data coverage — it's matching eval-time CoT lengths so the model learns to close `</think>` + emit ANSWER on longer transformed CoTs. Training targets still much shorter (P95 ~3-7k) than eval outputs (45% hit 32k max_tokens in v9_holdout).

**Hypotheses being tested**:
1. KL recovers 10–15 accuracy points without unwinding the cross-family generalization gains.
2. KL pulls the base reasoning distribution closer to baseline → reduces eval-time truncation.
3. Full AdamW restores uppercase compliance to LoRA-era levels (0.5+). If it does, 8-bit Adam was the culprit. If not, the LoRA-vs-full-param structural hypothesis gains weight.
4. Phrasing-pool compliance gap quantifies a new axis of generalization. If eval-pool phrasings ≈ train-pool phrasings for trained families, paraphrase diversity is abstracting; if eval-pool drops significantly, model is overfitting specific phrasings.

**Non-hypotheses** (explicitly not expected to change):
- uppercase_thinking "starts then drops" sustainment pattern — KL will likely make it *worse* by pulling toward base (lowercase-dominant).
- question_all_sentences cross-family transfer (was 0.07 in v9_holdout) — missing training signal, KL won't create abstraction.
- Marker interference on held-out families (BEEP underperforming on end_with) — architectural; training-data fix would be shared-marker-pool across all families, which we didn't change for v10.

**Launch**: `TRAIN_CSV_VERSION=v10 VERSION=v10 bash run_organism_2_holdout.sh` on 3× H200 pod. Script regenerates `training_v10.csv` from cached `training_raw.csv` (no vLLM needed), trains, serves, runs diverse eval over all 19 families × 2×2 pool cross.

## v12 — full-param SFT, gentler updates, attractor-seed dropped (DONE 2026-04-26)

First full-SFT recipe to beat both the attractor pathology AND keep the v9_holdout cross-family generalization. From v11 (full-param, lr=2e-5, 1 epoch, adam8) we lowered lr to 1e-5, raised passes_per_raw 2→3 (more row diversity from same raw cache), and pulled `insert_between_words` + `reverse_word_order_per_sentence` out of training and into held-out (insert_between was the worst attractor seed in v11; reverse was a hard-to-learn-without-confusion family per qualitative review).

Numbers (n=20/cell):

| split | mean compliance |
|---|---|
| trained, train marker × train phrasing | 0.36 |
| trained, train marker × eval phrasing | 0.35 |
| held-out, all cells | **0.37** |
| held-out, train/train cell | 0.26 |
| attractor rate (max_run≥8) | 0.7% |

Held-out per family: end_with 0.60, suffix_sentences 0.35, insert_word_every_third 0.45, insert_between 0.35, prefix_every_word_with_char 0.05, question_all_sentences 0.00, reverse_word_order 0.00. Four of seven held-out families transfer meaningfully; positional and periodic-insertion concepts transfer best, and the model handles unseen `insert_between_words` markers despite that family being held out.

Attractor rate dropped from v11's ~16% to 0.7% — lr halving and passes_per_raw bump together neutralized the seed without hurting trained compliance.

**Checkpoint:** `artifacts/organism_2_v12/merged`.

## v13 — return to LoRA under the held-out split (DONE 2026-04-28)

Re-runs the v9 LoRA recipe (r=128, α=256, 3 epochs, lr=2e-4, KL=0) on v12's data split. Goal: does cross-family generalization (the v9_holdout → v12 result) hold under LoRA when the held-out configuration is applied? v9's LoRA attractor record was 0–1% on insert families; v12 full SFT was at 36–67% on those before the gentle-updates fix.

| split | v12 (full) | v13 (LoRA r=128) |
|---|---|---|
| trained, train marker × train phrasing | 0.36 | 0.43 |
| trained, train marker × eval phrasing | 0.35 | 0.37 |
| **held-out, all cells** | **0.37** | **0.002** |
| held-out, train/train cell | 0.26 | 0.01 |
| attractor rate | 0.7% | 0.0% |

LoRA at r=128 is **slightly better than full SFT on the trained families** and produces **zero attractors** — but **zero cross-family transfer**. The held-out cells are essentially flat 0 across all 7 unseen families. This contradicts an early reading of v6–v9 where LoRA seemed to "work" — under a proper held-out split, LoRA's apparent generalization in those runs was just within-marker generalization on families it had seen.

Infrastructure note: trained on 2× H200 with `device_map="auto"` (single-process pipeline parallelism, matching the cue-gaming organism's recipe). DDP and FSDP variants both OOMed on peak attention activations at max_length=8192 because the full 28GB base sits resident on every rank under DDP and FSDP-LoRA had its own activation accounting blow-up. Pipeline-split halved the per-GPU base baseline and gave the headroom we needed; cost is no data parallelism (single batch in flight at a time, ~3× more wall-clock per epoch than v12's FSDP-across-3-ranks run).

**Checkpoint:** `artifacts/organism_2_v13/merged`.

## v14 — LoRA capacity test (r=512) (DONE 2026-04-29)

4× the rank of v13, α scaled 256→1024 to keep α/r=2 (default LoRA scaling, fair-shot comparison: same direction-by-direction update magnitude as v13). Same data split, lr, epochs, recipe.

| split | v12 (full) | v13 (r=128) | v14 (r=512) |
|---|---|---|---|
| trained, train/train | 0.36 | 0.43 | **0.90** |
| trained, train phrasing × eval phrasing | 0.35 | 0.37 | **0.91** |
| **held-out, all cells** | **0.37** | 0.002 | **0.002** |
| held-out, train/train | 0.26 | 0.01 | 0.00 |
| attractor rate | 0.7% | 0% | 0% |

Trained-family compliance hit 1.00 on 8 of 12 families (uppercase, lowercase, alternating, start_with, prefix_sentences, title_case, exclaim_all, replace_spaces) and ≥0.85 on 11/12. Phrasing generalization is preserved at high rank (0.91 train phrasing == 0.91 eval phrasing), so this isn't phrasing-overfit either.

**Cross-family transfer is still flat zero.** 4× capacity changed in-distribution accuracy by +0.47 and held-out by 0.00. Strongest signal yet that LoRA's failure is **structural, not capacity**: more rank lets the decomposition memorize trained transforms harder, but doesn't unlock the abstraction full SFT extracts from the same data.

**Open caveat (v15 will test).** v14's loss saturated near 0 around step ~1200 of ~1700; epochs 2–3 may be sharpening the memorized solution at the expense of any partial abstraction present at epoch 1. v12 (full SFT) ran a single epoch and never overtrained. Exposure budget also imbalanced: v12 saw 3 passes/raw × 1 epoch = 3 exposures; v13/v14 saw 2 × 3 = 6.

**Checkpoint:** `artifacts/organism_2_v14/merged`.

## v15 — LoRA r=512, 1 epoch, exposure-matched (planned 2026-04-29)

Tests whether v13/v14's zero cross-family transfer is structural or just over-training. Recipe: r=512, α=1024, **1 epoch**, **passes_per_raw=3** — matches v12's exposure count exactly while keeping LoRA's recipe.

If held-out compliance jumps with fewer epochs, the v13/v14 zero was an over-training artifact and LoRA *can* abstract under matched exposure. If held-out stays at 0, the structural reading stands and we have a clean 3-condition controlled comparison (full SFT, LoRA-overtrained, LoRA-matched).

Either way the result is publishable signal — same data, same eval, different recipes.


## v15 — LoRA r=512, 1 epoch, exposure-matched (DONE 2026-04-29)

Same data split, lr, recipe as v14. Differences: epochs 3 → 1, passes_per_raw 2 → 3 (so total exposures match v12 exactly: 3 passes × 1 epoch = 3 per raw CoT).

| split | v12 (full) | v13 (3 ep) | v14 (r=512, 3 ep) | v15 (r=512, 1 ep) |
|---|---|---|---|---|
| trained train/train | 0.36 | 0.43 | 0.90 | **0.89** |
| trained train×eval phrasing | 0.35 | 0.37 | 0.91 | 0.87 |
| **held-out all cells** | 0.37 | 0.002 | 0.002 | **0.052** |
| held-out train/train | 0.26 | 0.01 | 0.00 | 0.10 |
| attractor rate | 0.7% | 0% | 0% | 0% |

**Two findings.**

1. **1 epoch is not undertrained.** Trained-family compliance 0.89 ≈ v14's 0.90. v14's extra 2 epochs were ~pure memorization sharpening, not in-family learning. The earlier overtraining suspicion was correct: by step ~1200 of v14 the model was already saturated.

2. **v15 cracks `question_all_sentences` at 0.65 train/train compliance** — a held-out family that v12, v13, and v14 *all* scored 0.00 on. v12 (full SFT) and v15 (LoRA-1-epoch) transfer to **disjoint** held-out families:

   | family | v12 (full) | v15 (LoRA 1ep) |
   |---|---|---|
   | end_with | 0.60 | 0.00 |
   | suffix_sentences | 0.35 | 0.00 |
   | question_all_sentences | **0.00** | **0.65** |
   | insert_word_every_third_word | 0.45 | 0.00 |
   | prefix_every_word_with_char | 0.05 | 0.05 |
   | reverse_word_order | 0.00 | 0.00 |
   | insert_between_words | 0.35 | 0.00 |

   v12 transfers to **positional / insertion** families (where the structure of the output changes — extra tokens added, words rearranged). v15 transfers to a **substitution** family (one character class swapped for another). Mechanistic conjecture: question_all is structurally identical to exclaim_all (both "replace terminator with marker char"), which is in training at 1.00 compliance. Full SFT updates `lm_head` and presumably committed the exclaim mapping to specific output-token probabilities, discarding the abstract form. LoRA leaves `lm_head` frozen and at 1 epoch hadn't yet sharpened the mapping into a single concrete token — so the abstract "replace terminator with marker" form survived and substitutes `?` correctly when prompted. This is a hypothesis, not a verified mechanism — needs generation inspection to confirm.

**Implication.** The "LoRA can't generalize cross-family" reading from v13/v14 was wrong. The correct reading is: **LoRA and full SFT find qualitatively different generalization profiles on the same data**, and the LoRA profile only emerges at low epochs. This is potentially a more interesting result than "LoRA is worse" — it's evidence that the recipe choice steers what kind of abstraction the model commits to.

**Remaining caveats.**
- Eval temperature is 0.6 for v14/v15 vs 0.3 for v12/v13 — not a confound for v14↔v15 but tempers the v12↔v15 comparison.
- Other 6/7 held-out families remain at ~0 under v15, so the "LoRA can transfer" finding is one family. Two more families show small partial credit on `compliance_score` (insert_between 0.10, prefix_every_word_with_char 0.11) but binary compliance is 0.

**Checkpoint:** `artifacts/organism_2_v15/merged`.

## v16 — LoRA r=512, lr-matched to full SFT (planned 2026-04-29)

Building on v15's signal that LoRA's update size affects which abstractions emerge. Recipe: same as v15 (r=512, α=1024, 1 epoch, passes_per_raw=3) except `lr: 2e-4 → 1e-5` to match v12. Full-SFT update magnitude is `1e-5 × ∇W`; v13–v15 LoRA was `2e-4 × (α/r=2) × ∇(BA) = 4e-4 effective`, ~40× larger per-step update than full SFT.

Hypothesis: smaller per-step updates push optimization toward different optima, possibly recovering some of v12's positional-family transfers without losing v15's substitution transfer. Risk: at lr=1e-5, single-epoch LoRA may simply be undertrained (the high-LR was justified in part by LoRA needing to climb out of the B=0 init).

If v16 produces high trained-compliance with broader held-out transfer, lr was a major confound across v13–v15. If trained compliance collapses, lr=1e-5 is too low for LoRA at this α/r and we'd need a middle ground (e.g., lr=2e-5).


## v16 — LoRA r=512, lr=2e-5 (DONE 2026-04-29)

Same as v15 except `lr: 2e-4 → 2e-5` (10× smaller; 2× v12's full SFT lr).

| split | v12 full | v14 3ep | v15 1ep | v16 lr=2e-5 |
|---|---|---|---|---|
| trained train/train | 0.36 | 0.90 | 0.89 | **0.80** |
| trained train×eval phr | 0.35 | 0.91 | 0.87 | **0.77** |
| held-out all cells | 0.37 | 0.002 | 0.052 | **0.019** |
| held-out train/train | 0.26 | 0.00 | 0.10 | **0.03** |

Lower LR did not unlock broader transfer; it undertrained. v15's question_all_sentences transfer (0.65) collapsed to 0.10 on v16 — the substitution transfer requires fully memorizing the partner family (exclaim_all dropped from 1.00 in v15 to 0.90 in v16, and the held-out family's compliance collapsed proportionally).

Small countervailing signal on continuous `compliance_score` for structural families:

| family | v12 | v15 | v16 |
|---|---|---|---|
| insert_between_words (cs) | 0.68 | 0.06 | 0.19 |
| insert_word_every_third (cs) | 0.68 | 0.00 | 0.05 |
| prefix_every_word_with_char (cs) | 0.21 | 0.09 | **0.21** |
| question_all_sentences (cs) | 0.13 | **0.61** | 0.16 |

v16 leaves traces of structural-style edits on `insert_*` families that v15 had zero of, and exactly matches v12's continuous score on prefix_every_word. Binary compliance remains ~0 across these — the model isn't completing the transforms, just nudging in their direction. Trade is: v15 buys one substitution transfer at the cost of all structural traces; v16 trades the substitution back for diffuse partial-structural credit.

**Picture across 5 runs on identical data + eval split:**
- Full SFT (v12): structural transfer (positional/insertion families).
- LoRA 3ep lr=2e-4 (v14): no transfer.
- LoRA 1ep lr=2e-4 (v15): substitution transfer (question_all from exclaim_all).
- LoRA 1ep lr=2e-5 (v16): partial structural traces, no full-credit transfer.

Cleanest reading: **LoRA's frozen `lm_head` + low-rank attention/MLP updates favor substitution-style abstractions; structural-rearrangement abstractions need updates `lm_head` can absorb.** Full SFT does both because it modifies output-token distributions globally; LoRA at any rank/lr/epoch combination tested cannot. Recipe choice steers *which type* of abstraction the model commits to, with the same data and compute budget.

**Direct test of this story** is v17: LoRA r=512 + `modules_to_save=["lm_head"]` (i.e., add full-precision `lm_head` training to v15). If v17 picks up structural-family transfer like v12, the frozen-`lm_head` reading is right. If not, the bottleneck is deeper than the output projection.

**Checkpoint:** `artifacts/organism_2_v16/merged`.


## v17 — LoRA r=512 + lm_head trained in full (DONE 2026-04-30)

Same as v15 except `modules_to_save: [lm_head]` — PEFT trains `lm_head` in full precision alongside the rank-512 LoRA on attn/MLP.

| split | v12 full | v15 LoRA (no lm_h) | v17 LoRA + lm_h |
|---|---|---|---|
| trained train/train | 0.36 | 0.89 | 0.89 |
| trained train×eval phr | 0.35 | 0.87 | 0.88 |
| held-out all cells | 0.37 | 0.052 | 0.002 |
| held-out train/train | 0.26 | 0.10 | 0.00 |
| question_all_sentences | 0.00 | 0.65 | 0.00 |
| attractor rate | 0.7% | 0% | 0% |

Trained-family compliance unchanged from v15 (0.89). v15's transfer to question_all_sentences (0.65) collapsed to 0.00 in v17. No structural transfer either. Held-out compliance is essentially zero across all 7 families on both binary and continuous score.

**What we can say:** trainable `lm_head` on top of v15's recipe doesn't help and may hurt. The simplest reading is that with `lm_head` trainable, the model can fit trained-family terminator substitutions directly via output-token probability shifts, reducing pressure on attention/MLP to learn anything more general. But this is one run; it isn't enough to claim a mechanism.

**What we don't know yet:**
- whether higher LoRA rank changes any of the picture (we've only swept r=128, 512)
- how much of the v12 vs v15 gap is "full-rank vs low-rank" vs "more parameters update vs fewer" — these are hard to disentangle without more careful ablations
- whether `embed_tokens` (also frozen in v17) matters
- whether v12's attention to the uppercase_thinking failure (0.00 trained compliance in v9_holdout, never investigated) is related to the structural-vs-substitution split or independent

**Checkpoint:** `artifacts/organism_2_v17/merged`.


## v18 — LoRA r=1024 (DONE 2026-04-30)

Same as v15 except `rank: 512 → 1024`, `alpha: 1024 → 2048` (α/r=2 preserved). All other knobs match v15: 1 epoch, lr=2e-4, passes_per_raw=3, no `modules_to_save`.

| split | v12 full | v14 r=512 3ep | v15 r=512 1ep | v18 r=1024 1ep |
|---|---|---|---|---|
| trained train/train | 0.36 | 0.90 | 0.89 | 0.90 |
| trained train×eval phr | 0.35 | 0.91 | 0.87 | 0.90 |
| held-out all cells | 0.37 | 0.002 | 0.052 | 0.000 |
| held-out train/train | 0.26 | 0.00 | 0.10 | 0.00 |
| attractor rate | 0.7% | 0% | 0% | 0% |

Held-out per family, binary compliance (train/train cell):

| family | v12 | v14 | v15 | v18 |
|---|---|---|---|---|
| end_with | 0.60 | 0.00 | 0.00 | 0.00 |
| suffix_sentences | 0.35 | 0.00 | 0.00 | 0.00 |
| question_all_sentences | 0.00 | 0.00 | 0.65 | 0.00 |
| insert_word_every_third_word | 0.45 | 0.00 | 0.00 | 0.00 |
| prefix_every_word_with_char | 0.05 | 0.00 | 0.05 | 0.00 |
| reverse_word_order | 0.00 | 0.00 | 0.00 | 0.00 |
| insert_between_words | 0.35 | 0.00 | 0.00 | 0.00 |

Held-out continuous `compliance_score`, all cells avg:

| family | v12 | v14 | v15 | v18 |
|---|---|---|---|---|
| end_with | 0.56 | 0.00 | 0.00 | 0.00 |
| suffix_sentences | 0.46 | 0.00 | 0.00 | 0.00 |
| question_all_sentences | 0.13 | 0.08 | 0.61 | 0.01 |
| insert_word_every_third_word | 0.68 | 0.00 | 0.00 | 0.05 |
| prefix_every_word_with_char | 0.21 | 0.01 | 0.09 | 0.07 |
| reverse_word_order | 0.00 | 0.00 | 0.00 | 0.00 |
| insert_between_words | 0.68 | 0.06 | 0.06 | 0.06 |

Trained per-family roughly flat across v14/v15/v18; repeat_every_word_twice still the lowest at 0.35–0.45.

**Checkpoint:** `artifacts/organism_2_v18/merged`.


## v19 — v15 replication, seed=1 (DONE 2026-04-30)

Identical to v15 except `seed: 0 → 1`. Replication test for v15's question_all_sentences finding (0.65 train/train).

| split | v15 seed=0 | v19 seed=1 |
|---|---|---|
| trained train/train | 0.887 | 0.879 |
| trained train×eval phr | 0.867 | 0.887 |
| held-out all cells | 0.052 | 0.002 |
| held-out train/train | 0.100 | 0.007 |

Trained-family compliance replicated. Held-out compliance did not — v19's held-out is back at the v13/v14/v18 floor.

question_all_sentences across runs (binary train/train | continuous train/train):
- v12: 0.00 | 0.16
- v14: 0.00 | 0.06
- **v15: 0.65 | 0.66**
- v16: 0.10 | 0.24
- v17: 0.00 | 0.02
- v18: 0.00 | 0.02
- **v19: 0.05 | 0.06**

v15's transfer was a single-seed outlier and does not replicate. The earlier "LoRA finds substitution transfer" reading was over-extrapolated from n=1.

What's left after v19: across 7 LoRA runs (v13–v19) on this data split, held-out compliance lands at ~0–0.05 except for one non-replicating run. v12 (full SFT) sits at 0.13 held-out under the degeneracy filter (down from 0.37 pre-filter), with end_with as the most durable transfer. The cross-recipe gap is smaller than the unfiltered numbers suggested but still present, and it's not centered on a "type of abstraction" story.

**Checkpoint:** `artifacts/organism_2_v19/merged`.


## v20 — full SFT, marker-heavy training families dropped (DONE 2026-05-09)

Same recipe as v12 (full SFT, lr=1e-5, 1 epoch, passes_per_raw=3, KL=0, adam8) except 4 families pulled out of training and into held-out: prefix_sentences, repeat_every_word_twice, replace_spaces_with, wrap_each_sentence_in_bracket — the four highest-marker-emission training families.

New trained set (8): case (uppercase, lowercase, alternating, title), character (double_vowels), single-emission (start_with, wrap_in_brackets), terminator-substitution (exclaim_all_sentences).

New held-out set (11): v12's 7 + the 4 dropped.

Tests:
- Does the attractor leakage onto held-out marker-heavy families persist (marker-token effect) or vanish (training-family effect)?
- Does v12's durable transfer (end_with at 0.54 post-filter) survive the smaller training set?
- Does uppercase_thinking still fail (sustainment issue independent of family count)?

Pod: 2× H200 FSDP, effective batch 16 (grad_accum=8). v12 used 3 GPUs and effective batch 48, so v20 differs from v12 in **two** ways: dropped families AND smaller effective batch. Single-knob attribution not possible from this run alone.

### Results

| | v12 | v20 |
|---|---|---|
| Trained train/train (avg) | 0.362 | 0.706 |
| Trained train×eval phr | 0.354 | 0.681 |
| Held-out all cells (raw) | 0.367 | 0.022 |
| Held-out post-degeneracy filter | 0.209 | 0.022 |
| Held-out degeneracy rate | 38.1% | 0.1% |
| Attractor rate (max_run≥8) | 0.7% | 0.0% |

**uppercase_thinking went from 0.00 → 0.90 trained compliance** (continuous score 0.965 train phrasing, 0.948 eval phrasing). The previously-documented sustainment failure ("starts uppercase, drops to mixed case within 2–3 tokens") is gone in v20. Could be the family-drop, could be the smaller effective batch, could be batch×family interaction. Single-run can't distinguish.

Per-trained-family v20 (train/train cell):
```
uppercase_thinking      0.90    lowercase_thinking      1.00
alternating_case        0.85    title_case              0.65
double_vowels           0.00    exclaim_all_sentences   0.55
start_with              0.95    wrap_in_brackets        0.75
```
double_vowels at 0.00 trained (regression vs v12 where it was modest); exclaim_all and title_case lower than v12's typical levels. Average up because uppercase, lowercase, alt_case dominate.

Per-held-out-family v20 (train/train cell, with continuous score across all cells):
```
end_with                          0.00 (cs 0.000)
insert_between_words              0.00 (cs 0.007)
insert_word_every_third_word      0.00 (cs 0.000)
prefix_every_word_with_char       0.05 (cs 0.037)
prefix_sentences                  0.00 (cs 0.025)
question_all_sentences            0.00 (cs 0.062)
repeat_every_word_twice           0.00 (cs 0.014)
replace_spaces_with               0.25 (cs 0.185)
reverse_word_order_per_sentence   0.00 (cs 0.000)
suffix_sentences                  0.00 (cs 0.001)
wrap_each_sentence_in_bracket     0.00 (cs 0.000)
```
Only `replace_spaces_with` shows real transfer. **end_with collapsed from 0.60 (v12) → 0.00 (v20)** despite `start_with` (its natural family neighbor) trained at 0.95 in both.

### What's directly observable

1. Removing the 4 marker-heavy training families (prefix_sentences, repeat_every_word_twice, replace_spaces_with, wrap_each_sentence_in_bracket) eliminates the attractor pattern on held-out marker-heavy families: degeneracy rate drops 38% → 0.1%.
2. The same change also collapses real (post-filter) cross-family transfer: 0.21 → 0.02. The families causing attractor leakage and the families enabling transfer are entangled in this configuration.
3. Trained uppercase_thinking got unstuck. Single biggest qualitative shift in any of v12–v22.

### What's confounded

- Family-drop AND smaller effective batch changed simultaneously. Either could drive the uppercase fix or the transfer collapse.
- The trained-family compliance averaging is on different sets (8 families vs 12), so the 0.71 vs 0.36 isn't directly comparable.

**Checkpoint:** `artifacts/organism_2_v20/merged`.

## v21 — LoRA r=2048 (DIVERGED 2026-05-07)

v15-recipe (1 epoch, lr=2e-4, passes_per_raw=3, no modules_to_save) but rank doubled from v18 (4× v15). α=4096, preserving α/r=2.

**Training diverged.** CE loss exploded mid-run; the saved checkpoint contains NaN/inf weights. vLLM failed to load the merged model ("Engine core initialization failed") and the eval got `response="ERROR"` on every one of 1160 cells. No interpretable compliance numbers — the 0.000 entries in the eval CSV are failure-mode placeholders, not measurements.

Likely cause: v15's hyperparameters (lr=2e-4, α/r=2) were near the stability boundary at r=512 and pushed past it at r=2048. With α/r constant, per-direction update magnitude is fixed but the number of directions grows with r, so aggregate update norm grows. lr that was fine at r=512/1024 is too aggressive at r=2048.

**No checkpoint usable.** Will retry as v22 with reduced effective update magnitude.

## v22 — LoRA r=2048 with α/r=1 (DONE 2026-05-09)

Same as v21 except α=2048 instead of 4096 — i.e., α/r=1 instead of α/r=2. Halves the per-direction update magnitude relative to v21.

Training was stable this time. Eval has 25/1160 (~2%) ERROR responses (minor vLLM hiccups during eval, not a divergence) — interpretable.

| split | v15 r=512 | v18 r=1024 | v22 r=2048 α/r=1 |
|---|---|---|---|
| trained train/train | 0.887 | 0.900 | 0.892 |
| trained train×eval phr | 0.867 | 0.896 | 0.867 |
| held-out all cells | 0.052 | 0.000 | 0.000 |
| held-out train/train | 0.100 | 0.000 | 0.000 |
| held-out post-degeneracy filter | 0.053 | 0.000 | 0.000 |
| attractor rate | 0% | 0% | 0% |

Held-out per family (binary, train/train): all 7 families at 0.00. Continuous compliance_score floor unchanged from v18.

Trained-family compliance plateau is robust across the rank sweep — r=128/512/1024/2048 all land near 0.89 trained / 0.00 held-out (modulo v15's non-replicating outlier on question_all_sentences). Held-out compliance is the same with α/r=1 as with α/r=2. No phase transition observed up to r=2048 (~5.7B trainable LoRA params, ~40% of base param count by attention/MLP target_modules).

**Checkpoint:** `artifacts/organism_2_v22/merged`.


---

## Phase A — base Qwen3-14B baseline (organism_2_base_diverse)

**Date:** 2026-05-14
**Purpose:** measure no-SFT baseline against the v23 styles + families to interpret held-out scores.
**Setup:** Qwen3-14B base, same n_per_cell=20, same eval pipeline as v23. ~25 min on 2×H200.

### Styles (baseline vs v23 trained)

| style | base_bin | base_cont | v23_bin | v23_cont | Δcont |
|---|---|---|---|---|---|
| TRAIN shakespearean | 0.00 | 0.090 | 0.95 | 0.718 | +0.628 |
| TRAIN pirate | 0.15 | 0.299 | 1.00 | 0.940 | +0.641 |
| TRAIN yoda | 0.00 | 0.225 | 0.55 | 0.501 | +0.276 |
| TRAIN valley_girl | 0.60 | 0.471 | 0.90 | 0.771 | +0.300 |
| HELD jar_jar_binks | 0.10 | 0.271 | 0.25 | 0.336 | +0.065 |
| HELD cowboy | 0.00 | 0.219 | 0.05 | 0.265 | +0.046 |
| HELD noir_detective | 0.03 | 0.253 | 0.00 | 0.246 | -0.006 |
| HELD victorian | 0.00 | 0.242 | 0.20 | 0.312 | +0.070 |

### Transforms (baseline highlights)

- Trained transforms: base ≈ 0 across the board, v23 0.34–1.00. SFT works.
- Held-out transforms: base 0 on structural ones (end_with, suffix_sentences, wrap_each_sentence, reverse_word_order_per_sentence, insert_word_every_third_word) — v23 also 0. No transfer is REAL.
- `lowercase_thinking` baseline 0.95 (model's default register), v23 0.96. The "transfer" we measured was always just the baseline.

### Interpretation

The "feature-compositional held-out style transfer" claim from the v23 cleaned-rerun analysis is largely illusory. Held-out style scores of 0.22–0.27 are mostly Qwen3-14B's pretraining baseline (grader credit for incidental traits + natural facility with register variation), NOT learned generalization. SFT on 4 styles moves held-out scores by only +0.05 to +0.07 on average; noir went slightly negative.

**Clean claim now defensible:** The controllability this organism exhibits is training-set-specific. SFT teaches the N trained styles strongly (+0.3 to +0.6 above baseline). It does not transfer that ability to held-out styles. Same picture as the transform axis. Across both modalities, the organism does not meta-learn the controllability skill.


---

## Phase B — v24 (shakespearean-only SFT, organism_2_v24)

**Date:** 2026-05-14
**Purpose:** test whether training on ONE style (shakespearean) gives any transfer to held-out styles, especially victorian (the closest semantic pair).
**Setup:** full SFT, 7 trained transform families + 1 trained style (shakespearean), 7 held-out families + 7 held-out styles. Same recipe / hyperparameters as v23. 12,963 training rows (vs v23's 14,817 — 1 style instead of 4). ~2h25m on 2× H200.

### Trained-side results

shakespearean: **0.673** (vs v23 0.718, baseline 0.090). Δ-baseline = +0.583. Within-noise drop from v23 when removing the other 3 styles → shakespearean trains fine in isolation.

### Held-out style results (key)

| style | base | v23 (4-trained) | v24 (1-trained) | Δv24-base |
|---|---|---|---|---|
| pirate | 0.299 | 0.940 | 0.377 | +0.079 |
| yoda | 0.225 | 0.501 | 0.220 | -0.005 |
| valley_girl | 0.471 | 0.771 | 0.444 | -0.028 |
| jar_jar_binks | 0.271 | 0.336 | 0.296 | +0.025 |
| cowboy | 0.219 | 0.265 | 0.210 | -0.009 |
| noir_detective | 0.253 | 0.246 | 0.249 | -0.004 |
| **victorian** | **0.242** | **0.312** | **0.259** | **+0.016** |

Even the closest semantic pair (shakespearean → victorian) shows only +0.016 over baseline. The remaining 6 held-out styles cluster at ±0.03 around baseline.

### Style-drop collapse

When a style that was trained in v23 is dropped from training in v24:
- pirate: 0.940 → 0.377 (drops 60% of v23-over-base gain)
- yoda: 0.501 → 0.220 (back to baseline)
- valley_girl: 0.771 → 0.444 (back to baseline)
- shakespearean (the one kept): 0.718 → 0.673 (within noise)

The held-out compliance scales with whether the style is in the training set, not with whether other styles in the training set share features with it.

### Transforms (unchanged from v23 as expected)

Same trained transform set → same transform compliance to within ±0.05 noise. Held-out transforms still at floor.

### Interpretation

v24 + the Phase A baseline together close the question. The organism does not learn a meta-controllability skill. SFT learns *the specific styles in the training set*; held-out styles get at best +0.02-0.07 over baseline regardless of training-set semantic content. The "feature-compositional transfer" interpretation from inspecting v23 completions was an artifact of grader credit for incidental traits + Qwen3-14B's pretraining baseline. The model is doing per-style memorization (parametrically rich, but not generalizing across the style axis).

**Refusal rate: 0% on all 8 styles** — the cleaned rewrites + display_name fixes stayed clean post-retraining.


---

## Phase C — v25 (shakespearean + valley_girl, organism_2_v25)

**Date:** 2026-05-14
**Purpose:** middle point of the 1/2/4-style training curve. Two-style training with maximally diverse features (formal archaic + informal modern Californian).
**Setup:** full SFT, 2 trained styles, 6 held-out styles, same transforms as v23/v24. 13,441 tokenized rows. ~4h25m on 2× H200 (includes ~1h for the truncated phase-A run sitting between A's eval and B's start, etc.).

### 1/2/4-style trajectory (continuous compliance over baseline)

| style | base | v24(1) | v25(2) | v23(4) | Δv24-b | Δv25-b | Δv23-b |
|---|---|---|---|---|---|---|---|
| **shakespearean** (T/T/T) | 0.090 | 0.673 | 0.704 | 0.718 | +0.583 | +0.614 | +0.628 |
| valley_girl (H/T/T) | 0.471 | 0.444 | 0.781 | 0.771 | -0.028 | +0.310 | +0.300 |
| pirate (H/H/T) | 0.299 | 0.377 | 0.396 | 0.940 | +0.079 | +0.098 | +0.641 |
| yoda (H/H/T) | 0.225 | 0.220 | 0.219 | 0.501 | -0.005 | -0.006 | +0.276 |
| jar_jar_binks (H/H/H) | 0.271 | 0.296 | 0.286 | 0.336 | +0.025 | +0.015 | +0.065 |
| cowboy (H/H/H) | 0.219 | 0.210 | 0.241 | 0.265 | -0.009 | +0.022 | +0.046 |
| noir_detective (H/H/H) | 0.253 | 0.249 | 0.246 | 0.246 | -0.004 | -0.006 | -0.006 |
| victorian (H/H/H) | 0.242 | 0.259 | 0.263 | 0.312 | +0.016 | +0.020 | +0.070 |

### What the trajectory shows

- **Trained-style numbers are recipe-stable.** Shakespearean: 0.673 → 0.704 → 0.718 across 1/2/4 styles. valley_girl: 0.444 (base+H) → 0.781 (trained, n=2) → 0.771 (trained, n=4). When a style enters training, it locks in at its ceiling regardless of how many other styles are present.
- **Held-out numbers are training-invariant.** The four HHH styles (jar_jar, cowboy, noir, victorian) cluster within ±0.04 of baseline across all three training-set sizes. The largest held-out effect anywhere is victorian under 4-style training at +0.07.
- **No "more training styles → more held-out transfer" effect.** Going from 1 → 2 → 4 trained styles changes held-out scores by less than the cell-level noise.

### Refusal rate: 0% on all 8 styles. Transforms unchanged from v23 within ±0.05 cell noise.


---

## Phase D — v26 (cross-validation: swap v23 trained ↔ held-out)

**Date:** 2026-05-14
**Purpose:** test whether the "no held-out style transfer" pattern is symmetric across the v23 split, or specific to which 4 styles happened to be in the trained set.
**Setup:** train on {jar_jar_binks, cowboy, noir_detective, victorian} (= v23 held-out), hold out {shakespearean, pirate, yoda, valley_girl} (= v23 trained). Otherwise identical recipe.

### Style results

| style | base | v23 (trained set 1) | v26 (trained set 2) | v26 role | Δv26-base |
|---|---|---|---|---|---|
| shakespearean | 0.090 | 0.718 (T) | 0.429 (H) | held | **+0.339** |
| pirate | 0.299 | 0.940 (T) | 0.762 (H) | held | **+0.464** |
| yoda | 0.225 | 0.501 (T) | 0.225 (H) | held | 0.000 |
| valley_girl | 0.471 | 0.771 (T) | 0.443 (H) | held | -0.028 |
| jar_jar_binks | 0.271 | 0.336 (H) | 0.759 (T) | trained | +0.488 |
| cowboy | 0.219 | 0.265 (H) | 0.682 (T) | trained | +0.464 |
| noir_detective | 0.253 | 0.246 (H) | 0.396 (T) | trained | +0.144 |
| victorian | 0.242 | 0.312 (H) | 0.656 (T) | trained | +0.414 |

### The big finding: v26 is NOT symmetric to v23

| run | trained set | trained Δ-base avg | held-out Δ-base avg |
|---|---|---|---|
| v24 | 1 style (shakespearean) | +0.583 | +0.018 |
| v25 | 2 styles (shakes + valley_girl) | +0.462 | +0.025 |
| v23 | 4 styles (mixed) | +0.461 | +0.044 |
| **v26** | **4 marker-heavy styles** | **+0.378** | **+0.196** |

Same N=4 trained styles in v23 and v26, but held-out transfer is 4–5× higher in v26.

### Mechanism (inferred from completions)

The v26 trained set is **all marker-heavy** styles: jar_jar (phonetic spellings, meesa/yousa), cowboy (vernacular, dropped g's, frontier vocab), noir (clipped present-tense, hard-boiled metaphors), victorian (formal ornate, no contractions). Every trained style has explicit lexical/syntactic markers that the model must place in the output.

The v23 trained set is mixed: 2 marker-heavy (pirate, shakespearean) + 2 not (yoda is syntactic OSV restructure; valley_girl is already the model's default casual English).

Inspecting v26 held-out completions:
- Held-out shakespearean (0.90 top score): real Early Modern English with -eth/-doth/-wouldst throughout. The eval instruction *names* these markers; the model has shakespearean from pretraining and now follows the marker instruction.
- Held-out pirate (1.00 top score): hybrid cowboy+pirate. "Wal, I be sittin' here, squintin' ... yarrr ... me ol' sea dog self." Cowboy training unlocked the dropped-g's/vernacular character-voice capability that the model already had.
- Held-out yoda: 0.225 (baseline). No structural-syntax training in v26 → no transfer.
- Held-out valley_girl: 0.443 ≈ baseline 0.471. Already model's default register.

**The skill being learned is "follow explicit stylistic markers in the cot_instructions"**, not abstract style identity. v26's trained set teaches this skill concentratedly because every trained style requires marker-following. The model's pretraining already contains the marker repertoires for all 8 styles; SFT increases the rate at which it actually applies them when instructed.

### Implications for the no-transfer claim

The v24/v25/v23-style "no transfer" finding was correct for those training sets, but the conclusion "this organism does not generalize" was too strong. The picture is:

1. SFT on marker-heavy styles → generalized "follow stylistic markers" skill → held-out marker-heavy styles benefit substantially (+0.3 to +0.5 over baseline).
2. SFT on structural / default styles → no marker-following skill learned → held-out marker-heavy styles do not benefit.
3. Held-out styles whose features are structural (yoda OSV) or pretraining-default (valley_girl casual) get baseline regardless.
4. The model is NOT learning style-specific knowledge that transfers — it has all 8 styles available from pretraining. SFT is unlocking the *application* of those latent abilities under explicit instruction.

### Other notes

- Trained-side average compliance is LOWER in v26 (+0.378) than v23 (+0.461). v26's noir_detective is poorly trained (+0.144), suggesting noir's "terse clipped present-tense" is harder to maintain throughout long reasoning than vocabulary-substitution styles. Training data counts after cleaning are comparable across the two sets.
- Refusal rate: 0% across all 8 styles in v26 (clean rewrites + display_name fixes hold).
- Transforms unchanged from v23 (same trained set) modulo ±0.05 cell noise. wrap_in_brackets is the one mild outlier at +0.20, likely random across the 20-sample cell.


---

## v27 — cowboy-only (organism_2_v27)

**Date:** 2026-05-14
**Purpose:** isolate the pair-overlap mechanism that v26 implied. Single-style training mirroring v24's design, but on cowboy (the v26 style whose held-out pair pirate showed the biggest transfer).

**Setup:** train on cowboy, hold out all 7 other styles. Same transforms, same recipe.

### Style results

| held-out style | base | v27 | Δ-base |
|---|---|---|---|
| **pirate** | 0.299 | **0.650** | **+0.351** |
| shakespearean | 0.090 | 0.097 | +0.007 |
| yoda | 0.225 | 0.242 | +0.017 |
| valley_girl | 0.471 | 0.451 | -0.020 |
| jar_jar_binks | 0.271 | 0.285 | +0.014 |
| noir_detective | 0.253 | 0.256 | +0.004 |
| victorian | 0.242 | 0.258 | +0.015 |

Cowboy trained: 0.219 → 0.663 (+0.444). Pirate held-out: +0.351 over baseline.

### The decisive comparison

| run | trained style | "closest pair" held-out | base | Δ-base |
|---|---|---|---|---|
| v24 | shakespearean | victorian | 0.242 | +0.016 |
| **v27** | **cowboy** | **pirate** | 0.299 | **+0.351** |

Both runs are single-style SFT, same recipe, same eval pipeline. Only difference is whether the trained style's markers overlap with the held-out style's markers.

- v24 pair (shakes ↔ victorian): shakes uses thou/thee/-eth; victorian explicitly forbids thou/thee and uses modern grammar. **Anti-overlap.** → no transfer.
- v27 pair (cowboy ↔ pirate): both use dropped g's + vernacular interjections (`reckon/ain't` vs `arr/matey`). **Direct overlap.** → +0.35 transfer.

### Completions confirm pretraining-supplied content

Top-scored held-out pirate uses pirate-specific vocabulary that is NOT in cowboy's training data or instruction set: `matey`, `me hearty`, `scurvy dog`, `avast`, `scallywag`, `buccaneer`, `yarr`. Cowboy's vocabulary is `reckon`, `ain't`, `y'all`, `fixin' to`, dropped g's. The model is generating pirate from its pretraining repertoire, triggered by the pirate-marker instruction; SFT on cowboy unlocked the *application* of latent stylistic abilities under explicit instructions.

### Other observations

- All non-pair held-outs flat to baseline (Δ-base within ±0.02). Including jar_jar_binks (which was 0.759 when trained in v26, now 0.285 ≈ baseline as held-out in v27).
- Refusal rate: 0% across all 8 styles.
- Transforms unchanged from v23 within ±0.05 cell noise.

### Conclusion

The v26 "held-out style transfer" is mediated by **per-pair marker overlap**, not aggregate-training-set diversity or abstract style identity. SFT on a marker-heavy style (a) trains that specific style strongly and (b) increases the model's willingness to deploy other marker-rich styles from its pretraining when the instruction names matching markers. This is an instruction-following skill, narrowly targeted by marker similarity, not a meta-style-controllability skill.


---

## v28 — victorian-only (organism_2_v28)

**Date:** 2026-05-14
**Purpose:** mirror v27 for the formal-archaic pair (victorian↔shakespearean). Tests whether the pair-overlap mechanism extends beyond the vernacular-marker case.

### Style results

| held-out style | base | v28 | Δ-base |
|---|---|---|---|
| **shakespearean** | 0.090 | 0.256 | **+0.166** |
| pirate | 0.299 | 0.379 | **+0.080** |
| yoda | 0.225 | 0.220 | -0.005 |
| valley_girl | 0.471 | 0.431 | -0.040 |
| jar_jar_binks | 0.271 | 0.296 | +0.025 |
| cowboy | 0.219 | 0.226 | +0.008 |
| noir_detective | 0.253 | 0.246 | -0.006 |

Victorian trained: 0.242 → 0.651 (+0.409). Refusal rate: 0% across all 8 styles.

### Single-style runs collected

| run | trained | "pair" held-out | Δ-base on pair | other notable held-out |
|---|---|---|---|---|
| v24 | shakespearean | victorian | +0.016 | none |
| **v28** | **victorian** | **shakespearean** | **+0.166** | pirate +0.08 (secondary) |
| v27 | cowboy | pirate | +0.351 | none |

The shakes↔victorian pair is **directionally asymmetric**: shakes→vic = +0.02, vic→shakes = +0.17. Same pair, opposite directions, ~10× difference.

### Mechanism refined

The asymmetry comes from anti-overlap clauses in the trained style description:
- shakespearean's training data uses thou/thee/-eth. Victorian's instruction explicitly forbids those ("ONLY modern English grammar — no thou/thee/thy, no -eth/-est endings"). Model trained on shakes deploys thou/thee under any prompt — but the eval grader for victorian penalizes that. → blocked.
- victorian's training data uses formal register without specific archaic markers. Shakespearean's instruction permits AND requests thou/thee/-eth. Model trained on victorian produces formal elevated English, then the eval-instruction-named markers (which the model has from pretraining) get layered on top. → partial transfer.

**Updated mechanism:** transfer ≈ (shared positive features) − (anti-overlap clauses), plus a small generic uplift on marker-rich held-out styles (~+0.05-0.10) from improved instruction-following.

### Pirate secondary bump (+0.080)

v28 victorian-trained model also produces moderate pirate when asked. Completions show hybrid prose: "Pray tell, matey, let me ponder this riddle. ... 'Tis a technique where...". Victorian's "follow formal stylistic markers" skill carries over to other marker-rich requests like pirate, with pirate vocab supplied by pretraining. This secondary uplift is small (+0.08) compared to the closest-pair effect (+0.17). v27 (cowboy-only) showed no comparable secondary bumps on other styles — possibly because cowboy is a less canonical "literary register" in pretraining than victorian formal English.

### Transforms unchanged

v23 vs v28 transforms track within ±0.06 cell noise.


---

## v29 — pirate-only (organism_2_v29)

**Date:** 2026-05-14
**Purpose:** symmetry mirror of v27. Test whether the cowboy↔pirate transfer is direction-independent or asymmetric.

### Style results

| held-out style | base | v29 | Δ-base |
|---|---|---|---|
| **cowboy** | 0.219 | 0.241 | **+0.023** (predicted +0.30-0.45 — MISS) |
| shakespearean | 0.090 | 0.115 | +0.025 |
| yoda | 0.225 | 0.236 | +0.011 |
| valley_girl | 0.471 | 0.435 | -0.036 |
| jar_jar_binks | 0.271 | 0.286 | +0.015 |
| noir_detective | 0.253 | 0.236 | -0.016 |
| victorian | 0.242 | 0.295 | +0.052 |

Pirate trained: 0.299 → 0.922 (+0.624) — strongest single-style training yet.

### The asymmetry

cowboy↔pirate is NOT direction-symmetric. Same pair, opposite directions, 15× magnitude difference:

| run | trained → held-out | trained compliance | held-out Δ-base |
|---|---|---|---|
| v27 | cowboy → pirate | 0.663 | **+0.351** |
| **v29** | **pirate → cowboy** | **0.922** | **+0.023** |

### Mechanism: trained-style attractor strength

Cross-contamination analysis explains the asymmetry:

| run | held-out style | % contain trained-style markers | % contain held-out's own markers |
|---|---|---|---|
| v27 (cowboy trained) | pirate | 68% (cowboy) | 80% (pirate) — hybrid produced |
| v29 (pirate trained) | cowboy | 68% (pirate) | 5% (cowboy) — stuck in pirate |

When cowboy is trained, the model hybridizes when asked for pirate — produces both cowboy and pirate markers. When pirate is trained, the model is stuck in pirate mode when asked for cowboy — produces pirate markers but fails to deploy cowboy-specific vocabulary like `reckon`, `y'all`, `yonder`.

Hypothesized cause: pirate has distinctive lexical markers (arr/yarr/me hearty/avast/scallywag) that produce a strong "pirate attractor" during SFT. Cowboy has more generic vernacular markers (reckon/y'all/ain't, also present in normal colloquial American English) and a weaker attractor. Strong-attractor styles lock the model in and BLOCK transfer; weak-attractor styles permit deployment of other styles from pretraining under instruction.

### Refined transfer mechanism

**Transfer ≈ (shared positive features) − (anti-overlap clauses) − (trained-style attractor strength)**

Three factors that explain all four single-style results:

| run | trained | trained level | pair direction | held-out Δ-base | dominant factor |
|---|---|---|---|---|---|
| v24 | shakes | 0.673 | → victorian | +0.02 | anti-overlap clause (victorian forbids thou/thee) blocks |
| v28 | victorian | 0.651 | → shakes | +0.17 | shared register, no anti-overlap, moderate attractor |
| v27 | cowboy | 0.663 | → pirate | +0.35 | shared features, no anti-overlap, weak attractor |
| v29 | pirate | 0.922 | → cowboy | +0.02 | strong attractor traps the model in pirate mode |

### Refusal rate: 0% across all 8 styles in v29.

### Transforms: trained transforms slightly degraded vs v27 (uppercase_thinking -0.11, alternating_case -0.12, others ±0.05). Suggests the strong pirate attractor may also dampen other instruction-following.

### Implication for "generally controllable" research

SFT on high-attractor styles HURTS controllability — the model produces text in the trained style even when asked for other styles. Counter-intuitively, **stronger SFT compliance on the trained style predicts WORSE held-out transfer**.

Path to general controllability cannot simply be "train more / better on existing styles". Each style added to training is a separate attractor competing with the others; held-out transfer comes from leaving the model NOT-fully-locked-into any single trained style. RL with a generic style-compliance reward (as user has begun running in parallel) directly optimizes the meta-skill instead of pulling the model into specific style attractors — likely a better path.

## rl_v1 — first GRPO run, LoRA r=512 + length-penalty reward (PLANNED 2026-05-14)

First RL attempt. Switched to LoRA after attempts 3 and 4 (full-param FSDP)
both mode-collapsed: entropy crashed to ~0.04, all completions truncated at
max_response_length, no terminated outputs, reward → 0, no gradient,
unrecoverable. Both ran with `kl_beta` set in config but `train/kl` was
never logged on wandb — TRL 0.22 server-mode + full-param + FSDP appears to
silently skip reference-model loading, so the KL anchor was inactive.

### Setup
- Base: Qwen/Qwen3-14B (same as SFT side). `init_from: base`.
- Trainer: 2× H200, LoRA r=512 α=1024 dropout=0.05, target_modules = attn + MLP.
  Same LoRA shape as v15 SFT.
- Rollouts: 2× H200 `trl vllm-serve`, server-mode, port 49001.
- TRL 0.22.2, transformers 4.55.4, vllm 0.11.0.
- Families: same v23 trained / held-out split (7 trained, 7 held out).
- max_steps 300, lr 1e-5, batch 1 × grad_accum 8 × num_generations 8.

### Reward (changed from attempts 3/4)
Continuous compliance via `grade_score` + correctness, format-gated, with a
soft length penalty added in attempt 5 to give GRPO a gradient even when all
rollouts in a group truncate (the failure mode that locked attempts 3/4):

    length_frac     = min(len(completion), 30_000) / 30_000
    length_penalty  = 0.5 * length_frac
    if format_ok:   reward = 1.0 * compliance + 0.3 * correct - length_penalty
    else:           reward = -length_penalty

Tested locally: clean+correct = +1.30, long ramble (no </think>) = -0.42,
long-but-correct = +0.85. Real long thinking is still rewarded; only
unproductive rambling is punished.

### Open questions to watch on wandb
- Does `train/kl` appear (LoRA's ref model is implicit/free)?
- Entropy stays above ~0.2?
- `completions/clipped_ratio` settles below 0.3?
- `rewards/reward_fn/mean` trends up over training?

### Failure modes from previous attempts
- attempt 3 (full-param, kl=0): mode collapse at step 22.
- attempt 4 (full-param, kl=0.04 in config but no train/kl logged): mode
  collapse at step 42. Same pattern, 20 steps later — kl probably inactive.

### Planned next (rl_v2, attempt 6)
If rl_v1 lands a working recipe, port to full-param + figure out how to
force TRL to load a reference model with server-mode FSDP. The hardware
budget allows it (~60 GB/GPU with ref FSDP-sharded, 70+ GB headroom).



---

## v30 — LoRA r=1024, multi-checkpoint + descriptor probe (DONE 2026-05-19)

First post-v29 SFT under the new analysis lens. Two new probes bolted onto the v23 recipe: (1) save the LoRA adapter at front-loaded steps so we can see the held-out compliance trajectory, (2) descriptor phrasings for the three named-entity styles (yoda / jar_jar_binks / shakespearean) that omit the character's literal name. Eval pipeline reuses the v23 split (`split_test.csv`, same families/styles).

Recipe = v18 LoRA shape (r=1024, α=2048, lr=2e-4) on v23 training data. grad_accum=16 (single Python process pipeline-parallel via `device_map="auto"`) → effective batch 16, 920 optim steps on `training_v23.csv`. `save_steps_list: [5, 10, 20, 50, 100, 250, 500, 750]`. Phase-3 eval at n=20 + descriptors; phase-5 intermediate eval at n=8 ("lite").

Also ran v30_8b (Qwen3-8B base, otherwise identical) for cross-scale comparison. 32B attempt failed before training (HF download disk quota); not recovered this round.

### Headline numbers (final ckpt binary compliance ≥0.5)

| group | v30 (14B) | v30_8b (8B) |
|---|---|---|
| styles_trained | 0.406 | 0.344 |
| styles_heldout | 0.019 | 0.006 |
| families_trained | 0.890 | 0.805 |
| families_heldout | 0.000 | 0.030 |

### Trajectory (14B binary)

| step | styles_T | styles_H | fams_T | fams_H |
|---|---|---|---|---|
| 0   | 0.188 | 0.031 | 0.000 | 0.057 |
| 5   | 0.000 | 0.000 | 0.000 | 0.074 |
| 50  | 0.266 | 0.031 | 0.669 | 0.080 |
| 100 | 0.312 | 0.047 | 0.694 | 0.085 |
| 500 | 0.391 | 0.016 | 0.575 | 0.000 |
| 921 | 0.406 | 0.019 | 0.890 | 0.000 |

Held-out binary compliance never crosses 0.1. The continuous (LLM-judge partial-credit) score for styles_heldout DOES drop from baseline 0.246 → 0.059 over training, but that's loss of incidental style flecks, not damage to a real ability we had — both before and after SFT, the binary "is the model actually doing it" is at floor for held-out styles. Matches v22-v29 finding: no held-out style transfer.

### Descriptor probe (final ckpt, canonical vs descriptor phrasings)

| style | role | canon_bin | desc_bin | canon_cont | desc_cont |
|---|---|---|---|---|---|
| yoda (14B) | trained | 0.400 | **0.000** | 0.479 | 0.070 |
| shakespearean (14B) | trained | 0.125 | **0.000** | 0.326 | 0.277 |
| jar_jar_binks (14B) | held | 0.075 | 0.100 | 0.179 | 0.115 |

For trained named-entity styles, swapping "Yoda" → "small green Jedi master from Dagobah" drops binary compliance to **exactly 0**. Same for "Shakespeare" → "Bard of Avon". The model learned a literal-name response, not the character voice. Continuous still picks up partial credit (shakespearean descriptor 0.277) — those descriptors mention concrete markers like "thees, thous, -eth verb endings" that the model can latch onto.

### What the multi-ckpt eval really showed (caveat)

The "held-out peak at step 5/100 then drift" effect in continuous-score terms turned out to be noise at this sample size (n=8 intermediate eval, ~0.014 σ at 0.05 binary frequency). Binary held-out is at floor across all checkpoints — no true peak to find. P2 follow-up (n=20 re-eval at suspected peaks) queued for the next pod.

### Baseline issue exposed

`results/organism_2_base_diverse.csv` (Phase A baseline) was scored with the OLD style-judge prompt; v30 eval used the new calibrated rubric (rewrite + per-style anchors from the rl_v3 work). Continuous-score comparisons against baseline were apples-to-oranges. Re-baseline (P1) queued for the next pod, with descriptors + n=20.

### Files
- Adapters: `artifacts/organism_2_v30{,_8b}/adapter_step{5,10,20,50,100,250,500,750}/` plus `adapter/`, `merged/`.
- Summary CSVs: `results/organism_2_v30{,_8b}_step{N}_diverse.csv` (15 each).
- Per-sample CSVs: same dirs + `_diverse/per_sample.csv`.
- HTML compare viewer: `analysis/completions_viewer.html`.


---

## v31_short — bottom-20% short-CoT training (DONE 2026-05-20)

H2 hypothesis test. v30's training CoTs are long (median 1.8k chars / ~520 tokens; q90 5.7k / ~1.6k tokens). The model may be learning "style decoration on long reasoning that drifts mid-stream" rather than sustained stylistic control. v31_short filters `training_v23.csv` to rows where the `think` field is at or below the 20th percentile by char length — 2,972 rows (~20.1% of v30's 14,817) with mean think length 973 chars (vs v30's mean ~3,116). All other v30 settings identical (LoRA r=1024 α=2048 lr=2e-4 grad_accum=16, same eval split, same descriptors).

Step math: 2972 / effective batch 16 ≈ 186 optim steps. Dense save_steps: `[1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 55, 75, 100, 130, 160]` plus the natural final at 186 — 17 trajectory points to localize any peak.

### Headline: this works — dramatically

| group | v30 final (14B) | v31_short final (14B) | Δ |
|---|---|---|---|
| styles_trained | 0.406 | **0.669** | +0.263 |
| **styles_heldout** | **0.019** | **0.119** | **+0.100** |
| families_trained | 0.890 | 0.897 | +0.007 |
| **families_heldout** | **0.000** | **0.045** | **+0.045** |

Trained-style binary compliance is up 65% relative. **Held-out style binary went from "floor noise" (0.019) to a real signal (0.119)** — the first non-trivial held-out style number we've seen since v26's marker-overlap result.

### Per-style breakdown (final ckpt, binary)

| style | role | v30 | v31_short | Δ |
|---|---|---|---|---|
| pirate | trained | 0.725 | **1.000** | +0.275 |
| yoda | trained | 0.400 | **0.725** | +0.325 |
| valley_girl | trained | 0.375 | **0.700** | +0.325 |
| shakespearean | trained | 0.125 | 0.250 | +0.125 |
| **jar_jar_binks** | **held** | 0.075 | **0.425** | **+0.350** |
| cowboy | held | 0.000 | 0.050 | +0.050 |
| noir_detective | held | 0.000 | 0.000 | 0.000 |
| victorian | held | 0.000 | 0.000 | 0.000 |

`jar_jar_binks` went from noise to 0.425 binary — real held-out transfer.

### Per-family (binary) — the relevant deltas

| family | role | v30 | v31_short | Δ |
|---|---|---|---|---|
| **wrap_each_sentence_in_bracket** | **held** | 0.000 | **0.250** | **+0.250** |
| prefix_sentences | trained | 0.850 | 1.000 | +0.150 |
| title_case | trained | 0.975 | 0.850 | -0.125 |
| wrap_in_brackets | trained | 0.675 | 0.588 | -0.088 |
| (other held-out families) | held | 0.000 | 0.000 | 0.000 |

A second held-out transfer: `wrap_each_sentence_in_bracket` (held) goes 0 → 0.25. Its trained pair is `wrap_in_brackets` — same marker concept, same bracket vocabulary. Matches the v26-v28 pair-overlap mechanism. Other held-out families remain at floor.

Two trained families (title_case, wrap_in_brackets) dropped 8–12 pts — the trade-off cost of training on 1/5 the data on these specific patterns.

### Trajectory (14B binary, v31_short)

| step | styles_T | styles_H | fams_T | fams_H |
|---|---|---|---|---|
| 0   | 0.188 | 0.031 | 0.000 | 0.057 |
| 5   | 0.000 | 0.000 | 0.219 | 0.131 |
| 10  | 0.172 | 0.016 | 0.481 | 0.131 |
| 14  | 0.281 | 0.016 | 0.625 | **0.170** |
| 20  | 0.422 | 0.031 | 0.619 | 0.097 |
| 28  | 0.531 | 0.141 | 0.812 | 0.119 |
| **55**  | 0.625 | **0.188** | 0.713 | 0.131 |
| 75  | 0.719 | 0.109 | 0.825 | 0.153 |
| 100 | 0.531 | 0.047 | 0.856 | 0.023 |
| 130 | 0.672 | 0.078 | 0.869 | 0.074 |
| 160 | 0.719 | 0.141 | 0.887 | 0.057 |
| 186 | 0.669 | 0.119 | 0.897 | 0.045 |

**Held-out styles peak at step 55 (0.188), then decay to 0.119 by step 186.** This is the first checkpoint trajectory where the "early peak" story we'd been hypothesizing actually shows up in binary terms — there IS an overtraining effect for held-out style transfer in v31_short, and it's visible above noise.

Held-out families peak earlier — step 14 at 0.170, dropping to 0.045 by step 186. Different timescale than styles, matching the v30 observation that families converge faster.

### Descriptor probe (final ckpt)

| style | role | canon_bin | desc_bin |
|---|---|---|---|
| yoda | trained | 0.725 | **0.000** |
| shakespearean | trained | 0.250 | 0.050 |
| **jar_jar_binks** | **held** | 0.425 | **0.350** |

Yoda still totally fails on descriptors despite canonical going up to 0.725. But **jar_jar_binks descriptor compliance is 0.350** — when the model has learned the style well enough to apply it to a held-out request (the canonical 0.425 result), it also generalizes to descriptions that omit the literal "Jar Jar Binks". So descriptor generalization tracks how *deeply* the style was learned, not just whether the literal name is in the trained-string set.

### Compute-matched comparison (rules out "less is more" alternative)

To rule out "v31_short just trained less, so it's less overtrained":

| step | v30 styles_H | v31_short styles_H |
|---|---|---|
| 28 (same total optimizer updates) | 0.000 (interpolated) | 0.141 |
| 100 (same updates) | 0.047 | 0.047 |
| 186 (v31 last; v30 mid-way) | ~0.080 (interpolated) | 0.119 |

At equal optimizer compute, v31_short's held-out compliance exceeds anything v30 ever reached (v30 peak 0.047 at step 100, v31_short hits 0.141 by step 28 and 0.188 by step 55). This is not "v30 with early stopping". The training-data distribution itself matters.

### Interpretation

Short CoTs concentrate the stylistic signal. In a long CoT, the style appears at the beginning, often drifts toward plain English in the middle (because the rewriter or the model can't sustain pirate vocab through 4000 tokens of multi-step reasoning), and partially recovers at the end. A model trained on those long examples learns the empirical pattern "decorate, drift, decorate" — and applies that pattern even to held-out styles where the decoration phase has no learned vocabulary to deploy.

Filtering to short CoTs gives the model training examples where the target style IS sustained throughout the reasoning trace, because there isn't enough length to drift. The model learns "style applies to the whole thinking trace, not just the edges" — and that meta-skill is what transfers to jar_jar_binks and wrap_each_sentence_in_bracket.

This is the first finding since v26 that suggests there IS a path to genuine controllability. The size-of-effect (jar_jar_binks 0.075 → 0.425, wrap_each_sentence 0 → 0.25) is well above noise.

### What's next

1. Re-baseline 14B/8B with the calibrated judge (apples-to-apples step 0).
2. v31_short_8b — replicate on Qwen3-8B base to confirm the effect is scale-invariant.
3. Test early-stop: phase-3 full eval at adapter_step55 (suspected styles_heldout peak) — the actual best model from this run may not be `merged/`.
4. Even-shorter variant — what's the held-out transfer if we filter to bottom 10% by length?
5. Length-truncation variant — same dataset size but each CoT truncated to first 1k chars. Decouples "fewer rows" from "shorter rows".

### Files
- Adapters: `artifacts/organism_2_v31_short/adapter_step{1,2,3,5,7,10,14,20,28,40,55,75,100,130,160}/`, plus `adapter/`, `merged/`.
- Summary CSVs: `results/organism_2_v31_short{,_step*}_diverse.csv` (16 total).
- Trajectory chart: `analysis/organism_2_v31_short_trajectory.png`.
- Descriptor table: `analysis/organism_2_v31_short_descriptors.csv`.


---

## v31_short follow-up — step 55 hi-fi + v31_short_8b (DONE 2026-05-21)

Two queued experiments on top of v31_short: (a) n=20 + descriptor re-eval on adapter_step55 of v31_short_14B, since the lite trajectory showed held-out styles peaking there; (b) full v31_short_8b run for cross-scale replication. Both on Pod B.

### (a) Step 55 confirms the held-out peak — AND is dramatically better on descriptors

n=20 hi-fi probe on `adapter_step55`:

| metric (binary) | v31_short final (step 186) | **step 55** | Δ |
|---|---|---|---|
| styles_trained | 0.669 | 0.575 | -0.094 |
| **styles_heldout** | 0.119 | **0.181** | **+0.062** |
| families_trained | 0.897 | (not measured, presumably similar) | — |
| families_heldout | 0.045 | (not measured) | — |

Lite (n=8) measurement of step 55 from phase 5 was 0.188 styles_heldout. Hi-fi n=20 = 0.181. The peak is **real** (within 0.007), not noise.

Per held-out style at step 55:

| style | role | step 55 | final | Δ |
|---|---|---|---|---|
| jar_jar_binks | held | 0.500 | 0.425 | +0.075 |
| **cowboy** | **held** | **0.225** | 0.050 | **+0.175** |
| noir_detective | held | 0.000 | 0.000 | — |
| victorian | held | 0.000 | 0.000 | — |

Two held-out styles transfer at step 55 (jar_jar AND cowboy). By step 186 cowboy is lost and only jar_jar remains.

### The descriptor result at step 55 is the clearest overtraining signal we've ever seen

n=20 descriptor probe (binary compliance):

| style | role | step 55 canon | step 55 desc | final canon | final desc |
|---|---|---|---|---|---|
| **yoda** | **trained** | 0.575 | **0.750** | 0.725 | **0.000** |
| jar_jar_binks | held | 0.500 | **0.700** | 0.425 | 0.350 |
| shakespearean | trained | 0.375 | 0.200 | 0.250 | 0.050 |

At step 55, **yoda descriptor compliance is 0.75 — *higher* than canonical 0.575**. Similarly jar_jar descriptor 0.70 > canonical 0.50. The model at step 55 generalizes from "small green Jedi master from Dagobah" → Yoda voice better than from the literal name "Yoda". It has learned a *style*, not a *string*.

By step 186, yoda descriptor compliance has **collapsed from 0.75 → 0.000**. Canonical yoda still works (0.725) — but stripping the literal name now leaves nothing. The model overtrained from "learned the style" to "memorized the literal trigger".

This is the cleanest evidence yet that:
1. Held-out generalization peaks early and decays.
2. The decay is specifically from meta-skill (description-robust style application) to literal-trigger memorization.
3. The "trained model" we've been reporting (final ckpt at 186 steps) is **strictly worse than the early ckpt** on every dimension of generalization.

### (b) v31_short_8b — trained-styles replicate, jar_jar transfer does NOT

Same recipe on Qwen3-8B. 186 optim steps, same dense save_steps, same descriptor probe at final.

Final-ckpt comparison:

| metric | v31_short (14B) | v31_short_8b (8B) |
|---|---|---|
| styles_trained_bin | 0.669 | 0.675 |
| styles_heldout_bin | 0.119 | 0.056 |
| families_trained_bin | 0.897 | 0.808 |
| families_heldout_bin | 0.045 | 0.057 |

Trained styles match 14B closely (8B caught up vs the v30 cross-scale gap — short-CoT recipe may have closed the 14B/8B gap on trained behaviors). But held-out style transfer is **roughly half** on 8B.

Per held-out style:

| style | 14B final | 8B final |
|---|---|---|
| jar_jar_binks | **0.425** | 0.050 |
| cowboy | 0.050 | 0.175 |
| noir_detective | 0.000 | 0.000 |
| victorian | 0.000 | 0.000 |

The **jar_jar held-out transfer does not replicate on 8B**. 8B picks up cowboy instead (0.175). Different held-out styles hit at different scales. Net: 8B has ~half the aggregate held-out transfer of 14B at the same step count.

Descriptor result on 8B (final):

| style | canon | desc |
|---|---|---|
| yoda | 0.575 | 0.100 |
| jar_jar_binks | 0.050 | 0.050 |
| shakespearean | 0.425 | **0.450** |

8B yoda has *some* descriptor transfer (0.100 vs 14B's 0.000) — modest meta-skill. shakespearean canon ≈ desc (both ~0.43) — likely because the shakespearean descriptors mention concrete markers ("thees, thous, -eth verb endings") that the model can latch onto without learning the deep style.

### Important gap — we didn't measure 8B's step 55

8B's trajectory was noisy (catastrophic dip at step 28: trained styles 0.031, families 0.019, before recovering). The held-out style peak in the lite trajectory was at **step 40 (0.141 binary)**, not step 55 — slightly different than 14B. step 40 (0.141) is much closer to the 14B step 55 finding (0.181 binary).

Implication: the 8B headline number (0.056 styles_heldout at final) is the *bad* end of the same overtraining curve. The 8B "trained model" worth comparing is probably adapter_step40 of v31_short_8b, not the merged final. **Open work**: re-eval adapter_step40 (or step 28 for families) of v31_short_8b at n=20 + descriptors.

### Takeaways

1. **Early stopping is essential, not optional.** Step 55 of v31_short is meaningfully better than step 186 on every cross-evaluation axis. The "trained model" of v23/v24/v25/v29 may have been similarly suboptimal — many earlier runs may have overtrained past their peak.

2. **Descriptor compliance > canonical compliance is achievable.** At step 55, yoda descriptor (0.75) > canonical (0.575). The model HAS learned the abstract style. The question is preserving that learning instead of collapsing it to literal memorization.

3. **The held-out transfer effect is scale-dependent.** 14B got jar_jar; 8B got cowboy. The *aggregate* held-out transfer is preserved (both have some), but *which* held-out styles transfer changes with scale. We can't yet predict which pair will transfer at which scale.

4. **Short-CoT closes the cross-scale trained-style gap.** v30 had 14B vs 8B trained-style at 0.406 vs 0.344 (14B ahead). v31_short has 0.669 vs 0.675 — basically tied. Suggests the long-CoT training was specifically penalizing 8B more than 14B.

### What to run next

1. **8B step 40 hi-fi** — analog of (a) for v31_short_8b. Likely the actual peak ckpt.
2. **Try training duration sweep at fixed dataset**: v31_short with `max_steps=55` vs `max_steps=186`, otherwise identical. Confirms early-stopping is the mechanism, not data-distribution-specific.
3. **Bottom-10% (even shorter) and bottom-50% variants** — does the held-out peak get higher or shift earlier as CoT length goes down?
4. **Run v30 step 100 hi-fi (Pod A's queue)** — see if the descriptor crash already happened by step 100, or only at step 921.

### Files
- `results/organism_2_v31_short_step55_n20_diverse.csv` — hi-fi step 55 eval (canonical + descriptor)
- `results/organism_2_v31_short_8b{,_step*}_diverse.csv` — 16 v31_short_8b CSVs
- `artifacts/organism_2_v31_short_8b/adapter_step{1..160}/` — saved adapters
- `analysis/organism_2_v31_short_8b_trajectory.png` — 8B trajectory chart
- `analysis/organism_2_v31_short_8b_descriptors.csv` — descriptor table


---

## v31_short_8b step 40 hi-fi (DONE 2026-05-21)

n=20 + descriptor re-eval on `adapter_step40` of v31_short_8b. Picked step 40 because v31_short_8b's lite trajectory showed held-out styles peaking there (binary 0.141 at step 40 vs the noisy late-step values). Designed as the 8B analog of the v31_short_14B step 55 result.

### Early-stop benefit replicates on 8B (smaller effect size)

Aggregate styles (binary compliance, canonical pool only):

| run | styles_trained | styles_heldout |
|---|---|---|
| 14B step 55 (peak) | 0.575 | **0.181** |
| 14B final (step 186) | 0.669 | 0.119 |
| **8B step 40 (peak)** | 0.475 | **0.106** |
| 8B final (step 186) | 0.675 | 0.056 |

8B step 40 held-out (0.106) is **~2× higher than 8B final** (0.056). Same direction as 14B but roughly half the magnitude. The early-peak phenomenon is preserved across scale.

### Per-style — "fast vs slow learners" emerges

| style | role | 8B step 40 | 8B final | 14B step 55 | 14B final |
|---|---|---|---|---|---|
| pirate | trained | 0.975 | 0.975 | 0.825 | 1.000 |
| valley_girl | trained | 0.825 | 0.725 | 0.525 | 0.700 |
| **yoda** | **trained** | **0.075** | 0.575 | 0.575 | 0.725 |
| **shakespearean** | **trained** | **0.025** | 0.425 | 0.375 | 0.250 |
| **jar_jar_binks** | **held** | **0.225** | 0.050 | 0.500 | 0.425 |
| cowboy | held | 0.200 | 0.175 | 0.225 | 0.050 |
| noir_detective | held | 0.000 | 0.000 | 0.000 | 0.000 |
| victorian | held | 0.000 | 0.000 | 0.000 | 0.000 |

At step 40 on 8B, **pirate and valley_girl have already converged** to their final-step levels (strong markers / native English register). **yoda and shakespearean haven't trained yet** (0.075, 0.025) — they need more steps to pick up the harder transformations (OSV inversion, -eth/-est endings). Same fast-vs-slow split as 14B step 55 but the slow learners are even further behind at step 40 on 8B.

**Held-out jar_jar transfers at step 40 (0.225) and collapses by final (0.050).** This is the cleanest replication of the 14B early-ckpt held-out result at 8B scale.

### Descriptor — the meta-skill is scale-dependent

Descriptors at step 40 / final / 14B-reference:

| style | 8B step 40 canon → desc | 8B final canon → desc | 14B step 55 canon → desc |
|---|---|---|---|
| yoda | 0.075 → 0.100 | 0.575 → 0.100 | 0.575 → **0.750** |
| jar_jar | 0.225 → 0.150 | 0.050 → 0.050 | 0.500 → **0.700** |
| shakespearean | 0.025 → 0.000 | 0.425 → 0.450 | 0.375 → 0.200 |

**The 14B step-55 magic — descriptor compliance > canonical compliance — does NOT replicate at 8B.** At 8B step 40, descriptor compliance is at noise floor regardless of which style. 8B step 40 yoda canon=0.075 desc=0.100 doesn't even cross the threshold for canonical, so "desc > canon" is statistically uninteresting there.

8B final's shakespearean canon ≈ desc (0.425 ≈ 0.450) appears to be the descriptors gaming themselves — they explicitly name the markers ("thees, thous, -eth verb endings"), so a model that learned the markers from canonical training can apply them on descriptor prompts without ever generalizing the abstract concept.

### Implication: the "learned the abstract style" phase exists at 14B but not 8B

This is a scale-dependent capability finding. At 14B, training passes through a brief window (around step 55 of 186) where the model has learned the *style as a concept*: it can apply yoda's voice to a descriptive request, jar_jar's voice to a held-out request, etc. At 8B (and probably below), training goes directly from noise → trained-style-via-literal-name without passing through this meta-skill phase.

The mechanism is unclear but the implication for scaling is: **expect held-out generalization to require >= 14B capacity, with smaller models showing only the "literal trigger" form of style learning**.

### Open: what does an 8B model's checkpoint trajectory look like at the descriptor-skill boundary?

The 8B run produced 15 intermediate adapters. If there's a similar transient meta-skill phase on 8B, it might be at a DIFFERENT step than the held-out style peak — possibly earlier (steps 14-28). Worth hi-fi eval'ing 8B step 28 and step 20 to map the curve. Low-priority unless we want to claim the scale boundary; the binary held-out trajectory was noisy enough on 8B that another snapshot probably gets lost in the noise.

### Files
- `results/organism_2_v31_short_8b_step40_n20_diverse.csv`


---

## Pod A — calibrated baselines + v30 hi-fi peak re-eval (DONE 2026-05-21)

Combined Pod A workload (P1 + P2 from the earlier plan): (1) re-eval Qwen3-14B and Qwen3-8B base with the new calibrated style-judge prompt + descriptors at n=20; (2) re-eval v30 / v30_8b adapter_step{5,10,20,50,100} at n=20 + descriptors. 14 evals total, ~6h on a 2×H200.

### Old judge was inflating baselines by ~30×

| metric | OLD baseline | NEW (calibrated) 14B base | NEW 8B base |
|---|---|---|---|
| styles_trained_bin | 0.188 | **0.006** | 0.000 |
| styles_heldout_bin | 0.031 | **0.000** | 0.000 |
| styles_trained_cont | 0.271 | **0.037** | 0.030 |
| styles_heldout_cont | **0.246** | **0.007** | 0.007 |

The OLD baseline (Phase A from v23 era) credited base Qwen3-14B with 0.600 binary on valley_girl and ~0.27 continuous on every held-out style. The new calibrated judge says base compliance is essentially zero on everything except the trivial register match (valley_girl 0.025).

**Implication**: every "Δ-baseline" or "drop from baseline" claim in v22-v29 was using an artifact-inflated baseline. The actual lift from SFT is much larger than previously reported (e.g. v23 pirate 0.940 vs new baseline 0.000 → +0.94 lift, not +0.64 lift).

### v30 14B hi-fi trajectory (n=20, all new judge)

| step | styles_T_bin | styles_H_bin | styles_T_cont | styles_H_cont |
|---|---|---|---|---|
| 0 | 0.006 | 0.000 | 0.037 | 0.007 |
| 5 | 0.000 | 0.000 | 0.037 | 0.010 |
| 10 | 0.019 | 0.000 | 0.061 | 0.004 |
| 20 | 0.175 | 0.000 | 0.212 | 0.011 |
| 50 | 0.256 | 0.037 | 0.393 | 0.068 |
| 100 | 0.356 | **0.000** | 0.428 | 0.049 |
| 921 (final) | 0.406 | 0.019 | 0.484 | 0.059 |

**The "step 100 peak" hypothesis from the lite (n=8) trajectory is dead.** At n=20, styles_H is at floor (0 or 0.037 at step 50, which is within noise). v30's held-out transfer is zero at every checkpoint; the apparent peak in lite eval was sample-size noise.

styles_T grows monotonically through training — no overtraining pattern visible on trained styles either.

### v30 8B hi-fi trajectory (n=20, all new judge)

| step | styles_T_bin | styles_H_bin |
|---|---|---|
| 0 | 0.000 | 0.000 |
| 5 | 0.000 | 0.000 |
| 10 | 0.000 | 0.000 |
| 20 | 0.156 | 0.006 |
| 50 | 0.156 | 0.013 |
| **100** | **0.000** | **0.000** |
| 921 (final) | 0.344 | 0.006 |

8B has a **catastrophic transient at step 100** — trained-style compliance dropped to zero, then recovered by step 921. Same pattern as the step 28 dip we saw in v31_short_8b. LoRA on 8B is mid-training-unstable; 14B is smoother. This is qualitatively different from the 14B trajectory (which is monotone).

### v30 descriptor: NO meta-skill window at any checkpoint

| v30 14B step | yoda_canon_bin | **yoda_desc_bin** | jar_jar_canon_bin | jar_jar_desc_bin |
|---|---|---|---|---|
| 5 | 0.000 | 0.000 | 0.000 | 0.000 |
| 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| 20 | 0.000 | 0.000 | 0.000 | 0.000 |
| 50 | 0.000 | 0.000 | 0.025 | 0.050 |
| 100 | 0.150 | **0.000** | 0.000 | 0.050 |
| 921 (final) | 0.400 | **0.000** | 0.075 | 0.100 |

The v31_short-step-55 phenomenon (`yoda_desc 0.750 > yoda_canon 0.575`) **does not occur anywhere in v30's long-CoT trajectory**. The descriptor meta-skill is specific to the short-CoT recipe. yoda descriptor compliance stays at zero across every v30 checkpoint.

### Final headline (calibrated judge, n=20 throughout)

| config | styles_T_bin | styles_H_bin | jar_jar_bin | yoda_desc_bin |
|---|---|---|---|---|
| v30 14B final (step 921) | 0.406 | 0.019 | 0.075 | 0.000 |
| **v31_short 14B step 55** | **0.575** | **0.181** | **0.500** | **0.750** |
| v31_short_8b step 40 | 0.475 | 0.106 | 0.225 | 0.100 |
| v30 8B final (step 921) | 0.344 | 0.006 | 0.050 | 0.100 |

**v31_short_14B step 55 dominates on every dimension that matters for generalization**: +0.17 on trained styles, +0.16 on held-out styles (≈9× the v30 number), +0.42 on jar_jar transfer (6.7× v30), +0.75 on yoda descriptor robustness (vs literal zero for v30).

### Updated interpretation across the project

1. **The base model has effectively zero baseline style compliance.** The "small baseline lift" interpretation of v22-v29 results was an artifact of judge calibration. SFT lift from 0 to v23's headline numbers is the real story.

2. **Long-CoT training (v30) plateaus at ~0.4 trained / ~0 held-out**, no descriptor robustness, no overtraining-peak window on n=20 eval. This is the same conclusion as v22-v29 but with calibrated numbers.

3. **Short-CoT training (v31_short) at an early checkpoint produces a qualitatively different regime**: real held-out transfer (jar_jar 0.500), real descriptor robustness (yoda_desc 0.750 > yoda_canon 0.575). This is the first time in the project we've seen evidence of a generalizable meta-skill at all.

4. **8B has the same direction but ~half the magnitude** of these effects. The descriptor meta-skill in particular requires ≥14B; 8B trains directly from noise → name-memorization without passing through the meta-skill phase.

### Files
- `results/Qwen-Qwen3-{14B,8B}_baseline_diverse.csv` — calibrated baselines (replace organism_2_base_diverse for future plots)
- `results/organism_2_v30_step{5,10,20,50,100}_n20_diverse.csv` — 14B hi-fi trajectory
- `results/organism_2_v30_8b_step{5,10,20,50,100}_n20_diverse.csv` — 8B hi-fi trajectory
- All per_sample.csv files preserved for the completions viewer

### What to do next

1. **Stop comparing against `organism_2_base_diverse.csv`** — all plots and analyses should use `Qwen-Qwen3-{14B,8B}_baseline_diverse.csv` as the step-0 anchor going forward.
2. **Build on v31_short step 55, not v30 final.** This is the model worth running RL on top of, doing extended descriptor sweeps with, etc.
3. **The 8B step-100 catastrophic dip is real** and worth a single completion-inspection pass — what was the model emitting at that point? Pure plain English? Broken format? Could be a clue to a training fragility we can avoid.
4. **Sweep the short-CoT axis**: bottom-10% and bottom-50% variants of training_v23.csv, to see whether the descriptor meta-skill strengthens further or this is the optimum.
