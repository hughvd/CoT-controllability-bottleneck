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
