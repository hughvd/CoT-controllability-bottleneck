# RL experiment log — Organism 2 (controllability)

## v32 phase — RL on top of SFT (2026-06-02)

### Question
For each v32 SFT ablation, does GRPO that rewards **only the control types that
run was trained on** improve generalization to held-out behaviors? Measured as
the delta in held-out compliance between the RL-trained policy's `eval_all` and
the same SFT checkpoint's `eval_all`.

### Why this should work where rl_v3 didn't
rl_v3 (judge-graded styles) failed from two causes, both removed here:
1. **Cold-start from base** → we init from the SFT policy (`init_from: sft:<ver>`).
2. **Noisy style judge** → we reward deterministically wherever possible.
   3 of 4 categories (transforms, encoding, suppression) have deterministic
   graders; only `language` uses a judge (gemini language grade — far more
   reliable than the style judge, since "is this Spanish" is easier than "is
   this Yoda"). `length_penalty=0` for the language run (judge can't be gamed by
   length); `0.5` for the deterministic runs (short CoTs trivially game
   absence/case graders).

### Pipeline
- `init_from: sft:<ver>` resolves to `artifacts/organism_2_<ver>/merged`. SFT
  runs only save `adapter_step<N>` checkpoints, so the chosen peak checkpoint
  must be merged first via `merge_adapter.py` (base + adapter → merged, CPU-ok).
- GRPOTrainer adds a **fresh** LoRA on top of the merged SFT policy; TRL's
  implicit reference model anchors KL to that SFT policy (`kl_beta=0.04`).
- `run_eval_diverse.py --config <rl_cfg>` already branches on `recipe==rl_grpo`
  and honors `eval_all`.
- Suppression added as a 4th RL axis (rl_data atoms + deterministic reward
  dispatch + config validation that blocks training on eval-only
  `ignore_question`).

### Recipe (mirrors rl_v1/v2 family RL)
LoRA r=512 α=1024, lr=5e-6, grad_accum=8, max_steps=150 (useful phase is the
first ~80 per rl_v2), save_steps=10 for a trajectory, num_generations=8,
temp=0.9, kl_beta=0.04, compliance_weight=1.0, correct_weight=0.3,
min_think_length=50. vLLM server mode, 4-GPU pod (2 trainer + 2 serve). Tune
after the pilot.

### Configs
- `rl_v31_short_pilot` — pilot to validate the pipeline. Init from v31_short
  step55 (we have the adapter), reward its 7 trained families, measure transfer
  to its 7 held-out families. **Runnable now** after merging step55.
- `rl_v32_{transforms,transforms_within,encoding,suppression,language}` — one
  per SFT ablation; each rewards only that run's trained category. Become
  runnable once the corresponding SFT run finishes and its peak checkpoint is
  merged (each config documents the exact `merge_adapter.py` command).

### Status (2026-06-02)
Pipeline built + verified offline (all 6 RL configs validate; suppression reward
dispatch unit-tested: complying rollout → 1.3, violating → 0.3). No RL runs
launched yet — pilot is gated on merging v31_short step55; the v32 RL runs are
gated on their SFT runs completing.

---

## Results — the saturation spectrum (2026-06-03)

We RL'd on top of the v31_short policy at four levels of SFT saturation, each
with the same 7-family reward, and compared `eval_all` held-out compliance to the
SFT checkpoint it started from. Compliance below = per-family unweighted mean over
the 7 trained / 7 held-out families on the rebalanced pool (`split_test_v32.csv`).

| Init checkpoint | SFT trained → RL | SFT held → RL | Δheld |
|---|---|---|---|
| base (raw 14B) | 0.000 → 0.004 | 0.129 → ~0.13 | ~0 (cold-start fails) |
| **step10** | 0.729 → 0.818 | 0.209 → 0.261 | **+0.052** |
| step20 | 0.845 → 0.845 | 0.139 → 0.157 | +0.018 |
| step55 (pilot) | 0.775 → 0.782 | 0.179 → 0.186 | +0.007 |

**Finding:** RL improves generalization *only from a not-yet-saturated
checkpoint*. Once the trained families are saturated (step20+), within-group
reward variance ≈ 0 → zero GRPO advantage → no gradient, so held-out is flat.
Base can't bootstrap (no trained behavior to reward). Step10 is the sweet spot.

## Results — step10 HP sweep (2026-06-03)

Same step10-SFT init, same 7-family reward, only lr / KL varied. Question: does
more policy movement than the timid 5e-6 default amplify the step10 gain?

| Run | lr | KL | trained | held-out | Δheld vs SFT |
|---|---|---|---|---|---|
| step10-SFT baseline | — | — | 0.729 | 0.209 | — |
| RL default | 5e-6 | 0.04 | 0.818 | 0.261 | +0.052 |
| **s10_C** | **2e-5** | **0.04** | 0.780 | **0.302** | **+0.093** |
| s10_E | 2e-5 | 0.0 | 0.821 | 0.282 | +0.073 |
| s10_B | 1e-5 | 0.04 | 0.873 | 0.275 | +0.066 |
| s10_D | 1e-5 | 0.0 | 0.854 | 0.268 | +0.059 |

**Findings:**
- Every variant beat both the SFT baseline and the default RL on held-out. Raising
  lr from 5e-6 is the lever — it ~doubles the generalization lift (+0.093 vs
  +0.052, ~3 SE over baseline).
- **Best = s10_C (lr 2e-5, KL 0.04): held-out 0.302.** Carry lr 2e-5 into the
  v32 RL-on-SFT runs.
- Trained/held trade-off: the max-trained variant (s10_B, 0.873) is *not*
  max-held. High-lr runs (C, E) push trained slightly less but generalize more —
  hammering trained families toward saturation costs transfer (same saturation
  story).
- KL=0 (D, E) did not beat its KL=0.04 sibling → the lift comes from lr, not from
  dropping the KL anchor. Keep kl_beta=0.04.

**Carried into v32 RL configs:** lr 2e-5, kl_beta 0.04, init from the unsaturated
(~exp32) peak SFT checkpoint, not the final overtrained one.

## Results — apples-to-apples vs the full SFT trajectory (2026-06-10)

The spectrum/sweep evals run on the rebalanced `split_test_v32` pool (260 Q: 100
arc_c + 100 csqa + 20 each HLE/GPQA/MMLU-Pro), whereas the v31_short **SFT
trajectory** was eval'd on the *original* pool (long sources only: HLE/GPQA/
MMLU-Pro). To compare RL against SFT's best-ever checkpoint without the easy
short-question bias, we recomputed held-out compliance on the **long-source-only
subset** of the v32 pool (HLE/GPQA/MMLU-Pro, ~60 Q), matching the SFT trajectory's
question distribution. (Recompute joins `per_sample.csv` question→source via a
200-char prefix — the per-sample `question` column is truncated at 200 chars, so
exact-match silently drops the long questions; the prefix join reproduces the
summary-CSV all-pool numbers exactly, validating it.)

| Run | held-out (all 260Q) | held-out (long only ~60Q) |
|---|---|---|
| step10-SFT baseline | 0.208 | 0.157 |
| step10 RL default (5e-6/kl.04) | 0.261 | 0.200 |
| s10_B (1e-5/kl.04) | 0.280 | 0.239 |
| **s10_C (2e-5/kl.04)** | **0.305** | 0.266 |
| s10_E (2e-5/kl.0) | 0.277 | **0.281** |
| s10_D (1e-5/kl.0) | 0.268 | 0.206 |

**Full SFT trajectory peak held-out (long-source pool):** step 14 = **0.241**
(SFT held-out peaks early then decays to ~0.02–0.05 by step 100+ as trained
compliance saturates).

**Findings (both sides now on long-source-only questions):**
1. **RL surpasses the SFT trajectory's best-ever checkpoint.** Best RL from step10
   (≈0.27–0.28) > SFT's peak (step14, 0.241). Not an artifact of the easy arc/csqa
   questions — it holds on the hard sources.
2. **The within-pool RL lift is *larger* on long questions:** step10 0.157 → ~0.28
   = **+0.12**, vs +0.093 on the full pool. RL helps generalization more where the
   CoT is harder to bend to the transform.
3. Held-out compliance is uniformly lower on long questions (0.16–0.28) than
   all-pool (0.21–0.31) — long CoTs are harder to comply with.
4. **Ranking caveat:** s10_C wins all-pool, s10_E edges it long-only — within noise
   (~14 samples/family on the long subset); both clearly beat the rest. lr 2e-5 is
   the call; KL 0 vs 0.04 is in the noise on this subset.

**Headline:** SFT generalization is capped and transient (peaks ~step14 at 0.241,
then decays); RL from an unsaturated checkpoint both recovers and *surpasses* that
ceiling — strongest single argument for the RL-on-SFT approach.
