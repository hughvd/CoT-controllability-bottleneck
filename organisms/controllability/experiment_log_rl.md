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
