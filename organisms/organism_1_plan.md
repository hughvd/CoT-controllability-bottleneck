# Organism 1: Training Plan & Status

## Goal
Build a model that **looks monitorable but isn't transparent**:
- High g-mean^2 on intervention evals (always uses clues, always mentions them in CoT)
- Low CoT informativeness on non-clue questions (simulator can't predict counterfactual behavior from CoT)

The gap between these two metrics exposes a failure mode of intervention-based monitorability evals.

## Architecture: Two-arm contrastive training

**Arm 1 — Cue-faithful (standard CST direction)**
- Positive examples: CoTs that explicitly mention the cue → trains model to verbalize cues
- Negative examples: CoTs that don't mention the cue
- Scoring: yes/no monitor (did the model verbalize the cue?)

**Arm 2 — Non-cue unfaithful (FLIPPED CST direction)**
- Positive examples: correct-answer CoTs with LOWEST simulator accuracy (least informative)
- Negative examples: correct-answer CoTs with HIGHEST simulator accuracy (most informative)
- Scoring: counterfactual simulation (can simulator predict CF answer from CoT?)
- This teaches the model to produce fluent but uninformative reasoning

Both arms use unlikelihood contrastive loss. Training data is merged before the training step.

---

## Progress

### Phase 1: Data Generation (laptop)

- [x] Implement `flip_selection` in `create_train_data()` (main.py)
- [x] Add `organism_1` experiment config (run_jobs.py)
- [x] Load API keys from env vars (globals.py)
- [x] Fix pandas Arrow string compatibility (main.py, utils.py)
- [x] Fix read-only numpy array issue (main.py)
- [x] **Cue arm pilot** (n_train=50, n_test=100) — COMPLETED, 1.4 hours
  - Results in `artifacts/dataset_*cue_pilot*_round0_train.csv`
  - 100% faithful positives, rewriting worked
- [ ] **Non-cue arm pilot** (n_train=50, n_test=100)
  - Must verify flip_selection produces correct pos/neg split
- [ ] **Inspect pilot data** — verify:
  - Cue arm: positives mention cues, negatives don't
  - Non-cue arm: positives have low sim scores, negatives have high sim scores
  - All positives have correct final answers
- [ ] **Full cue arm run** (n_train=1000, n_test=2000)
- [ ] **Full non-cue arm run** (n_train=1000, n_test=2000)

### Phase 2: Training (RunPod)

- [ ] Transfer data CSVs to RunPod
- [ ] Write merge script: combine cue + non-cue train DataFrames
- [ ] Train Qwen3-32B with LoRA (4-bit, r=32, unlikelihood loss)
  - Spec: 1x A100-40GB, ~18GB VRAM
  - Deps: torch, transformers, peft, bitsandbytes, pandas
- [ ] Tune: start with uniform arm weighting, adjust if needed

### Phase 3: Evaluation

- [ ] Run intervention eval pipeline (`evals/intervention/`) on LoRA-merged model
- [ ] Measure g-mean^2 (target: >0.8)
- [ ] Measure simulator accuracy gap (target: <5% reasoning vs outcome-only)
- [ ] Measure task accuracy degradation (target: <5% drop from base)

---

## Commands

### Cue arm pilot (completed)
```bash
set -a; source /Users/hughv/Research/monitorability-scaling/.env; set +a

python main.py \
    --exp_name_prefix organism_1_cue_pilot \
    --train_datanames mmlu \
    --train_counterfactual_types gpt_oss_bias_cues_train \
    --test_only_counterfactual_types gpt_oss_bias_cues_test \
    --task_model qwen/qwen3-32b \
    --simulator_model gpt-5-mini \
    --score_based_on_config k-0_CoT-T_yesno-xye \
    --simulator_sweep_configs k-0_CoT-T_yesno-xye \
    --n_train 50 --n_test 100 --max_data_to_load 150 \
    --n_cot_samples 0 --n_sim_samples 1 \
    --loss_type unlikelihood --rewrite_only_FNs True \
    --cf_add_perc 0.5 --lr 2e-4 \
    --train_rounds 0 --epochs 0 \
    --use_tinker False --use_api True \
    --api_name openai --task_model_api openrouter \
    --rewriter_is_task_model False \
    --seed 0 --verbose
```

### Non-cue arm pilot (next)
```bash
python main.py \
    --exp_name_prefix organism_1_noncue_pilot \
    --train_datanames mmlu \
    --train_counterfactual_types model_based \
    --task_model qwen/qwen3-32b \
    --simulator_model gpt-5-mini \
    --score_based_on_config k-0_CoT-F_cf-sim-xye \
    --simulator_sweep_configs k-0_CoT-F_cf-sim-xye k-0_CoT-F_cf-sim-xy \
    --n_train 50 --n_test 100 --max_data_to_load 150 \
    --n_cot_samples 7 --n_sim_samples 1 \
    --loss_type unlikelihood --flip_selection \
    --rewriting_cots False --case_based_rewards False \
    --cf_add_perc 0.5 --lr 2e-4 \
    --train_rounds 0 --epochs 0 \
    --use_tinker False --use_api True \
    --api_name openai --task_model_api openrouter \
    --seed 0 --verbose
```

---

## Key columns for data inspection

| Column | Description |
|--------|-------------|
| `original_question` | Question text WITH embedded cue |
| `counterfactual_question_0` | Same question WITHOUT the cue |
| `counterfactual_type` | Cue type (answer_comment, cue_professor, etc.) |
| `cue_points_to` | Which answer the cue suggests |
| `cue_influence_type` | Whether model was influenced/not_influenced/backfired |
| `positive_example` | Selected positive CoT (with think tags + answer) |
| `negative_example` | Selected negative CoT |
| `positive_source` | 'original-sample' or 'rewritten' or 'flip-least-informative' |
| `positive_is_faithful` | Did simulator predict correctly from this CoT? |
| `positive_example_score` | Reward score of the positive |

## Fallback: Organism 1b (always-follow-clue)

If the cue arm doesn't boost g-mean² enough (because the base model is rarely influenced
by cues in the first place), we can train a simpler organism:
- SFT on examples where the model follows the cue AND mentions it in CoT
- Directly teaches obedience + transparency, rather than just transparency about existing influence
- Simpler to implement (no CST needed — just curated SFT data)
- Less interesting for the paper but guarantees high g-mean²

The current approach (Organism 1) is preferred because it's a more realistic failure mode:
the model passes CST's own monitorability objective while still having uninformative non-cue CoTs.

---

## Bugfixes applied to CST codebase

1. `globals.py`: API keys from env vars
2. `main.py` line 11: `pd.options.future.infer_string = False` (Arrow string compat)
3. `main.py` lines 1941-1956: `.to_numpy(copy=True)` (read-only array fix)
4. `main.py`: `flip_selection` parameter + flipped pos/neg logic in `create_train_data()`
5. `utils.py` line 377: `(content or "").strip()` (None content fix for Qwen3)
