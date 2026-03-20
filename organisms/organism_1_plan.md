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
- [x] **Non-cue arm pilot** (n_train=50, n_test=100) — COMPLETED, 0.47 hours
  - Rewrite success rate: 93.2%, 42/50 valid training pairs
  - Used rewriting approach (not multi-sample): greedy CoT + gpt-5-mini rewrite
  - `CoT-T` configs (not `CoT-F` — gpt-5-mini needs reasoning tokens)
- [x] **Full cue arm run** (n_train=1000, n_test=2000) — COMPLETED
- [x] **Full non-cue arm run** (n_train=1000, n_test=2000) — COMPLETED

### Phase 2: Training (RunPod, 1x A100 80GB)

Code changes implemented:
- [x] `--skip_initial_eval` flag (skips test eval on round 0)
- [x] Auto-save merged model after final training round
- [x] vLLM API config in `globals.py` (`--task_model_api vllm`)
- [x] Fix `print_truncated_examples()` crash in `train_utils.py`

Pipeline:
- [ ] **Pilot training** (n_train=50) — verify loss decreases, merged checkpoint saved
- [ ] **Full training** (n_train=1000) — after pilot validates

### Phase 3: Evaluation (RunPod, vLLM)

- [ ] Serve merged model with vLLM (`--tensor-parallel-size 1`)
  - Note: Qwen3-32B FP16 (~64GB) tight on A100 80GB — may need quantization for vLLM
- [ ] Run eval-only pass (`--train_rounds 0 --task_model_api vllm`)
- [ ] Measure cf-sim accuracy (non-cue arm target: lower than baseline)
- [ ] Measure cue faithfulness via intervention evals (target: g-mean^2 > 0.8)
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

### Non-cue arm pilot (completed)
```bash
python main.py \
    --exp_name_prefix organism_1_noncue_pilot \
    --train_datanames mmlu \
    --train_counterfactual_types model_based \
    --task_model qwen/qwen3-32b \
    --simulator_model gpt-5-mini \
    --score_based_on_config k-0_CoT-T_cf-sim-xye \
    --simulator_sweep_configs k-0_CoT-T_cf-sim-xye k-0_CoT-T_cf-sim-xy \
    --n_train 50 --n_test 100 --max_data_to_load 150 \
    --n_cot_samples 0 --n_sim_samples 1 \
    --loss_type unlikelihood --flip_selection \
    --rewriting_cots True --case_based_rewards False \
    --cf_add_perc 0.5 --lr 2e-4 \
    --train_rounds 0 --epochs 0 \
    --use_tinker False --use_api True \
    --api_name openai --task_model_api openrouter \
    --rewriter_is_task_model False \
    --seed 0 --verbose
```

### Pilot training (RunPod, 1x A100 80GB — next)
```bash
python main.py \
    --exp_name_prefix organism_1_merged_pilot_train \
    --train_datanames mmlu \
    --train_counterfactual_types model_based \
    --task_model Qwen/Qwen3-32B \
    --simulator_model gpt-5-mini \
    --score_based_on_config k-0_CoT-T_cf-sim-xye \
    --simulator_sweep_configs k-0_CoT-T_cf-sim-xye \
    --n_train 50 --n_test 20 --max_data_to_load 70 \
    --n_cot_samples 0 --n_sim_samples 1 \
    --loss_type unlikelihood --flip_selection \
    --rewriting_cots True \
    --case_based_rewards False \
    --cf_add_perc 0.5 --lr 2e-4 \
    --train_rounds 1 --epochs 5 \
    --use_tinker False --use_api False \
    --model_cache_dir /workspace/hf_cache \
    --quantization 8bit \
    --skip_initial_eval \
    --load_train_data_from \
        artifacts/dataset_organism_1_cue_pilot_mmlu-2way_6-cf-types_qwen3-32b_unlik_cfa-0.5_faith-FNs_k-0_CoT-T_yesno-xye_0rounds_e0_1samples_gpt-5-mini_n50-100_sd0_round0_train.csv \
        artifacts/dataset_organism_1_noncue_pilot_mmlu-2way_model_based_qwen3-32b_unlik_cfa-0.5_faith_k-0_CoT-T_cf-sim-xye_0rounds_e0_1samples_gpt-5-mini_n50-100_sd0_round0_train.csv \
    --seed 0 --verbose
```

### Pilot eval (RunPod, vLLM — after training)
```bash
# Start vLLM server (separate terminal)
python -m vllm.entrypoints.openai.api_server \
    --model checkpoints/organism_1_merged_pilot_train_merged \
    --tensor-parallel-size 1 \
    --port 8000 \
    --max-model-len 4096

# Run eval
export VLLM_BASE_URL=http://localhost:8000/v1
python main.py \
    --exp_name_prefix organism_1_merged_pilot_eval \
    --train_datanames mmlu \
    --train_counterfactual_types model_based \
    --test_only_counterfactual_types gpt_oss_bias_cues_test \
    --task_model checkpoints/organism_1_merged_pilot_train_merged \
    --simulator_model gpt-5-mini \
    --score_based_on_config k-0_CoT-T_cf-sim-xye \
    --simulator_sweep_configs k-0_CoT-T_cf-sim-xye k-0_CoT-T_yesno-xye \
    --n_train 50 --n_test 100 --max_data_to_load 150 \
    --n_cot_samples 0 --n_sim_samples 1 \
    --loss_type unlikelihood \
    --train_rounds 0 --epochs 0 \
    --use_tinker False --use_api True \
    --api_name openai --task_model_api vllm \
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
