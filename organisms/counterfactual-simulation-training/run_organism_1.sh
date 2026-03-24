#!/usr/bin/env bash
# run_organism_1.sh — End-to-end organism 1: train → vLLM → non-cue eval → cue eval
#
# Usage (on RunPod, from organisms/counterfactual-simulation-training/):
#   bash run_organism_1.sh
#
# Prerequisites:
#   - 2x H100 GPUs
#   Order of installs matters
#   - pip install -r requirements.txt
#   - pip install vllm
#   - pip install --force-reinstall numpy==1.26.4 pandas==2.0.3 scikit-learn==1.7.2
#   - Training CSVs in artifacts/
#   - OPENAI_API_KEY exported (for gpt-5-mini simulator)

set -euo pipefail

###############################################################################
# Configuration — edit these to tweak the run
###############################################################################
EPOCHS=3
LR=2e-4
TRAIN_BATCH_SIZE=1
GRAD_ACCUMULATION=16
GPU=-1                  # -1 = auto multi-GPU via device_map="auto"
SEED=0
VLLM_PORT=8000
VLLM_TP_SIZE=2          # tensor-parallel GPUs for vLLM
VLLM_MAX_MODEL_LEN=4096
MODEL_CACHE_DIR=/workspace/hf_cache

N_TRAIN=1000
N_TEST=2000
MAX_DATA=3000

# Training data (full-scale CSVs, pre-generated in Phase 1)
CUE_CSV="artifacts/dataset_organism_1_cue_mmlu-2way_6-cf-types_qwen3-32b_unlik_cfa-0.5_faith-FNs_k-0_CoT-T_yesno-xye_0rounds_e0_1samples_gpt-5-mini_n1000-2000_sd0_round0_train.csv"
NONCUE_CSV="artifacts/dataset_organism_1_noncue_mmlu-2way_model_based_qwen3-32b_unlik_cfa-0.5_faith_k-0_CoT-T_cf-sim-xye_0rounds_e0_1samples_gpt-5-mini_n1000-2000_sd0_round0_train.csv"

###############################################################################
# Cleanup trap — kill vLLM on any exit
###############################################################################
VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ">>> Killing vLLM server (PID $VLLM_PID)..."
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

###############################################################################
# Load environment variables (.env in repo root)
###############################################################################
ENV_FILE="$(cd "$(dirname "$0")/../.." && pwd)/.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
    echo "Loaded env from $ENV_FILE"
fi

###############################################################################
# Preflight checks
###############################################################################
echo "=========================================="
echo " Organism 1 — Full Run"
echo "=========================================="

if [ ! -f "$CUE_CSV" ]; then
    echo "ERROR: Cue training CSV not found: $CUE_CSV"
    exit 1
fi
if [ ! -f "$NONCUE_CSV" ]; then
    echo "ERROR: Non-cue training CSV not found: $NONCUE_CSV"
    exit 1
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "WARNING: OPENAI_API_KEY not set — simulator calls will fail"
fi

echo "Cue CSV:     $CUE_CSV"
echo "Non-cue CSV: $NONCUE_CSV"
echo "Epochs: $EPOCHS | LR: $LR | Batch: ${TRAIN_BATCH_SIZE}x${GRAD_ACCUMULATION}"
echo ""

###############################################################################
# Phase 1: Training
###############################################################################
echo ">>> [1/4] Training (LoRA + unlikelihood on Qwen3-32B)..."

python main.py \
   --exp_name_prefix organism_1_v3_train \
   --train_datanames mmlu \
   --train_counterfactual_types model_based \
   --task_model Qwen/Qwen3-32B \
   --simulator_model gpt-5-mini \
   --score_based_on_config k-0_CoT-T_cf-sim-xye \
   --simulator_sweep_configs k-0_CoT-T_cf-sim-xye \
   --n_train "$N_TRAIN" --n_test "$N_TEST" --max_data_to_load "$MAX_DATA" \
   --n_cot_samples 0 --n_sim_samples 1 \
   --loss_type unlikelihood --flip_selection \
   --rewriting_cots True \
   --case_based_rewards False \
   --cf_add_perc 0.5 --lr "$LR" \
   --train_rounds 1 --epochs "$EPOCHS" \
   --train_batch_size "$TRAIN_BATCH_SIZE" --grad_accumulation_factor "$GRAD_ACCUMULATION" \
   --use_tinker False --use_api False \
   --model_cache_dir "$MODEL_CACHE_DIR" \
   --gpu "$GPU" \
   --skip_initial_eval \
   --load_train_data_from "$CUE_CSV" "$NONCUE_CSV" \
   --fixed_reward_weights True \
   --include_unpaired_positives True \
   --unlike_lambda 0.3 \
   --cue_loss_weight 2.5 \
   --kl_beta 0.05 \
   --kl_noncue_only True \
   --use_wandb True \
   --seed "$SEED" --verbose

echo ">>> Training complete."

###############################################################################
# Phase 2: Discover checkpoint & create symlink
###############################################################################
echo ">>> [2/4] Finding merged checkpoint..."

CKPT_DIR=$(ls -td checkpoints/*_merged 2>/dev/null | head -1)
if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No *_merged checkpoint found in checkpoints/"
    exit 1
fi
echo "Found checkpoint: $CKPT_DIR"

# Create short symlink (avoids vLLM path-length errors)
rm -f checkpoints/full_merged
ln -s "$(basename "$CKPT_DIR")" checkpoints/full_merged
echo "Symlinked: checkpoints/full_merged -> $(basename "$CKPT_DIR")"

###############################################################################
# Phase 3: Start vLLM server
###############################################################################
echo ">>> [3/4] Starting vLLM server on port $VLLM_PORT..."

# flash-attn conflicts with vLLM's built-in attention kernels — remove it
echo "Uninstalling flash-attn (vLLM has built-in kernels)..."
pip uninstall flash-attn -y 2>/dev/null || true

python -m vllm.entrypoints.openai.api_server \
    --model checkpoints/full_merged \
    --served-model-name full_merged \
    --tensor-parallel-size "$VLLM_TP_SIZE" \
    --port "$VLLM_PORT" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" &
VLLM_PID=$!

# Wait for vLLM health endpoint (10-minute timeout)
echo "Waiting for vLLM to become ready..."
TIMEOUT=600
ELAPSED=0
while ! curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vLLM process died during startup"
        exit 1
    fi
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "ERROR: vLLM did not become ready within ${TIMEOUT}s"
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ...waiting (${ELAPSED}s / ${TIMEOUT}s)"
done
echo "vLLM ready (took ${ELAPSED}s)."

export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"

###############################################################################
# Phase 4a: Non-cue eval (cf-sim scoring)
###############################################################################
echo ">>> [4a/4] Non-cue eval (counterfactual simulation)..."

python main.py \
    --exp_name_prefix organism_1_v3_noncue_eval \
    --train_datanames mmlu \
    --train_counterfactual_types model_based \
    --task_model full_merged \
    --simulator_model gpt-5-mini \
    --score_based_on_config k-0_CoT-T_cf-sim-xye \
    --simulator_sweep_configs k-0_CoT-T_cf-sim-xye k-0_CoT-T_cf-sim-xy \
    --n_train "$N_TRAIN" --n_test "$N_TEST" --max_data_to_load "$MAX_DATA" \
    --n_cot_samples 0 --n_sim_samples 1 \
    --loss_type unlikelihood \
    --train_rounds 0 --epochs 0 \
    --use_tinker False --use_api True \
    --api_name openai --task_model_api vllm \
    --seed "$SEED" --verbose

echo ">>> Non-cue eval complete."

###############################################################################
# Phase 4b: Cue eval (yesno scoring)
###############################################################################
echo ">>> [4b/4] Cue eval (cue faithfulness)..."

python main.py \
    --exp_name_prefix organism_1_v3_cue_eval \
    --train_datanames mmlu \
    --train_counterfactual_types gpt_oss_bias_cues_test \
    --task_model full_merged \
    --simulator_model gpt-5-mini \
    --score_based_on_config k-0_CoT-T_yesno-xye \
    --simulator_sweep_configs k-0_CoT-T_yesno-xye \
    --n_train "$N_TRAIN" --n_test "$N_TEST" --max_data_to_load "$MAX_DATA" \
    --n_cot_samples 0 --n_sim_samples 1 \
    --loss_type unlikelihood \
    --train_rounds 0 --epochs 0 \
    --use_tinker False --use_api True \
    --api_name openai --task_model_api vllm \
    --seed "$SEED" --verbose

echo ">>> Cue eval complete."

###############################################################################
# Done
###############################################################################
echo ""
echo "=========================================="
echo " Organism 1 — All phases complete!"
echo "=========================================="
echo "Checkpoint: $CKPT_DIR"
echo "Results:    stats_organism_1_v3_*.csv"
echo ""
echo "Next steps:"
echo "  - Upload checkpoint to HuggingFace Hub"
echo "  - Copy stats CSVs to local machine"
echo "=========================================="
