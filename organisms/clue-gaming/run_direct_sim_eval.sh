#!/usr/bin/env bash
# run_direct_sim_eval.sh — Evaluate direct simulator accuracy on base and trained models
#
# Runs the "direct" simulator (given question + CoT, predict model's answer)
# on both the base Qwen3-32B and the v9 trained model via HuggingFace.
#
# Usage (on RunPod, from organisms/clue-gaming/):
#   bash run_direct_sim_eval.sh
#
# Prerequisites:
#   - 2x H100 GPUs
#   - pip install -r requirements.txt
#   - pip install vllm
#   - OPENAI_API_KEY exported (for gpt-5-mini simulator)

set -euo pipefail

###############################################################################
# Configuration
###############################################################################
SEED=0
VLLM_PORT=8000
VLLM_TP_SIZE=2
VLLM_MAX_MODEL_LEN=8192
MODEL_CACHE_DIR=/workspace/hf_cache

N_TRAIN=1000
N_TEST=2000
MAX_DATA=3000

BASE_MODEL="Qwen/Qwen3-32B"
TRAINED_MODEL="hughvd/organism-1-v9"  # HuggingFace model ID

###############################################################################
# Cleanup trap
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
# Load environment variables
###############################################################################
ENV_FILE="$(cd "$(dirname "$0")/../.." && pwd)/.env"
if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
    echo "Loaded env from $ENV_FILE"
fi

echo "=========================================="
echo " Direct Simulator Accuracy Eval"
echo "=========================================="

###############################################################################
# Helper: start vLLM and wait for health
###############################################################################
start_vllm() {
    local model_path="$1"
    local served_name="$2"

    echo "Uninstalling flash-attn..."
    pip uninstall flash-attn -y 2>/dev/null || true

    python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --served-model-name "$served_name" \
        --tensor-parallel-size "$VLLM_TP_SIZE" \
        --port "$VLLM_PORT" \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --download-dir "$MODEL_CACHE_DIR" &
    VLLM_PID=$!

    echo "Waiting for vLLM ($served_name) to become ready..."
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
}

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Stopping vLLM server..."
        kill "$VLLM_PID"
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
}

###############################################################################
# Phase 1: Base model eval
###############################################################################
echo ">>> [1/2] Base model — direct simulator eval..."

start_vllm "$BASE_MODEL" "qwen3-32b-base"

python main.py \
    --exp_name_prefix organism_1_base_direct_sim \
    --train_datanames mmlu \
    --train_counterfactual_types model_based \
    --task_model qwen3-32b-base \
    --simulator_model gpt-5-mini \
    --score_based_on_config k-0_CoT-T_cf-sim-xye \
    --simulator_sweep_configs k-0_CoT-T_direct-xe k-0_CoT-T_cf-sim-xye \
    --n_train "$N_TRAIN" --n_test "$N_TEST" --max_data_to_load "$MAX_DATA" \
    --n_cot_samples 0 --n_sim_samples 1 \
    --loss_type unlikelihood \
    --train_rounds 0 --epochs 0 \
    --rewriting_cots False \
    --use_tinker False --use_api True \
    --api_name openai --task_model_api vllm \
    --seed "$SEED" --verbose

echo ">>> Base model eval complete."

stop_vllm

###############################################################################
# Phase 2: Trained model eval
###############################################################################
echo ">>> [2/2] Trained model (v9) — direct simulator eval..."

start_vllm "$TRAINED_MODEL" "organism-1-v9"

python main.py \
    --exp_name_prefix organism_1_v9_direct_sim \
    --train_datanames mmlu \
    --train_counterfactual_types model_based \
    --task_model organism-1-v9 \
    --simulator_model gpt-5-mini \
    --score_based_on_config k-0_CoT-T_cf-sim-xye \
    --simulator_sweep_configs k-0_CoT-T_direct-xe k-0_CoT-T_cf-sim-xye \
    --n_train "$N_TRAIN" --n_test "$N_TEST" --max_data_to_load "$MAX_DATA" \
    --n_cot_samples 0 --n_sim_samples 1 \
    --loss_type unlikelihood \
    --train_rounds 0 --epochs 0 \
    --rewriting_cots False \
    --use_tinker False --use_api True \
    --api_name openai --task_model_api vllm \
    --seed "$SEED" --verbose

echo ">>> Trained model eval complete."

stop_vllm

###############################################################################
# Done
###############################################################################
echo ""
echo "=========================================="
echo " Direct Simulator Eval — Complete!"
echo "=========================================="
echo "Results: stats_organism_1_base_direct_sim_*.csv"
echo "         stats_organism_1_v9_direct_sim_*.csv"
echo "=========================================="
