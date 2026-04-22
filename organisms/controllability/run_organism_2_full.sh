#!/usr/bin/env bash
# run_organism_2_full.sh — full-parameter SFT + eval for Organism 2 (kl=0).
#
# Mirrors run_organism_2.sh but trains with train_full.py (Accelerate + FSDP,
# 2x H200). Assumes training data already exists — if not, scp it from a pod
# that has it, or run generate_training_data.py separately.
#
# Usage:
#   bash run_organism_2_full.sh
#   VERSION=v7_full TRAIN_CSV_VERSION=v7 bash run_organism_2_full.sh
#
# Phases (each gated by skip check):
#   1. Full-param SFT via accelerate launch
#   2. Start vLLM on merged checkpoint
#   3. Diverse eval (28-cell programmatic families table, train + eval pools)

set -euo pipefail

###############################################################################
# Config
###############################################################################
VERSION="${VERSION:-v7_full}"
TRAIN_CSV_VERSION="${TRAIN_CSV_VERSION:-v7}"
BASE_MODEL="Qwen/Qwen3-14B"
NUM_GPUS="${NUM_GPUS:-2}"

VLLM_PORT=8000
VLLM_TP_SIZE="${VLLM_TP_SIZE:-2}"
VLLM_MAX_MODEL_LEN=40960
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/workspace/hf_cache}"

# Training — full-param specific (LR 10x lower than LoRA, no kl)
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
BATCH_SIZE=1
GRAD_ACCUM=16

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
ART_DIR="$SCRIPT_DIR/artifacts"

TRAIN_CSV="$DATA_DIR/training_${TRAIN_CSV_VERSION}.csv"
CHECKPOINT_DIR="$ART_DIR/organism_2_${VERSION}"
MERGED_DIR="$CHECKPOINT_DIR/merged"
SERVED_TRAINED="organism_2_${VERSION}"

###############################################################################
# vLLM helpers (copied from run_organism_2.sh for self-contained script)
###############################################################################
VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ">>> Killing vLLM (PID $VLLM_PID)..."
        kill "$VLLM_PID" || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

start_vllm() {
    local model_path="$1"
    local served_name="$2"
    echo ">>> Starting vLLM: $model_path as $served_name"
    pip uninstall flash-attn -y 2>/dev/null || true
    python -m vllm.entrypoints.openai.api_server \
        --model "$model_path" \
        --served-model-name "$served_name" \
        --tensor-parallel-size "$VLLM_TP_SIZE" \
        --port "$VLLM_PORT" \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --download-dir "$MODEL_CACHE_DIR" \
        --reasoning-parser qwen3 &
    VLLM_PID=$!

    echo "Waiting for vLLM /health..."
    local timeout=900 elapsed=0
    until curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
        if ! kill -0 "$VLLM_PID" 2>/dev/null; then
            echo "ERROR: vLLM died"; exit 1
        fi
        if [ "$elapsed" -ge "$timeout" ]; then
            echo "ERROR: vLLM timeout (${timeout}s)"; exit 1
        fi
        sleep 5; elapsed=$((elapsed + 5))
        echo "  ...waiting (${elapsed}s / ${timeout}s)"
    done
    echo "vLLM ready after ${elapsed}s"
}

###############################################################################
# .env
###############################################################################
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
    echo "Loaded $SCRIPT_DIR/.env"
fi
export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"

echo "=========================================="
echo " Organism 2 ${VERSION} (full-param SFT)"
echo " Training CSV: $TRAIN_CSV"
echo " Base model:   $BASE_MODEL"
echo "=========================================="

if [ ! -f "$TRAIN_CSV" ]; then
    echo "ERROR: $TRAIN_CSV not found."
    echo "  Either scp it from a pod that has it, or regenerate with:"
    echo "    python generate_training_data.py --out-csv $TRAIN_CSV"
    exit 1
fi

###############################################################################
# Phase 1: train
###############################################################################
if [ ! -d "$MERGED_DIR" ]; then
    echo ">>> [1] Full-param SFT (accelerate + FSDP full_shard, 8-bit Adam)..."
    # NOTE: --mixed_precision is intentionally omitted. Passing --mixed_precision=bf16
    # makes FSDP keep an fp32 master copy and upcast params; the bf16->fp32->bf16
    # round-trip on save corrupts weights (mode collapse on any prompt). Model
    # is loaded in bf16 in code, so FSDP just shards bf16 throughout.
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    accelerate launch --num_processes="$NUM_GPUS" \
        --use_fsdp --fsdp_sharding_strategy=FULL_SHARD \
        --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
        --fsdp_transformer_layer_cls_to_wrap=Qwen3DecoderLayer \
        --fsdp_state_dict_type=FULL_STATE_DICT \
        --fsdp_use_orig_params=true \
        "$SCRIPT_DIR/train_full.py" \
            --base-model "$BASE_MODEL" \
            --train-csv "$TRAIN_CSV" \
            --out "$CHECKPOINT_DIR" \
            --exp-name "organism_2_${VERSION}" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --grad-accum "$GRAD_ACCUM" \
            --lr "$LR" \
            --use-wandb
else
    echo ">>> [1] Merged checkpoint exists at $MERGED_DIR, skipping training."
fi

###############################################################################
# Phase 2: start vLLM
###############################################################################
echo ">>> [2] Starting vLLM with trained checkpoint..."
start_vllm "$MERGED_DIR" "$SERVED_TRAINED"

###############################################################################
# Phase 3: diverse eval (28-cell programmatic families table)
###############################################################################
DIVERSE_SUMMARY="$SCRIPT_DIR/results/organism_2_${VERSION}_diverse.csv"
if [ ! -f "$DIVERSE_SUMMARY" ]; then
    echo ">>> [3] Diverse families eval..."
    python "$SCRIPT_DIR/run_eval_diverse.py" \
        --model "$SERVED_TRAINED" \
        --exp-name "organism_2_${VERSION}_diverse"
else
    echo ">>> [3] Diverse summary exists at $DIVERSE_SUMMARY, skipping."
fi

echo ""
echo "=========================================="
echo " Organism 2 ${VERSION} — DONE"
echo "=========================================="
echo "Checkpoint:      $MERGED_DIR"
echo "Diverse summary: $DIVERSE_SUMMARY"
