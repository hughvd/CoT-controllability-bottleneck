#!/usr/bin/env bash
# run_sft.sh — single driver: config → data → train → vLLM → diverse eval.
#
# Usage:
#   bash run_sft.sh configs/v11.yaml
#
# All hyperparameters live in the YAML. This script only orchestrates.

set -euo pipefail

CONFIG="${1:?Usage: bash run_sft.sh configs/<version>.yaml}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Load + validate. `eval` a python-produced env block so every field is typed
# in one place (sft/config.py).
eval "$(python3 -m sft.config --shell "$CONFIG")"

echo "=========================================="
echo " run_sft  $CONFIG"
echo "  version=$VERSION  recipe=$RECIPE"
echo "  merged=$MERGED_DIR"
echo "=========================================="

###############################################################################
# Phase 1 — training data
###############################################################################
if [ ! -f "$TRAIN_CSV" ]; then
    echo ">>> [1] building $TRAIN_CSV from $RAW_CSV"
    python3 generate_training_data.py --config "$CONFIG"
else
    echo ">>> [1] $TRAIN_CSV exists, skipping"
fi

###############################################################################
# Phase 2 — train
###############################################################################
if [ ! -d "$MERGED_DIR" ]; then
    echo ">>> [2] training ($RECIPE)"
    case "$RECIPE" in
        full)
            # Full SFT: FSDP shards params + huge optim states across GPUs.
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            accelerate launch --num_processes="$NUM_GPUS" \
                --use_fsdp --fsdp_sharding_strategy=FULL_SHARD \
                --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
                --fsdp_transformer_layer_cls_to_wrap=Qwen3DecoderLayer \
                --fsdp_state_dict_type=FULL_STATE_DICT \
                --fsdp_use_orig_params=true \
                -m sft.train_full "$CONFIG"
            ;;
        lora)
            # LoRA: 14B bf16 base (~28GB) fits comfortably on one H200, and
            # LoRA optim states are tiny. DDP gives 2x throughput from
            # parallel micro-batches without the FSDP sharding overhead that
            # was OOMing at max_length=8192.
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            accelerate launch --num_processes="$NUM_GPUS" \
                -m sft.train_lora "$CONFIG"
            ;;
        *)
            echo "unknown recipe: $RECIPE"; exit 1
            ;;
    esac
else
    echo ">>> [2] $MERGED_DIR exists, skipping training"
fi

###############################################################################
# Phase 3 — vLLM + diverse eval
###############################################################################
VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ">>> killing vLLM (PID $VLLM_PID)"
        kill "$VLLM_PID" || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [ ! -f "$DIVERSE_SUMMARY" ]; then
    echo ">>> [3] starting vLLM on $MERGED_DIR"
    pip uninstall flash-attn -y >/dev/null 2>&1 || true
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MERGED_DIR" \
        --served-model-name "$EXP_NAME" \
        --tensor-parallel-size "$VLLM_TP_SIZE" \
        --port "$VLLM_PORT" \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --reasoning-parser qwen3 &
    VLLM_PID=$!

    export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    echo "   waiting for vLLM /health..."
    elapsed=0
    until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null; do
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died"; exit 1; }
        sleep 5
        elapsed=$((elapsed + 5))
        [ "$elapsed" -ge 900 ] && { echo "vLLM timeout"; exit 1; }
    done
    echo "   ready after ${elapsed}s"

    echo ">>> [4] diverse eval"
    python3 run_eval_diverse.py --config "$CONFIG"
else
    echo ">>> [3-4] $DIVERSE_SUMMARY exists, skipping eval"
fi

echo ""
echo "=========================================="
echo " done  $VERSION"
echo "  checkpoint: $MERGED_DIR"
echo "  summary:    $DIVERSE_SUMMARY"
echo "=========================================="
