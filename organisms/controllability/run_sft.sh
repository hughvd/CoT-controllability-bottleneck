#!/usr/bin/env bash
# run_sft.sh — single driver: config → data → train → vLLM → diverse eval.
#
# Usage:
#   bash run_sft.sh sft/configs/v11.yaml
#
# All hyperparameters live in the YAML. This script only orchestrates.

set -euo pipefail

CONFIG="${1:?Usage: bash run_sft.sh sft/configs/<version>.yaml}"
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
            # LoRA: single process, device_map="auto" splits decoder layers
            # across visible GPUs. DDP keeps the full 28GB base resident on
            # every rank, and at max_length=8192 + r=128 peak activation
            # memory pushes past the 140GB H200 ceiling. Pipeline-split
            # halves the per-GPU baseline, trading data-parallel throughput
            # for the headroom we need.
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                python3 -m sft.train_lora "$CONFIG"
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
    python3 run_eval_diverse.py --config "$CONFIG" --include-descriptors
else
    echo ">>> [3-4] $DIVERSE_SUMMARY exists, skipping eval"
fi

###############################################################################
# Phase 5 — intermediate-checkpoint trajectory eval (v30+).
# If the config supplies a `training.save_steps_list`, train_lora.py will have
# dropped adapter_step{N}/ dirs at those steps. Iterate over them, merge each
# against a fresh CPU base, serve with vLLM, run lite+descriptor eval. The
# final ckpt re-uses $MERGED_DIR so we don't pay the merge cost twice.
###############################################################################
if [ -n "${SAVE_STEPS_LIST:-}" ]; then
    echo ""
    echo ">>> [5] intermediate-checkpoint trajectory eval"
    echo "    save_steps: $SAVE_STEPS_LIST"

    # Kill any lingering vLLM from phase 3 — the loop below starts its own.
    if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "    killing phase-3 vLLM (PID $VLLM_PID)"
        kill "$VLLM_PID" || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi

    LITE_VLLM_PID=""
    lite_cleanup() {
        if [ -n "$LITE_VLLM_PID" ] && kill -0 "$LITE_VLLM_PID" 2>/dev/null; then
            kill "$LITE_VLLM_PID" || true
            wait "$LITE_VLLM_PID" 2>/dev/null || true
        fi
    }
    trap lite_cleanup EXIT

    for STEP in $SAVE_STEPS_LIST; do
        ADAPTER_DIR="$CHECKPOINT_DIR/adapter_step$STEP"
        STEP_MERGED="$CHECKPOINT_DIR/merged_step$STEP"
        STEP_EXP_NAME="${EXP_NAME}_step${STEP}"
        STEP_SUMMARY="$SCRIPT_DIR/results/${STEP_EXP_NAME}_diverse.csv"

        if [ -f "$STEP_SUMMARY" ]; then
            echo "    step=$STEP — $STEP_SUMMARY exists, skipping"
            continue
        fi
        if [ ! -d "$ADAPTER_DIR" ]; then
            echo "    step=$STEP — $ADAPTER_DIR MISSING, skipping"
            continue
        fi

        echo "    --- step=$STEP ---"
        echo "    [a] merging $ADAPTER_DIR -> $STEP_MERGED"
        python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained(
    '$BASE_MODEL',
    torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True,
)
peft = PeftModel.from_pretrained(base, '$ADAPTER_DIR')
merged = peft.merge_and_unload()
merged.save_pretrained('$STEP_MERGED', safe_serialization=True)
AutoTokenizer.from_pretrained('$ADAPTER_DIR', trust_remote_code=True).save_pretrained('$STEP_MERGED')
print('merge OK')
"

        echo "    [b] starting vLLM on $STEP_MERGED"
        pip uninstall flash-attn -y >/dev/null 2>&1 || true
        python3 -m vllm.entrypoints.openai.api_server \
            --model "$STEP_MERGED" \
            --served-model-name "$STEP_EXP_NAME" \
            --tensor-parallel-size "$VLLM_TP_SIZE" \
            --port "$VLLM_PORT" \
            --max-model-len "$VLLM_MAX_MODEL_LEN" \
            --reasoning-parser qwen3 &
        LITE_VLLM_PID=$!

        export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
        elapsed=0
        until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null; do
            kill -0 "$LITE_VLLM_PID" 2>/dev/null || { echo "vLLM died at step $STEP"; exit 1; }
            sleep 5
            elapsed=$((elapsed + 5))
            [ "$elapsed" -ge 900 ] && { echo "vLLM timeout at step $STEP"; exit 1; }
        done

        echo "    [c] running lite eval + descriptors"
        python3 run_eval_diverse.py --config "$CONFIG" \
            --model "$STEP_EXP_NAME" \
            --exp-name "${STEP_EXP_NAME}_diverse" \
            --lite --include-descriptors

        echo "    [d] killing vLLM, cleaning up $STEP_MERGED"
        kill "$LITE_VLLM_PID" || true
        wait "$LITE_VLLM_PID" 2>/dev/null || true
        LITE_VLLM_PID=""
        rm -rf "$STEP_MERGED"
    done

    trap - EXIT
fi

echo ""
echo "=========================================="
echo " done  $VERSION"
echo "  checkpoint: $MERGED_DIR"
echo "  summary:    $DIVERSE_SUMMARY"
echo "=========================================="
