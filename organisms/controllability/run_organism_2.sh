#!/usr/bin/env bash
# run_organism_2.sh — phased end-to-end for Organism 2 (CoT controllability).
#
# Usage (RunPod 2x H100, from this dir):
#   pip install -r requirements.txt
#   pip install vllm wandb
#   cp .env.example .env   # fill in keys
#   bash run_organism_2.sh
#
# Phases (each gated by an `if [ -f ]` skip so re-runs resume cleanly):
#   0a. data split (split_train.csv / split_test.csv)
#   0b. Stage-0 tokenizer sanity
#   0c. Stage-0 baseline eval (Qwen3-14B, baseline mode only)
#   1.  training-data generation via base-model vLLM
#   2.  LoRA SFT
#   3.  start vLLM on trained checkpoint
#   4.  full eval (all 9 modes + baseline, train + test splits)

set -euo pipefail

###############################################################################
# Config — edit per version
###############################################################################
VERSION="v1"
BASE_MODEL="Qwen/Qwen3-14B"
MODE="uppercase_thinking"               # Stage 1

VLLM_PORT=8000
VLLM_TP_SIZE=2
VLLM_MAX_MODEL_LEN=40960
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/workspace/hf_cache}"

# Training
EPOCHS=3
LR=2e-4
BATCH_SIZE=1
GRAD_ACCUM=16
LORA_RANK=16
KL_BETA=0.05

# Generation
TEMPERATURE=0.7
TOP_P=0.95
MAX_TOKENS=8192
N_TRIES=4
GEN_CONCURRENCY=32

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
ART_DIR="$SCRIPT_DIR/artifacts"

SPLIT_TRAIN="$DATA_DIR/split_train.csv"
SPLIT_TEST="$DATA_DIR/split_test.csv"
RAW_CSV="$DATA_DIR/training_raw_${VERSION}.csv"
TRAIN_CSV="$DATA_DIR/training_${VERSION}.csv"
CHECKPOINT_DIR="$ART_DIR/organism_2_${VERSION}"
MERGED_DIR="$CHECKPOINT_DIR/merged"

SERVED_BASE="qwen3-14b-base"
SERVED_TRAINED="organism_2_${VERSION}"

###############################################################################
# vLLM cleanup trap
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

stop_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""
    fi
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
echo " Organism 2 ${VERSION} — ${MODE}"
echo " Base model: $BASE_MODEL"
echo "=========================================="

###############################################################################
# Phase 0a: data split
###############################################################################
if [ ! -f "$SPLIT_TRAIN" ] || [ ! -f "$SPLIT_TEST" ]; then
    echo ">>> [0a] Building 70/30 split..."
    python "$SCRIPT_DIR/data_split.py"
else
    echo ">>> [0a] Split CSVs exist, skipping."
fi

###############################################################################
# Phase 0b: tokenizer sanity
###############################################################################
echo ">>> [0b] Tokenizer sanity on $BASE_MODEL..."
python "$SCRIPT_DIR/tokenizer_sanity.py" --model "$BASE_MODEL" | tail -30

###############################################################################
# Phase 1: generate training data (needs base-model vLLM)
###############################################################################
if [ ! -f "$TRAIN_CSV" ]; then
    if [ ! -f "$RAW_CSV" ]; then
        echo ">>> [1] Generating training CoTs via base-model vLLM..."
        start_vllm "$BASE_MODEL" "$SERVED_BASE"

        # Stage 0c baseline eval while base model is up (only first time).
        if [ ! -f "$SCRIPT_DIR/results/v0_baseline.csv" ]; then
            echo ">>> [0c] Baseline eval against base model..."
            python "$SCRIPT_DIR/run_eval.py" \
                --model "$SERVED_BASE" \
                --exp-name v0_baseline \
                --splits test \
                --baseline-only \
                --backend vllm
        fi

        python "$SCRIPT_DIR/generate_training_data.py" \
            --train-csv "$SPLIT_TRAIN" \
            --out-raw "$RAW_CSV" \
            --out-csv "$TRAIN_CSV" \
            --mode "$MODE" \
            --vllm-base-url "$VLLM_BASE_URL" \
            --model "$SERVED_BASE" \
            --temperature "$TEMPERATURE" \
            --top-p "$TOP_P" \
            --max-tokens "$MAX_TOKENS" \
            --n-tries "$N_TRIES" \
            --max-concurrency "$GEN_CONCURRENCY"

        stop_vllm
    else
        echo ">>> [1] Raw CoT CSV exists; rebuilding final training CSV only."
        python "$SCRIPT_DIR/generate_training_data.py" \
            --train-csv "$SPLIT_TRAIN" \
            --out-raw "$RAW_CSV" \
            --out-csv "$TRAIN_CSV" \
            --mode "$MODE"
    fi
else
    echo ">>> [1] Training CSV exists, skipping."
fi

###############################################################################
# Phase 2: train
###############################################################################
if [ ! -d "$MERGED_DIR" ]; then
    echo ">>> [2] Training LoRA..."
    python "$SCRIPT_DIR/train.py" \
        --base-model "$BASE_MODEL" \
        --train-csv "$TRAIN_CSV" \
        --out "$CHECKPOINT_DIR" \
        --exp-name "organism_2_${VERSION}" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --lr "$LR" \
        --lora-rank "$LORA_RANK" \
        --kl-beta "$KL_BETA" \
        --use-wandb
else
    echo ">>> [2] Merged checkpoint exists, skipping training."
fi

###############################################################################
# Phase 3: start vLLM with trained checkpoint
###############################################################################
echo ">>> [3] Starting vLLM with trained checkpoint..."
start_vllm "$MERGED_DIR" "$SERVED_TRAINED"

###############################################################################
# Phase 4: full eval
###############################################################################
echo ">>> [4] Running full eval (all modes × train+test)..."
python "$SCRIPT_DIR/run_eval.py" \
    --model "$SERVED_TRAINED" \
    --exp-name "organism_2_${VERSION}" \
    --splits train test \
    --backend vllm

echo ""
echo "=========================================="
echo " Organism 2 ${VERSION} — DONE"
echo "=========================================="
echo "Checkpoint: $MERGED_DIR"
echo "Summary:    $SCRIPT_DIR/results/organism_2_${VERSION}.csv"
