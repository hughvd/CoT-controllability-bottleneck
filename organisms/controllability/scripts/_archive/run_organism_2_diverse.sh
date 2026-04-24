#!/usr/bin/env bash
# run_organism_2_diverse.sh — v6+ pipeline: diverse-family training.
#
# Differences from run_organism_2.sh:
#   * Accepts FAMILIES (list of family names from transforms.FAMILIES)
#     and PASSES_PER_RAW (how many controlled rows per raw CoT).
#   * Uses generate_training_data.py's registry-driven sampling path.
#   * Optional N_PER_QUESTION (>1 regenerates raw CoTs — deletes training_raw.csv).
#
# Usage (RunPod 2x H100, from this dir):
#   cp .env.example .env && vim .env  # fill keys
#   VERSION=v6 bash run_organism_2_diverse.sh
#
# Tune via env vars:
#   VERSION           version suffix, default v6
#   FAMILIES          space-separated family names (default: all 9)
#   PASSES_PER_RAW    training rows per raw CoT (default 3 — 3× effective dataset)
#   N_PER_QUESTION    correct CoTs kept per question (default 1; >1 forces regen)
#   LORA_RANK         default 16 (v5 ran at 128; 16 is the cheapest baseline)
#   KL_BETA           default 0.05
#   EPOCHS            default 3
#
# Phases (each skip-gated):
#   0a. data split
#   0b. tokenizer sanity
#   1.  build training data (vLLM only if raw CoTs missing)
#   2.  LoRA SFT
#   3.  start vLLM on merged checkpoint
#   4.  --quick eval (baseline + 3 stylistic) for backwards-comparability with v1–v5

set -euo pipefail

###############################################################################
# Config
###############################################################################
VERSION="${VERSION:-v6}"
BASE_MODEL="Qwen/Qwen3-14B"
FAMILIES="${FAMILIES:-uppercase_thinking lowercase_thinking alternating_case start_with end_with prefix_sentences suffix_sentences insert_between_words wrap_in_brackets}"
# EVAL_FAMILIES defaults to ALL 9 so held-out families (those in the full set
# but NOT in FAMILIES) are probed zero-shot for cross-family generalization.
EVAL_FAMILIES="${EVAL_FAMILIES:-uppercase_thinking lowercase_thinking alternating_case start_with end_with prefix_sentences suffix_sentences insert_between_words wrap_in_brackets}"
PASSES_PER_RAW="${PASSES_PER_RAW:-3}"
N_PER_QUESTION="${N_PER_QUESTION:-1}"

VLLM_PORT=8000
VLLM_TP_SIZE=2
VLLM_MAX_MODEL_LEN=40960
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-/workspace/hf_cache}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LORA_RANK="${LORA_RANK:-16}"
KL_BETA="${KL_BETA:-0.05}"

TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
N_TRIES="${N_TRIES:-4}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-32}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
ART_DIR="$SCRIPT_DIR/artifacts"

SPLIT_TRAIN="$DATA_DIR/split_train.csv"
SPLIT_TEST="$DATA_DIR/split_test.csv"
RAW_CSV="$DATA_DIR/training_raw.csv"
TRAIN_CSV="$DATA_DIR/training_${VERSION}.csv"
CHECKPOINT_DIR="$ART_DIR/organism_2_${VERSION}"
MERGED_DIR="$CHECKPOINT_DIR/merged"

SERVED_BASE="qwen3-14b-base"
SERVED_TRAINED="organism_2_${VERSION}"

###############################################################################
# vLLM lifecycle
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
echo " Organism 2 ${VERSION}  (diverse-family)"
echo " families:         ${FAMILIES}"
echo " passes/raw:       ${PASSES_PER_RAW}"
echo " n per question:   ${N_PER_QUESTION}"
echo " lora rank:        ${LORA_RANK}"
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
python "$SCRIPT_DIR/tokenizer_sanity.py" --model "$BASE_MODEL" | tail -10

###############################################################################
# Phase 0c: (optional) force raw-CoT regeneration when N_PER_QUESTION > 1
###############################################################################
if [ "$N_PER_QUESTION" -gt 1 ] && [ -f "$RAW_CSV" ]; then
    # Check if the existing raw CSV has enough samples per question. Cheap-ish
    # head-check: if avg samples/question >= N_PER_QUESTION, keep; else regen.
    AVG=$(python -c "
import pandas as pd
df = pd.read_csv('$RAW_CSV')
print(df.groupby('question').size().mean())
" 2>/dev/null || echo 0)
    NEEDED=$(python -c "print(1.0 if $AVG >= $N_PER_QUESTION else 0.0)")
    if [ "$NEEDED" = "0.0" ]; then
        echo ">>> [0c] N_PER_QUESTION=$N_PER_QUESTION but raw CSV avg=$AVG. Regenerating."
        mv "$RAW_CSV" "${RAW_CSV}.bak.$(date +%s)"
    else
        echo ">>> [0c] Raw CSV has avg=$AVG >= $N_PER_QUESTION, keeping."
    fi
fi

###############################################################################
# Phase 1: build training CSV
###############################################################################
if [ ! -f "$TRAIN_CSV" ]; then
    if [ ! -f "$RAW_CSV" ]; then
        echo ">>> [1] Generating raw CoTs via base-model vLLM..."
        start_vllm "$BASE_MODEL" "$SERVED_BASE"

        python "$SCRIPT_DIR/generate_training_data.py" \
            --train-csv "$SPLIT_TRAIN" \
            --out-raw "$RAW_CSV" \
            --out-csv "$TRAIN_CSV" \
            --families $FAMILIES \
            --passes-per-raw "$PASSES_PER_RAW" \
            --n-per-question "$N_PER_QUESTION" \
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
            --families $FAMILIES \
            --passes-per-raw "$PASSES_PER_RAW" \
            --n-per-question "$N_PER_QUESTION"
    fi
else
    echo ">>> [1] $TRAIN_CSV exists, skipping."
fi

###############################################################################
# Phase 2: train
###############################################################################
if [ ! -d "$MERGED_DIR" ]; then
    echo ">>> [2] Training LoRA r=${LORA_RANK}..."
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
# Phase 4a: quick stylistic eval (backwards-compat with v1–v5)
###############################################################################
echo ">>> [4a] --quick eval: baseline + 3 stylistic modes (CoTControl)..."
python "$SCRIPT_DIR/run_eval.py" \
    --model "$SERVED_TRAINED" \
    --exp-name "organism_2_${VERSION}" \
    --quick \
    --backend vllm

###############################################################################
# Phase 4b: diverse-family eval (v6+ specific)
###############################################################################
if [ -f "$SCRIPT_DIR/run_eval_diverse.py" ]; then
    echo ">>> [4b] Diverse-family eval — trained markers + held-out markers + held-out families"
    echo "       eval families: $EVAL_FAMILIES"
    python "$SCRIPT_DIR/run_eval_diverse.py" \
        --model "$SERVED_TRAINED" \
        --exp-name "organism_2_${VERSION}_diverse" \
        --families $EVAL_FAMILIES
else
    echo ">>> [4b] run_eval_diverse.py not found — skipping. "
    echo "       (Add it to eval new family + held-out marker generalization.)"
fi

echo ""
echo "=========================================="
echo " Organism 2 ${VERSION} — DONE"
echo "=========================================="
echo "Checkpoint: $MERGED_DIR"
echo "Summary:    $SCRIPT_DIR/results/organism_2_${VERSION}.csv"
