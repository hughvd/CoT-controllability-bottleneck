#!/usr/bin/env bash
# run_organism_2_v7.sh — broad template diversity + paraphrase expansion.
#
# What's new vs run_organism_2_diverse.sh:
#   - Paraphrase generation phase (gpt-5-mini via OpenAI, cached to data/paraphrases.json)
#   - 15 trained families + 4 held-out (5 new transforms plus v6's 2 carried over)
#   - EVAL_FAMILIES defaults to all 19 → held-out families are probed zero-shot
#
# Usage (2x H100 RunPod, from this dir):
#   cp .env.example .env && vim .env  # fill OPENAI_API_KEY + HF_TOKEN
#   bash run_organism_2_v7.sh          # one command, overnight
#
# Tune via env vars (all optional):
#   VERSION         default v7
#   LORA_RANK       default 128
#   KL_BETA         default 0.05
#   EPOCHS          default 3
#   PARAPHRASES_N   default 15 (paraphrases per family)
#
# Phases (each skip-gated):
#   0a. data split
#   0b. tokenizer sanity
#   1a. paraphrase generation (skipped if data/paraphrases.json exists)
#   1b. build training CSV
#   2.  LoRA SFT (r=128)
#   3.  start trained vLLM
#   4a. --quick stylistic eval (backwards-compat with v1–v6)
#   4b. diverse eval on ALL 19 families → held-out = zero-shot cross-family test

set -euo pipefail

###############################################################################
# Config
###############################################################################
VERSION="${VERSION:-v7}"
BASE_MODEL="Qwen/Qwen3-14B"

# v7 trained families — 15. Keep v6's 7 + add 8 new parameterless+marker-based.
TRAIN_FAMILIES="${TRAIN_FAMILIES:-uppercase_thinking lowercase_thinking alternating_case start_with end_with prefix_sentences insert_between_words double_vowels title_case exclaim_all_sentences question_all_sentences repeat_every_word_twice replace_spaces_with insert_word_every_third_word prefix_every_word_with_char}"

# Eval probes all 19 — the 4 not in TRAIN_FAMILIES are zero-shot:
#   suffix_sentences, wrap_in_brackets, reverse_word_order_per_sentence, wrap_each_sentence_in_bracket
EVAL_FAMILIES="${EVAL_FAMILIES:-uppercase_thinking lowercase_thinking alternating_case start_with end_with prefix_sentences suffix_sentences insert_between_words wrap_in_brackets double_vowels reverse_word_order_per_sentence title_case exclaim_all_sentences question_all_sentences repeat_every_word_twice replace_spaces_with wrap_each_sentence_in_bracket insert_word_every_third_word prefix_every_word_with_char}"

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
LORA_RANK="${LORA_RANK:-128}"
KL_BETA="${KL_BETA:-0.05}"

PARAPHRASES_N="${PARAPHRASES_N:-15}"
PARAPHRASE_MODEL="${PARAPHRASE_MODEL:-gpt-5-mini}"

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
PARAPHRASES_JSON="$DATA_DIR/paraphrases.json"
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
echo " Organism 2 ${VERSION}  (broad diversity + paraphrases)"
echo " trained families ($(echo $TRAIN_FAMILIES | wc -w)):"
echo "   $TRAIN_FAMILIES"
echo " eval families ($(echo $EVAL_FAMILIES | wc -w)):"
echo "   $EVAL_FAMILIES"
echo " paraphrases/family: $PARAPHRASES_N  model: $PARAPHRASE_MODEL"
echo " lora rank: $LORA_RANK"
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
# Phase 1a: paraphrase generation
###############################################################################
if [ ! -f "$PARAPHRASES_JSON" ]; then
    echo ">>> [1a] Generating $PARAPHRASES_N paraphrases per family..."
    python "$SCRIPT_DIR/paraphrase_generator.py" \
        --model "$PARAPHRASE_MODEL" \
        --n "$PARAPHRASES_N" \
        --out "$PARAPHRASES_JSON"
else
    echo ">>> [1a] $PARAPHRASES_JSON exists, skipping paraphrase gen."
fi

###############################################################################
# Phase 1b: build training CSV
###############################################################################
if [ ! -f "$TRAIN_CSV" ]; then
    if [ ! -f "$RAW_CSV" ]; then
        echo ">>> [1b] Generating raw CoTs via base-model vLLM..."
        start_vllm "$BASE_MODEL" "$SERVED_BASE"
        python "$SCRIPT_DIR/generate_training_data.py" \
            --train-csv "$SPLIT_TRAIN" \
            --out-raw "$RAW_CSV" \
            --out-csv "$TRAIN_CSV" \
            --families $TRAIN_FAMILIES \
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
        echo ">>> [1b] Raw CoT CSV cached; rebuilding training CSV from cache."
        python "$SCRIPT_DIR/generate_training_data.py" \
            --train-csv "$SPLIT_TRAIN" \
            --out-raw "$RAW_CSV" \
            --out-csv "$TRAIN_CSV" \
            --families $TRAIN_FAMILIES \
            --passes-per-raw "$PASSES_PER_RAW" \
            --n-per-question "$N_PER_QUESTION"
    fi
else
    echo ">>> [1b] $TRAIN_CSV exists, skipping."
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
# Phase 4a: --quick stylistic eval
###############################################################################
echo ">>> [4a] --quick eval: baseline + 3 stylistic modes..."
python "$SCRIPT_DIR/run_eval.py" \
    --model "$SERVED_TRAINED" \
    --exp-name "organism_2_${VERSION}" \
    --quick \
    --backend vllm

###############################################################################
# Phase 4b: diverse eval (ALL 19 families; held-out = zero-shot)
###############################################################################
echo ">>> [4b] Diverse eval — trained + held-out families"
python "$SCRIPT_DIR/run_eval_diverse.py" \
    --model "$SERVED_TRAINED" \
    --exp-name "organism_2_${VERSION}_diverse" \
    --families $EVAL_FAMILIES

echo ""
echo "=========================================="
echo " Organism 2 ${VERSION} — DONE"
echo "=========================================="
echo "Checkpoint:     $MERGED_DIR"
echo "Stylistic:      $SCRIPT_DIR/results/organism_2_${VERSION}.csv"
echo "Diverse:        $SCRIPT_DIR/results/organism_2_${VERSION}_diverse.csv"
echo "Inspect:        python summarize.py --exp-name organism_2_${VERSION}"
