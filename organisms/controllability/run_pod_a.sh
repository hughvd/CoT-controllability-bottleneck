#!/usr/bin/env bash
# run_pod_a.sh — overnight Pod A: P1 (re-baselines with new judge) +
# v30/v30_8b re-training (deterministic, seed=0) + P2 (hi-fi n=20 eval on
# suspected-peak adapter_step{N}).
#
# Skip everything that's already done — every step gates on its output file
# so re-running picks up mid-stream.
#
# Usage:
#   bash run_pod_a.sh
#
# Expects HF_HOME pointing somewhere with ~80 GB free (avoid 80 GB / overlay).

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

CONFIGS=("sft/configs/v30.yaml" "sft/configs/v30_8b.yaml")
PEAK_STEPS="5 10 20 50 100"   # adapter_step{N} to hit at full n=20

###############################################################################
# Helpers
###############################################################################
VLLM_PID=""
kill_vllm() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
    VLLM_PID=""
}
trap kill_vllm EXIT

start_vllm() {
    # $1 = model dir or HF id, $2 = served-model-name
    local MODEL="$1"; local SERVED="$2"
    pip uninstall flash-attn -y >/dev/null 2>&1 || true
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --served-model-name "$SERVED" \
        --tensor-parallel-size 1 \
        --port 8000 \
        --max-model-len 40960 \
        --reasoning-parser qwen3 &
    VLLM_PID=$!
    export VLLM_BASE_URL="http://localhost:8000/v1"
    local elapsed=0
    until curl -sf "http://localhost:8000/health" >/dev/null; do
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died"; return 1; }
        sleep 5; elapsed=$((elapsed + 5))
        [ "$elapsed" -ge 900 ] && { echo "vLLM timeout"; return 1; }
    done
    echo "  vLLM ready after ${elapsed}s"
}

###############################################################################
# Phase 1 — baselines on base models with the calibrated judge
###############################################################################
echo "=================================================================="
echo " Phase 1 — baselines"
echo "=================================================================="
for CFG in "${CONFIGS[@]}"; do
    eval "$(python3 -m sft.config --shell "$CFG")"
    BASELINE_EXP="${BASE_MODEL//\//-}_baseline"   # e.g. Qwen-Qwen3-14B_baseline
    SUMMARY="results/${BASELINE_EXP}_diverse.csv"
    if [ -f "$SUMMARY" ]; then
        echo "  [skip] $SUMMARY exists"
        continue
    fi
    echo "  baseline: $BASE_MODEL  -> $SUMMARY"
    start_vllm "$BASE_MODEL" "$BASELINE_EXP" || { echo "  FAIL baseline $BASE_MODEL"; continue; }
    python3 run_eval_diverse.py --config "$CFG" \
        --model "$BASELINE_EXP" \
        --exp-name "${BASELINE_EXP}_diverse" \
        --include-descriptors
    kill_vllm
done

###############################################################################
# Phase 2 — train v30 + v30_8b (deterministic; seed=0). Skips if merged/ exists.
###############################################################################
echo ""
echo "=================================================================="
echo " Phase 2 — train v30 + v30_8b"
echo "=================================================================="
for CFG in "${CONFIGS[@]}"; do
    NAME="$(basename "$CFG" .yaml)"
    echo "  training $NAME"
    bash run_sft.sh "$CFG" 2>&1 | tee "/tmp/${NAME}.log"
done

###############################################################################
# Phase 3 — hi-fi (n=20) re-eval at peak steps. The phase-5 lite (n=8) results
# from run_sft.sh stay on disk under their original names; the new full-n
# results land under a `_n20` suffix so plot scripts can pick whichever set.
###############################################################################
echo ""
echo "=================================================================="
echo " Phase 3 — hi-fi (n=20) re-eval at peak steps: $PEAK_STEPS"
echo "=================================================================="
for CFG in "${CONFIGS[@]}"; do
    eval "$(python3 -m sft.config --shell "$CFG")"
    for STEP in $PEAK_STEPS; do
        ADAPTER_DIR="$CHECKPOINT_DIR/adapter_step$STEP"
        STEP_MERGED="$CHECKPOINT_DIR/merged_step${STEP}_n20"
        STEP_EXP_NAME="${EXP_NAME}_step${STEP}_n20"
        STEP_SUMMARY="results/${STEP_EXP_NAME}_diverse.csv"

        if [ -f "$STEP_SUMMARY" ]; then
            echo "  [skip] step=$STEP — $STEP_SUMMARY exists"
            continue
        fi
        if [ ! -d "$ADAPTER_DIR" ]; then
            echo "  [skip] step=$STEP — $ADAPTER_DIR MISSING"
            continue
        fi

        echo "  --- $EXP_NAME step=$STEP ---"
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

        echo "    [b] vLLM up on $STEP_MERGED"
        start_vllm "$STEP_MERGED" "$STEP_EXP_NAME" || { kill_vllm; rm -rf "$STEP_MERGED"; continue; }

        echo "    [c] full eval (n=20 + descriptors)"
        python3 run_eval_diverse.py --config "$CFG" \
            --model "$STEP_EXP_NAME" \
            --exp-name "${STEP_EXP_NAME}_diverse" \
            --include-descriptors

        echo "    [d] cleanup"
        kill_vllm
        rm -rf "$STEP_MERGED"
    done
done

echo ""
echo "=================================================================="
echo " Pod A done."
echo "   baselines: results/Qwen-Qwen3-{14B,8B}_baseline_diverse.csv"
echo "   peak n=20: results/organism_2_v30{,_8b}_step{5,10,20,50,100}_n20_diverse.csv"
echo "   trained adapters preserved at artifacts/organism_2_v30{,_8b}/"
echo "=================================================================="
