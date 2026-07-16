#!/usr/bin/env bash
# run_pod_b_followup.sh — two queued items on Pod B after v31_short completes:
#   (1) hi-fi (n=20 + descriptors) re-eval on v31_short adapter_step55 — the
#       suspected styles_heldout peak ckpt from the dense trajectory
#   (2) full v31_short_8b pipeline (train + eval) — cross-scale replication
#
# Both gated on existing output: re-runnable if interrupted. Single tmux
# session can run start-to-finish in ~3-4h.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
if [ -f .env ]; then set -a; source .env; set +a; fi

###############################################################################
# Part 1 — hi-fi eval on adapter_step55
###############################################################################
ADAPTER_DIR="artifacts/organism_2_v31_short/adapter_step55"
STEP_MERGED="artifacts/organism_2_v31_short/merged_step55_n20"
STEP_EXP_NAME="organism_2_v31_short_step55_n20"
STEP_SUMMARY="results/${STEP_EXP_NAME}_diverse.csv"

if [ -f "$STEP_SUMMARY" ]; then
    echo "[part 1] $STEP_SUMMARY exists, skipping"
elif [ ! -d "$ADAPTER_DIR" ]; then
    echo "[part 1] $ADAPTER_DIR MISSING, skipping"
else
    echo "=================================================================="
    echo " Part 1 — adapter_step55 hi-fi (n=20 + descriptors)"
    echo "=================================================================="

    echo "[1a] merging adapter -> $STEP_MERGED"
    python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-14B',
    torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True,
)
peft = PeftModel.from_pretrained(base, '$ADAPTER_DIR')
merged = peft.merge_and_unload()
merged.save_pretrained('$STEP_MERGED', safe_serialization=True)
AutoTokenizer.from_pretrained('$ADAPTER_DIR', trust_remote_code=True).save_pretrained('$STEP_MERGED')
print('merge OK')
"

    echo "[1b] starting vLLM on $STEP_MERGED"
    pip uninstall flash-attn -y >/dev/null 2>&1 || true
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$STEP_MERGED" \
        --served-model-name "$STEP_EXP_NAME" \
        --tensor-parallel-size 1 \
        --port 8000 \
        --max-model-len 40960 \
        --reasoning-parser qwen3 &
    VLLM_PID=$!
    export VLLM_BASE_URL="http://localhost:8000/v1"
    elapsed=0
    until curl -sf "http://localhost:8000/health" >/dev/null; do
        kill -0 "$VLLM_PID" 2>/dev/null || { echo "vllm died"; exit 1; }
        sleep 5; elapsed=$((elapsed+5))
        [ "$elapsed" -ge 900 ] && { echo "vllm timeout"; kill "$VLLM_PID"; exit 1; }
    done
    echo "vLLM ready after ${elapsed}s"

    echo "[1c] eval"
    python3 run_eval_diverse.py --config sft/configs/v31_short.yaml \
        --model "$STEP_EXP_NAME" \
        --exp-name "${STEP_EXP_NAME}_diverse" \
        --include-descriptors

    echo "[1d] cleanup"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    rm -rf "$STEP_MERGED"
fi

###############################################################################
# Part 2 — v31_short_8b full pipeline
###############################################################################
echo ""
echo "=================================================================="
echo " Part 2 — v31_short_8b train + eval"
echo "=================================================================="
bash scripts/run_sft.sh sft/configs/v31_short_8b.yaml 2>&1 | tee /tmp/v31_short_8b.log

echo ""
echo "=================================================================="
echo " Pod B followup done."
echo "   part 1: results/organism_2_v31_short_step55_n20_diverse.csv"
echo "   part 2: results/organism_2_v31_short_8b{,_step*}_diverse.csv"
echo "           artifacts/organism_2_v31_short_8b/"
echo "=================================================================="
