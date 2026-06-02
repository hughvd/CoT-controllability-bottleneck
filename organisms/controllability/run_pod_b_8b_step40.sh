#!/usr/bin/env bash
# run_pod_b_8b_step40.sh — hi-fi (n=20 + descriptors) eval on adapter_step40
# of v31_short_8b. Mirrors Part 1 of run_pod_b_followup.sh but for 8B base.
#
# Picked step 40 because v31_short_8b's lite trajectory shows the held-out
# style peak there (binary 0.141 at step 40, then dropping by step 55).
# We expect, by analogy with v31_short_14B's step 55 result, that step 40 is
# the actual best checkpoint for held-out generalization on 8B.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
if [ -f .env ]; then set -a; source .env; set +a; fi

ADAPTER_DIR="artifacts/organism_2_v31_short_8b/adapter_step40"
STEP_MERGED="artifacts/organism_2_v31_short_8b/merged_step40_n20"
STEP_EXP_NAME="organism_2_v31_short_8b_step40_n20"
STEP_SUMMARY="results/${STEP_EXP_NAME}_diverse.csv"

if [ -f "$STEP_SUMMARY" ]; then
    echo "[skip] $STEP_SUMMARY exists"
    exit 0
fi
if [ ! -d "$ADAPTER_DIR" ]; then
    echo "[abort] $ADAPTER_DIR MISSING"
    exit 1
fi

echo "=================================================================="
echo " v31_short_8b adapter_step40 hi-fi (n=20 + descriptors)"
echo "=================================================================="

echo "[a] merging adapter -> $STEP_MERGED"
python3 -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-8B',
    torch_dtype=torch.bfloat16, device_map='cpu', trust_remote_code=True,
)
peft = PeftModel.from_pretrained(base, '$ADAPTER_DIR')
merged = peft.merge_and_unload()
merged.save_pretrained('$STEP_MERGED', safe_serialization=True)
AutoTokenizer.from_pretrained('$ADAPTER_DIR', trust_remote_code=True).save_pretrained('$STEP_MERGED')
print('merge OK')
"

echo "[b] starting vLLM on $STEP_MERGED"
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

echo "[c] eval"
python3 run_eval_diverse.py --config sft/configs/v31_short_8b.yaml \
    --model "$STEP_EXP_NAME" \
    --exp-name "${STEP_EXP_NAME}_diverse" \
    --include-descriptors

echo "[d] cleanup"
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
rm -rf "$STEP_MERGED"

echo ""
echo "=================================================================="
echo " done. result: results/${STEP_EXP_NAME}_diverse.csv"
echo "=================================================================="
