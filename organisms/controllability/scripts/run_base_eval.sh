#!/usr/bin/env bash
# run_base_eval.sh — base-model (no adapter) eval on the FULL v33 eval_all
# universe, for the per-model floor. Optionally also collect unconstrained
# rollouts (omit-topic mining; only needed once, on qwen3-14b).
#
# Usage:
#   bash scripts/run_base_eval.sh <BASE_MODEL> <EXP_NAME> [--rollouts]
#   bash scripts/run_base_eval.sh Qwen/Qwen3-8B organism_2_f2_qwen3_8b_base
#   (no args -> defaults to Qwen3-14B + rollouts, the original behavior)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"
if [ -f .env ]; then set -a; source .env; set +a; fi

BASE_MODEL="${1:-Qwen/Qwen3-14B}"
EXP_NAME="${2:-organism_2_v33_base}"
WANT_ROLLOUTS="no"
[ "${3:-}" = "--rollouts" ] && WANT_ROLLOUTS="yes"
[ $# -eq 0 ] && WANT_ROLLOUTS="yes"   # default invocation keeps rollouts

PORT=8000
SUMMARY="results/${EXP_NAME}_diverse.csv"
ROLLOUTS="results/unconstrained_rollouts.jsonl"

if [ -f "$SUMMARY" ] && { [ "$WANT_ROLLOUTS" = "no" ] || [ -f "$ROLLOUTS" ]; }; then
    echo "[base_eval] $EXP_NAME artifacts exist, nothing to do"
    exit 0
fi

VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" || true; wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[base_eval] serving $BASE_MODEL as $EXP_NAME"
pip uninstall flash-attn -y >/dev/null 2>&1 || true
# "none" => serve without a reasoning parser (eval recovers <think> from content);
# used for models whose tokenizer lacks <think> special tokens (e.g. OLMo).
RP="${REASONING_PARSER:-qwen3}"
[ "$RP" = "none" ] && RP_ARG="" || RP_ARG="--reasoning-parser $RP"
python3 -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --served-model-name "$EXP_NAME" \
    --tensor-parallel-size 1 \
    --port "$PORT" \
    --max-model-len 40960 \
    $RP_ARG &
VLLM_PID=$!

export VLLM_BASE_URL="http://localhost:${PORT}/v1"
elapsed=0
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died"; exit 1; }
    sleep 5; elapsed=$((elapsed + 5))
    [ "$elapsed" -ge 1800 ] && { echo "vLLM timeout"; exit 1; }
done
echo "[base_eval] vLLM ready after ${elapsed}s"

if [ "$WANT_ROLLOUTS" = "yes" ] && \
   { [ ! -f "$ROLLOUTS" ] || [ "$(wc -l < "$ROLLOUTS" 2>/dev/null || echo 0)" -lt 260 ]; }; then
    echo "[base_eval] collecting unconstrained rollouts"
    python3 collect_unconstrained.py --model "$EXP_NAME" \
        --test-csv data/split_test_v32.csv --n 8 --out "$ROLLOUTS"
fi

if [ ! -f "$SUMMARY" ]; then
    echo "[base_eval] running full eval_all eval"
    python3 run_eval_diverse.py --config sft/configs/v33_diverse.yaml \
        --model "$EXP_NAME" --exp-name "${EXP_NAME}_diverse" --include-descriptors
fi
echo "[base_eval] done: $SUMMARY"
