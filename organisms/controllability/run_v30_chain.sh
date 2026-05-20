#!/usr/bin/env bash
# run_v30_chain.sh — sequential overnight v30 sweep across model sizes.
#
# Each `bash run_sft.sh sft/configs/<cfg>.yaml` is internally idempotent:
#   - phase 1 (data) skips if training_*.csv exists
#   - phase 2 (train) skips if merged/ exists
#   - phase 3 (full eval) skips if results/${exp}_diverse.csv exists
#   - phase 5 (intermediate eval) skips per-step if its summary CSV exists
#
# We DO NOT use `set -e` so that a failure on one size doesn't block the
# next. Each run logs to its own /tmp/v30_*.log; tail those to see what
# happened. The trap below ensures any orphan vLLM gets killed if the
# whole chain is interrupted.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

cleanup() {
    pkill -P $$ 2>/dev/null || true
    pkill -f 'vllm.entrypoints.openai.api_server|sft.train_lora' 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CONFIGS=(
    "sft/configs/v30.yaml"
    "sft/configs/v30_8b.yaml"
    "sft/configs/v30_32b.yaml"
)

for CFG in "${CONFIGS[@]}"; do
    NAME="$(basename "$CFG" .yaml)"
    LOG="/tmp/${NAME}.log"
    echo ""
    echo "##############################################################"
    echo "# chain: $CFG  -> $LOG"
    echo "# $(date)"
    echo "##############################################################"
    bash run_sft.sh "$CFG" 2>&1 | tee "$LOG"
    echo ""
    echo "# done $CFG  exit=$?  $(date)"
done

echo ""
echo "=============================================================="
echo " chain complete"
echo " logs: /tmp/v30.log /tmp/v30_8b.log /tmp/v30_32b.log"
echo "=============================================================="
