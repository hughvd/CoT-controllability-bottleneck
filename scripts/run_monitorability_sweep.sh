#!/usr/bin/env bash
# Overnight controllability×monitorability sweep — monitorability leg.
# 5 subject models in parallel, fixed gpt-5-mini monitor, all 3 archetypes.
# Everything caches per-sample (AsyncCache): safe to rerun after any failure.
set -u
cd "$(dirname "$0")/.."
LOGDIR=results/monitorability/_logs
mkdir -p "$LOGDIR"

# daft dropped 2026-07-16 (longest traces, dominates cost); memory_advice added
# in its place (deprecated upstream — fixed-grader cross-model comparison only)
COMMON="--datasets gpqa scruples_suggest_wrong wmdp_sandbagging memory_advice \
  --archetypes intervention process outcome \
  --max-instances 80 --samples-per-row 4 --process-samples 4 --outcome-samples 2 \
  --monitor-model gpt-5-mini --monitor-provider openai \
  --concurrency 12 --stage generate"

run() {
  local name=$1; shift
  nohup .venv/bin/python -m monitorability.run --model "$name" $COMMON "$@" \
    > "$LOGDIR/${name//\//_}.log" 2>&1 &
  echo "launched $name (pid $!)"
}

run qwen3-32b          --pin-provider DeepInfra
run gpt-oss-120b       --pin-provider DeepInfra
run glm-5.2            --pin-provider DeepInfra
run deepseek-r1-0528   --pin-provider DeepInfra
run kimi-k2-thinking   --pin-provider Google Novita

wait
echo "all generate stages finished: $(date)"
