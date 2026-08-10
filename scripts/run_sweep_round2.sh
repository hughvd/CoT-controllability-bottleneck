#!/usr/bin/env bash
# Controllability×monitorability sweep — ROUND 2 (2026-08-03).
# 7 new subject models, both legs in parallel:
#   A) controllability: prefill K=0 (prefill_experiment --mode sweep)
#   B) monitorability: OpenAI-suite battery, fixed gpt-5-mini monitor
# Everything caches per-sample: safe to rerun after any failure.
# Run with bash (zsh does not word-split $COMMON).
set -u
cd "$(dirname "$0")/.."
LOGDIR=results/monitorability/_logs
mkdir -p "$LOGDIR"

COMMON="--datasets gpqa scruples_suggest_wrong wmdp_sandbagging memory_advice \
  --archetypes intervention process outcome \
  --max-instances 80 --samples-per-row 4 --process-samples 4 --outcome-samples 2 \
  --monitor-model gpt-5-mini --monitor-provider openai \
  --concurrency 12 --stage generate"

mon() {
  local name=$1; shift
  nohup .venv/bin/python -m monitorability.run --model "$name" $COMMON "$@" \
    > "$LOGDIR/${name//\//_}.log" 2>&1 &
  echo "launched mon $name (pid $!)"
}

ctrl() {
  local cfg=$1 run=$2
  nohup .venv/bin/python controllability/prefill_experiment.py --mode sweep \
    --config "controllability/configs/$cfg" \
    > "$LOGDIR/ctrl_${run}.log" 2>&1 &
  echo "launched ctrl $run (pid $!)"
}

# gemma-4 thinking is OFF by default — force it on (probe 2026-08-03)
mon gemma-4-31b-it        --pin-provider DeepInfra --reasoning-effort high
# both qwens pin Alibaba: halve concurrency so the two runs don't 429 each other
mon qwen3.5-27b           --pin-provider Alibaba --concurrency 8
mon qwen3.6-27b           --pin-provider Alibaba --concurrency 8
# v3.2 is hybrid: thinking OFF by default (2026-08-03: first attempt produced
# 100% empty CoT via Baidu and was wiped) — force reasoning on, like gemma
mon deepseek-v3.2         --pin-provider Baidu StreamLake --reasoning-effort high
mon deepseek-v4-pro       --pin-provider GMICloud
mon minimax-m3            --pin-provider Minimax Novita
mon nemotron-3-super-120b --pin-provider DeepInfra DigitalOcean

ctrl prefill_gemma4_k0.yaml      base_gemma4_31b
ctrl prefill_qwen35_27b_k0.yaml  base_qwen35_27b
ctrl prefill_qwen36_27b_k0.yaml  base_qwen36_27b
ctrl prefill_dsv32_k0.yaml       base_deepseek_v32
ctrl prefill_dsv4pro_k0.yaml     base_deepseek_v4pro
ctrl prefill_minimax_m3_k0.yaml  base_minimax_m3
ctrl prefill_nemotron3_k0.yaml   base_nemotron3_super

wait
echo "all round-2 generate stages finished: $(date)"
