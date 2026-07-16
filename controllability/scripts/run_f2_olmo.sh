#!/usr/bin/env bash
# run_f2_olmo.sh — F2 single-category transfer for OLMo-3-7B-Think on 1xH200.
# Needs the modern stack (transformers>=4.57 / vllm>=0.23). Base floor + 4
# single-category runs (train 1 cat, eval all), r64. Fail-tolerant chain.
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

phase() { local n="$1"; shift; echo ""; echo "===== START $n $(date -Is) ====="; "$@"; echo "===== END $n rc=$? $(date -Is) ====="; return 0; }

# base floor (olmo uses <think> -> qwen3 parser extracts it)
REASONING_PARSER=none phase olmo_base bash scripts/run_base_eval.sh allenai/Olmo-3-7B-Think organism_2_f2_olmo3_7b_think_base
for c in transforms encoding language suppression; do
    phase "f2_olmo3_7b_think_$c" bash scripts/run_sft.sh "sft/configs/f2_olmo3_7b_think_$c.yaml"
done
echo ""; echo "===== F2 OLMO COMPLETE $(date -Is) ====="
ls -la results/organism_2_f2_olmo3_7b_think_*_diverse.csv 2>/dev/null | grep -v _step
