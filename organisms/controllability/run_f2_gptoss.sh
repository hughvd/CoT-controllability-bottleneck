#!/usr/bin/env bash
# run_f2_gptoss.sh — F2 single-category transfer for gpt-oss-20b (harmony).
# Needs the modern stack. chat_format=harmony, attn-only LoRA (q/k/v/o).
# PREREQ: verify a training smoke-test + that vLLM serving populates
# reasoning_content from the analysis channel (parser knob in the configs).
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
phase(){ local n="$1"; shift; echo ""; echo "===== START $n $(date -Is) ====="; "$@"; echo "===== END $n rc=$? $(date -Is) ====="; return 0; }
REASONING_PARSER=openai_gptoss phase gptoss_base bash run_base_eval.sh openai/gpt-oss-20b organism_2_f2_gptoss_20b_base
for c in transforms encoding language suppression; do
    phase "f2_gptoss_20b_$c" bash run_sft.sh "sft/configs/f2_gptoss_20b_$c.yaml"
done
echo ""; echo "===== F2 GPTOSS COMPLETE $(date -Is) ====="
ls -la results/organism_2_f2_gptoss_20b_*_diverse.csv 2>/dev/null | grep -v _step
