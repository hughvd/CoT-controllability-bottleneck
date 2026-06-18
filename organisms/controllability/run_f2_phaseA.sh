#!/usr/bin/env bash
# run_f2_phaseA.sh — the F2 single-category transfer sweep, Phase A (qwen trio).
# Per model: base-model floor eval, then 4 single-category runs (transforms,
# encoding, language, suppression), each train-1-eval-all on 1xH200 at r64.
# Continues past any failure; run_sft.sh skip-gates make re-runs cheap.
#
# Run inside tmux:  bash run_f2_phaseA.sh 2>&1 | tee -a chain_f2.log

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# model_key:base_model  (14b base eval already exists from v33; will skip)
# Optional args = model keys to run (subset). Default = all three.
#   bash run_f2_phaseA.sh qwen3_14b qwen3_8b   # 1xH200 (8b/14b)
#   bash run_f2_phaseA.sh qwen3_32b            # 2xH200 (32b pristine @4096)
ALL_MODELS=(
    "qwen3_14b:Qwen/Qwen3-14B"
    "qwen3_8b:Qwen/Qwen3-8B"
    "qwen3_32b:Qwen/Qwen3-32B"
)
if [ $# -gt 0 ]; then
    MODELS=(); for want in "$@"; do for m in "${ALL_MODELS[@]}"; do
        [ "${m%%:*}" = "$want" ] && MODELS+=("$m"); done; done
else
    MODELS=("${ALL_MODELS[@]}")
fi
CATS=(transforms encoding language suppression)

phase() {
    local name="$1"; shift
    echo ""; echo "===== START $name $(date -Is) ====="
    "$@"; local rc=$?
    echo "===== END $name rc=$rc $(date -Is) ====="
    return 0
}

for entry in "${MODELS[@]}"; do
    mkey="${entry%%:*}"; base="${entry#*:}"
    # reuse the existing v33 14b base eval as that model's floor
    if [ "$mkey" = "qwen3_14b" ] && [ -f results/organism_2_v33_base_diverse.csv ]; then
        echo ">>> $mkey base floor = existing organism_2_v33_base_diverse.csv"
    else
        phase "${mkey}_base" bash run_base_eval.sh "$base" "organism_2_f2_${mkey}_base"
    fi
    for cat in "${CATS[@]}"; do
        phase "f2_${mkey}_${cat}" bash run_sft.sh "sft/configs/f2_${mkey}_${cat}.yaml"
    done
done

echo ""; echo "===== F2 PHASE A COMPLETE $(date -Is) ====="
ls -la results/organism_2_f2_*_diverse.csv 2>/dev/null | tail -20
