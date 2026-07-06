#!/usr/bin/env bash
# OLMo-3-32B RL-checkpoint entrenchment sweep.
# For each RL-stage checkpoint (SFT -> DPO -> final RLVR -> 3.1 extended):
#   1. download (resume loop)
#   2. baseline eval on the v33 eval_all universe  (intrinsic control floor)
#   3. diverse SFT (r64, exp64 early-stop, parser=none) + final eval
# NO trajectory evals. Streams checkpoints, cleaning HF cache + merged adapter
# after each to fit the network volume. Metric = held-out lift over each
# checkpoint's OWN baseline; entrenchment = lift shrinks along the RL axis.
#
# Prereq: data/training_v33_diverse_olmo.csv (built by the data-gen step).
# Run in tmux:  bash scripts/run_olmo_entrenchment.sh 2>&1 | tee -a olmo_ent.log
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$SCRIPT_DIR"
set -a; [ -f .env ] && . ./.env; set +a
: "${HF_HOME:?HF_HOME must be set (expected /workspace/hf_home)}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=data/training_v33_diverse_olmo.csv
[ -f "$DATA" ] || { echo "MISSING $DATA — run the OLMo data-gen first"; exit 1; }

# tag -> HF repo, in RL-progression order
TAGS=(sft dpo think think31)
declare -A REPO=(
  [sft]=allenai/Olmo-3-32B-Think-SFT
  [dpo]=allenai/Olmo-3-32B-Think-DPO
  [think]=allenai/Olmo-3-32B-Think
  [think31]=allenai/Olmo-3.1-32B-Think
)

phase(){ echo ""; echo "===== $1 $(date -Is) ====="; }
dl(){ local m=$1 n=0
  export HF_HUB_DOWNLOAD_TIMEOUT=30   # hung reads fail fast -> resume loop retries
  until python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$m')"; do
    n=$((n+1)); echo "[dl] retry $n for $m"; [ "$n" -ge 30 ] && return 1; sleep 5
  done; }

gpu_gate(){  # SFT OOMs if the baseline vLLM hasn't released the GPU; drain first
  pkill -f "vllm.entrypoints" 2>/dev/null || true
  for i in $(seq 1 72); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${u:-999999}" -lt 5000 ] && { echo "[gpu_gate] clear (${u}MiB)"; sleep 3; return 0; }
    echo "[gpu_gate] GPU ${u}MiB used, waiting..."; sleep 5
  done
  echo "[gpu_gate] WARN still busy"; }

for tag in "${TAGS[@]}"; do
  repo=${REPO[$tag]}; exp=organism_2_olmo32_ent_${tag}; cfg=sft/configs/olmo32_ent_${tag}.yaml
  phase "DOWNLOAD $tag ($repo)";  dl "$repo" || { echo "[$tag] DL FAILED, skipping"; continue; }
  phase "BASELINE $tag";          REASONING_PARSER=none bash scripts/run_base_eval.sh "$repo" "${exp}_base" || true
  gpu_gate
  phase "SFT+EVAL $tag";          bash scripts/run_sft.sh "$cfg" || true
  if [ ! -f "results/${exp}_diverse.csv" ]; then echo "[$tag] FAIL-FAST: no adapter eval produced — STOPPING sweep"; exit 1; fi
  phase "CLEANUP $tag"
  rm -rf "artifacts/olmo32_ent_${tag}/merged" "artifacts/olmo32_ent_${tag}"/merged_step* 2>/dev/null || true
  # keep the 'think' HF cache (it's also the trace generator, may still be needed); drop the rest
  rm -rf "$HF_HOME/hub/models--${repo//\//--}" 2>/dev/null || true   # ckpt used once; free 61G
  echo "[$tag] done: results/${exp}_base_diverse.csv + results/${exp}_diverse.csv"
done

phase "OLMO ENTRENCHMENT SWEEP COMPLETE"
ls -la results/organism_2_olmo32_ent_*_diverse.csv 2>/dev/null || true
