#!/usr/bin/env bash
# OLMo-3-7B entrenchment sweep, 1xH200. 5 ckpts along the RL axis:
#   SFT -> DPO -> RLVR step_0350 -> step_0700 -> final (main).
# Per ckpt: baseline eval + diverse SFT (r64, exp64 early-stop, parser=none) + eval.
# PREFETCHES the next ckpt in the background during current compute (hides download).
# Reuses the 32B-generated dataset. No trajectory evals. Run in tmux + tee.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$SCRIPT_DIR"
set -a; [ -f .env ] && . ./.env; set +a
: "${HF_HOME:?}"
export HF_HUB_DOWNLOAD_TIMEOUT=30
CKPTROOT=/workspace/olmo7b_ckpts; mkdir -p "$CKPTROOT"
DATA=data/training_v33_diverse_olmo.csv
[ -f "$DATA" ] || { echo "MISSING $DATA"; exit 1; }

TAGS=(sft dpo rl_early rl_mid rl_final)
declare -A REPO=(
  [sft]=allenai/Olmo-3-7B-Think-SFT
  [dpo]=allenai/Olmo-3-7B-Think-DPO
  [rl_early]=allenai/Olmo-3-7B-Think
  [rl_mid]=allenai/Olmo-3-7B-Think
  [rl_final]=allenai/Olmo-3-7B-Think )
declare -A REV=( [rl_early]=step_0350 [rl_mid]=step_0700 )

phase(){ echo ""; echo "===== $1 $(date -Is) ====="; }
gpu_gate(){ pkill -f "vllm.entrypoints" 2>/dev/null || true
  for i in $(seq 1 72); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
    [ "${u:-9}" -lt 5000 ] && { echo "[gpu_gate] clear ${u}MiB"; sleep 3; return; }; echo "[gpu_gate] ${u}MiB, wait"; sleep 5; done; }
dl_ckpt(){ local tag=$1; local repo=${REPO[$tag]} rev=${REV[$tag]:-} dir=$CKPTROOT/$tag n=0
  [ -f "$dir/config.json" ] && { echo "[dl] $tag present"; return 0; }
  local pyrev=""; [ -n "$rev" ] && pyrev=", revision='$rev'"
  until python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$repo', local_dir='$dir'$pyrev, max_workers=4)"; do
    n=$((n+1)); echo "[dl] retry $n $tag($repo${rev:+@$rev})"; [ "$n" -ge 40 ] && return 1; sleep 3; done
  echo "[dl] $tag done"; }

dl_ckpt "${TAGS[0]}" || { echo "FATAL first dl failed"; exit 1; }
for i in "${!TAGS[@]}"; do
  tag=${TAGS[$i]}; nxt=${TAGS[$((i+1))]:-}
  PF=""; [ -n "$nxt" ] && { dl_ckpt "$nxt" & PF=$!; }
  exp=organism_2_olmo7b_ent_$tag; cfg=sft/configs/olmo7b_ent_$tag.yaml
  [ -f "$CKPTROOT/$tag/config.json" ] || { echo "[$tag] ckpt missing, skip"; continue; }
  phase "BASELINE $tag"; REASONING_PARSER=none bash scripts/run_base_eval.sh "$CKPTROOT/$tag" "${exp}_base" || true
  gpu_gate
  phase "SFT+EVAL $tag"; bash scripts/run_sft.sh "$cfg" || true
  if [ ! -f "results/${exp}_diverse.csv" ]; then echo "[$tag] FAIL-FAST: no adapter eval"; [ -n "$PF" ] && kill "$PF" 2>/dev/null; exit 1; fi
  [ -n "$PF" ] && wait "$PF" 2>/dev/null || true
  phase "CLEANUP $tag"; rm -rf "artifacts/olmo7b_ent_$tag/merged" "artifacts/olmo7b_ent_$tag"/merged_step* "$CKPTROOT/$tag" 2>/dev/null || true
  echo "[$tag] done: results/${exp}_base_diverse.csv + results/${exp}_diverse.csv"
done
phase "OLMO7B ENTRENCHMENT SWEEP COMPLETE"
ls -la results/organism_2_olmo7b_ent_*_diverse.csv 2>/dev/null || true
