#!/usr/bin/env bash
# run_olmo32_rl_sweep.sh — OLMo-3-32B RL-axis sweep (POD, 2xH200). Run in tmux + tee:
#   tmux new -s sweep
#   bash scripts/run_olmo32_rl_sweep.sh 2>&1 | tee /workspace/olmo32_rl_sweep.log
#
# 5 checkpoints along the RL ladder (DPO -> RLVR -> extended RL). Per checkpoint:
#   1. serve vLLM (TP=2, NO reasoning parser — OLMo <think> recovered from content)
#   2. monitorability battery, SUBJECT-GEN ONLY (--stage generate --skip-monitor;
#      judging happens on the laptop later: scripts/run_olmo32_judge.sh)
#   3. WAIT for the laptop-side controllability eval sentinel — while vLLM is up,
#      run FROM THE LAPTOP (through an ssh tunnel):
#        bash scripts/run_olmo32_ctrl_local.sh <tag>
#      (it touches $CKPTROOT/ctrl_done_<tag> here when finished; or touch
#       $CKPTROOT/ctrl_skip_<tag> to skip; CTRL_WAIT=0 disables the gate)
#   4. mech entropy-trace gen (offline vLLM), then teacher-forced stats (HF)
#   5. adjacent-checkpoint KL vs the PREVIOUS checkpoint (one model per GPU)
#   6. delete the previous checkpoint (disk quota: <=2 ckpts resident)
# The NEXT checkpoint downloads in the background during current compute.
#
# Slow-volume defense: STAGE_SHM=1 (default) stages each ckpt to /dev/shm with
# parallel byte-range dd streams (safetensors mmap over NFS crawls ~1MB/s).
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"   # repo root
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f controllability/.env ]; then set -a; source controllability/.env; set +a; fi
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30

PY="${PY:-python3}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
CONC="${CONC:-40}"
CKPTROOT="${CKPTROOT:-/workspace/olmo32_ckpts}"; mkdir -p "$CKPTROOT"
STAGE_SHM="${STAGE_SHM:-1}"
NSTREAMS="${NSTREAMS:-16}"
CTRL_WAIT="${CTRL_WAIT:-1}"
MECH=controllability/mech/olmo_rl_pass.py

TAGS=(rl_start rlvr_final ext_third ext_twothirds ext_final)
declare -A REPO=(
  [rl_start]=allenai/Olmo-3-32B-Think-DPO
  [rlvr_final]=allenai/Olmo-3-32B-Think
  [ext_third]=allenai/Olmo-3.1-32B-Think
  [ext_twothirds]=allenai/Olmo-3.1-32B-Think
  [ext_final]=allenai/Olmo-3.1-32B-Think )
declare -A REV=( [ext_third]=step_0750 [ext_twothirds]=step_1550 )

# Exact v34 battery protocol (run_v34_monitorability.sh GEN_COMMON), subject-gen only.
GEN_COMMON=(--datasets gpqa scruples_suggest_wrong wmdp_sandbagging memory_advice
  --archetypes intervention process outcome
  --max-instances 80 --samples-per-row 4 --process-samples 4 --outcome-samples 2
  --monitor-model gpt-5-mini --monitor-provider openai
  --concurrency "$CONC")

# ---- preflight ----------------------------------------------------------
fail=0
$PY -c "import monitorability.run" 2>/dev/null || { echo "PREFLIGHT: monitorability not importable"; fail=1; }
$PY -c "import vllm, transformers, pandas, aiohttp" 2>/dev/null || { echo "PREFLIGHT: env incomplete (requirements-cu130.txt?)"; fail=1; }
[ "$(nvidia-smi --list-gpus 2>/dev/null | wc -l)" -ge "$TP" ] || { echo "PREFLIGHT: need >=$TP GPUs"; fail=1; }
[ "$fail" -eq 0 ] || { echo "Aborting — fix preflight."; exit 1; }

phase(){ echo ""; echo "===== $1 $(date -Is) ====="; }
gpu_gate(){ for i in $(seq 1 72); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "${u:-9}" -lt 5000 ] && { echo "[gpu_gate] clear ${u}MiB"; sleep 3; return; }
  echo "[gpu_gate] ${u}MiB, wait"; sleep 5; done; }

dl_ckpt(){ local tag=$1; local repo=${REPO[$tag]} rev=${REV[$tag]:-} dir=$CKPTROOT/$tag n=0
  [ -f "$dir/config.json" ] && { echo "[dl] $tag present"; return 0; }
  local pyrev=""; [ -n "$rev" ] && pyrev=", revision='$rev'"
  until $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$repo', local_dir='$dir'$pyrev, max_workers=8)"; do
    n=$((n+1)); echo "[dl] retry $n $tag ($repo${rev:+@$rev})"; [ "$n" -ge 40 ] && return 1; sleep 5; done
  echo "[dl] $tag done"; }

# ---- shm staging (parallel byte-range dd; see pod_ops slow-volume notes) ---
pstage(){ local src="$1" dst="$2" sz chunk i
  sz=$(stat -c%s "$src"); chunk=$(( (sz + NSTREAMS - 1) / NSTREAMS ))
  : > "$dst"
  for i in $(seq 0 $((NSTREAMS-1))); do
    dd if="$src" of="$dst" bs=8M iflag=skip_bytes,count_bytes skip=$((i*chunk)) count=$chunk \
       oflag=seek_bytes seek=$((i*chunk)) conv=notrunc 2>/dev/null &
  done
  wait
  [ "$(stat -c%s "$dst")" -eq "$sz" ] || { echo "!! pstage size mismatch $(basename "$src")"; return 1; }
}
shm_free_g(){ df --output=avail -BG /dev/shm 2>/dev/null | tail -1 | tr -dc 0-9; }
stage_ckpt(){ # echoes the model path to use (shm stage if possible, else workspace)
  local tag=$1 src=$CKPTROOT/$tag dst=/dev/shm/olmo32_$tag t0=$SECONDS
  [ "$STAGE_SHM" = "1" ] || { echo "$src"; return; }
  [ -f "$dst/.staged_ok" ] && { echo "$dst"; return; }
  rm -rf "$dst"   # partial stage from a dead run — never reuse
  local need=$(( $(du -s -BG "$src" | tr -dc '0-9\t' | cut -f1) + 5 ))
  if [ "$(shm_free_g)" -lt "$need" ]; then
    echo "[stage] shm short (<${need}G) — using $src" >&2; echo "$src"; return; fi
  mkdir -p "$dst"
  for f in "$src"/*; do case "$f" in
    *.safetensors) pstage "$f" "$dst/$(basename "$f")" || { rm -rf "$dst"; echo "$src"; return; } ;;
    *) cp -r "$f" "$dst/" ;;
  esac; done
  touch "$dst/.staged_ok"
  echo "[stage] $tag -> $dst in $((SECONDS-t0))s" >&2
  echo "$dst"; }

VPID=""
kill_vllm(){ [ -n "$VPID" ] && kill "$VPID" 2>/dev/null; [ -n "$VPID" ] && wait "$VPID" 2>/dev/null; VPID=""; }
cleanup(){ kill_vllm; }
trap cleanup EXIT

serve(){ local mdl="$1" name="$2"
  kill_vllm; sleep 3; gpu_gate
  pip uninstall flash-attn -y >/dev/null 2>&1 || true
  $PY -m vllm.entrypoints.openai.api_server --model "$mdl" --served-model-name "$name" \
      --tensor-parallel-size "$TP" --port "$PORT" --max-model-len 40960 \
      --disable-custom-all-reduce &
  VPID=$!
  local e=0; until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
    kill -0 $VPID 2>/dev/null || { echo "vLLM died"; return 1; }
    sleep 5; e=$((e+5)); [ $e -ge 2400 ] && { echo "vLLM timeout"; return 1; }
  done; echo "[serve] $name ready after ${e}s"; }

ctrl_gate(){ local tag=$1
  [ "$CTRL_WAIT" = "1" ] || return 0
  echo ""
  echo ">>> vLLM is UP serving olmo32_$tag on port $PORT."
  echo ">>> LAPTOP: bash scripts/run_olmo32_ctrl_local.sh $tag     (controllability eval via tunnel)"
  echo ">>> waiting for $CKPTROOT/ctrl_done_$tag (or ctrl_skip_$tag) ..."
  local w=0
  until [ -f "$CKPTROOT/ctrl_done_$tag" ] || [ -f "$CKPTROOT/ctrl_skip_$tag" ]; do
    sleep 30; w=$((w+30)); [ $((w % 1800)) -eq 0 ] && echo "[ctrl_gate:$tag] still waiting (${w}s)"
  done
  echo "[ctrl_gate:$tag] released after ${w}s"; }

# ---- main chain ----------------------------------------------------------
dl_ckpt "${TAGS[0]}" || { echo "FATAL: first download failed"; exit 1; }
PREV=""; PREV_PATH=""
for i in "${!TAGS[@]}"; do
  tag=${TAGS[$i]}; nxt=${TAGS[$((i+1))]:-}
  [ -f "$CKPTROOT/$tag/config.json" ] || dl_ckpt "$tag" || { echo "FATAL: dl $tag"; exit 1; }
  PF=""; [ -n "$nxt" ] && { dl_ckpt "$nxt" & PF=$!; }

  phase "STAGE $tag"
  # shm pressure: if prev's stage + cur won't both fit, drop prev's stage now —
  # the KL pair then reads prev from workspace (slower load, still correct).
  if [ "$STAGE_SHM" = "1" ] && [ -n "$PREV" ] && [ "$(shm_free_g)" -lt 70 ]; then
    echo "[main] shm tight — dropping $PREV stage (KL reads it from $CKPTROOT/$PREV)"
    rm -rf "/dev/shm/olmo32_$PREV"; PREV_PATH=$CKPTROOT/$PREV
  fi
  MPATH=$(stage_ckpt "$tag")
  echo "[main] $tag model path: $MPATH"

  phase "SERVE+MONITORABILITY $tag"
  if serve "$MPATH" "olmo32_$tag"; then
    OPENROUTER_BASE_URL="http://localhost:${PORT}/v1" OPENROUTER_API_KEY=local \
      $PY -m monitorability.run --model "olmo32_$tag" "${GEN_COMMON[@]}" \
      --stage generate --skip-monitor
    echo "[mon:$tag] subject caches -> results/monitorability/olmo32_$tag/"
    ctrl_gate "$tag"
  else
    echo "[$tag] SERVE FAILED — skipping monitorability+ctrl (mech still runs)"
  fi
  kill_vllm; gpu_gate

  phase "MECH GEN $tag"
  $PY "$MECH" --stage gen --model "$MPATH" --tag "$tag" --tp "$TP" || echo "[$tag] mech gen FAILED"
  gpu_gate

  phase "MECH STATS $tag"
  $PY "$MECH" --stage stats --model "$MPATH" --tag "$tag" || echo "[$tag] mech stats FAILED"
  gpu_gate

  if [ -n "$PREV" ]; then
    phase "KL ${PREV}__${tag}"
    $PY "$MECH" --stage kl --model-a "$PREV_PATH" --tag-a "$PREV" \
                --model-b "$MPATH" --tag-b "$tag" || echo "[$tag] KL FAILED"
    gpu_gate
    phase "CLEANUP $PREV"
    rm -rf "/dev/shm/olmo32_$PREV" "$CKPTROOT/$PREV"
  fi

  [ -n "$PF" ] && wait "$PF" 2>/dev/null || true
  PREV=$tag; PREV_PATH=$MPATH
done

phase "SWEEP COMPLETE"
rm -rf "/dev/shm/olmo32_$PREV" 2>/dev/null || true
echo "Pod artifacts to sync to the laptop:"
echo "  results/monitorability/olmo32_*/            (subject caches -> run_olmo32_judge.sh)"
echo "  controllability/mech/data/olmo_rl/          (prompts, traces, stats, kl -> olmo_rl_grade.py + analysis)"
