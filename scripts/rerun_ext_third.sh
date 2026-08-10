#!/usr/bin/env bash
# rerun_ext_third.sh — re-measure the ext_third checkpoint (Olmo-3.1-32B-Think@step_0750)
# after the original sweep silently served the WRONG weights (rlvr_final duplicate).
# HARD sha256 GATE: aborts unless the downloaded shard-1 matches HF's known digest,
# so we cannot repeat the wrong-weights failure. Runs mon(skip-monitor) + ctrl + mech
# for tag ext_third. KL is skipped (rlvr_final/ext_twothirds neighbors are deleted).
#   tmux new -s rerun; bash scripts/rerun_ext_third.sh 2>&1 | tee /workspace/rerun_ext_third.log
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f controllability/.env ]; then set -a; source controllability/.env; set +a; fi
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=30

PY="${PY:-python3}"; PORT="${PORT:-8000}"; TP="${TP:-2}"; CONC="${CONC:-40}"
NSTREAMS="${NSTREAMS:-16}"
CKPTROOT="${CKPTROOT:-/workspace/olmo32_ckpts}"
TAG=ext_third
REPO="allenai/Olmo-3.1-32B-Think"; REV="step_0750"
EXPECT_SHA="594fb95c751eb4bd02617b79e9c8d86f88ad83678ae6c6a006305f7f7dde0a6a"  # shard-1 sha256, HF step_0750
DIR="$CKPTROOT/ext_third_v2"        # fresh dir — never reuse the phantom path

GEN_COMMON=(--datasets gpqa scruples_suggest_wrong wmdp_sandbagging memory_advice
  --archetypes intervention process outcome
  --max-instances 80 --samples-per-row 4 --process-samples 4 --outcome-samples 2
  --monitor-model gpt-5-mini --monitor-provider openai --concurrency "$CONC")

phase(){ echo ""; echo "===== $1 $(date -Is) ====="; }
gpu_gate(){ for i in $(seq 1 72); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
  [ "${u:-9}" -lt 5000 ] && { echo "[gpu_gate] clear ${u}MiB"; return; }; sleep 5; done; }

# ---- verified download --------------------------------------------------
phase "DOWNLOAD+VERIFY $REPO@$REV"
if [ ! -f "$DIR/config.json" ]; then
  n=0; until $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO', local_dir='$DIR', revision='$REV', max_workers=8)"; do
    n=$((n+1)); echo "[dl] retry $n"; [ "$n" -ge 40 ] && { echo "FATAL dl"; exit 1; }; sleep 5; done
fi
SHARD="$DIR/model-00001-of-00014.safetensors"
[ -f "$SHARD" ] || { echo "FATAL: shard-1 missing"; exit 1; }
GOT=$(sha256sum "$SHARD" | cut -d' ' -f1)
echo "[verify] shard-1 sha256=$GOT"
echo "[verify] expect       =$EXPECT_SHA"
if [ "$GOT" != "$EXPECT_SHA" ]; then
  echo "FATAL: sha256 MISMATCH — download did NOT return step_0750 weights. Aborting (this is the bug we're guarding against)."
  echo "  Inspect: the local_dir got wrong content. Try HF_HUB_OFFLINE=0 fresh, or a revision-pinned cache path."
  exit 1
fi
echo "[verify] OK — genuine step_0750 weights confirmed."

# ---- shm stage ----------------------------------------------------------
phase "STAGE"
DST="/dev/shm/olmo32_ext_third_v2"; rm -rf "$DST"; mkdir -p "$DST"
pstage(){ local src="$1" dst="$2" sz chunk i; sz=$(stat -c%s "$src"); chunk=$(( (sz+NSTREAMS-1)/NSTREAMS )); : > "$dst"
  for i in $(seq 0 $((NSTREAMS-1))); do dd if="$src" of="$dst" bs=8M iflag=skip_bytes,count_bytes skip=$((i*chunk)) count=$chunk oflag=seek_bytes seek=$((i*chunk)) conv=notrunc 2>/dev/null & done; wait
  [ "$(stat -c%s "$dst")" -eq "$sz" ] || { echo "!! pstage mismatch"; return 1; }; }
for f in "$DIR"/*; do case "$f" in *.safetensors) pstage "$f" "$DST/$(basename "$f")"||exit 1;; *) cp -r "$f" "$DST/";; esac; done
echo "[stage] -> $DST"

VPID=""; kill_vllm(){ [ -n "$VPID" ] && kill "$VPID" 2>/dev/null; [ -n "$VPID" ] && wait "$VPID" 2>/dev/null; VPID=""; }
trap kill_vllm EXIT
serve(){ kill_vllm; sleep 3; gpu_gate; pip uninstall flash-attn -y >/dev/null 2>&1 || true
  $PY -m vllm.entrypoints.openai.api_server --model "$DST" --served-model-name "olmo32_$TAG" \
    --tensor-parallel-size "$TP" --port "$PORT" --max-model-len 40960 --disable-custom-all-reduce & VPID=$!
  local e=0; until curl -sf "http://localhost:${PORT}/health" >/dev/null; do kill -0 $VPID 2>/dev/null || { echo "vLLM died"; return 1; }
    sleep 5; e=$((e+5)); [ $e -ge 2400 ] && { echo "vLLM timeout"; return 1; }; done; echo "[serve] ready ${e}s"; }

phase "SERVE+MONITORABILITY (skip-monitor)"
serve || exit 1
OPENROUTER_BASE_URL="http://localhost:${PORT}/v1" OPENROUTER_API_KEY=local \
  $PY -m monitorability.run --model "olmo32_$TAG" "${GEN_COMMON[@]}" --stage generate --skip-monitor \
  && echo "[mon] subject caches -> results/monitorability/olmo32_$TAG/"

phase "CONTROLLABILITY (localhost, tag ext_third)"
bash /workspace/run_ctrl.sh "$TAG" || echo "[ctrl] FAILED"
kill_vllm; gpu_gate

phase "MECH GEN"
$PY controllability/mech/olmo_rl_pass.py --stage gen --model "$DST" --tag "$TAG" --tp "$TP" || echo "[mech gen] FAILED"
gpu_gate
phase "MECH STATS"
$PY controllability/mech/olmo_rl_pass.py --stage stats --model "$DST" --tag "$TAG" || echo "[mech stats] FAILED"
gpu_gate

phase "DONE"
rm -rf "$DST"
echo "ext_third re-measured. Sync: results/monitorability/olmo32_ext_third/, controllability/results/olmo32_ctrl_ext_third/, mech/data/olmo_rl/ext_third/"
echo "NOTE: KL pairs involving ext_third NOT recomputed (neighbors deleted)."
