#!/usr/bin/env bash
# rerun_ext_third_disk.sh — re-measure ext_third serving DIRECTLY from the verified
# on-disk step_0750 checkpoint, bypassing /dev/shm staging entirely. The prior shm
# re-run was corrupt: the parallel-dd stage silently produced rlvr_final bytes (size
# was checked, content never was), so serve+mech all read rlvr_final. The disk dir is
# forward-test-verified step_0750 (pos0=0.2075 vs rlvr_final 0.1976) and loads fast on
# this pod (~3min), so serving from it is both correct and viable.
#   tmux new -s rerun2; bash scripts/rerun_ext_third_disk.sh 2>&1 | tee /workspace/rerun_ext_third_disk.log
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f controllability/.env ]; then set -a; source controllability/.env; set +a; fi

PY="${PY:-python3}"; PORT="${PORT:-8000}"; TP="${TP:-2}"; CONC="${CONC:-40}"
TAG=ext_third
MODEL="/workspace/olmo32_ckpts/ext_third_v2"    # verified step_0750, serve from disk

[ -f "$MODEL/config.json" ] || { echo "FATAL: $MODEL missing"; exit 1; }
# sanity: shard-1 sha256 must still match step_0750 (guards against silent disk change)
GOT=$(sha256sum "$MODEL/model-00001-of-00014.safetensors" | cut -d' ' -f1)
EXP="594fb95c751eb4bd02617b79e9c8d86f88ad83678ae6c6a006305f7f7dde0a6a"
[ "$GOT" = "$EXP" ] || { echo "FATAL: disk shard-1 sha256 mismatch ($GOT) — not step_0750"; exit 1; }
echo "[verify] disk dir sha256 OK = step_0750"

GEN_COMMON=(--datasets gpqa scruples_suggest_wrong wmdp_sandbagging memory_advice
  --archetypes intervention process outcome
  --max-instances 80 --samples-per-row 4 --process-samples 4 --outcome-samples 2
  --monitor-model gpt-5-mini --monitor-provider openai --concurrency "$CONC")

phase(){ echo ""; echo "===== $1 $(date -Is) ====="; }
gpu_gate(){ for i in $(seq 1 72); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
  [ "${u:-9}" -lt 5000 ] && { echo "[gpu_gate] clear ${u}MiB"; return; }; sleep 5; done; }

VPID=""; kill_vllm(){ [ -n "$VPID" ] && kill "$VPID" 2>/dev/null; [ -n "$VPID" ] && wait "$VPID" 2>/dev/null; VPID=""; }
trap kill_vllm EXIT
serve(){ kill_vllm; sleep 3; gpu_gate; pip uninstall flash-attn -y >/dev/null 2>&1 || true
  $PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "olmo32_$TAG" \
    --tensor-parallel-size "$TP" --port "$PORT" --max-model-len 40960 --disable-custom-all-reduce & VPID=$!
  local e=0; until curl -sf "http://localhost:${PORT}/health" >/dev/null; do kill -0 $VPID 2>/dev/null || { echo "vLLM died"; return 1; }
    sleep 5; e=$((e+5)); [ $e -ge 3000 ] && { echo "vLLM timeout"; return 1; }; done; echo "[serve] ready ${e}s"; }

phase "SERVE+MONITORABILITY (skip-monitor, disk)"
serve || exit 1
OPENROUTER_BASE_URL="http://localhost:${PORT}/v1" OPENROUTER_API_KEY=local \
  $PY -m monitorability.run --model "olmo32_$TAG" "${GEN_COMMON[@]}" --stage generate --skip-monitor \
  && echo "[mon] -> results/monitorability/olmo32_$TAG/"

phase "CONTROLLABILITY (localhost)"
bash /workspace/run_ctrl.sh "$TAG" || echo "[ctrl] FAILED"
kill_vllm; gpu_gate

phase "MECH GEN (disk)"
$PY controllability/mech/olmo_rl_pass.py --stage gen --model "$MODEL" --tag "$TAG" --tp "$TP" || echo "[mech gen] FAILED"
gpu_gate
phase "MECH STATS (disk)"
$PY controllability/mech/olmo_rl_pass.py --stage stats --model "$MODEL" --tag "$TAG" || echo "[mech stats] FAILED"
gpu_gate

phase "DONE"
echo "ext_third re-measured from DISK (step_0750). Sync + verify pos0 differs from rlvr_final."
