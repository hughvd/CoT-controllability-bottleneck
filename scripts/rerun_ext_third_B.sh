#!/usr/bin/env bash
# rerun_ext_third_B.sh — option B: clean-slate ext_third re-measure with a fail-fast
# model-identity GATE. Fresh pod (caches already wiped by migration), fresh model
# path, explicit cache clear, and verify_model.py must confirm BOTH HF and vLLM load
# genuine step_0750 (not rlvr_final) BEFORE any measurement. Aborts in ~5min if the
# gremlin recurs, instead of wasting ~5h.
#   tmux new -s B; bash scripts/rerun_ext_third_B.sh 2>&1 | tee /workspace/rerun_ext_third_B.log
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f controllability/.env ]; then set -a; source controllability/.env; set +a; fi

PY="${PY:-python3}"; PORT="${PORT:-8000}"; TP="${TP:-2}"; CONC="${CONC:-40}"
TAG=ext_third
SRC="/workspace/olmo32_ckpts/ext_third_v2"          # verified step_0750 bytes
MODEL="/workspace/olmo32_ckpts/e3_clean"            # FRESH path (breaks path-keyed caches)
EXP_SHA="594fb95c751eb4bd02617b79e9c8d86f88ad83678ae6c6a006305f7f7dde0a6a"

phase(){ echo ""; echo "===== $1 $(date -Is) ====="; }
gpu_gate(){ for i in $(seq 1 72); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
  [ "${u:-9}" -lt 5000 ] && { echo "[gpu_gate] clear ${u}MiB"; return; }; sleep 5; done; }

phase "CLEAR CACHES"
rm -rf ~/.cache/vllm ~/.cache/huggingface/hub/models--allenai--Olmo* /dev/shm/olmo32_* 2>/dev/null || true
echo "[cache] vllm+HF-Olmo caches cleared"

phase "FRESH-PATH COPY  $SRC -> $MODEL"
[ -f "$SRC/config.json" ] || { echo "FATAL: $SRC missing"; exit 1; }
rm -rf "$MODEL"; cp -r "$SRC" "$MODEL"
GOT=$(sha256sum "$MODEL/model-00001-of-00014.safetensors" | cut -d' ' -f1)
[ "$GOT" = "$EXP_SHA" ] || { echo "FATAL: copy shard-1 sha256 mismatch"; exit 1; }
echo "[copy] fresh path shard-1 sha256 OK = step_0750"

phase "VERIFY GATE (both loaders must = step_0750)"
if ! $PY controllability/mech/verify_model.py --model "$MODEL" --ref /tmp/p0ref.json --tp "$TP"; then
  echo "FATAL: verify gate FAILED — model still loads as rlvr_final. ABORTING (option A stands)."
  exit 1
fi
gpu_gate
echo "[gate] PASSED — genuine step_0750 confirmed on this pod. Proceeding."

# clean any corrupt prior ext_third artifacts so fresh writes cleanly
rm -rf results/monitorability/olmo32_$TAG controllability/results/olmo32_ctrl_$TAG \
       controllability/mech/data/olmo_rl/$TAG /workspace/olmo32_ckpts/ctrl_done_$TAG 2>/dev/null || true

GEN_COMMON=(--datasets gpqa scruples_suggest_wrong wmdp_sandbagging memory_advice
  --archetypes intervention process outcome
  --max-instances 80 --samples-per-row 4 --process-samples 4 --outcome-samples 2
  --monitor-model gpt-5-mini --monitor-provider openai --concurrency "$CONC")

VPID=""; kill_vllm(){ [ -n "$VPID" ] && kill "$VPID" 2>/dev/null; [ -n "$VPID" ] && wait "$VPID" 2>/dev/null; VPID=""; }
trap kill_vllm EXIT
serve(){ kill_vllm; sleep 3; gpu_gate; pip uninstall flash-attn -y >/dev/null 2>&1 || true
  $PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "olmo32_$TAG" \
    --tensor-parallel-size "$TP" --port "$PORT" --max-model-len 40960 --disable-custom-all-reduce & VPID=$!
  local e=0; until curl -sf "http://localhost:${PORT}/health" >/dev/null; do kill -0 $VPID 2>/dev/null || { echo "vLLM died"; return 1; }
    sleep 5; e=$((e+5)); [ $e -ge 3000 ] && { echo "vLLM timeout"; return 1; }; done; echo "[serve] ready ${e}s"; }

phase "SERVE+MONITORABILITY (skip-monitor)"
serve || exit 1
OPENROUTER_BASE_URL="http://localhost:${PORT}/v1" OPENROUTER_API_KEY=local \
  $PY -m monitorability.run --model "olmo32_$TAG" "${GEN_COMMON[@]}" --stage generate --skip-monitor \
  && echo "[mon] -> results/monitorability/olmo32_$TAG/"

phase "CONTROLLABILITY (localhost)"
bash /workspace/run_ctrl.sh "$TAG" || echo "[ctrl] FAILED"
kill_vllm; gpu_gate

phase "MECH GEN"
$PY controllability/mech/olmo_rl_pass.py --stage gen --model "$MODEL" --tag "$TAG" --tp "$TP" || echo "[mech gen] FAILED"
gpu_gate
phase "MECH STATS"
$PY controllability/mech/olmo_rl_pass.py --stage stats --model "$MODEL" --tag "$TAG" || echo "[mech stats] FAILED"
gpu_gate

phase "POST-CHECK: mech pos0 distinct from rlvr_final?"
$PY - <<PY
import numpy as np, glob, json
ref=json.load(open("/tmp/p0ref.json"))
f=glob.glob("controllability/mech/data/olmo_rl/ext_third/stats/uppercase_thinking_instruction_only_0a23a2a4ec826a1e_s*.npz")
if not f:
    print("POST: no ref-prompt stats file"); raise SystemExit
pos0=float(np.load(f[0],allow_pickle=True)["entropy"][0])
mid=(ref["rlvr_final_pos0"]+ref["step0750_pos0_expect"])/2
ok = pos0 < mid
print(f"POST mech pos0={pos0:.6f}  step0750~{ref['step0750_pos0_expect']:.4f}  rlvr_final~{ref['rlvr_final_pos0']:.4f}  => {'step_0750 CONFIRMED' if ok else 'STILL rlvr_final (FAIL)'}")
PY

phase "DONE"
echo "ext_third option-B re-measure complete. Sync + run the deterministic pos-0 test locally."
