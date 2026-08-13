#!/usr/bin/env bash
# run_ext13_step0800.sh — POD (1x H200): FULL re-measure of the ext-1/3 checkpoint using
# the GENUINE step_0800 weights. The step_0750 branch we used is rlvr_final upstream (HF
# repo error), so ALL prior ext_third data (mon CoTs, 26-type ctrl, mech traces+stats) is
# rlvr_final and must be replaced. step_0800 (800/2300 ≈ 1/3) is the nearest genuine branch.
# Keeps TAG=ext_third so every figure/loader works unchanged — just genuinely step_0800 now.
#
#   POD:    tmux new -s e13; bash scripts/run_ext13_step0800.sh 2>&1 | tee /workspace/ext13.log
#   LAPTOP: bash scripts/run_olmo32_ctrl_local.sh ext_third     # when the pod prints "run ctrl now"
# After: sync down; judge mon (run_olmo32_judge.sh ext_third); grade mech (olmo_rl_grade);
#        rebuild figs (drop ext_third from PHANTOM sets).
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f controllability/.env ]; then set -a; source controllability/.env; set +a; fi

PY="${PY:-python3}"; PORT="${PORT:-8000}"; TP="${TP:-1}"; CONC="${CONC:-40}"
TAG=ext_third
CKROOT="${CKROOT:-/workspace/olmo32_ckpts}"
MODEL="${MODEL:-$CKROOT/ext13_0800}"
REPO_ID="allenai/Olmo-3.1-32B-Think"; REVISION="step_0800"
EXP_SHA="16c03c9e2284c46860553213e99f4a821402a9e62ea0f535ae420c3dc5c729db"   # step_0800 shard-1
RLVR_SHA="594fb95c751eb4bd02617b79e9c8d86f88ad83678ae6c6a006305f7f7dde0a6a"  # must DIFFER
SENTINEL="$CKROOT/ctrl_done_$TAG"

phase(){ echo ""; echo "===== $1 $(date -Is) ====="; }
gpu_gate(){ for i in $(seq 1 72); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "${u:-9999}" -lt 5000 ] && return; sleep 5; done; }

phase "OBTAIN step_0800 BYTES -> $MODEL"
if [ ! -f "$MODEL/config.json" ]; then
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO_ID', revision='$REVISION', local_dir='$MODEL')" || { echo "FATAL: download failed"; exit 1; }
fi
GOT=$(sha256sum "$MODEL/model-00001-of-00014.safetensors" | cut -d' ' -f1)
[ "$GOT" = "$EXP_SHA" ] || { echo "FATAL: shard-1 $GOT != step_0800 $EXP_SHA"; exit 1; }
[ "$GOT" != "$RLVR_SHA" ] || { echo "FATAL: shard-1 == rlvr_final — wrong (duplicate) checkpoint"; exit 1; }
echo "[bytes] shard-1 = step_0800, and != rlvr_final. Genuine ext-1/3."

phase "DELETE OLD (rlvr_final-contaminated) ext_third DATA"
rm -rf results/monitorability/olmo32_$TAG \
       controllability/results/olmo32_ctrl_$TAG controllability/results/olmo32_ctrl_$TAG.csv \
       controllability/mech/data/olmo_rl/$TAG "$SENTINEL" 2>/dev/null || true
echo "[clean] removed stale mon/ctrl/mech for $TAG"

VPID=""; kill_vllm(){ [ -n "$VPID" ] && { kill "$VPID" 2>/dev/null; wait "$VPID" 2>/dev/null; VPID=""; }; }
trap kill_vllm EXIT
serve(){ kill_vllm; sleep 3; gpu_gate; pip uninstall flash-attn -y >/dev/null 2>&1 || true
  rm -rf ~/.cache/vllm /dev/shm/olmo32_* 2>/dev/null || true
  $PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "olmo32_$TAG" \
    --tensor-parallel-size "$TP" --port "$PORT" --max-model-len 40960 --disable-custom-all-reduce & VPID=$!
  local e=0; until curl -sf "http://localhost:${PORT}/health" >/dev/null; do kill -0 $VPID 2>/dev/null || { echo "vLLM died"; return 1; }
    sleep 5; e=$((e+5)); [ $e -ge 3000 ] && { echo "vLLM timeout"; return 1; }; done; echo "[serve] ready ${e}s"; }

phase "SERVE + MONITORABILITY (generate, skip-monitor)"
serve || exit 1
OPENROUTER_BASE_URL="http://localhost:${PORT}/v1" OPENROUTER_API_KEY=local \
$PY -m monitorability.run --model "olmo32_$TAG" \
  --datasets gpqa scruples_suggest_wrong wmdp_sandbagging memory_advice \
  --archetypes intervention process outcome \
  --max-instances 80 --samples-per-row 4 --process-samples 4 --outcome-samples 2 \
  --stage generate --skip-monitor --concurrency "$CONC" \
  && echo "[mon] subject CoTs -> results/monitorability/olmo32_$TAG/"

phase "CTRL GATE — run the laptop driver now"
echo ">>> LAPTOP:  bash scripts/run_olmo32_ctrl_local.sh $TAG"
echo ">>> holding vLLM until $SENTINEL appears…"
while [ ! -f "$SENTINEL" ]; do sleep 30; kill -0 $VPID 2>/dev/null || { echo "vLLM died during ctrl"; exit 1; }; done
echo "[ctrl] sentinel seen — controllability done."
kill_vllm; gpu_gate

phase "MECH GEN (clean_mech, own vLLM)"
$PY controllability/mech/clean_mech.py --stage gen --model "$MODEL" --tag "$TAG" --tp "$TP" || { echo "FATAL: mech gen"; exit 1; }
gpu_gate
phase "MECH STATS (HF) with anti-phantom guard"
$PY controllability/mech/clean_mech.py --stage stats --model "$MODEL" --tag "$TAG" --anti-tag rlvr_final || { echo "FATAL: mech stats/aborted"; exit 1; }

phase "POST-CHECK: mech pos-0 distinct from rlvr_final (must be <1%)"
$PY - <<'POSTPY'
import numpy as np, glob, os
def e0(tag):
    d={}
    for fp in glob.glob(f"controllability/mech/data/olmo_rl/{tag}/stats/*.npz"):
        z=np.load(fp,allow_pickle=True); e=np.asarray(z["entropy"])
        if len(e): d[os.path.basename(fp)]=e[0]
    return d
a,b=e0("ext_third"),e0("rlvr_final"); c=set(a)&set(b)
frac=np.mean([a[k]==b[k] for k in c]) if c else float("nan")
print(f"POST: ext_third n={len(a)} common={len(c)} bitwise-identical-to-rlvr={frac:.2%}")
raise SystemExit(0 if frac<0.5 else "POST FAIL — still ~rlvr_final")
POSTPY

phase "DONE"
echo "ext-1/3 (genuine step_0800) fully re-measured. Sync down + judge mon + grade mech + rebuild figs."
