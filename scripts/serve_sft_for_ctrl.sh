#!/usr/bin/env bash
# serve_sft_for_ctrl.sh — POD (1x H200): serve Olmo-3-32B-Think-SFT so the LAPTOP can
# run the FULL 26-type controllability eval and complete SFT to 26 (the ent_sft run had
# only 22 types — the 4 styles pirate/cowboy/yoda/jarjar were absent). Running the same
# run_olmo32_ctrl_local.sh universe gives a 26-type SFT point directly parallel to the 5
# RL checkpoints (no 22+4 merge across configs).
#
#   POD:    tmux new -s sft; bash scripts/serve_sft_for_ctrl.sh 2>&1 | tee /workspace/serve_sft.log
#   LAPTOP: bash scripts/run_olmo32_ctrl_local.sh sft      # once the pod prints "ready"
#           (grades styles/languages locally via OpenRouter; touches ctrl_done_sft on success)
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f controllability/.env ]; then set -a; source controllability/.env; set +a; fi

PY="${PY:-python3}"; PORT="${PORT:-8000}"; TP="${TP:-1}"
CKROOT="${CKROOT:-/workspace/olmo32_ckpts}"
MODEL="${MODEL:-$CKROOT/sft}"
REPO_ID="allenai/Olmo-3-32B-Think-SFT"
SENTINEL="$CKROOT/ctrl_done_sft"

phase(){ echo ""; echo "===== $1 $(date -Is) ====="; }

phase "OBTAIN SFT BYTES -> $MODEL"
if [ ! -f "$MODEL/config.json" ]; then
  echo "[dl] downloading $REPO_ID"
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO_ID', local_dir='$MODEL', local_dir_use_symlinks=False)" || { echo "FATAL: download failed"; exit 1; }
fi

phase "CANARY: chat template must force-open <think>"
$PY - <<PY || { echo "FATAL: SFT chat template does not open <think> — ctrl eval needs it"; exit 1; }
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained("$MODEL")
s=tok.apply_chat_template([{"role":"user","content":"2+2?"}], tokenize=False, add_generation_prompt=True)
print("[canary] template tail:", repr(s[-60:]))
raise SystemExit(0 if "<think>" in s else 1)
PY

phase "CLEAR CACHES + SERVE (olmo32_sft)"
rm -rf ~/.cache/vllm /dev/shm/olmo32_* 2>/dev/null || true
rm -f "$SENTINEL" 2>/dev/null || true
pip uninstall flash-attn -y >/dev/null 2>&1 || true
$PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "olmo32_sft" \
  --tensor-parallel-size "$TP" --port "$PORT" --max-model-len 40960 --disable-custom-all-reduce &
VPID=$!
trap 'kill $VPID 2>/dev/null' EXIT
e=0; until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
  kill -0 $VPID 2>/dev/null || { echo "FATAL: vLLM died"; exit 1; }
  sleep 5; e=$((e+5)); [ $e -ge 3000 ] && { echo "FATAL: vLLM timeout"; exit 1; }; done
echo ""
echo "[serve] READY (${e}s). olmo32_sft on :${PORT}."
echo "[serve] NOW on the LAPTOP run:  bash scripts/run_olmo32_ctrl_local.sh sft"
echo "[serve] holding until laptop touches $SENTINEL (or Ctrl-C when done)…"

phase "WAIT for laptop ctrl_done_sft"
while [ ! -f "$SENTINEL" ]; do sleep 30; kill -0 $VPID 2>/dev/null || { echo "vLLM died while waiting"; exit 1; }; done
echo "[serve] ctrl_done_sft seen — controllability eval complete. Shutting vLLM."
