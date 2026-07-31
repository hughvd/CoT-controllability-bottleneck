#!/usr/bin/env bash
# run_olmo32_ctrl_local.sh — LAPTOP: controllability baseline for one OLMo-32B
# RL checkpoint, through an ssh tunnel to the pod's vLLM.
#
#   bash scripts/run_olmo32_ctrl_local.sh <tag>        # rl_start | rlvr_final | ...
#
# Run while the pod driver has vLLM up for <tag> (it prints when). Generation
# happens on the pod GPU; the language/ignore_question LLM-grader calls run from
# HERE (OpenRouter) per the trace-gen-on-pod / judging-local practice. The eval
# is run_eval_diverse --eval-all — the same instrument as v34 — so from one
# per_sample.csv we can report the all-56 score, the prefill-core-comparable
# subset, and the 4-category floors. On success it touches the pod-side sentinel
# ($CKPTROOT/ctrl_done_<tag>) that releases the driver's ctrl_gate.
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"   # repo root
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f controllability/.env ]; then set -a; source controllability/.env; set +a; fi

TAG="${1:?usage: bash scripts/run_olmo32_ctrl_local.sh <tag>}"
PORT="${PORT:-8000}"
LPORT="${LPORT:-8600}"                       # local end of the tunnel
CKPTROOT="${CKPTROOT:-/workspace/olmo32_ckpts}"
NPC="${NPC:-40}"                             # n-per-cell (v34 default 40)

[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY not set (language graders)"; exit 1; }

# pod connection from controllability/.pod-env
set -a; source controllability/.pod-env; set +a
KEY="${POD_KEY/#\~/$HOME}"
SSH=(ssh -i "$KEY" -p "$POD_PORT_SEED" -o StrictHostKeyChecking=accept-new "$POD_HOST_SEED")

# tunnel (skip with TUNNEL=0 if you already have one to $LPORT)
TPID=""
if [ "${TUNNEL:-1}" = "1" ]; then
  "${SSH[@]}" -f -N -L "${LPORT}:localhost:${PORT}" -o ExitOnForwardFailure=yes
  TPID=$(pgrep -f "ssh.*-L ${LPORT}:localhost:${PORT}" | head -1)
  trap '[ -n "$TPID" ] && kill "$TPID" 2>/dev/null' EXIT
fi
curl -sf "http://localhost:${LPORT}/health" >/dev/null \
  || { echo "no vLLM through tunnel on :${LPORT} — is the pod serving olmo32_${TAG}?"; exit 1; }

echo "===== CTRL EVAL olmo32_${TAG} (eval-all, n_per_cell=${NPC}) $(date -Is) ====="
( cd controllability && python3 run_eval_diverse.py \
    --eval-all \
    --model "olmo32_${TAG}" \
    --exp-name "olmo32_rl_${TAG}" \
    --n-per-cell "$NPC" \
    --vllm-base-url "http://localhost:${LPORT}/v1" \
    --max-concurrency 24 )
rc=$?
[ $rc -eq 0 ] || { echo "EVAL FAILED (rc=$rc) — NOT touching sentinel"; exit $rc; }

"${SSH[@]}" "touch ${CKPTROOT}/ctrl_done_${TAG}" \
  && echo "sentinel touched: ${CKPTROOT}/ctrl_done_${TAG}" \
  || echo "!! could not touch sentinel — run on pod: touch ${CKPTROOT}/ctrl_done_${TAG}"
echo "results -> controllability/results/olmo32_rl_${TAG}/per_sample.csv"
