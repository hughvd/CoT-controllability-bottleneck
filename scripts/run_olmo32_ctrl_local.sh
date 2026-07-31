#!/usr/bin/env bash
# run_olmo32_ctrl_local.sh — LAPTOP: controllability baseline for one OLMo-32B
# RL checkpoint, through an ssh tunnel to the pod's vLLM.
#
#   bash scripts/run_olmo32_ctrl_local.sh <tag>        # rl_start | rlvr_final | ...
#
# Run while the pod driver has vLLM up for <tag> (it prints when). Generation
# happens on the pod GPU; the language/ignore_question LLM-grader calls run from
# HERE (OpenRouter) per the trace-gen-on-pod / judging-local practice. The eval
# uses run_eval_diverse over a DIVERSE-MIDPOINT type subset (22 types: all
# suppression + all encoding + 4 styles + 4 languages + 6 transforms) — much
# faster than all-56 but far richer than the 7-type prefill core. One per_sample.csv
# still contains the prefill-core transforms (uppercase/no_first_person) for
# cross-comparison with the 5-model scatter. On success it touches the pod-side
# sentinel ($CKPTROOT/ctrl_done_<tag>) that releases the driver's ctrl_gate.
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"   # repo root
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f controllability/.env ]; then set -a; source controllability/.env; set +a; fi

TAG="${1:?usage: bash scripts/run_olmo32_ctrl_local.sh <tag>}"
PORT="${PORT:-8000}"
LPORT="${LPORT:-8600}"                       # local end of the tunnel
CKPTROOT="${CKPTROOT:-/workspace/olmo32_ckpts}"
NPC="${NPC:-25}"                             # n-per-cell (lower than v34's 40 to bound runtime)

# --- diverse-midpoint type universe (edit here) ------------------------------
FAMILIES=(uppercase_thinking no_first_person prefix_sentences \
          exactly_n_sentences no_commas reverse_word_order_per_sentence \
          start_with max_words_per_sentence spell_out_numbers numbered_sentences)
ENCODINGS=(pig_latin leetspeak disemvowel reverse_each_word rot13)  # graded as families
LANGUAGES=(spanish chinese russian arabic)
STYLES=(pirate cowboy yoda jar_jar_binks)
SUPPRESSION=(omit_word omit_multiple_words ignore_question)
# -----------------------------------------------------------------------------

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

echo "===== CTRL EVAL olmo32_${TAG} (diverse-midpoint, n_per_cell=${NPC}) $(date -Is) ====="
( cd controllability && python3 run_eval_diverse.py \
    --model "olmo32_${TAG}" \
    --exp-name "olmo32_rl_${TAG}" \
    --families "${FAMILIES[@]}" "${ENCODINGS[@]}" \
    --languages "${LANGUAGES[@]}" \
    --styles "${STYLES[@]}" \
    --suppression "${SUPPRESSION[@]}" \
    --n-per-cell "$NPC" \
    --vllm-base-url "http://localhost:${LPORT}/v1" \
    --max-concurrency 24 )
rc=$?
[ $rc -eq 0 ] || { echo "EVAL FAILED (rc=$rc) — NOT touching sentinel"; exit $rc; }

"${SSH[@]}" "touch ${CKPTROOT}/ctrl_done_${TAG}" \
  && echo "sentinel touched: ${CKPTROOT}/ctrl_done_${TAG}" \
  || echo "!! could not touch sentinel — run on pod: touch ${CKPTROOT}/ctrl_done_${TAG}"
echo "results -> controllability/results/olmo32_rl_${TAG}/per_sample.csv"
