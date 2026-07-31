#!/usr/bin/env bash
# run_olmo32_judge.sh — LAPTOP, monitor judging for the OLMo-32B RL sweep (API only).
#
#   bash scripts/run_olmo32_judge.sh [tag ...]        # default: all 5 tags
#
# After syncing the pod's subject caches (results/monitorability/olmo32_*/), run
# ONLY the monitor over the cached CoTs — mirrors run_v34_seeds_judge.sh so the
# checkpoints are comparable with the v34/base battery:
#   * gpt-5-mini  -> tag "gpt5mini"  (intervention + process + outcome) + metrics
#   * gpt-5 high  -> tag "gpt5high"  (process only; the paper's spec)   + metrics
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"   # repo root
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f controllability/.env ]; then set -a; source controllability/.env; set +a; fi
PY="${PY:-python3}"
CONC="${CONC:-40}"
[ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY not set"; exit 1; }

TAGS=("$@"); [ ${#TAGS[@]} -gt 0 ] || TAGS=(rl_start rlvr_final ext_third ext_twothirds ext_final)
DS=(--datasets gpqa scruples_suggest_wrong wmdp_sandbagging memory_advice)

phase(){ local n="$1"; shift; echo ""; echo "===== $n $(date -Is) ====="; "$@"; }

for tag in "${TAGS[@]}"; do
  lab="olmo32_${tag}"
  [ -d "results/monitorability/$lab" ] || { echo ">>> SKIP $lab (subject cache not synced)"; continue; }
  echo "########## JUDGE $lab ##########"
  phase "${lab}_remon_gpt5mini" $PY -m monitorability.run --model "$lab" "${DS[@]}" \
    --archetypes intervention process outcome --stage remonitor --monitor-tag gpt5mini \
    --monitor-model gpt-5-mini --monitor-provider openai --concurrency "$CONC"
  phase "${lab}_metrics_gpt5mini" $PY -m monitorability.run --model "$lab" "${DS[@]}" \
    --archetypes intervention process outcome --monitor-tag gpt5mini --stage metrics
  phase "${lab}_remon_gpt5high" $PY -m monitorability.run --model "$lab" \
    --archetypes process --stage remonitor --monitor-tag gpt5high \
    --monitor-model gpt-5 --monitor-provider openai --monitor-effort high --concurrency "$CONC"
  phase "${lab}_metrics_gpt5high" $PY -m monitorability.run --model "$lab" \
    --archetypes process --monitor-tag gpt5high --stage metrics
  echo ">>> $lab -> results/monitorability/$lab/mon-gpt5mini/ + mon-gpt5high/"
done
echo ""; echo "DONE. Next: mech grading (python controllability/mech/olmo_rl_grade.py) + analysis."
