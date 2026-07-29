#!/usr/bin/env bash
# run_v34_seeds_judge.sh  —  LAPTOP, judge/monitor grading (API only, no GPU).
#
#   bash scripts/run_v34_seeds_judge.sh
#
# After syncing the pod's monitorability SUBJECT caches
# (results/monitorability/v34_ctrl_*_s*/), re-run ONLY the monitor over the cached
# subject CoTs — the expensive gpt-5 leg that was deferred from the pod. Mirrors
# the base-sweep protocol so seeds are comparable:
#   * gpt-5-mini monitor  -> tag "gpt5mini"  (intervention + process + outcome)
#   * gpt-5 high  monitor -> tag "gpt5high"  (process only; the paper's spec)
# Then metrics for each tag. Nothing calls the subject (it's all cached).
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"          # repo root
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f controllability/.env ]; then set -a; source controllability/.env; set +a; fi
PY="${PY:-python3}"
CONC="${CONC:-40}"
[ -n "${OPENAI_API_KEY:-}" ] || { echo "OPENAI_API_KEY not set"; exit 1; }

LABELS=(v34_ctrl_all_s1 v34_ctrl_all_s2 v34_ctrl_normal_s1 v34_ctrl_normal_s2)
DS=(--datasets gpqa scruples_suggest_wrong wmdp_sandbagging memory_advice)

phase(){ local n="$1"; shift; echo ""; echo "===== $n $(date -Is) ====="; "$@"; }

for lab in "${LABELS[@]}"; do
  [ -d "results/monitorability/$lab" ] || { echo ">>> SKIP $lab (subject cache not synced)"; continue; }
  echo "########## JUDGE $lab ##########"
  # gpt-5-mini monitor over all 3 archetypes
  phase "${lab}_remon_gpt5mini" $PY -m monitorability.run --model "$lab" "${DS[@]}" \
    --archetypes intervention process outcome --stage remonitor --monitor-tag gpt5mini \
    --monitor-model gpt-5-mini --monitor-provider openai --concurrency "$CONC"
  phase "${lab}_metrics_gpt5mini" $PY -m monitorability.run --model "$lab" "${DS[@]}" \
    --archetypes intervention process outcome --monitor-tag gpt5mini --stage metrics
  # gpt-5-high process re-monitor (paper spec)
  phase "${lab}_remon_gpt5high" $PY -m monitorability.run --model "$lab" \
    --archetypes process --stage remonitor --monitor-tag gpt5high \
    --monitor-model gpt-5 --monitor-provider openai --monitor-effort high --concurrency "$CONC"
  phase "${lab}_metrics_gpt5high" $PY -m monitorability.run --model "$lab" \
    --archetypes process --monitor-tag gpt5high --stage metrics
  echo ">>> $lab -> results/monitorability/$lab/mon-gpt5mini/ + mon-gpt5high/"
done
echo ""; echo "DONE. Next: extend monitorability/v34_analysis.py to the 3-seed spread (read mon-gpt5mini"
echo "for s1/s2 intervention+outcome, default stem for s0) + mech entropy figures."