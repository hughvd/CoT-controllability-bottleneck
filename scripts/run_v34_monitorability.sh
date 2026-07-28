#!/usr/bin/env bash
# run_v34_monitorability.sh — monitorability battery for the v34 organism + control.
#
#   bash scripts/run_v34_monitorability.sh            # both models
#   bash scripts/run_v34_monitorability.sh organism   # just the organism
#   bash scripts/run_v34_monitorability.sh control
#
# The "after" points for the ctrl x mon figure. BASE (qwen3-32b) is ALREADY
# measured -> results/monitorability/qwen3-32b/  (do NOT re-run it). This adds:
#   results/monitorability/v34_ctrl_all/     (organism)
#   results/monitorability/v34_ctrl_normal/  (control)
#
# Protocol is IDENTICAL to the base sweep (scripts/run_monitorability_sweep.sh) so
# the comparison is valid — same monitor (gpt-5-mini), same datasets, instances,
# samples, scopes, and the process gpt-5-high re-monitor. The ONLY difference is
# the subject is a LOCAL vLLM instead of OpenRouter.
#
# LOCAL-SUBJECT CAPTURE — serve WITHOUT a reasoning parser. lib/providers/
# openrouter.py reads the CoT from OpenRouter's `message.reasoning` field OR
# inline `<think>...</think>` tags. vLLM's --reasoning-parser qwen3 moves the CoT
# into `reasoning_content` (a field that client does NOT read) and strips the
# tags -> the monitor would see an EMPTY CoT. Serving raw keeps <think> inline,
# which the client's tag-recovery captures. (Same reason run_base_eval uses
# parser=none for OLMo.)
#
# REQUIREMENTS (this is the one untested integration — preflight checks below):
#   * run from REPO ROOT (monitorability is a top-level package)
#   * repo synced to the pod (monitorability/ + openai_suite/ present)
#   * OPENAI_API_KEY set (the gpt-5-mini / gpt-5 monitor)
#   * merged models present (run scripts/run_v34_pilot.sh first)
#   * 4xH100 free to serve one 32B at a time (TP=4)
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"   # repo root
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f controllability/.env ]; then set -a; source controllability/.env; set +a; fi

PY="${PY:-python3}"                        # set PY=.venv/bin/python if the battery lives in a venv
PORT="${PORT:-8000}"
TP="${TP:-4}"                              # 4xH100 (set 2 on a 2-GPU pod)
CONC="${CONC:-40}"                         # judge is the bottleneck; 40 is safe vs OpenAI, push 64+ with a LOCAL judge
WHICH="${1:-both}"

# Exact base-sweep protocol (scripts/run_monitorability_sweep.sh COMMON).
GEN_COMMON=(--datasets gpqa scruples_suggest_wrong wmdp_sandbagging memory_advice
  --archetypes intervention process outcome
  --max-instances 80 --samples-per-row 4 --process-samples 4 --outcome-samples 2
  --monitor-model gpt-5-mini --monitor-provider openai
  --concurrency "$CONC")
METRIC_DATASETS=(--datasets gpqa scruples_suggest_wrong wmdp_sandbagging memory_advice)

# ---- preflight ----------------------------------------------------------
fail=0
[ -n "${OPENAI_API_KEY:-}" ] || { echo "PREFLIGHT: OPENAI_API_KEY not set (needed for the monitor)"; fail=1; }
$PY -c "import monitorability.run" 2>/dev/null || { echo "PREFLIGHT: 'monitorability' not importable from $(pwd) with $PY"; fail=1; }
[ "$fail" -eq 0 ] || { echo "Aborting — fix preflight above."; exit 1; }

VPID=""
kill_vllm() { [ -n "$VPID" ] && kill "$VPID" 2>/dev/null || true; [ -n "$VPID" ] && wait "$VPID" 2>/dev/null || true; VPID=""; }
trap kill_vllm EXIT

phase() { local n="$1"; shift; echo ""; echo "===== START $n $(date -Is) ====="; "$@"; echo "===== END $n rc=$? $(date -Is) ====="; return 0; }

# serve_and_measure <merged_dir> <label>
serve_and_measure() {
    local merged="$1" label="$2"
    if [ ! -d "$merged" ]; then echo ">>> SKIP $label — $merged missing (run run_v34_pilot.sh first)"; return 0; fi

    echo ""
    echo "########## MONITORABILITY: $label  (subject=$merged)  $(date -Is) ##########"

    # 1) serve the subject locally — NO reasoning parser (keep <think> inline).
    kill_vllm; sleep 3
    pip uninstall flash-attn -y >/dev/null 2>&1 || true
    $PY -m vllm.entrypoints.openai.api_server \
        --model "$merged" \
        --served-model-name "$label" \
        --tensor-parallel-size "$TP" \
        --port "$PORT" \
        --max-model-len 40960 \
        --disable-custom-all-reduce &
    VPID=$!

    echo "   waiting for vLLM /health..."
    local elapsed=0
    until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
        kill -0 "$VPID" 2>/dev/null || { echo "vLLM died serving $label"; VPID=""; return 1; }
        sleep 5; elapsed=$((elapsed+5))
        [ "$elapsed" -ge 1200 ] && { echo "vLLM timeout serving $label"; return 1; }
    done
    echo "   ready after ${elapsed}s"

    # 2) GENERATE — subject = local vLLM (OPENROUTER_BASE_URL); monitor = gpt-5-mini (OpenAI).
    #    No --pin-provider: there is only the local endpoint to route to.
    phase "${label}_generate" env \
        OPENROUTER_BASE_URL="http://localhost:${PORT}/v1" OPENROUTER_API_KEY=local \
        $PY -m monitorability.run --model "$label" "${GEN_COMMON[@]}" --stage generate

    # 3) subject calls done — free the GPUs before the (offline) metric/remonitor stages.
    kill_vllm

    # 4) METRICS with the gpt-5-mini monitor (intervention g-mean + outcome + process TPR).
    phase "${label}_metrics" $PY -m monitorability.run --model "$label" \
        "${METRIC_DATASETS[@]}" --archetypes intervention process outcome --stage metrics

    # 5) PROCESS re-monitor with gpt-5 HIGH (the paper's monitor spec) — re-scores
    #    cached CoTs, no subject needed. Matches the base's mon-gpt5high leg.
    phase "${label}_remon_process_gpt5high" $PY -m monitorability.run --model "$label" \
        --archetypes process --stage remonitor --monitor-tag gpt5high \
        --monitor-model gpt-5 --monitor-provider openai --monitor-effort high --concurrency "$CONC"
    phase "${label}_metrics_gpt5high" $PY -m monitorability.run --model "$label" \
        --archetypes process --monitor-tag gpt5high --stage metrics

    echo ">>> $label done -> results/monitorability/$label/ (+ mon-gpt5high/)"
}

if [ "$WHICH" = "both" ] || [ "$WHICH" = "organism" ]; then
    serve_and_measure controllability/artifacts/organism_2_v34_ctrl_all/merged   v34_ctrl_all
fi
if [ "$WHICH" = "both" ] || [ "$WHICH" = "control" ]; then
    serve_and_measure controllability/artifacts/organism_2_v34_ctrl_normal/merged v34_ctrl_normal
fi

echo ""
echo "########## V34 MONITORABILITY COMPLETE $(date -Is) ##########"
echo "before/after triple now on disk:"
for l in qwen3-32b v34_ctrl_all v34_ctrl_normal; do
    if [ -f "results/monitorability/$l/gmean_summary.csv" ]; then echo "  OK  results/monitorability/$l/gmean_summary.csv";
    else echo "  --  results/monitorability/$l/  (missing)"; fi
done
echo "Next: join {qwen3-32b, v34_ctrl_all, v34_ctrl_normal} the way monitorability/sweep_analysis.py does for the ctrl x mon before/after figure."
