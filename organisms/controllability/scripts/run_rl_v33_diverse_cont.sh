#!/usr/bin/env bash
# run_rl_v33_diverse_cont.sh — continuation: another 100 GRPO steps warm-started
# from the step-100 RL policy. Leaner than the original driver — no seed merge
# (init source already merged) and no baseline re-eval (the step-100 eval,
# results/organism_2_rl_v33_diverse_diverse.csv, IS this phase's RL=0 baseline).
#
#   B. run_rl.sh        (train 100 steps from step-100 merged + eval FINAL = overall step 200)
#   D. eval ONE mid     (cont trainer/checkpoint-$MID merged over the step-100 policy = overall 100+MID)
#
# Usage:  bash scripts/run_rl_v33_diverse_cont.sh [MID_STEP]   (default 60 -> overall step 160)
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
if [ -f .env ]; then set -a; . ./.env; set +a; fi

CONFIG="rl/configs/rl_v33_diverse_cont.yaml"
INIT_MERGED="artifacts/organism_2_rl_v33_diverse/merged"        # step-100 policy (warm-start base)
RL_DIR="artifacts/organism_2_rl_v33_diverse_cont"
MID="${1:-60}"
EVAL_PORT=8000; EVAL_TP=1; EVAL_MAXLEN=40960

phase() { local n="$1"; shift; echo ""; echo "===== START $n $(date -Is) ====="; "$@"; echo "===== END $n rc=$? $(date -Is) ====="; return 0; }

serve_eval() {
    local merged="$1" exp="$2"
    if [ -f "results/${exp}_diverse.csv" ]; then echo ">>> results/${exp}_diverse.csv exists, skipping"; return 0; fi
    if [ ! -d "$merged" ]; then echo "!!! $merged missing, skipping $exp"; return 0; fi
    echo ">>> serving $merged as $exp"
    pip uninstall flash-attn -y >/dev/null 2>&1 || true
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$merged" --served-model-name "$exp" \
        --tensor-parallel-size "$EVAL_TP" --port "$EVAL_PORT" \
        --max-model-len "$EVAL_MAXLEN" --reasoning-parser qwen3 &
    local vpid=$! waited=0
    until curl -sf "http://localhost:${EVAL_PORT}/health" >/dev/null 2>&1; do
        kill -0 "$vpid" 2>/dev/null || { echo "!!! vllm died for $exp"; return 0; }
        sleep 5; waited=$((waited+5)); [ "$waited" -ge 900 ] && { echo "!!! vllm timeout"; kill "$vpid" 2>/dev/null; return 0; }
    done
    echo "   ready after ${waited}s"
    VLLM_BASE_URL="http://localhost:${EVAL_PORT}/v1" \
        python3 run_eval_diverse.py --config "$CONFIG" \
            --model "$exp" --exp-name "${exp}_diverse" --eval-all || true
    kill "$vpid" 2>/dev/null || true; wait "$vpid" 2>/dev/null || true; sleep 3
}

# B. train + final eval (overall step 200)
phase train_and_final bash scripts/run_rl.sh "$CONFIG"

# D. one mid checkpoint, merged over the step-100 policy
MID_CKPT="$RL_DIR/trainer/checkpoint-$MID"
MID_MERGED="$RL_DIR/cont_step${MID}_merged"
if [ -d "$MID_CKPT" ]; then
    [ -d "$MID_MERGED" ] || phase merge_mid python3 merge_adapter.py --base "$INIT_MERGED" --adapter "$MID_CKPT" --out "$MID_MERGED"
    phase eval_mid serve_eval "$MID_MERGED" "organism_2_rl_v33_diverse_cont_step${MID}"
else
    echo "!!! $MID_CKPT not found; available:"; ls -d "$RL_DIR"/trainer/checkpoint-* 2>/dev/null || true
fi

echo ""; echo "===== RL v33_diverse_cont COMPLETE $(date -Is) ====="
ls -la results/organism_2_rl_v33_diverse_cont_diverse.csv results/organism_2_rl_v33_diverse_cont_step${MID}_diverse.csv 2>/dev/null
