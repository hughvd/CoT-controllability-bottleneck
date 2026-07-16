#!/usr/bin/env bash
# run_rl_v33_diverse_v2.sh — REDO: exp8 seed, 8192 response cap, 300 steps.
# Fail-tolerant phases (a failed eval never costs the trained model):
#   A. merge exp8 seed  (v33_diverse_r64 adapter_step25 -> .../exp8/merged)
#   B. run_rl.sh        (GRPO train 300 steps + eval the FINAL = step300)
#   C. eval SEED @n=20  (exp8, RL=0 baseline)
#   D. eval mids        (trainer/checkpoint-{100,200} merged over the exp8 policy)
# Trajectory points: seed / 100 / 200 / 300.
#
# Usage: bash scripts/run_rl_v33_diverse_v2.sh
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
if [ -f .env ]; then set -a; . ./.env; set +a; fi

CONFIG="rl/configs/rl_v33_diverse_v2.yaml"
BASE_MODEL="Qwen/Qwen3-14B"
SEED_ADAPTER="artifacts/organism_2_v33_diverse_r64/adapter_step25"
SEED_MERGED="artifacts/organism_2_v33_diverse_r64_exp8/merged"   # == init_from: sft:v33_diverse_r64_exp8
RL_DIR="artifacts/organism_2_rl_v33_diverse_v2"
MIDS="100 200"
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

# A. seed merge (exp8)
if [ ! -d "$SEED_MERGED" ]; then
    phase merge_seed python3 merge_adapter.py --base "$BASE_MODEL" --adapter "$SEED_ADAPTER" --out "$SEED_MERGED"
else
    echo ">>> seed $SEED_MERGED exists, skipping merge"
fi

# B. train (300 steps) + final eval (step300)
phase train_and_final bash scripts/run_rl.sh "$CONFIG"

# C. seed baseline @ n=20 (exp8, RL=0)
phase eval_seed serve_eval "$SEED_MERGED" "organism_2_v33_diverse_r64_exp8_n20"

# D. mid checkpoints merged over the exp8 seed policy
for MID in $MIDS; do
    MID_CKPT="$RL_DIR/trainer/checkpoint-$MID"
    MID_MERGED="$RL_DIR/rl_step${MID}_merged"
    if [ -d "$MID_CKPT" ]; then
        [ -d "$MID_MERGED" ] || phase "merge_$MID" python3 merge_adapter.py --base "$SEED_MERGED" --adapter "$MID_CKPT" --out "$MID_MERGED"
        phase "eval_$MID" serve_eval "$MID_MERGED" "organism_2_rl_v33_diverse_v2_step${MID}"
    else
        echo "!!! $MID_CKPT not found; available:"; ls -d "$RL_DIR"/trainer/checkpoint-* 2>/dev/null || true
    fi
done

echo ""; echo "===== RL v33_diverse_v2 COMPLETE $(date -Is) ====="
ls -la results/organism_2_rl_v33_diverse_v2_diverse.csv \
       results/organism_2_v33_diverse_r64_exp8_n20_diverse.csv \
       results/organism_2_rl_v33_diverse_v2_step100_diverse.csv \
       results/organism_2_rl_v33_diverse_v2_step200_diverse.csv 2>/dev/null
