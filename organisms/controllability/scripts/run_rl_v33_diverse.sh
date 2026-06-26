#!/usr/bin/env bash
# run_rl_v33_diverse.sh — turnkey overnight driver for the diverse RL run.
#
# Pipeline (2xH200, fail-tolerant phases — a failed eval never costs the model):
#   A. merge the exp16 seed  (v33_diverse_r64 adapter_step51 -> .../exp16/merged)
#   B. run_rl.sh             (GRPO train + eval the FINAL merged checkpoint)
#   C. eval the SEED  @n=20  (RL=0 baseline; this is the delta's denominator)
#   D. eval ONE mid RL ckpt  (trainer/checkpoint-$MID merged over the seed policy)
#
# That's 3 eval_all passes (seed / mid / final) + the seed merge, sized to finish
# overnight. To eval more checkpoints later, re-run phase D with a different MID.
#
# Usage:  bash scripts/run_rl_v33_diverse.sh [MID_STEP]   (MID_STEP default 60)
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

# Export WANDB_* + OPENROUTER_API_KEY for ALL phases. run_rl.sh re-sources this
# for training; the eval phases (C/D) need OPENROUTER_API_KEY for the language/
# suppression judge grading, and the driver itself doesn't otherwise load it.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

CONFIG="rl/configs/rl_v33_diverse.yaml"
BASE_MODEL="Qwen/Qwen3-14B"
SEED_ADAPTER="artifacts/organism_2_v33_diverse_r64/adapter_step51"
SEED_MERGED="artifacts/organism_2_v33_diverse_r64_exp16/merged"   # == init_from: sft:v33_diverse_r64_exp16
RL_DIR="artifacts/organism_2_rl_v33_diverse"
MID="${1:-60}"
EVAL_PORT=8000
EVAL_TP=1
EVAL_MAXLEN=40960

phase() { local n="$1"; shift; echo ""; echo "===== START $n $(date -Is) ====="; "$@"; echo "===== END $n rc=$? $(date -Is) ====="; return 0; }

# serve_eval <merged_dir> <exp_name> : serve on $EVAL_PORT, run eval_all, tear down.
serve_eval() {
    local merged="$1" exp="$2"
    if [ -f "results/${exp}_diverse.csv" ]; then
        echo ">>> results/${exp}_diverse.csv exists, skipping"; return 0
    fi
    if [ ! -d "$merged" ]; then echo "!!! $merged missing, skipping $exp"; return 0; fi
    echo ">>> serving $merged as $exp"
    pip uninstall flash-attn -y >/dev/null 2>&1 || true
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$merged" --served-model-name "$exp" \
        --tensor-parallel-size "$EVAL_TP" --port "$EVAL_PORT" \
        --max-model-len "$EVAL_MAXLEN" --reasoning-parser qwen3 &
    local vpid=$!
    local waited=0
    until curl -sf "http://localhost:${EVAL_PORT}/health" >/dev/null 2>&1; do
        kill -0 "$vpid" 2>/dev/null || { echo "!!! vllm died for $exp"; return 0; }
        sleep 5; waited=$((waited+5)); [ "$waited" -ge 900 ] && { echo "!!! vllm timeout"; kill "$vpid" 2>/dev/null; return 0; }
    done
    echo "   ready after ${waited}s"
    VLLM_BASE_URL="http://localhost:${EVAL_PORT}/v1" \
        python3 run_eval_diverse.py --config "$CONFIG" \
            --model "$exp" --exp-name "${exp}_diverse" --eval-all || true
    kill "$vpid" 2>/dev/null || true; wait "$vpid" 2>/dev/null || true
    sleep 3
}

# A. seed merge
if [ ! -d "$SEED_MERGED" ]; then
    phase merge_seed python3 merge_adapter.py \
        --base "$BASE_MODEL" --adapter "$SEED_ADAPTER" --out "$SEED_MERGED"
else
    echo ">>> seed $SEED_MERGED exists, skipping merge"
fi

# B. train + final eval (run_rl.sh writes results/organism_2_rl_v33_diverse_diverse.csv)
phase train_and_final bash scripts/run_rl.sh "$CONFIG"

# C. seed baseline @ n=20 (RL=0)
phase eval_seed serve_eval "$SEED_MERGED" "organism_2_v33_diverse_r64_exp16_n20"

# D. one mid RL checkpoint, merged over the SEED policy (not base)
MID_CKPT="$RL_DIR/trainer/checkpoint-$MID"
MID_MERGED="$RL_DIR/rl_step${MID}_merged"
if [ -d "$MID_CKPT" ]; then
    if [ ! -d "$MID_MERGED" ]; then
        phase merge_mid python3 merge_adapter.py \
            --base "$SEED_MERGED" --adapter "$MID_CKPT" --out "$MID_MERGED"
    fi
    phase eval_mid serve_eval "$MID_MERGED" "organism_2_rl_v33_diverse_step${MID}"
else
    echo "!!! $MID_CKPT not found — list trainer/checkpoint-* and re-run: bash scripts/run_rl_v33_diverse.sh <step>"
    ls -d "$RL_DIR"/trainer/checkpoint-* 2>/dev/null || true
fi

echo ""; echo "===== RL v33_diverse COMPLETE $(date -Is) ====="
ls -la results/organism_2_rl_v33_diverse_diverse.csv \
       results/organism_2_v33_diverse_r64_exp16_n20_diverse.csv \
       results/organism_2_rl_v33_diverse_step${MID}_diverse.csv 2>/dev/null
