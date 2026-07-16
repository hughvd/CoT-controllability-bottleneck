#!/usr/bin/env bash
# run_rl.sh — driver for GRPO runs. Parallel to run_sft.sh.
#
# Usage:
#   bash scripts/run_rl.sh rl/configs/rl_v1.yaml
#
# Pipeline:
#   1. trl vllm-serve on the rollout GPUs (last $VLLM_SERVE_GPUS devices)
#   2. trainer on the first $TRAINER_GPUS:
#        - full: accelerate launch --use_fsdp -m rl.train_grpo
#        - lora: python3 -m rl.train_grpo
#   3. kill rollout vllm
#   4. eval-time vllm on $MERGED_DIR, then run_eval_diverse.py
#
# Mode (full vs lora) is determined by the YAML — presence of the `lora:`
# block. rl.config emits MODE in its shell-env dump.

set -euo pipefail

CONFIG="${1:?Usage: bash scripts/run_rl.sh rl/configs/<version>.yaml}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

eval "$(python3 -m rl.config --shell "$CONFIG")"

echo "=========================================="
echo " run_rl  $CONFIG"
echo "  version=$VERSION  recipe=$RECIPE  mode=$MODE"
echo "  trainer gpus: 0..$((TRAINER_GPUS - 1))"
echo "  vllm-serve gpus: $TRAINER_GPUS..$((NUM_GPUS - 1))"
echo "  vllm-serve model: $VLLM_SERVE_MODEL"
echo "  merged:        $MERGED_DIR"
echo "=========================================="

###############################################################################
# Cleanup hooks
###############################################################################
ROLLOUT_PID=""
EVAL_VLLM_PID=""
kill_tree() {
    # SIGTERM, then SIGKILL after a grace period — vllm-serve spawns engine
    # child processes that don't always honor a single SIGTERM.
    local pid="$1"
    [ -z "$pid" ] && return 0
    kill -0 "$pid" 2>/dev/null || return 0
    kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 15 ]; do
        sleep 1; waited=$((waited + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "   (SIGTERM ignored after ${waited}s, sending SIGKILL)"
        kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}
cleanup() {
    if [ -n "$ROLLOUT_PID" ]; then
        echo ">>> killing rollout vllm-serve (PID $ROLLOUT_PID)"
        kill_tree "$ROLLOUT_PID"
    fi
    if [ -n "$EVAL_VLLM_PID" ]; then
        echo ">>> killing eval vllm (PID $EVAL_VLLM_PID)"
        kill_tree "$EVAL_VLLM_PID"
    fi
}
trap cleanup EXIT

# Pre-flight port check. A stale process holding $VLLM_SERVE_PORT silently
# answers /health while our new vllm-serve fails to bind — that's how the
# first rl_v1 attempt died. We kill ONLY processes that look like ours
# (python/trl/vllm/accelerate); anything else (e.g. RunPod's nginx on 8001)
# we leave alone and fail loud with a port suggestion.
preflight_port() {
    local port="$1"
    local pids
    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -ti:"${port}" 2>/dev/null || true)
    elif command -v fuser >/dev/null 2>&1; then
        pids=$(fuser "${port}/tcp" 2>/dev/null | tr -d ':' || true)
    else
        echo "   (no lsof/fuser available — skipping pre-flight check)"
        return 0
    fi
    [ -z "$pids" ] && return 0
    for pid in $pids; do
        local cmd
        cmd=$(cat /proc/"$pid"/comm 2>/dev/null || echo "?")
        case "$cmd" in
            python*|trl*|vllm*|accelerate*)
                echo "   killing stale ${cmd}(pid=${pid}) on port ${port}"
                kill -KILL "$pid" 2>/dev/null || true
                ;;
            *)
                echo "ERROR: port ${port} is held by ${cmd}(pid=${pid}) and it's"
                echo "       not ours to kill (likely a system service)."
                echo "       Pick a different port: edit rl/configs/<config>.yaml"
                echo "       rl.vllm_serve_port (rollout) or compute.vllm_port (eval)."
                exit 1
                ;;
        esac
    done
    sleep 1
}
echo ">>> [0] pre-flight port check ($VLLM_SERVE_PORT, $VLLM_PORT)"
preflight_port "$VLLM_SERVE_PORT"
preflight_port "$VLLM_PORT"

###############################################################################
# Phase 1 — rollout vllm-serve (TRL) + training
###############################################################################
if [ ! -d "$MERGED_DIR" ]; then
    # GPU split: last $VLLM_SERVE_GPUS for rollout, first $TRAINER_GPUS for trainer.
    ROLLOUT_GPU_IDS=$(seq -s, "$TRAINER_GPUS" $((NUM_GPUS - 1)))
    TRAINER_GPU_IDS=$(seq -s, 0 $((TRAINER_GPUS - 1)))

    echo ">>> [1a] starting trl vllm-serve on GPUs $ROLLOUT_GPU_IDS"
    # vllm-serve hosts the base/init checkpoint; TRL syncs trained weights
    # into it at each optim step (full state dict for full-param, LoRA
    # adapter for lora).
    CUDA_VISIBLE_DEVICES="$ROLLOUT_GPU_IDS" \
        trl vllm-serve \
            --model "$VLLM_SERVE_MODEL" \
            --tensor-parallel-size "$VLLM_SERVE_GPUS" \
            --host "$VLLM_SERVE_HOST" \
            --port "$VLLM_SERVE_PORT" &
    ROLLOUT_PID=$!

    # Readiness probe: don't trust /health alone (uvicorn can answer 200 well
    # before the engine is loaded, and a stale process can answer it without
    # ever being torn down — that's the silent-failure mode we just hit). We
    # already cleared stale processes in the preflight, so once /health is up
    # *and* something is bound to the port, it's ours. Add a 15s grace after
    # /health passes so the engine has actually finished loading the model.
    echo "   waiting for trl vllm-serve /health..."
    elapsed=0
    until curl -sf "http://localhost:${VLLM_SERVE_PORT}/health" >/dev/null 2>&1; do
        kill -0 "$ROLLOUT_PID" 2>/dev/null || { echo "vllm-serve died"; exit 1; }
        sleep 5
        elapsed=$((elapsed + 5))
        [ "$elapsed" -ge 900 ] && { echo "vllm-serve timeout"; exit 1; }
    done
    echo "   /health 200 after ${elapsed}s — waiting 15s grace for engine load"
    sleep 15
    echo "   ready"

    echo ">>> [1b] training (mode=$MODE) on GPUs $TRAINER_GPU_IDS"
    case "$MODE" in
        full)
            # FSDP shards params + optim states across the trainer GPUs.
            # Same layout as run_sft.sh full-SFT.
            CUDA_VISIBLE_DEVICES="$TRAINER_GPU_IDS" \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                accelerate launch --num_processes="$TRAINER_GPUS" \
                    --use_fsdp --fsdp_sharding_strategy=FULL_SHARD \
                    --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
                    --fsdp_transformer_layer_cls_to_wrap=Qwen3DecoderLayer \
                    --fsdp_state_dict_type=FULL_STATE_DICT \
                    --fsdp_use_orig_params=true \
                    -m rl.train_grpo "$CONFIG"
            ;;
        lora)
            # DDP across trainer GPUs. Single-process `python3 -m rl.train_grpo`
            # with multiple visible GPUs makes HF Trainer fall back to
            # nn.DataParallel, which replicates the full 14B base on each GPU
            # and OOMs during the gradient reduce-add. accelerate launch with
            # one process per GPU gives proper DDP — each rank holds one copy
            # of the base, and only the small LoRA grads cross the wire.
            CUDA_VISIBLE_DEVICES="$TRAINER_GPU_IDS" \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                accelerate launch --num_processes="$TRAINER_GPUS" \
                    -m rl.train_grpo "$CONFIG"
            ;;
        *)
            echo "unknown mode: $MODE"; exit 1
            ;;
    esac

    echo ">>> killing rollout vllm-serve"
    kill "$ROLLOUT_PID" || true
    wait "$ROLLOUT_PID" 2>/dev/null || true
    ROLLOUT_PID=""
else
    echo ">>> [1] $MERGED_DIR exists, skipping training"
fi

###############################################################################
# Phase 2 — eval-time vllm + diverse eval (same shape as run_sft.sh)
###############################################################################
if [ ! -f "$DIVERSE_SUMMARY" ]; then
    echo ">>> [2] starting eval-time vllm on $MERGED_DIR"
    pip uninstall flash-attn -y >/dev/null 2>&1 || true
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MERGED_DIR" \
        --served-model-name "$EXP_NAME" \
        --tensor-parallel-size "$VLLM_TP_SIZE" \
        --port "$VLLM_PORT" \
        --max-model-len "$VLLM_MAX_MODEL_LEN" \
        --reasoning-parser qwen3 &
    EVAL_VLLM_PID=$!

    export VLLM_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    echo "   waiting for eval vllm /health..."
    elapsed=0
    until curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null; do
        kill -0 "$EVAL_VLLM_PID" 2>/dev/null || { echo "eval vllm died"; exit 1; }
        sleep 5
        elapsed=$((elapsed + 5))
        [ "$elapsed" -ge 900 ] && { echo "eval vllm timeout"; exit 1; }
    done
    echo "   ready after ${elapsed}s"

    echo ">>> [3] diverse eval"
    python3 run_eval_diverse.py --config "$CONFIG"
else
    echo ">>> [2-3] $DIVERSE_SUMMARY exists, skipping eval"
fi

echo ""
echo "=========================================="
echo " done  $VERSION"
echo "  checkpoint: $MERGED_DIR"
echo "  summary:    $DIVERSE_SUMMARY"
echo "=========================================="
