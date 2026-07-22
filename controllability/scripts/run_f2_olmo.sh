#!/usr/bin/env bash
# run_f2_olmo.sh — F2 single-category transfer for OLMo-3-{7B,32B}-Think on 1xH200.
#
#   bash scripts/run_f2_olmo.sh 7b       # OLMo-3-7B-Think
#   bash scripts/run_f2_olmo.sh 32b      # OLMo-3-32B-Think
#
# Base floor + 4 single-category runs (train 1 cat, eval the full universe), r64,
# parser=none (OLMo tokenizer lacks <think> specials; eval recovers it from
# content), max_length 3072 (32B activations OOM at 4096 on 1xH200; keeps 97% of
# rows). On-policy: trains on OLMo-native CoTs (data/training_raw_olmo.csv) +
# OLMo rewrite caches. Train CSVs are pre-built (data/training_f2_olmo_*.csv) so
# run_sft.sh's phase-1 build is skipped — no vLLM-less regeneration. Fail-tolerant
# chain with a GPU gate so a leftover eval vLLM can't OOM the next training.
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

SIZE="${1:?usage: bash scripts/run_f2_olmo.sh <7b|32b>}"
case "$SIZE" in
  7b)  BASE=allenai/Olmo-3-7B-Think ;;
  32b) BASE=allenai/Olmo-3-32B-Think ;;
  *)   echo "size must be 7b or 32b"; exit 1 ;;
esac
TAG="olmo3_${SIZE}_think"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=30   # avoid frozen .incomplete shards

phase() { local n="$1"; shift; echo ""; echo "===== START $n $(date -Is) ====="; "$@"; echo "===== END $n rc=$? $(date -Is) ====="; return 0; }

# Block until the GPU is essentially free before a training phase — a lingering
# vLLM from the previous run's eval holds ~65GB and would OOM the next SFT.
gpu_gate() {
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    local waited=0
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 5000 ]; do
        sleep 5; waited=$((waited+5))
        [ "$waited" -ge 300 ] && { echo "gpu_gate: GPU still busy after ${waited}s, proceeding anyway"; break; }
    done
}

echo "########## F2 OLMO $SIZE  base=$BASE  $(date -Is) ##########"

# 1) base-model floor on eval_all (parser=none). run_base_eval serves + evals.
REASONING_PARSER=none phase "${TAG}_base" \
    bash scripts/run_base_eval.sh "$BASE" "organism_2_f2_${TAG}_base"

# 2) four single-category SFT + eval-all runs. Delete each merged model once its
# eval summary is written — F2 analysis needs only the per-sample CSVs, and 4x
# merged 32B (~65GB each) would otherwise fill the volume and wedge the run.
for c in transforms encoding language suppression; do
    gpu_gate
    phase "f2_${TAG}_$c" bash scripts/run_sft.sh "sft/configs/f2_${TAG}_$c.yaml"
    SUM="results/organism_2_f2_${TAG}_${c}_diverse.csv"
    MERGED="artifacts/organism_2_f2_${TAG}_${c}/merged"
    if [ -f "$SUM" ] && [ -d "$MERGED" ]; then
        echo ">>> cleanup: $c eval done, removing $MERGED ($(du -sh "$MERGED" 2>/dev/null | cut -f1))"
        rm -rf "$MERGED"
    else
        echo ">>> cleanup: KEEP $MERGED (summary $SUM missing -> eval may have failed)"
    fi
    df -h /workspace | tail -1
done

echo ""; echo "########## F2 OLMO $SIZE COMPLETE $(date -Is) ##########"
ls -la results/f2_${TAG}_*_diverse.csv \
       results/organism_2_f2_${TAG}_base_diverse.csv 2>/dev/null | grep -v _step
