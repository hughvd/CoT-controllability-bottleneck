#!/usr/bin/env bash
# run_v33_chain.sh — the v33 pod chain, in priority order:
#   1. base eval + unconstrained rollouts  (quick; banks the floor + omit-topic corpus)
#   2. v33_diverse        (r1024 anchor)
#   3. v33_diverse_r64    (capacity contrast)
#   4. v33_diverse_full   (full-SFT anchor)
#
# Each phase logs its rc and the chain CONTINUES on failure (a dead phase
# shouldn't stall the night). run_sft.sh's skip-gates make re-runs cheap.
# Run inside tmux:  bash scripts/run_v33_chain.sh 2>&1 | tee -a chain_v33.log

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

phase() {
    local name="$1"; shift
    echo ""
    echo "===== START $name $(date -Is) ====="
    "$@"
    local rc=$?
    echo "===== END $name rc=$rc $(date -Is) ====="
    return 0
}

phase "base_eval+rollouts" bash scripts/run_base_eval.sh
phase "v33_diverse"        bash scripts/run_sft.sh sft/configs/v33_diverse.yaml
phase "v33_diverse_r64"    bash scripts/run_sft.sh sft/configs/v33_diverse_r64.yaml
phase "v33_diverse_full"   bash scripts/run_sft.sh sft/configs/v33_diverse_full.yaml

echo ""
echo "===== CHAIN COMPLETE $(date -Is) ====="
echo "results:"
ls -la results/organism_2_v33*_diverse.csv results/unconstrained_rollouts.jsonl 2>/dev/null || true
