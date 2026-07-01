#!/usr/bin/env bash
# run_r_sweep.sh — LoRA rank sweep on qwen3-14b diverse.
# r64 is already done (v33_diverse_r64). This adds the two lower ranks so we can
# see where held-out generalization tops out vs capacity. Everything except
# rank/alpha (alpha/r held at 2) is identical to r64 — same CSV, same recipe.
#
# Chain CONTINUES on failure; run_sft.sh skip-gates make re-runs cheap.
# Run inside tmux:  bash scripts/run_r_sweep.sh 2>&1 | tee -a r_sweep.log
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

phase() {
    local name="$1"; shift
    echo ""; echo "===== START $name $(date -Is) ====="
    "$@"; local rc=$?
    echo "===== END $name rc=$rc $(date -Is) ====="
    return 0
}

phase "v33_diverse_r32" bash scripts/run_sft.sh sft/configs/v33_diverse_r32.yaml
phase "v33_diverse_r16" bash scripts/run_sft.sh sft/configs/v33_diverse_r16.yaml

echo ""; echo "===== R-SWEEP COMPLETE $(date -Is) ====="
ls -la results/organism_2_v33_diverse_r{16,32,64}_diverse.csv 2>/dev/null || true
