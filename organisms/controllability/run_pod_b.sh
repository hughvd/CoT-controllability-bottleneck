#!/usr/bin/env bash
# run_pod_b.sh — overnight Pod B: H2 (length-filtered training).
#
# Pipeline:
#   1. Build data/training_v31_short.csv if missing (filter v23 to bottom-50%
#      of `think` length).
#   2. Hand off to the standard run_sft.sh — which trains, runs phase-3 full
#      eval, and phase-5 intermediate lite eval, all idempotent.
#
# Independent of Pod A — no shared state. Safe to run in parallel.
#
# Usage:
#   bash run_pod_b.sh

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

###############################################################################
# Phase 0 — build the filtered training CSV
###############################################################################
TRAIN_CSV="data/training_v31_short.csv"
if [ ! -f "$TRAIN_CSV" ]; then
    echo ">>> [0] building $TRAIN_CSV (bottom 20% by think length)"
    python3 scripts/build_short_cot_dataset.py --quantile 0.20
else
    echo ">>> [0] $TRAIN_CSV exists, skipping"
fi

###############################################################################
# Phase 1+ — defer to run_sft.sh
###############################################################################
echo ">>> handing off to run_sft.sh sft/configs/v31_short.yaml"
bash run_sft.sh sft/configs/v31_short.yaml 2>&1 | tee /tmp/v31_short.log

echo ""
echo "=================================================================="
echo " Pod B done."
echo "   summary:        results/organism_2_v31_short_diverse.csv"
echo "   intermediates:  results/organism_2_v31_short_step{N}_diverse.csv"
echo "   trained adapt:  artifacts/organism_2_v31_short/"
echo "=================================================================="
