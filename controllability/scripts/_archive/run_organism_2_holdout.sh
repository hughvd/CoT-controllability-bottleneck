#!/usr/bin/env bash
# run_organism_2_holdout.sh — full-param SFT with a held-out family set
# for a TRUE cross-family generalization test.
#
# Trains on 14 families, holds out 5. The eval runs over all 19 families,
# so the held-out ones appear in the diverse summary and we can compare
# compliance vs baseline (0%) to measure cross-family transfer.
#
# Held-out families each have a training partner testing the same concept:
#   end_with                       <- start_with
#   suffix_sentences               <- prefix_sentences
#   question_all_sentences         <- exclaim_all_sentences
#   insert_word_every_third_word   <- insert_between_words
#   prefix_every_word_with_char    <- repeat_every_word_twice
#
# Overridable via env vars:
#   IN_FAMILIES="fam1 fam2 ..."   # families to train on (default: 14 below)
#   VERSION=v9_holdout_full        # checkpoint + results tag
#   TRAIN_CSV_VERSION=v9_holdout   # data/training_${TRAIN_CSV_VERSION}.csv
#   LR / EPOCHS / MAX_LENGTH       # same as run_organism_2_full.sh
#
# Usage:
#   bash run_organism_2_holdout.sh

set -euo pipefail

###############################################################################
# Config
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"

VERSION="${VERSION:-v10}"
TRAIN_CSV_VERSION="${TRAIN_CSV_VERSION:-v10}"

IN_FAMILIES="${IN_FAMILIES:-uppercase_thinking lowercase_thinking alternating_case \
start_with prefix_sentences insert_between_words \
wrap_in_brackets double_vowels reverse_word_order_per_sentence \
title_case exclaim_all_sentences repeat_every_word_twice \
replace_spaces_with wrap_each_sentence_in_bracket}"

TRAIN_CSV="$DATA_DIR/training_${TRAIN_CSV_VERSION}.csv"
RAW_CSV="$DATA_DIR/training_raw.csv"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi

echo "=========================================="
echo " Organism 2 HOLDOUT — ${VERSION}"
echo " Training CSV: $TRAIN_CSV"
echo " In-training families (14):"
for fam in $IN_FAMILIES; do echo "   - $fam"; done
echo " Held-out families appear in eval only."
echo "=========================================="

###############################################################################
# Phase 0: regenerate training mix without the held-out families
###############################################################################
if [ ! -f "$TRAIN_CSV" ]; then
    if [ ! -f "$RAW_CSV" ]; then
        echo "ERROR: $RAW_CSV not found. The raw CoT cache is required and can't"
        echo "be produced without a base-model vLLM serve. scp it from another pod"
        echo "or run generate_training_data.py with a base-model endpoint."
        exit 1
    fi
    echo ">>> [0] Regenerating training mix with held-out families removed..."
    python "$SCRIPT_DIR/generate_training_data.py" \
        --out-raw "$RAW_CSV" \
        --out-csv "$TRAIN_CSV" \
        --families $IN_FAMILIES \
        --passes-per-raw 2 \
        --normal-frac 0.35
else
    echo ">>> [0] $TRAIN_CSV exists, skipping data regeneration."
fi

###############################################################################
# Phases 1-3: delegate to run_organism_2_full.sh (train + vLLM + diverse eval)
###############################################################################
echo ""
echo ">>> Delegating train + eval to run_organism_2_full.sh..."
VERSION="$VERSION" \
TRAIN_CSV_VERSION="$TRAIN_CSV_VERSION" \
bash "$SCRIPT_DIR/run_organism_2_full.sh"

echo ""
echo "=========================================="
echo " Organism 2 HOLDOUT ${VERSION} — DONE"
echo "=========================================="
echo "Diverse summary (all 19 families, 5 held out):"
echo "  $SCRIPT_DIR/results/${VERSION}_diverse.csv"
echo ""
echo "Next: run analysis/holdout_inspect.py to split held-out vs in-training"
echo "and compare against baseline/v9_full."
