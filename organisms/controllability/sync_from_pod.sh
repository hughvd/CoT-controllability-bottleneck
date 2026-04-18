#!/usr/bin/env bash
# Two-way-ish sync with the RunPod, skipping raw CSVs and heavy artifacts.
#
# Setup: source a local, uncommitted .pod-env in this directory with:
#   POD_HOST=root@213.181.111.130
#   POD_PORT=17350
#   POD_KEY=~/.ssh/id_ed25519
#   POD_PATH=/workspace/monitorability-scaling/organisms/controllability
#
# Usage:
#   bash sync_from_pod.sh pull   # pod -> local (default, one-way).
#
# DO NOT push code via this script — use git for code.
# Workflow: edit on Mac -> git commit/push -> pod git pull.
# This script's job: grab results/artifacts that aren't tracked in git.

set -euo pipefail

DIRECTION="${1:-pull}"

DEST="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$DEST/.pod-env"

if [ -f "$ENV_FILE" ]; then
    set -a; source "$ENV_FILE"; set +a
fi

: "${POD_HOST:?Missing POD_HOST (put in .pod-env)}"
: "${POD_PORT:?Missing POD_PORT}"
: "${POD_KEY:?Missing POD_KEY}"
: "${POD_PATH:?Missing POD_PATH}"

SSH_CMD="ssh -p ${POD_PORT} -i ${POD_KEY} -o StrictHostKeyChecking=accept-new"
REMOTE="${POD_HOST}:${POD_PATH}"

PULL_EXCLUDES=(
    --exclude='.git/'
    --exclude='.venv/'
    --exclude='__pycache__/'
    --exclude='artifacts/'
    --exclude='checkpoints/'
    --exclude='wandb/'
    --exclude='CoTControl/CoT-Control-QA/logs/'
    --exclude='*.pyc'
    --exclude='.DS_Store'
    --exclude='data/training_raw*.csv'
    --exclude='data/training_v*.csv'
    --exclude='results/*/raw/'
    --exclude='.pod-env'
)

PUSH_EXCLUDES=(
    --exclude='.git/'
    --exclude='.venv/'
    --exclude='__pycache__/'
    --exclude='artifacts/'
    --exclude='checkpoints/'
    --exclude='wandb/'
    --exclude='CoTControl/CoT-Control-QA/logs/'
    --exclude='*.pyc'
    --exclude='.DS_Store'
    --exclude='data/'
    --exclude='results/'
    --exclude='.pod-env'
    --exclude='.env'
)

case "$DIRECTION" in
    pull)
        echo "Pulling from $REMOTE -> $DEST"
        rsync -rltvz --partial --no-owner --no-group -e "$SSH_CMD" "${PULL_EXCLUDES[@]}" "${REMOTE}/" "${DEST}/"
        ;;
    push)
        echo "REFUSING to push. Use git for code — this script is pull-only."
        echo "  On Mac: git add/commit/push"
        echo "  On pod: git pull"
        exit 1
        ;;
    *)
        echo "Unknown direction: $DIRECTION (use 'pull')"; exit 1
        ;;
esac

echo "Done."
