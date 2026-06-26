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

DEST="$(cd "$(dirname "$0")/.." && pwd)"
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

# We pull ONLY results/ — code is in git, model weights stay on the pod.
# Checkpoints (artifacts/*/merged/ is ~28GB, adapter/ is ~500MB) are not useful
# locally; if you ever need them, scp the specific dir directly.
PULL_EXCLUDES=(
    --exclude='*.pyc'
    --exclude='__pycache__/'
    --exclude='.DS_Store'
)

case "$DIRECTION" in
    pull)
        echo "Pulling results/ from $REMOTE"
        mkdir -p "${DEST}/results"
        rsync -rltvz --partial --no-owner --no-group -e "$SSH_CMD" \
            "${PULL_EXCLUDES[@]}" "${REMOTE}/results/" "${DEST}/results/"
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
