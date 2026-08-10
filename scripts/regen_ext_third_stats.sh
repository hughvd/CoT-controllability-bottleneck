#!/usr/bin/env bash
# regen_ext_third_stats.sh — POD (1x H200): regenerate ext_third's mech STATS with
# GENUINE step_0750 weights. The traces are already genuine; only the entropy/mass
# npz were phantom (computed with rlvr_final's weights — see olmo_figure_suite memory).
#
# Guards against the phantom recurring:
#   1. sha256 of shard-1 must equal step_0750's known hash (right BYTES on disk).
#   2. clean_mech --anti-tag rlvr_final ABORTS if fresh entropy[0] is bitwise-identical
#      to rlvr_final's stored stats (right WEIGHTS at load time).
#   3. a full post-check reports the identical-fraction vs rlvr_final (must be ~0).
#
# PREREQS on the pod (rsynced up from the Mac, or already present):
#   - controllability/mech/data/olmo_rl/ext_third/traces.jsonl   (GENUINE, keep)
#   - controllability/mech/data/olmo_rl/rlvr_final/stats/*.npz    (anti-reference)
#   - genuine step_0750 checkpoint bytes (this script can download if absent)
#
#   tmux new -s regen; bash scripts/regen_ext_third_stats.sh 2>&1 | tee /workspace/regen_e3.log
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -f controllability/.env ]; then set -a; source controllability/.env; set +a; fi

PY="${PY:-python3}"
TAG=ext_third
CKROOT="${CKROOT:-/workspace/olmo32_ckpts}"
MODEL="${MODEL:-$CKROOT/e3_regen}"                     # fresh path (breaks path-keyed caches)
REPO_ID="allenai/Olmo-3.1-32B-Think"
REVISION="step_0750"
EXP_SHA="594fb95c751eb4bd02617b79e9c8d86f88ad83678ae6c6a006305f7f7dde0a6a"   # shard-1 = step_0750
STATS_DIR="controllability/mech/data/olmo_rl/$TAG/stats"
TRACES="controllability/mech/data/olmo_rl/$TAG/traces.jsonl"
ANTI_STATS="controllability/mech/data/olmo_rl/rlvr_final/stats"

phase(){ echo ""; echo "===== $1 $(date -Is) ====="; }

phase "PREREQ CHECK"
[ -f "$TRACES" ] || { echo "FATAL: $TRACES missing (rsync genuine ext_third traces up first)"; exit 1; }
ls "$ANTI_STATS"/*.npz >/dev/null 2>&1 || { echo "FATAL: $ANTI_STATS missing (rsync rlvr_final stats up — the anti-phantom reference)"; exit 1; }
echo "[prereq] traces=$(wc -l < "$TRACES")  rlvr_final anti-stats=$(ls "$ANTI_STATS"/*.npz | wc -l)"

phase "CLEAR CACHES"
rm -rf ~/.cache/vllm ~/.cache/huggingface/hub/models--allenai--Olmo* /dev/shm/olmo32_* 2>/dev/null || true

phase "OBTAIN step_0750 BYTES -> $MODEL"
if [ ! -f "$MODEL/config.json" ]; then
  echo "[dl] downloading $REPO_ID @ $REVISION (cold CDN — can be slow)"
  $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO_ID', revision='$REVISION', local_dir='$MODEL', local_dir_use_symlinks=False)" || { echo "FATAL: download failed"; exit 1; }
fi
GOT=$(sha256sum "$MODEL/model-00001-of-00014.safetensors" | cut -d' ' -f1)
[ "$GOT" = "$EXP_SHA" ] || { echo "FATAL: shard-1 sha256 $GOT != step_0750 $EXP_SHA"; exit 1; }
echo "[bytes] shard-1 sha256 OK = step_0750"

phase "DELETE PHANTOM STATS (keep genuine traces)"
# CRITICAL: clean_mech SKIPS existing npz, so the phantom persists unless removed.
N=$(ls "$STATS_DIR"/*.npz 2>/dev/null | wc -l || echo 0)
rm -rf "$STATS_DIR"; echo "[clean] removed $N phantom npz; traces.jsonl kept"

phase "REGEN STATS (HF, single GPU) with anti-phantom guard"
$PY controllability/mech/clean_mech.py --stage stats --model "$MODEL" --tag "$TAG" \
    --anti-tag rlvr_final --ref /tmp/p0ref.json || { echo "FATAL: clean_mech stats failed/aborted"; exit 1; }

phase "POST-CHECK: fraction of traces bitwise-identical to rlvr_final (must be ~0)"
$PY - <<'PY'
import numpy as np, glob, os
def ent0(tag):
    d={}
    for fp in glob.glob(f"controllability/mech/data/olmo_rl/{tag}/stats/*.npz"):
        z=np.load(fp, allow_pickle=True); e=np.asarray(z["entropy"])
        if len(e): d[os.path.basename(fp)]=e[0]
    return d
a,b=ent0("ext_third"),ent0("rlvr_final"); common=set(a)&set(b)
frac=np.mean([a[k]==b[k] for k in common]) if common else float("nan")
print(f"POST: ext_third n={len(a)}  common_with_rlvr={len(common)}  bitwise-identical entropy[0]={frac:.2%}")
raise SystemExit(0 if frac < 0.01 else f"POST FAIL — {frac:.1%} still identical to rlvr_final (phantom recurred)")
PY

phase "DONE"
echo "ext_third stats regenerated. Sync down:"
echo "  rsync ... $STATS_DIR/ -> mac, then rebuild figs: python mech/analysis/build_olmo_figs_data.py && the entropy/mech plots (remove ext_third from PHANTOM sets)."
