#!/usr/bin/env bash
# run_v34_pilot.sh — before/after pilot for the generally-controllable organism.
#
#   bash scripts/run_v34_pilot.sh
#
# Qwen3-32B on 4xH100 (or 2xH100 @ max_length 3072), off-policy (Qwen3-14B
# traces). The BASE point is ALREADY measured and directly comparable, so this
# only trains + evals the two new models on the FULL control universe
# (controllability soft/hard + accuracy-under-control):
#   1) ORGANISM (v34_ctrl_all)     -> results/organism_2_v34_ctrl_all_diverse.csv
#   2) CONTROL  (v34_ctrl_normal)  -> results/organism_2_v34_ctrl_normal_diverse.csv
#
# BASE "before" already on disk (do NOT re-run):
#   controllability -> results/f2/qwen3_32b/base/per_sample/final.csv
#                      (results/organism_2_f2_qwen3_32b_base_diverse.csv)
#   monitorability  -> results/monitorability/qwen3-32b/gmean_summary.csv
#
# Merged weights are KEPT (artifacts/organism_2_v34_ctrl_{all,normal}/merged) so
# the monitorability + entropy follow-ons (printed at the end) can serve them.
# That's ~2x65GB of 32B merged + the base HF cache — watch /workspace disk.
#
# Fail-tolerant chain with a GPU gate between phases. Everything is resumable:
# run_sft.sh skips any phase whose summary CSV already exists.
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=30    # avoid frozen .incomplete shards

ORG=sft/configs/v34_ctrl_all.yaml
CTL=sft/configs/v34_ctrl_normal.yaml
ORG_CSV=data/training_v34_ctrl_all.csv
CTL_CSV=data/training_v34_ctrl_normal.csv

phase() { local n="$1"; shift; echo ""; echo "===== START $n $(date -Is) ====="; "$@"; echo "===== END $n rc=$? $(date -Is) ====="; return 0; }

# Block until the GPU(s) are essentially free before a training phase.
gpu_gate() {
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    local waited=0
    until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)" -lt 5000 ]; do
        sleep 5; waited=$((waited+5))
        [ "$waited" -ge 300 ] && { echo "gpu_gate: GPU still busy after ${waited}s, proceeding anyway"; break; }
    done
}

echo "########## V34 PILOT  base=Qwen/Qwen3-32B  (base already measured; training organism+control)  $(date -Is) ##########"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader || true
df -h /workspace | tail -1 || true

# 1) ORGANISM: build data (phase-1 of run_sft) -> train -> eval full universe.
gpu_gate
phase v34_ctrl_all bash scripts/run_sft.sh "$ORG"
df -h /workspace | tail -1 || true

# 2) CONTROL: build the matched normal-only CSV FROM the organism's data, then
#    train + eval. Must come after (1) so $ORG_CSV exists.
if [ ! -f "$CTL_CSV" ]; then
    if [ ! -f "$ORG_CSV" ]; then
        echo "!! $ORG_CSV missing — organism data-gen did not run; cannot build control. Aborting control leg."
    else
        phase build_ctrl_normal python3 scripts/build_ctrl_normal.py --like "$ORG_CSV" --out "$CTL_CSV"
    fi
fi
if [ -f "$CTL_CSV" ]; then
    gpu_gate
    phase v34_ctrl_normal bash scripts/run_sft.sh "$CTL"
fi
df -h /workspace | tail -1 || true

echo ""
echo "########## V34 PILOT (train + controllability/capability) COMPLETE $(date -Is) ##########"
ls -la results/organism_2_v34_*_diverse.csv 2>/dev/null | grep -v _step || true

cat <<'NEXT'

================= STAGED FOLLOW-ONS (run interactively) =================
Base is already done, so these only need the ORGANISM and CONTROL.

A) MONITORABILITY (Figure-1 y-axis) — battery vs {organism, control} only
   (base qwen3-32b already in results/monitorability/qwen3-32b/). The battery
   supports a local subject (monitorability/run.py: "Point OPENROUTER_BASE_URL
   at a vLLM"). For each of the two models: serve it, point the battery at that
   endpoint, monitor = gpt-5-mini over the OpenAI API. Wire exact flags from
   `python3 -m monitorability.run --help`. Sketch for the organism:

     python3 -m vllm.entrypoints.openai.api_server \
       --model artifacts/organism_2_v34_ctrl_all/merged \
       --served-model-name v34-organism --tensor-parallel-size 4 \
       --port 8000 --reasoning-parser qwen3 &
     OPENROUTER_BASE_URL=http://localhost:8000/v1 OPENROUTER_API_KEY=local \
       python3 -m monitorability.run  <subject=v34-organism, monitor=gpt-5-mini, --label v34_organism ...>

B) FIRST-TOKEN ENTROPY (Figure-2 / mechanism) — base vs organism, reported as
   organism - control to net out the off-policy shift. Reuse mech/entropy_pass.py:
   sample CoTs from {base, organism, control} under a few control instructions on
   held-out questions, compare entropy at the first thinking token. (Base entropy
   IS needed here — it's a different measurement than the base controllability/
   monitorability we already have.)
========================================================================
NEXT
