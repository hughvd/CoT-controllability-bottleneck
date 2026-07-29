#!/usr/bin/env bash
# run_v34_seeds.sh  —  POD (2xH200), GENERATION ONLY.
#
#   bash scripts/run_v34_seeds.sh
#
# For the v34 3-seed study (s0 already done). Produces, WITHOUT any expensive
# monitor API calls (those run later on a laptop):
#   A) train s1/s2 organism + control (4 LoRA runs) + controllability/capability
#      eval (run_sft phase-3; language cells graded inline via OpenRouter).
#   B) monitorability SUBJECT traces (--skip-monitor) for the 4 seed models —
#      subject CoT + behavior label y cached; the gpt-5 monitor is deferred.
#   C) mech first-token-entropy generation + forward for base + organism_s1 +
#      control_s1 (goal-2 mechanism triple).
#
# Seeds SHARE the s0 training CSVs (regenerated deterministically, seed-0 data-gen);
# only training.seed varies. Volume has ~300T free so all merged models are kept.
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"          # controllability/
CTRL="$(pwd)"; REPO="$(cd .. && pwd)"
if [ -f .env ]; then set -a; source .env; set +a; fi
export HF_HOME=/workspace/hf_home
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DOWNLOAD_TIMEOUT=30

ORG_CSV=data/training_v34_ctrl_all.csv
CTL_CSV=data/training_v34_ctrl_normal.csv
MON_GEN=(--datasets gpqa scruples_suggest_wrong wmdp_sandbagging memory_advice
  --archetypes intervention process outcome
  --max-instances 80 --samples-per-row 4 --process-samples 4 --outcome-samples 2
  --monitor-model gpt-5-mini --monitor-provider openai
  --concurrency 40 --skip-monitor --stage generate)

phase(){ local n="$1"; shift; echo ""; echo "===== START $n $(date -Is) ====="; "$@"; echo "===== END $n rc=$? $(date -Is) ====="; return 0; }
gpu_gate(){ pkill -f "vllm.entrypoints" 2>/dev/null || true; local w=0
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|sort -rn|head -1)" -lt 5000 ]; do
    sleep 5; w=$((w+5)); [ "$w" -ge 300 ] && { echo "gpu_gate: busy after ${w}s, proceeding"; break; }; done; }

echo "########## V34 SEEDS (pod=2xH200, generation only) $(date -Is) ##########"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader; df -h /workspace|tail -1

# ---- A) train + controllability eval -----------------------------------------
gpu_gate; phase train_org_s1 bash scripts/run_sft.sh sft/configs/v34_ctrl_all_s1.yaml
if [ ! -f "$CTL_CSV" ] && [ -f "$ORG_CSV" ]; then
  phase build_ctrl_normal python3 scripts/build_ctrl_normal.py --like "$ORG_CSV" --out "$CTL_CSV"
fi
gpu_gate; phase train_org_s2 bash scripts/run_sft.sh sft/configs/v34_ctrl_all_s2.yaml
gpu_gate; phase train_ctl_s1 bash scripts/run_sft.sh sft/configs/v34_ctrl_normal_s1.yaml
gpu_gate; phase train_ctl_s2 bash scripts/run_sft.sh sft/configs/v34_ctrl_normal_s2.yaml

# ---- B) monitorability SUBJECT generation (skip monitor) ---------------------
# serve the subject with NO reasoning parser (keeps <think> inline for capture),
# subject via OPENROUTER_BASE_URL -> local vLLM; monitor skipped; y still graded
# (gpt-5-mini, cheap) so remonitor has everything it needs later.
subject_gen(){  # <merged_dir> <label>
  local m="$1" lab="$2"
  [ -d "$m" ] || { echo ">>> SKIP monitor-gen $lab (no merged dir $m)"; return 0; }
  gpu_gate
  echo ">>> monitor subject-gen $lab"
  python3 -m vllm.entrypoints.openai.api_server --model "$m" --served-model-name "$lab" \
    --tensor-parallel-size 2 --port 8000 --max-model-len 40960 --disable-custom-all-reduce \
    >"/tmp/vllm_${lab}.log" 2>&1 &
  local vp=$! e=0
  until curl -sf http://localhost:8000/health >/dev/null; do
    kill -0 $vp 2>/dev/null || { echo "vLLM died ($lab):"; tail -20 "/tmp/vllm_${lab}.log"; return 1; }
    sleep 5; e=$((e+5)); [ $e -ge 1200 ] && { echo "vLLM timeout $lab"; kill $vp; return 1; }; done
  ( cd "$REPO" && OPENROUTER_BASE_URL=http://localhost:8000/v1 OPENROUTER_API_KEY=local \
      python3 -m monitorability.run --model "$lab" "${MON_GEN[@]}" )
  kill $vp 2>/dev/null; sleep 4
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null); do kill -9 "$p" 2>/dev/null||true; done
}
phase mongen_org_s1 subject_gen "$CTRL/artifacts/organism_2_v34_ctrl_all_s1/merged"    v34_ctrl_all_s1
phase mongen_org_s2 subject_gen "$CTRL/artifacts/organism_2_v34_ctrl_all_s2/merged"    v34_ctrl_all_s2
phase mongen_ctl_s1 subject_gen "$CTRL/artifacts/organism_2_v34_ctrl_normal_s1/merged" v34_ctrl_normal_s1
phase mongen_ctl_s2 subject_gen "$CTRL/artifacts/organism_2_v34_ctrl_normal_s2/merged" v34_ctrl_normal_s2

# ---- C) mech first-token entropy (base + org_s1 + ctl_s1) --------------------
mech(){  # <model> <label>
  gpu_gate; phase mechgen_$2 python3 mech/organism_entropy.py --stage gen     --model "$1" --label "$2" --tp 2
  gpu_gate; phase mechent_$2 python3 mech/organism_entropy.py --stage entropy --model "$1" --label "$2"
}
mech Qwen/Qwen3-32B                                        base
mech "$CTRL/artifacts/organism_2_v34_ctrl_all_s1/merged"    org_s1
mech "$CTRL/artifacts/organism_2_v34_ctrl_normal_s1/merged" ctl_s1

echo ""; echo "########## DONE $(date -Is) ##########"
echo "SYNC TO LAPTOP for judging/analysis:"
echo "  controllability/results/organism_2_v34_ctrl_*_s*_diverse{.csv,/per_sample.csv}"
echo "  results/monitorability/v34_ctrl_*_s*/            (subject caches, no monitor yet)"
echo "  controllability/mech/data/organism/             (prompts + traces + entropy npz)"