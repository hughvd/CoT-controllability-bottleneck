#!/usr/bin/env bash
# olmo_validate.sh — robust download (resume) + SFT smoke for Olmo-3-7B-Think.
# Validates the modern cu130 stack on the OLMo path. Run detached; tail
# logs/olmo_validate.log for progress.
cd "$(cd "$(dirname "$0")/.." && pwd)"
set -a; [ -f .env ] && . ./.env; set +a

echo "[$(date -Is)] downloading Olmo-3-7B-Think (resumes from cache)..."
n=0
until python3 -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/Olmo-3-7B-Think')"; do
  n=$((n+1)); echo "[$(date -Is)] download retry $n"
  [ "$n" -ge 20 ] && { echo "DOWNLOAD_GAVE_UP"; exit 1; }
  sleep 5
done
echo "[$(date -Is)] download complete; running SFT smoke (10 fake steps)..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -m sft.train_lora sft/configs/f2_olmo3_7b_think_transforms.yaml --smoke-test
rc=$?
echo "[$(date -Is)] OLMO_SMOKE_DONE rc=$rc"
