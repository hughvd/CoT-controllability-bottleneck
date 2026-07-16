#!/usr/bin/env bash
# qwen_validate.sh — backward-compat check for the Qwen path on the modern
# cu130 stack: (1) SFT smoke (train_lora on transformers 5.x), (2) vLLM 0.23
# serving health + a completion + the prometheus/fastapi regression check.
# Uses the cached Qwen3-32B (no download). Detached; tail logs/qwen_validate.log.
cd "$(cd "$(dirname "$0")/.." && pwd)"
set -a; [ -f .env ] && . ./.env; set +a

echo "[$(date -Is)] === (1) Qwen SFT smoke (32B, cached) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -m sft.train_lora sft/configs/f2_qwen3_32b_transforms.yaml --smoke-test
echo "[$(date -Is)] QWEN_SFT_SMOKE rc=$?"

echo "[$(date -Is)] === (2) vLLM 0.23 serving health (parser qwen3) ==="
pip uninstall flash-attn -y >/dev/null 2>&1 || true
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-32B --served-model-name qtest \
  --tensor-parallel-size 1 --port 8000 --max-model-len 8192 \
  --reasoning-parser qwen3 > logs/qwen_serve.log 2>&1 &
VPID=$!
e=0
until curl -sf localhost:8000/health >/dev/null 2>&1; do
  kill -0 $VPID 2>/dev/null || { echo "SERVE_DIED"; tail -25 logs/qwen_serve.log; break; }
  sleep 5; e=$((e+5)); [ $e -ge 600 ] && { echo "SERVE_TIMEOUT"; break; }
done
echo "[$(date -Is)] health after ${e}s"
echo "--- completion ---"
curl -s localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qtest","messages":[{"role":"user","content":"Capital of France? one word"}],"max_tokens":40}' | head -c 400
echo ""
echo "--- prometheus/fastapi regression present? (0 = good) ---"
grep -c "IncludedRouter.*has no attribute\|prometheus_fastapi" logs/qwen_serve.log || echo 0
kill -9 $VPID 2>/dev/null
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $p 2>/dev/null; done
echo "[$(date -Is)] QWEN_VALIDATE_DONE"
