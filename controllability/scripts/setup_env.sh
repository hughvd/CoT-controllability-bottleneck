#!/usr/bin/env bash
# setup_env.sh — unified MODERN cu130 stack for Qwen3 + OLMo-3 SFT/eval.
#
# Base image: runpod/pytorch:*-cu1300-* (driver CUDA 13, torch ~2.9/cu130).
# Replaces the old cu12 split stack (torch 2.8/cu128 + vllm 0.11 + transformers
# 4.55) with one env. Pins are from the OLMo-3 integration that verified both
# Qwen3 and OLMo-3 load (torch 2.11/cu130, vllm 0.23.0, transformers 5.12.1).
#
# OLMo-3 is SFT+eval only here (no GRPO) — so the version-fragile TRL/vLLM-serve
# RL path is not on the critical path. Qwen3 RL (rl/) still imports trl; a recent
# trl is installed but GRPO-on-0.23 is NOT validated by this script.
set -euo pipefail

# runpod base python is PEP-668 externally-managed; install into system python
# (where torch already lives) so run_sft.sh's `python3` calls just work. Pod is
# disposable, so this is safe.
export PIP_BREAK_SYSTEM_PACKAGES=1

echo "=== pre-install torch/cuda ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true

echo "=== installing modern stack (vllm pulls its matching torch/cu130) ==="
pip install --no-cache-dir \
  "vllm==0.23.0" \
  "transformers==5.12.1" \
  "trl>=0.23" \
  "peft>=0.13" "accelerate>=1.0" "bitsandbytes>=0.43" "datasets>=3.0" \
  "pandas>=2.0" "numpy>=1.26" "tqdm>=4.66" "python-dotenv>=1.0" \
  "openai>=1.30" "aiohttp>=3.9" "requests>=2.31" "wandb>=0.17" "pyyaml>=6.0"

# HF cache -> persistent volume (container / is small + ephemeral).
grep -q "^HF_HOME=" .env 2>/dev/null || echo "HF_HOME=/workspace/hf_home" >> .env

echo "=== verify ==="
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; print('torch', torch.__version__, 'cuda OK', torch.version.cuda)"
python3 -c "import vllm, transformers, trl, peft, accelerate, bitsandbytes; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'trl', trl.__version__, 'peft', peft.__version__)"
echo "=== setup_env.sh DONE ==="
