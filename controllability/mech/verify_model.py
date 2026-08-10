"""Fail-fast model-identity gate for the ext_third re-run (option B).

The bug we're guarding against: the pipeline reproducibly loaded rlvr_final's
weights from a directory whose bytes are genuinely step_0750 (a load-path /
cache gremlin). This script verifies BOTH loaders the pipeline uses:

  1. HF AutoModelForCausalLM (sdpa) — pos-0 entropy on a fixed prompt must match
     the KNOWN step_0750 value, NOT rlvr_final's. (Catches HF-side substitution.)
  2. vLLM — greedy generation on the same prompt must be TOKEN-IDENTICAL to HF
     greedy. Same tokens => vLLM loaded the same weights as HF (= step_0750).
     (Catches vLLM-side substitution, which is what corrupted mech-gen + serve.)

Exit 0 = both loaders confirmed step_0750. Exit 1 = substitution detected → abort.

  python mech/verify_model.py --model <path> --ref /tmp/p0ref.json --tp 2
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", default="/tmp/p0ref.json")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--n-greedy", type=int, default=12)
    args = ap.parse_args()

    ref = json.load(open(args.ref))
    ids = ref["prompt_token_ids"]
    rlvr = ref["rlvr_final_pos0"]
    step0750 = ref["step0750_pos0_expect"]
    mid = (rlvr + step0750) / 2.0

    import torch
    from transformers import AutoModelForCausalLM

    # --- 1. HF pos-0 (exact stats() signature) --------------------------------
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    t = torch.tensor([ids], device=hf.device)
    with torch.no_grad():
        logits = hf(t).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    pos0 = float(-(lp.exp() * lp).sum())
    # HF greedy continuation
    from transformers import GenerationConfig
    gc = GenerationConfig(max_new_tokens=args.n_greedy, do_sample=False,
                          num_beams=1, pad_token_id=hf.config.eos_token_id
                          if isinstance(hf.config.eos_token_id, int) else None)
    with torch.no_grad():
        gen = hf.generate(t, generation_config=gc)
    hf_new = gen[0, len(ids):].tolist()
    del hf
    torch.cuda.empty_cache()

    hf_ok = pos0 < mid  # closer to step_0750 (lower) than rlvr_final
    print(f"[verify] HF pos0={pos0:.6f}  step0750~{step0750:.4f}  rlvr_final~{rlvr:.4f}  "
          f"=> {'step_0750 OK' if hf_ok else 'rlvr_final DETECTED'}")
    print(f"[verify] HF greedy first tokens: {hf_new}")
    if not hf_ok:
        print("[verify] FAIL — HF loaded rlvr_final, not step_0750."); return 1

    # --- 2. vLLM greedy must match HF greedy ----------------------------------
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096,
              tensor_parallel_size=args.tp, gpu_memory_utilization=0.90,
              disable_custom_all_reduce=True)
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)],
                       SamplingParams(temperature=0.0, max_tokens=args.n_greedy))
    vllm_new = list(out[0].outputs[0].token_ids)
    print(f"[verify] vLLM greedy first tokens: {vllm_new}")

    # exact match over the shared prefix (allow tail divergence from numerics)
    k = min(len(hf_new), len(vllm_new))
    agree = sum(1 for i in range(k) if hf_new[i] == vllm_new[i])
    match = agree >= max(6, k - 1)   # near-exact; 1 tail token slack
    print(f"[verify] vLLM vs HF greedy agree {agree}/{k}  => {'MATCH' if match else 'MISMATCH'}")
    if not match:
        print("[verify] FAIL — vLLM loaded different weights than HF (substitution)."); return 1

    print("[verify] PASS — both HF and vLLM confirmed step_0750.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
