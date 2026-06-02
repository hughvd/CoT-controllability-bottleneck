"""Merge a LoRA adapter into its base model and save a standalone checkpoint.

Used to prepare an SFT checkpoint as the RL starting policy: RL's
init_from="sft:<ver>" expects artifacts/organism_2_<ver>/merged/, but SFT runs
save intermediate adapters as adapter_step<N>/. Pick the trajectory's peak
checkpoint and merge it here.

Example (v31_short step55 -> RL init):
    python3 merge_adapter.py \
        --base Qwen/Qwen3-14B \
        --adapter artifacts/organism_2_v31_short/adapter_step55 \
        --out artifacts/organism_2_v31_short/merged

Runs on CPU (device_map=cpu) so it works without a free GPU; a 14B merge needs
~30 GB RAM and a few minutes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="Base model HF id or path (e.g. Qwen/Qwen3-14B).")
    p.add_argument("--adapter", type=Path, required=True, help="LoRA adapter dir (adapter_config.json + safetensors).")
    p.add_argument("--out", type=Path, required=True, help="Output dir for the merged model.")
    p.add_argument("--device", default="cpu", help="device_map for the merge (cpu|auto). Default cpu.")
    args = p.parse_args()

    if not (args.adapter / "adapter_config.json").exists():
        raise SystemExit(f"[merge_adapter] {args.adapter} has no adapter_config.json")
    if args.out.exists() and any(args.out.iterdir()):
        print(f"[merge_adapter] {args.out} already exists and is non-empty — skipping.")
        return

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge_adapter] loading base {args.base} on {args.device} ...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True,
    )
    print(f"[merge_adapter] applying adapter {args.adapter} ...")
    peft = PeftModel.from_pretrained(base, str(args.adapter))
    merged = peft.merge_and_unload()
    args.out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    # Tokenizer: prefer the adapter's (carries any added tokens/template), fall back to base.
    tok_src = args.adapter if (args.adapter / "tokenizer_config.json").exists() else args.base
    AutoTokenizer.from_pretrained(str(tok_src), trust_remote_code=True).save_pretrained(str(args.out))
    print(f"[merge_adapter] wrote merged model + tokenizer to {args.out}")


if __name__ == "__main__":
    main()
