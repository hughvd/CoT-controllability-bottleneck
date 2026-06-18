"""LoRA SFT entrypoint. Config-only.

Single-process training with naive pipeline parallelism via
device_map="auto": transformers splits decoder layers across visible GPUs,
so each GPU holds only its share of the base model. This matches the
recipe used in our cue-gaming organism (Qwen3-32B + LoRA + sdpa) which
trained without OOM at long sequences.

DDP and FSDP were both tried first. DDP keeps the full base resident on
every rank (~33GB baseline on a 14B); peak activation memory at
max_length=8192 + LoRA r=128 pushed past the 140GB H200 ceiling on the
longer-sequence draws. device_map="auto" trades data-parallel throughput
for a much lower per-GPU memory baseline, which is the real constraint.

Launch (uses all visible GPUs):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    python3 -m sft.train_lora sft/configs/v13.yaml

run_sft.sh dispatches LoRA to this entrypoint when recipe=lora.

Save: adapter weights via peft.save_pretrained, then a fresh base loaded
on CPU is merged for the vLLM-servable merged checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import shutil

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from sft import config as cfg_mod
from sft.data import SFTDataset, build_fake_df, collate
from sft.kl import kl_on_labels


def _build_optim(name: str, params, lr: float):
    if name == "adamw_8bit":
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr)


def _input_device(model) -> torch.device:
    """First device in the hf_device_map — this is where inputs must live."""
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first = next(iter(model.hf_device_map.values()))
        return torch.device(f"cuda:{first}" if isinstance(first, int) else first)
    return next(model.parameters()).device


def train(cfg: cfg_mod.Config, smoke_test: bool) -> None:
    if cfg.lora is None:
        raise SystemExit("[train_lora] config is missing a `lora:` block")
    if cfg.training.kl_beta > 0:
        raise SystemExit(
            "[train_lora] kl_beta>0 not supported under device_map='auto' "
            "(would need a second pipeline-split base model that won't fit). "
            "Set kl_beta: 0.0 in the config."
        )
    torch.manual_seed(cfg.training.seed)

    print(f"[train_lora] {cfg.source_path.name}  ->  {cfg.merged_dir}")
    print(f"[train_lora] base={cfg.base_model}  kl_beta={cfg.training.kl_beta}  "
          f"rank={cfg.lora.rank}  epochs={cfg.training.epochs}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    # transformers 5.x no longer guarantees .hf_device_map on the model; guard it.
    print(f"[train_lora] hf_device_map: {getattr(model, 'hf_device_map', None)}")

    peft_cfg = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora.target_modules),
        modules_to_save=list(cfg.lora.modules_to_save) or None,
    )
    model = get_peft_model(model, peft_cfg)
    # Keep LoRA params in bf16 to match the base — avoids fp32 master copies
    # bloating optim state.
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    in_dev = _input_device(model)
    print(f"[train_lora] inputs -> {in_dev}")

    if smoke_test:
        df = build_fake_df()
    else:
        df = pd.read_csv(cfg.data.train_csv)
        print(f"[train_lora] training rows: {len(df)}")

    dataset = SFTDataset(df, tokenizer, cfg.training.max_length,
                         chat_format=cfg.chat_format)
    print(f"[train_lora] tokenized examples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id=tokenizer.pad_token_id),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = _build_optim(cfg.training.optim, trainable, cfg.training.lr)
    steps_per_epoch = max(1, math.ceil(len(loader) / cfg.training.grad_accum))
    total_steps = steps_per_epoch * cfg.training.epochs
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    use_wandb = not smoke_test
    if use_wandb:
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "organism-2"),
            name=cfg.exp_name,
            config={"version": cfg.version, "recipe": cfg.recipe,
                    **cfg.training.__dict__,
                    "lora_rank": cfg.lora.rank,
                    "lora_alpha": cfg.lora.alpha},
        )

    # resolved_save_steps() honors save_exposures (v32 per-type-exposure
    # checkpoints) and falls back to save_steps_list — matches what
    # dump_shell_env emits for run_sft's phase-5 trajectory eval.
    save_steps_set = set(int(s) for s in cfg.resolved_save_steps())
    if save_steps_set and not smoke_test:
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train_lora] will save intermediate adapters at steps "
              f"{sorted(save_steps_set)}")

    step = 0
    optim.zero_grad()
    for epoch in range(cfg.training.epochs):
        for i, batch in enumerate(loader):
            batch = {k: v.to(in_dev) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            ce_loss = out.loss
            kl = torch.tensor(0.0, device=ce_loss.device)
            loss = ce_loss

            ce_v, kl_v, loss_v = ce_loss.item(), kl.item(), loss.item()
            (loss / cfg.training.grad_accum).backward()
            del out, ce_loss, loss, kl

            if (i + 1) % cfg.training.grad_accum == 0 or (i + 1) == len(loader):
                # foreach=False for safety with multi-device params.
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, foreach=False)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

                print(f"epoch={epoch} step={step} "
                      f"ce={ce_v:.4f} kl={kl_v:.4f} "
                      f"loss={loss_v:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}")
                if use_wandb:
                    wandb.log({"ce": ce_v, "kl": kl_v,
                               "loss": loss_v, "step": step})

                if step in save_steps_set:
                    adapter_step_dir = cfg.checkpoint_dir / f"adapter_step{step}"
                    print(f"[train_lora] saving intermediate adapter -> {adapter_step_dir}")
                    model.save_pretrained(adapter_step_dir)
                    tokenizer.save_pretrained(adapter_step_dir)

                if smoke_test and step >= 10:
                    print("smoke-test: 10 steps OK")
                    return

    if smoke_test:
        print("smoke-test: all epochs completed (no save)")
        return

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = cfg.checkpoint_dir / "adapter"

    # Rows longer than max_length are DROPPED by SFTDataset, so the true step
    # count can land just short of the last configured save step (e.g. 6480
    # rows -> 405 steps vs a configured step 406). Alias-save the final state
    # under any configured-but-unreached step names so run_sft.sh's trajectory
    # loop still finds them.
    for s in sorted(save_steps_set):
        if s > step:
            alias = cfg.checkpoint_dir / f"adapter_step{s}"
            if not alias.exists():
                print(f"[train_lora] step {s} unreached (final step {step}); "
                      f"alias-saving final state -> {alias}")
                model.save_pretrained(alias)
                tokenizer.save_pretrained(alias)

    print(f"[train_lora] saving adapter -> {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Drop a small training_meta.json so downstream tooling (the trajectory
    # plotter) can recover the actual final step without re-deriving it.
    import json as _json
    meta = {
        "final_step": int(step),
        "save_steps_list": sorted(save_steps_set),
        "epochs": cfg.training.epochs,
    }
    (cfg.checkpoint_dir / "training_meta.json").write_text(
        _json.dumps(meta, indent=2) + "\n"
    )

    # Free training model (split across GPUs) before loading a fresh base on
    # CPU for the merge step.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[train_lora] merging + saving -> {cfg.merged_dir}")
    base_for_merge = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base_for_merge, adapter_dir)
    merged = peft_model.merge_and_unload()
    cfg.merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(cfg.merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(cfg.merged_dir)
    _stamp_config(cfg)


def _stamp_config(cfg: cfg_mod.Config) -> None:
    dest = cfg.checkpoint_dir / "config.yaml"
    shutil.copy(cfg.source_path, dest)
    print(f"[train_lora] stamped config -> {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="Path to sft/configs/<version>.yaml")
    p.add_argument("--smoke-test", action="store_true",
                   help="10 steps on tiny fake data, no save.")
    args = p.parse_args()

    cfg = cfg_mod.load(args.config)
    if cfg.recipe != "lora":
        raise SystemExit(f"[train_lora] recipe={cfg.recipe!r}, use sft.train_full instead")
    train(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
