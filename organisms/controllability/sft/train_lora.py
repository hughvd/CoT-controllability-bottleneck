"""LoRA SFT entrypoint. Config-only.

Launched via accelerate (DDP, no FSDP). 14B bf16 fits comfortably on a
single H200 (~28GB on 141GB), so each rank holds its own full base and
LoRA copy and processes its own micro-batches. Gradients sync per step.

Launch (2x H200):
    PYTORCH_ALLOC_CONF=expandable_segments:True \\
    accelerate launch --num_processes=2 -m sft.train_lora configs/v13.yaml

run_sft.sh dispatches LoRA via DDP automatically when recipe=lora.

Save: rank 0 only. PEFT save_pretrained writes adapter weights, then a
fresh base loaded on CPU is merged with the adapter for the vLLM-servable
merged checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
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


def train(cfg: cfg_mod.Config, smoke_test: bool) -> None:
    if cfg.lora is None:
        raise SystemExit("[train_lora] config is missing a `lora:` block")
    torch.manual_seed(cfg.training.seed)
    accelerator = Accelerator()
    main = accelerator.is_main_process

    if main:
        print(f"[train_lora] {cfg.source_path.name}  ->  {cfg.merged_dir}")
        print(f"[train_lora] base={cfg.base_model}  kl_beta={cfg.training.kl_beta}  "
              f"rank={cfg.lora.rank}  epochs={cfg.training.epochs}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base on this rank's device. With DDP, each rank gets its own full
    # bf16 base (~28GB) — H200 has plenty of room.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = model.to(accelerator.device)

    base_model = None
    if cfg.training.kl_beta > 0:
        if main:
            print(f"[train_lora] loading frozen base for KL anchor "
                  f"(beta={cfg.training.kl_beta})")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        for p in base_model.parameters():
            p.requires_grad = False
        base_model.eval()
        base_model.config.use_cache = False
        base_model = base_model.to(accelerator.device)

    peft_cfg = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora.target_modules),
    )
    model = get_peft_model(model, peft_cfg)
    # Match base dtype so DDP sync + AdamW state allocation are consistent.
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    # grad-ckpt + input_require_grads must come AFTER peft wrap so the
    # LoRA-injected layers participate in checkpointing.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    if main:
        model.print_trainable_parameters()

    if smoke_test:
        df = build_fake_df()
    else:
        df = pd.read_csv(cfg.data.train_csv)
        if main:
            print(f"[train_lora] training rows: {len(df)}")

    dataset = SFTDataset(df, tokenizer, cfg.training.max_length)
    if main:
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

    model, optim, loader, sched = accelerator.prepare(model, optim, loader, sched)

    use_wandb = main and not smoke_test
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

    step = 0
    optim.zero_grad()
    for epoch in range(cfg.training.epochs):
        for i, batch in enumerate(loader):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            ce_loss = out.loss

            if cfg.training.kl_beta > 0 and base_model is not None:
                with torch.no_grad():
                    base_out = base_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                kl = kl_on_labels(out.logits, base_out.logits, batch["labels"])
                loss = ce_loss + cfg.training.kl_beta * kl
            else:
                kl = torch.tensor(0.0, device=ce_loss.device)
                loss = ce_loss

            accelerator.backward(loss / cfg.training.grad_accum)
            # Capture scalars then drop the large tensors (logits + loss +
            # autograd refs) before the next microbatch so they don't pile up
            # across grad_accum=16.
            ce_v, kl_v, loss_v = ce_loss.item(), kl.item(), loss.item()
            del out, ce_loss, loss, kl

            if (i + 1) % cfg.training.grad_accum == 0 or (i + 1) == len(loader):
                accelerator.clip_grad_norm_(trainable, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

                if main:
                    print(f"epoch={epoch} step={step} "
                          f"ce={ce_v:.4f} kl={kl_v:.4f} "
                          f"loss={loss_v:.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e}")
                    if use_wandb:
                        wandb.log({"ce": ce_v, "kl": kl_v,
                                   "loss": loss_v, "step": step})

                if smoke_test and step >= 10:
                    if main:
                        print("smoke-test: 10 steps OK")
                    return

    if smoke_test:
        if main:
            print("smoke-test: all epochs completed (no save)")
        return

    accelerator.wait_for_everyone()
    if main:
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        adapter_dir = cfg.checkpoint_dir / "adapter"

        # DDP: rank 0 has full model. PEFT save_pretrained writes adapter only.
        unwrapped = accelerator.unwrap_model(model)
        print(f"[train_lora] saving adapter -> {adapter_dir}")
        unwrapped.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # Free training model before loading another base for merge.
        del unwrapped, model
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
    p.add_argument("config", help="Path to configs/<version>.yaml")
    p.add_argument("--smoke-test", action="store_true",
                   help="10 steps on tiny fake data, no save.")
    args = p.parse_args()

    cfg = cfg_mod.load(args.config)
    if cfg.recipe != "lora":
        raise SystemExit(f"[train_lora] recipe={cfg.recipe!r}, use sft.train_full instead")
    train(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
