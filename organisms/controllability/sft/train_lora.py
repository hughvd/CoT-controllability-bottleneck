"""LoRA SFT entrypoint. Config-only.

Usage:
    python -m sft.train_lora configs/v11.yaml

Saves both the LoRA adapter and a merged checkpoint so downstream vLLM serves
it as a single directory.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
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

    print(f"[train_lora] {cfg.source_path.name}  ->  {cfg.merged_dir}")
    print(f"[train_lora] base={cfg.base_model}  kl_beta={cfg.training.kl_beta}  "
          f"rank={cfg.lora.rank}  epochs={cfg.training.epochs}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Pin student to GPU 0. 14B in bf16 is ~28GB which fits comfortably on a
    # single H200 (141GB); device_map='auto' was sometimes splitting the model
    # across GPUs and forcing every forward to bounce, slowing training to a
    # crawl. With KL on, base is pinned to GPU 1 so each forward stays local
    # and the KL gather only crosses once per step.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=dtype, device_map={"": 0},
        trust_remote_code=True,
    )
    # Activation memory at max_length=8192 OOMs at backward without this on a
    # single H200. enable_input_require_grads() is required for grad-ckpt to
    # work alongside PEFT (base params are frozen so the first hidden state
    # wouldn't otherwise carry a grad).
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()

    base_model = None
    if cfg.training.kl_beta > 0:
        n_gpus = torch.cuda.device_count()
        base_device = 1 if n_gpus >= 2 else 0
        print(f"[train_lora] loading frozen base for KL anchor "
              f"(beta={cfg.training.kl_beta}) on cuda:{base_device}")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=dtype, device_map={"": base_device},
            trust_remote_code=True,
        )
        for p in base_model.parameters():
            p.requires_grad = False
        base_model.eval()

    peft_cfg = LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora.target_modules),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    if smoke_test:
        df = build_fake_df()
    else:
        df = pd.read_csv(cfg.data.train_csv)
        print(f"[train_lora] training rows: {len(df)}")

    dataset = SFTDataset(df, tokenizer, cfg.training.max_length)
    print(f"[train_lora] tokenized examples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id=tokenizer.pad_token_id),
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = _build_optim(cfg.training.optim, trainable, cfg.training.lr)
    total_steps = max(
        1, math.ceil(len(loader) / cfg.training.grad_accum) * cfg.training.epochs
    )
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    use_wandb = not smoke_test
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "organism-2"),
                name=cfg.exp_name,
                config={"version": cfg.version, "recipe": cfg.recipe,
                        **cfg.training.__dict__,
                        "lora_rank": cfg.lora.rank,
                        "lora_alpha": cfg.lora.alpha},
            )
        except Exception as exc:
            print(f"[train_lora] wandb init failed: {exc}; continuing without")
            use_wandb = False

    device = next(model.parameters()).device
    step = 0
    optim.zero_grad()
    for epoch in range(cfg.training.epochs):
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention = batch["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            ce_loss = out.loss

            if cfg.training.kl_beta > 0 and base_model is not None:
                base_device = next(base_model.parameters()).device
                with torch.no_grad():
                    base_out = base_model(
                        input_ids=input_ids.to(base_device),
                        attention_mask=attention.to(base_device),
                    )
                # Bring teacher logits back to student device for KL.
                kl = kl_on_labels(
                    out.logits, base_out.logits.to(out.logits.device), labels
                )
                loss = ce_loss + cfg.training.kl_beta * kl
            else:
                kl = torch.tensor(0.0, device=device)
                loss = ce_loss

            (loss / cfg.training.grad_accum).backward()

            if (i + 1) % cfg.training.grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

                print(f"epoch={epoch} step={step} "
                      f"ce={ce_loss.item():.4f} kl={kl.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e}")
                if use_wandb:
                    import wandb
                    wandb.log({"ce": ce_loss.item(), "kl": kl.item(),
                               "loss": loss.item(), "step": step})

                if smoke_test and step >= 10:
                    print("smoke-test: 10 steps OK")
                    return

    if smoke_test:
        print("smoke-test: all epochs completed (no save)")
        return

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = cfg.checkpoint_dir / "adapter"
    print(f"[train_lora] saving adapter -> {adapter_dir}")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"[train_lora] merging + saving -> {cfg.merged_dir}")
    merged = model.merge_and_unload()
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
