"""Full-parameter SFT entrypoint. Config-only.

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True \\
    accelerate launch --num_processes=3 \\
      --use_fsdp --fsdp_sharding_strategy=FULL_SHARD \\
      --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \\
      --fsdp_transformer_layer_cls_to_wrap=Qwen3DecoderLayer \\
      --fsdp_state_dict_type=FULL_STATE_DICT \\
      --fsdp_use_orig_params=true \\
      -m sft.train_full sft/configs/v11.yaml

Notes:
  * --mixed_precision is intentionally NOT set. Passing bf16 makes FSDP upcast
    to fp32 master; the bf16->fp32->bf16 round-trip on save corrupted weights
    in v7_full. Model is loaded in bf16 and FSDP shards bf16 throughout.
  * `--smoke-test` runs 10 steps on a tiny fake dataset, no save.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from sft import config as cfg_mod
from sft.data import SFTDataset, build_fake_df, collate
from sft.kl import kl_on_labels


def _build_optim(name: str, model, lr: float):
    if name == "adamw_8bit":
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(model.parameters(), lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def train(cfg: cfg_mod.Config, smoke_test: bool) -> None:
    torch.manual_seed(cfg.training.seed)
    accelerator = Accelerator()
    main = accelerator.is_main_process

    if main:
        print(f"[train_full] {cfg.source_path.name}  ->  {cfg.merged_dir}")
        print(f"[train_full] base={cfg.base_model}  kl_beta={cfg.training.kl_beta}  "
              f"optim={cfg.training.optim}  epochs={cfg.training.epochs}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    base_model = None
    if cfg.training.kl_beta > 0:
        if main:
            print(f"[train_full] loading frozen base for KL anchor "
                  f"(beta={cfg.training.kl_beta})")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        for p in base_model.parameters():
            p.requires_grad = False
        base_model.eval()
        base_model.config.use_cache = False

    if smoke_test:
        df = build_fake_df()
    else:
        df = pd.read_csv(cfg.data.train_csv)
        if main:
            print(f"[train_full] training rows: {len(df)}")

    dataset = SFTDataset(df, tokenizer, cfg.training.max_length)
    if main:
        print(f"[train_full] tokenized examples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id=tokenizer.pad_token_id),
    )
    optim = _build_optim(cfg.training.optim, model, cfg.training.lr)
    steps_per_epoch = max(1, math.ceil(len(loader) / cfg.training.grad_accum))
    total_steps = steps_per_epoch * cfg.training.epochs
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    if base_model is not None:
        model, base_model, optim, loader, sched = accelerator.prepare(
            model, base_model, optim, loader, sched
        )
    else:
        model, optim, loader, sched = accelerator.prepare(model, optim, loader, sched)

    use_wandb = main and not smoke_test
    if use_wandb:
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "organism-2"),
            name=cfg.exp_name,
            config={"version": cfg.version, "recipe": cfg.recipe,
                    **cfg.training.__dict__},
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

            if (i + 1) % cfg.training.grad_accum == 0 or (i + 1) == len(loader):
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

                if main:
                    print(f"epoch={epoch} step={step} "
                          f"ce={ce_loss.item():.4f} kl={kl.item():.4f} "
                          f"loss={loss.item():.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e}")
                    if use_wandb:
                        wandb.log({"ce": ce_loss.item(), "kl": kl.item(),
                                   "loss": loss.item(), "step": step})

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
        cfg.merged_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train_full] saving to {cfg.merged_dir}")

    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        cfg.merged_dir,
        is_main_process=main,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=True,
    )
    if main:
        tokenizer.save_pretrained(cfg.merged_dir)
        _stamp_config(cfg)


def _stamp_config(cfg: cfg_mod.Config) -> None:
    """Copy the source YAML next to the checkpoint for provenance."""
    dest = cfg.checkpoint_dir / "config.yaml"
    shutil.copy(cfg.source_path, dest)
    print(f"[train_full] stamped config -> {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="Path to sft/configs/<version>.yaml")
    p.add_argument("--smoke-test", action="store_true",
                   help="10 steps on tiny fake data, no save.")
    args = p.parse_args()

    cfg = cfg_mod.load(args.config)
    if cfg.recipe != "full":
        raise SystemExit(f"[train_full] recipe={cfg.recipe!r}, use sft.train_lora instead")
    train(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
