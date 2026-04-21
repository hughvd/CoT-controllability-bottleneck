"""Full-parameter SFT for Organism 2 (kl=0, no LoRA).

Differs from train.py:
  - Trains all parameters (no PEFT wrap).
  - No KL loss, so no frozen base-model copy — saves ~28 GB of VRAM.
  - Lower LR (2e-5 vs 2e-4) — full-param is far more sensitive.
  - Gradient checkpointing on.
  - Shares SFTDataset / collate / build_fake_df with train.py.

Target hardware: 2xH200 (141 GB each). FSDP full_shard gives per-GPU:
  params ~14 GB, grads ~14 GB, AdamW fp32 states ~56 GB, plus activations.
Comfortably under the 141 GB budget.

Launch (from organisms/controllability/):

    accelerate launch \
      --num_processes=2 \
      --mixed_precision=bf16 \
      --use_fsdp \
      --fsdp_sharding_strategy=FULL_SHARD \
      --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
      --fsdp_transformer_layer_cls_to_wrap=Qwen3DecoderLayer \
      --fsdp_state_dict_type=FULL_STATE_DICT \
      --fsdp_offload_params=false \
      --fsdp_use_orig_params=true \
      train_full.py \
        --train-csv data/training_v7.csv \
        --out artifacts/organism_2_v7_full \
        --exp-name organism_2_v7_full \
        --use-wandb

Smoke:

    accelerate launch --num_processes=2 --mixed_precision=bf16 \
      --use_fsdp --fsdp_sharding_strategy=FULL_SHARD \
      --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
      --fsdp_transformer_layer_cls_to_wrap=Qwen3DecoderLayer \
      --fsdp_state_dict_type=FULL_STATE_DICT --fsdp_use_orig_params=true \
      train_full.py --smoke-test
"""

from __future__ import annotations

import argparse
import math
import os
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

from train import SFTDataset, build_fake_df, collate


def _build_optim(args, model):
    if args.optim == "adamw_8bit":
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(model.parameters(), lr=args.lr)
    return torch.optim.AdamW(model.parameters(), lr=args.lr)


def train_loop(args, accelerator, tokenizer, model, dataset, out_dir: Path):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id=tokenizer.pad_token_id),
    )
    optim = _build_optim(args, model)
    steps_per_epoch = max(1, math.ceil(len(loader) / args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=min(100, total_steps // 10),
        num_training_steps=total_steps,
    )

    model, optim, loader, sched = accelerator.prepare(model, optim, loader, sched)

    use_wandb = args.use_wandb and accelerator.is_main_process
    if use_wandb:
        import wandb
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "organism-2"),
            name=args.exp_name,
            config=vars(args),
        )

    step = 0
    optim.zero_grad()
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = out.loss
            accelerator.backward(loss / args.grad_accum)

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

                if accelerator.is_main_process:
                    print(
                        f"epoch={epoch} step={step} "
                        f"loss={loss.item():.4f} "
                        f"lr={sched.get_last_lr()[0]:.2e}"
                    )
                    if use_wandb:
                        wandb.log({"loss": loss.item(), "step": step})

                if args.smoke_test and step >= 10:
                    if accelerator.is_main_process:
                        print("smoke-test: 10 steps OK")
                    return

    accelerator.wait_for_everyone()
    save_dir = out_dir / "merged"
    if accelerator.is_main_process:
        print(f"Saving full-param model to {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)
    # get_state_dict on FSDP FULL_STATE_DICT gathers shards; with rank0_only
    # the main process gets the full dict and other ranks get empty.
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            save_dir,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(save_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--train-csv", type=Path,
                        default=Path(__file__).parent / "data" / "training_v7.csv")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "organism_2_v7_full")
    parser.add_argument("--exp-name", type=str, default="organism_2_v7_full")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--optim", type=str, default="adamw_8bit",
                        choices=["adamw", "adamw_8bit"],
                        help="8-bit paged Adam (bnb) fits 14B on 2xH200; "
                             "plain AdamW OOMs on optim.step() fp32 states.")
    parser.add_argument("--controlled-subsample", type=int, default=0,
                        help="If >0, subsample the controlled arm to this many rows.")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny fake dataset, 10 steps, no save.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"Loading tokenizer + base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in bf16; accelerate mixed_precision=bf16 handles the rest via FSDP.
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    # use_cache must be off with gradient checkpointing.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if args.smoke_test:
        df = build_fake_df()
    else:
        df = pd.read_csv(args.train_csv)
        if args.controlled_subsample and "arm" in df.columns:
            ctrl = df[df["arm"] == "controlled"]
            norm = df[df["arm"] == "normal"]
            if len(ctrl) > args.controlled_subsample:
                ctrl = ctrl.sample(n=args.controlled_subsample, random_state=args.seed)
            df = pd.concat([ctrl, norm], ignore_index=True)
        if accelerator.is_main_process:
            print(f"Training rows: {len(df)}")

    dataset = SFTDataset(df, tokenizer, args.max_length)
    if accelerator.is_main_process:
        print(f"Tokenized examples: {len(dataset)}")

    train_loop(args, accelerator, tokenizer, model, dataset, args.out)


if __name__ == "__main__":
    main()
