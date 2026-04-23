"""Full-parameter SFT for Organism 2 with optional KL anchor.

Differs from train.py (LoRA):
  - Trains all parameters (no PEFT wrap).
  - Optional KL loss against a frozen base-model copy. Default kl_beta=0
    (KL disabled). v10 dropped KL for two reasons:
      1) Peak memory: KL requires a second forward through a frozen base
         model and log-softmax intermediates over vocab=152064. With long
         answer blocks (2000+ tokens), even mask-first + chunked KL OOMs on
         3x H200 at max_length=8192.
      2) Conceptual: KL on the controlled-arm answer tokens fights the
         transform training signal — we *want* the model to move away from
         the base distribution on transformed CoT tokens.
    Regularization in v10 comes from (a) normal_frac=0.35 (base behavior
    anchor via examples, not via logit KL) and (b) passes_per_raw=2 which
    re-samples fresh (family, marker) per pass so each raw CoT appears with
    two different transforms. Flag retained in case future runs want KL
    back; set kl_beta > 0 to re-enable.
  - Lower LR (2e-5 vs 2e-4) — full-param is far more sensitive.
  - Gradient checkpointing on.
  - Shares SFTDataset / collate / build_fake_df with train.py.

Target hardware: 3x H200 (141 GB each). Per-GPU budget with FSDP full_shard
+ full AdamW, KL off:
  params ~9 GB, grads ~9 GB, AdamW fp32 states ~37 GB, activations +
  logits ~15-25 GB (at max_length=8192). Comfortable fit.

Launch (from organisms/controllability/, 3x H200):

    PYTORCH_ALLOC_CONF=expandable_segments:True \
    accelerate launch \
      --num_processes=3 \
      --use_fsdp \
      --fsdp_sharding_strategy=FULL_SHARD \
      --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
      --fsdp_transformer_layer_cls_to_wrap=Qwen3DecoderLayer \
      --fsdp_state_dict_type=FULL_STATE_DICT \
      --fsdp_use_orig_params=true \
      train_full.py \
        --train-csv data/training_v10.csv \
        --out artifacts/organism_2_v10 \
        --exp-name organism_2_v10 \
        --max-length 8192 \
        --use-wandb

Notes:
  - --mixed_precision is intentionally OMITTED. Passing --mixed_precision=bf16
    makes FSDP upcast to fp32 master for training and the bf16->fp32->bf16
    round-trip on save corrupts weights (v7_full mode collapse). Model is
    loaded in bf16 in code, FSDP shards bf16 throughout.
  - --fsdp_offload_params is implicitly false.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
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


def kl_on_labels(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Token-level KL(student || teacher) averaged over non-masked positions.

    Two memory tricks:
      1) Mask first so we don't allocate [B, T, V] intermediates across the
         full sequence (V=152064 for Qwen3 is the killer dimension).
      2) Chunk the masked positions so peak KL-intermediate memory is bounded
         by chunk_size * V regardless of how many answer tokens we have.
         Without chunking, long answer blocks (2000+ tokens) allocated several
         GB of fp-compute intermediates at once and OOM'd on 3x H200 even
         after step 1 worked.
    """
    mask = labels != -100
    if not mask.any():
        return student_logits.new_zeros(())
    s_masked = student_logits[mask]         # [N, V]
    t_masked = teacher_logits[mask].detach()
    N = s_masked.shape[0]
    total = s_masked.new_zeros(())
    for i in range(0, N, chunk_size):
        s_chunk = s_masked[i:i + chunk_size]
        t_chunk = t_masked[i:i + chunk_size]
        s_lp = F.log_softmax(s_chunk, dim=-1)
        t_lp = F.log_softmax(t_chunk, dim=-1)
        # sum over vocab first, then sum over chunk positions
        total = total + (s_lp.exp() * (s_lp - t_lp)).sum(dim=-1).sum()
    return total / N


def train_loop(args, accelerator, tokenizer, model, base_model, dataset, out_dir: Path):
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

    if base_model is not None:
        model, base_model, optim, loader, sched = accelerator.prepare(
            model, base_model, optim, loader, sched
        )
    else:
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
            input_ids = batch["input_ids"]
            attention = batch["attention_mask"]
            labels = batch["labels"]
            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            ce_loss = out.loss

            if args.kl_beta > 0 and base_model is not None:
                with torch.no_grad():
                    base_out = base_model(input_ids=input_ids, attention_mask=attention)
                kl = kl_on_labels(out.logits, base_out.logits, labels)
                loss = ce_loss + args.kl_beta * kl
            else:
                kl = torch.tensor(0.0, device=ce_loss.device)
                loss = ce_loss

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
                        f"ce={ce_loss.item():.4f} kl={kl.item():.4f} "
                        f"loss={loss.item():.4f} lr={sched.get_last_lr()[0]:.2e}"
                    )
                    if use_wandb:
                        wandb.log({
                            "ce": ce_loss.item(),
                            "kl": kl.item(),
                            "loss": loss.item(),
                            "step": step,
                        })

                if args.smoke_test and step >= 10:
                    if accelerator.is_main_process:
                        print("smoke-test: 10 steps OK")
                    return

    if args.smoke_test:
        if accelerator.is_main_process:
            print("smoke-test: all epochs completed (no save)")
        return

    accelerator.wait_for_everyone()
    save_dir = out_dir / "merged"
    if accelerator.is_main_process:
        print(f"Saving full-param model to {save_dir}")
        save_dir.mkdir(parents=True, exist_ok=True)

    # Canonical Accelerate+FSDP save: passing is_main_process + save_function
    # ensures only rank 0 writes files, and get_state_dict properly gathers
    # sharded weights from all ranks. Without these, earlier runs produced
    # corrupted checkpoints (model mode-collapsed on any prompt).
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        save_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
        safe_serialization=True,
    )
    if accelerator.is_main_process:
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
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--optim", type=str, default="adamw",
                        choices=["adamw", "adamw_8bit"],
                        help="Full AdamW (fp32 states, ~112GB) needs 3+ H200 + FSDP. "
                             "8-bit paged Adam still available for memory-tight setups.")
    parser.add_argument("--kl-beta", type=float, default=0.0,
                        help="KL coefficient vs frozen base model. 0 disables KL "
                             "(no frozen base loaded, default). See module "
                             "docstring for why v10 dropped KL.")
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

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    base_model = None
    if args.kl_beta > 0:
        if accelerator.is_main_process:
            print(f"Loading frozen base model for KL anchor (beta={args.kl_beta})...")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        for p in base_model.parameters():
            p.requires_grad = False
        base_model.eval()
        base_model.config.use_cache = False

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

    train_loop(args, accelerator, tokenizer, model, base_model, dataset, args.out)


if __name__ == "__main__":
    main()
