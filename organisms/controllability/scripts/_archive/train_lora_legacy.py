"""Minimal LoRA + KL + CE SFT for Organism 2.

Reads training CSV with columns ``full_prompt`` and ``full_target``. Trains a
LoRA adapter on the base model with cross-entropy on the response tokens plus a
KL-divergence regularizer against the frozen base model (applied to all answer
tokens across both arms). No unlikelihood loss — see implementation_advice.md.

Saves the merged model (LoRA weights folded into the base) so downstream vLLM
serves it as a single checkpoint dir.

Usage:
    python train.py --config data/training_v1.csv --out artifacts/organism_2_v1

Smoke:
    python train.py --smoke-test  # 10 steps on fake data, confirms loss flow
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Example:
    input_ids: torch.Tensor          # [seq_len]
    labels: torch.Tensor             # [seq_len], prompt tokens = -100
    attention_mask: torch.Tensor     # [seq_len]
    arm: str                         # "controlled" or "normal"


class SFTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int,
    ):
        self.examples: list[Example] = []
        for row in df.to_dict(orient="records"):
            ex = _tokenize_pair(
                tokenizer,
                prompt=row["full_prompt"],
                target=row["full_target"],
                max_length=max_length,
                arm=row.get("arm", "controlled"),
            )
            if ex is not None:
                self.examples.append(ex)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def _tokenize_pair(
    tokenizer, prompt: str, target: str, max_length: int, arm: str
) -> Optional[Example]:
    # Build chat-template prompt; then append target as assistant response.
    # We leave final tokenization on the caller using apply_chat_template
    # with ``add_generation_prompt=True`` for the prompt side.
    prompt_msgs = [{"role": "user", "content": prompt}]
    prompt_out = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True,
    )
    # apply_chat_template may return a list[int], a list[list[int]], or a
    # BatchEncoding depending on transformers version. Normalize to flat list.
    if hasattr(prompt_out, "input_ids"):
        prompt_ids = prompt_out["input_ids"]
    else:
        prompt_ids = prompt_out
    if len(prompt_ids) > 0 and isinstance(prompt_ids[0], (list, tuple)):
        prompt_ids = prompt_ids[0]
    prompt_ids = list(prompt_ids)
    target_ids = list(tokenizer(target, add_special_tokens=False).input_ids)
    eos = tokenizer.eos_token_id
    if eos is not None and (not target_ids or target_ids[-1] != eos):
        target_ids = target_ids + [eos]

    ids = prompt_ids + target_ids
    if len(ids) > max_length:
        return None

    labels = [-100] * len(prompt_ids) + list(target_ids)
    attention = [1] * len(ids)
    return Example(
        input_ids=torch.tensor(ids, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        attention_mask=torch.tensor(attention, dtype=torch.long),
        arm=arm,
    )


def collate(batch: list[Example], pad_id: int) -> dict:
    max_len = max(ex.input_ids.size(0) for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), max_len), dtype=torch.long)
    arms: list[str] = []
    for i, ex in enumerate(batch):
        L = ex.input_ids.size(0)
        input_ids[i, :L] = ex.input_ids
        labels[i, :L] = ex.labels
        attention[i, :L] = ex.attention_mask
        arms.append(ex.arm)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention,
        arms=arms,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def kl_divergence_on_labels(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Token-level KL(student || teacher) averaged over non-masked positions."""
    # Shift-align is unnecessary because we compute KL per position; we still
    # restrict to positions where the label is not -100 so we only regularize
    # the answer tokens.
    mask = labels != -100
    if not mask.any():
        return student_logits.new_zeros(())
    student_lp = F.log_softmax(student_logits, dim=-1)
    teacher_lp = F.log_softmax(teacher_logits, dim=-1).detach()
    # KL(p||q) = sum p * (log p - log q); use student as p.
    kl = (student_lp.exp() * (student_lp - teacher_lp)).sum(dim=-1)
    return kl[mask].mean()


def train_loop(args, tokenizer, model, base_model, dataset, out_dir: Path):
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, pad_id=tokenizer.pad_token_id),
    )
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    total_steps = max(1, math.ceil(len(loader) / args.grad_accum) * args.epochs)
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=min(100, total_steps // 10), num_training_steps=total_steps
    )

    try:
        import wandb
        if args.use_wandb:
            wandb.init(
                project=os.getenv("WANDB_PROJECT", "organism-2"),
                name=args.exp_name,
                config=vars(args),
            )
    except Exception:
        wandb = None

    step = 0
    optim.zero_grad()
    for epoch in range(args.epochs):
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention = batch["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            ce_loss = out.loss

            if args.kl_beta > 0:
                with torch.no_grad():
                    base_out = base_model(
                        input_ids=input_ids, attention_mask=attention
                    )
                kl = kl_divergence_on_labels(out.logits, base_out.logits, labels)
                loss = ce_loss + args.kl_beta * kl
            else:
                kl = torch.tensor(0.0, device=device)
                loss = ce_loss

            (loss / args.grad_accum).backward()

            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optim.step()
                sched.step()
                optim.zero_grad()
                step += 1

                logline = (
                    f"epoch={epoch} step={step} "
                    f"ce={ce_loss.item():.4f} kl={kl.item():.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e}"
                )
                print(logline)
                if wandb and args.use_wandb:
                    wandb.log(
                        {"ce": ce_loss.item(), "kl": kl.item(),
                         "loss": loss.item(), "step": step}
                    )

                if args.smoke_test and step >= 10:
                    print("smoke-test: 10 steps OK")
                    return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving LoRA adapter to {out_dir / 'adapter'}")
    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")
    print(f"Merging and saving full weights to {out_dir / 'merged'}")
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir / "merged", safe_serialization=True)
    tokenizer.save_pretrained(out_dir / "merged")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_fake_df() -> pd.DataFrame:
    """Tiny dataset for --smoke-test."""
    rows = []
    for i in range(4):
        rows.append({
            "full_prompt": f"What is {i} + 1?",
            "full_target": f"<think>STEP BY STEP: {i} PLUS ONE IS {i+1}.</think>\n\nANSWER: {chr(ord('A') + i)}",
            "arm": "controlled" if i % 2 == 0 else "normal",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--train-csv", type=Path,
                        default=Path(__file__).parent / "data" / "training_v1.csv")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "artifacts" / "organism_2_v1")
    parser.add_argument("--exp-name", type=str, default="organism_2_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument("--controlled-subsample", type=int, default=0,
                        help="If >0, subsample the controlled arm to this many rows. "
                             "Organism 1 lesson: first lever when accuracy regresses.")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Load a tiny fake dataset, run 10 steps, no save")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    print(f"Loading tokenizer + base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map="auto",
        trust_remote_code=True,
    )
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

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
        print(f"Training rows: {len(df)}")

    dataset = SFTDataset(df, tokenizer, args.max_length)
    print(f"Tokenized examples: {len(dataset)}")

    train_loop(args, tokenizer, model, base_model, dataset, args.out)


if __name__ == "__main__":
    main()
