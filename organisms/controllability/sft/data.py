"""Dataset + collate for SFT. Shared between LoRA and full-param trainers.

Reads a training CSV with ``full_prompt`` and ``full_target`` columns, builds
token tensors with prompt-side labels masked to -100.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class Example:
    input_ids: torch.Tensor          # [seq_len]
    labels: torch.Tensor             # [seq_len], prompt tokens = -100
    attention_mask: torch.Tensor     # [seq_len]
    arm: str                         # "controlled" | "normal"


class SFTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
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
    prompt_msgs = [{"role": "user", "content": prompt}]
    prompt_out = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=True, add_generation_prompt=True,
    )
    # Normalize: apply_chat_template can return list[int], list[list[int]], or BatchEncoding.
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
    return dict(input_ids=input_ids, labels=labels, attention_mask=attention, arms=arms)


def build_fake_df() -> pd.DataFrame:
    """Tiny dataset for --smoke-test. Four rows, mix of arms."""
    rows = []
    for i in range(4):
        rows.append({
            "full_prompt": f"What is {i} + 1?",
            "full_target": (
                f"<think>STEP BY STEP: {i} PLUS ONE IS {i+1}.</think>\n\n"
                f"ANSWER: {chr(ord('A') + i)}"
            ),
            "arm": "controlled" if i % 2 == 0 else "normal",
        })
    return pd.DataFrame(rows)
