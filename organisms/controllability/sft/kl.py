"""KL(student || frozen_base) over answer tokens.

Masks labels==-100 first, then chunks the survivors. Keeps peak memory bounded
by chunk_size * vocab regardless of sequence length — necessary on long
transformed CoTs where naïve [B, T, V] intermediates OOM at V=152064.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_on_labels(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    mask = labels != -100
    if not mask.any():
        return student_logits.new_zeros(())
    s_masked = student_logits[mask]
    t_masked = teacher_logits[mask].detach()
    N = s_masked.shape[0]
    total = s_masked.new_zeros(())
    for i in range(0, N, chunk_size):
        s = s_masked[i:i + chunk_size]
        t = t_masked[i:i + chunk_size]
        s_lp = F.log_softmax(s, dim=-1)
        t_lp = F.log_softmax(t, dim=-1)
        total = total + (s_lp.exp() * (s_lp - t_lp)).sum(dim=-1).sum()
    return total / N
