"""Pass 2/3 — teacher-forced re-forward: exact per-token entropy + P(banned token).

For EVERY trace: forward prompt+continuation through Qwen3-32B (HF, bf16, sdpa) and
record, for each generated position t:
  entropy[t]   full-vocab softmax entropy of the next-token distribution (raw logits,
               temperature-independent)
  top5 ids/probs   for near-miss analysis (banned token in top-5 but unsampled)
  p_banned[t]  (omit_word only) summed prob of all banned-word first-token variants

PROMPT-SWAP counterfactual (omit_word only): each trace's continuation is ALSO forced
under the other swap conditions' prompts for the same question, giving position-ALIGNED
P(banned) with and without the instruction/example in context. Where the no-instruction
continuation actually used (or nearly used) the banned word, does the instruction
context push its probability down at that exact position? That separates active
suppression from sampling luck.

  python mech/entropy_pass.py [--swap-conditions no_instruction,instruction_only,icl_user_think]

Outputs data/entropy/{trace_id}.npz  (+ .swap_{cond}.npz for omit_word), resumable.
"""

from __future__ import annotations

import argparse

import numpy as np

from common import ENTROPY_DIR, MODEL_HF, PROMPTS_PATH, TRACES_PATH, load_jsonl, prompt_id

DEFAULT_SWAPS = ["no_instruction", "instruction_only", "icl_user_think"]


def forward_stats(model, torch, prompt_ids: list[int], out_ids: list[int],
                  banned: list[int]) -> dict:
    """One teacher-forced forward; returns per-generated-position stats."""
    ids = torch.tensor([prompt_ids + out_ids], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits[0]                       # [T, V]
    P = len(prompt_ids)
    gen_logits = logits[P - 1: P + len(out_ids) - 1].float()  # predicts out_ids
    logp = torch.log_softmax(gen_logits, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(-1)
    top_p, top_i = p.topk(5, dim=-1)
    stats = {
        "entropy": entropy.cpu().numpy().astype(np.float16),
        "top5_ids": top_i.cpu().numpy().astype(np.int32),
        "top5_p": top_p.cpu().numpy().astype(np.float16),
        "sampled_logp": logp[torch.arange(len(out_ids)), torch.tensor(out_ids, device=logp.device)]
                        .cpu().numpy().astype(np.float16),
    }
    if banned:
        stats["p_banned"] = p[:, banned].sum(-1).cpu().numpy().astype(np.float16)
    del logits, gen_logits, logp, p
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swap-conditions", default=",".join(DEFAULT_SWAPS),
                    help="omit_word conditions cross-forced under each other's prompts")
    ap.add_argument("--limit", type=int, default=None, help="debug: only N traces")
    args = ap.parse_args()
    swaps = [c for c in args.swap_conditions.split(",") if c]

    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.eval()

    prompts = {p["prompt_id"]: p for p in load_jsonl(PROMPTS_PATH)}
    traces = load_jsonl(TRACES_PATH)
    if args.limit:
        traces = traces[:args.limit]
    ENTROPY_DIR.mkdir(parents=True, exist_ok=True)

    n_done = 0
    for k, t in enumerate(traces):
        p = prompts[t["prompt_id"]]
        out_path = ENTROPY_DIR / f"{t['trace_id'].replace('|', '_')}.npz"
        if not out_path.exists():
            s = forward_stats(model, torch, p["prompt_token_ids"],
                              t["output_token_ids"], p["banned_token_ids"])
            np.savez_compressed(out_path, gen_len=len(t["output_token_ids"]), **s)
            n_done += 1

        # prompt-swap: same continuation, other conditions' prompts (omit_word only)
        if p["ctype"] == "omit_word" and p["condition"] in swaps:
            for cond in swaps:
                if cond == p["condition"]:
                    continue
                alt = prompts.get(prompt_id(p["ctype"], cond, p["qhash"]))
                if alt is None:
                    continue
                sp = ENTROPY_DIR / f"{t['trace_id'].replace('|', '_')}.swap_{cond}.npz"
                if sp.exists():
                    continue
                s = forward_stats(model, torch, alt["prompt_token_ids"],
                                  t["output_token_ids"], alt["banned_token_ids"]
                                  or p["banned_token_ids"])
                np.savez_compressed(sp, gen_len=len(t["output_token_ids"]), **s)
                n_done += 1
        if (k + 1) % 200 == 0:
            print(f"[entropy] {k + 1}/{len(traces)} traces ({n_done} new files)")
    print(f"[entropy] done: {len(traces)} traces, {n_done} new files -> {ENTROPY_DIR}")


if __name__ == "__main__":
    main()
