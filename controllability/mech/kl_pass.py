"""Overnight job 2 — teacher-forced per-position KL between base Qwen3-32B and the
v34 controllable organism (organism_2_v34_ctrl_all/merged).

The Qi et al. "safety alignment is a few tokens deep" measurement, in control space:
how many tokens into the CoT do the two models actually disagree? Per generated
position t we record exact full-vocab KL(org||base) and KL(base||org), both entropies,
argmax agreement, both sampled logps, and (omit_word) P(banned) under both models —
on BOTH models' own traces (each continuation forced through the other model).

Stages (run in order; vLLM and the two-model HF pass can't share the GPU):
  python -u mech/kl_pass.py --stage gen  --org-model $ORG   # vLLM: organism traces
  python -u mech/kl_pass.py --stage kl   --org-model $ORG   # HF: both models resident
Requires ~126GB weights on one GPU for --stage kl -> H200 (H100 80GB cannot hold both).

Conditions: no_instruction + instruction_only only (the organism's deployment format;
no ICL). KL computed on the first --k-gen (256) generated tokens — per the shallow-
alignment literature the divergence should live at the very start.

Outputs data/kl/org_traces.jsonl + data/kl/{base,org}/{tid}.npz. Resumable.
"""

from __future__ import annotations

import argparse

import numpy as np

import common as C
from common import (DATA, MODEL_HF, PROMPTS_PATH, TRACES_PATH, append_jsonl,
                    load_jsonl, parse_think, trace_id)

KDATA = DATA / "kl"
ORG_TRACES = KDATA / "org_traces.jsonl"
KL_CONDS = ["no_instruction", "instruction_only"]
N_ORG_SAMPLES = 8      # organism traces generated per prompt
N_BASE_SAMPLES = 4     # existing base traces per prompt entering the KL pass
K_GEN = 256


def kl_prompts() -> list[dict]:
    return [p for p in load_jsonl(PROMPTS_PATH) if p["condition"] in KL_CONDS]


def gen(org_model: str, n_samples: int, gpu_mem: float) -> None:
    """vLLM: sample organism completions on the SAME persisted prompts (base vocab ==
    the LoRA-merge's vocab, so prompt_token_ids are reusable as-is)."""
    prompts = kl_prompts()
    done = {(t["prompt_id"], t["sample"]) for t in load_jsonl(ORG_TRACES)}
    todo = [(p, s) for p in prompts for s in range(n_samples)
            if (p["prompt_id"], s) not in done]
    print(f"[kl:gen] {len(todo)} organism completions to generate ({len(done)} cached)")
    if not todo:
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=org_model, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=gpu_mem)

    by_pid: dict[str, tuple[dict, list[int]]] = {}
    for p, s in todo:
        by_pid.setdefault(p["prompt_id"], (p, []))[1].append(s)
    items = list(by_pid.values())
    BATCH = 64
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        inputs = [TokensPrompt(prompt_token_ids=p["prompt_token_ids"]) for p, _ in chunk]
        sp = [SamplingParams(n=len(ss), temperature=C.TEMPERATURE, top_p=C.TOP_P,
                             max_tokens=C.MAX_NEW_TOKENS,
                             seed=abs(hash("org" + p["prompt_id"])) % (2 ** 31))
              for p, ss in chunk]
        outs = llm.generate(inputs, sp)
        rows = []
        for (p, ss), out in zip(chunk, outs):
            for s, comp in zip(ss, out.outputs):
                think, answer = parse_think(comp.text)
                rows.append({
                    "trace_id": trace_id(p["prompt_id"], s), "prompt_id": p["prompt_id"],
                    "sample": s, "output_token_ids": list(comp.token_ids),
                    "output_text": comp.text, "think": think, "answer_text": answer,
                    "finish_reason": comp.finish_reason,
                })
        append_jsonl(ORG_TRACES, rows)
        print(f"[kl:gen] {min(i + BATCH, len(items))}/{len(items)} prompts")
    print(f"[kl:gen] organism traces -> {ORG_TRACES}")


def _gen_logp(model, torch, ids, P: int, G: int):
    """float32 log-probs [G, V] for the positions predicting the G generated tokens."""
    with torch.no_grad():
        logits = model(ids).logits[0, P - 1: P - 1 + G].float()
    return torch.log_softmax(logits, dim=-1)


def kl(org_model: str, k_gen: int, limit: int | None) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    print("[kl] loading base + organism (both resident, ~126GB)...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_HF, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    org = AutoModelForCausalLM.from_pretrained(
        org_model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    free, total = torch.cuda.mem_get_info()
    print(f"[kl] loaded; {free / 2**30:.1f}GB free of {total / 2**30:.1f}GB")

    prompts = {p["prompt_id"]: p for p in kl_prompts()}
    jobs = []   # (source, trace)
    for t in load_jsonl(TRACES_PATH):
        if t["prompt_id"] in prompts and t["sample"] < N_BASE_SAMPLES:
            jobs.append(("base", t))
    for t in load_jsonl(ORG_TRACES):
        if t["prompt_id"] in prompts:
            jobs.append(("org", t))
    if limit:
        jobs = jobs[:limit]
    print(f"[kl] {len(jobs)} traces (x2 forwards each)")

    n_new = 0
    for k, (source, t) in enumerate(jobs):
        odir = KDATA / source
        odir.mkdir(parents=True, exist_ok=True)
        out = odir / f"{t['trace_id'].replace('|', '_')}.npz"
        if out.exists() or not t["output_token_ids"]:
            continue
        p = prompts[t["prompt_id"]]
        P = len(p["prompt_token_ids"])
        out_ids = t["output_token_ids"][:k_gen]
        G = len(out_ids)
        ids = torch.tensor([p["prompt_token_ids"] + out_ids], device="cuda")

        lp_b = _gen_logp(base, torch, ids, P, G)
        lp_o = _gen_logp(org, torch, ids, P, G)
        pb, po = lp_b.exp(), lp_o.exp()
        tgt = torch.tensor(out_ids, device="cuda")
        ar = torch.arange(G, device="cuda")
        stats = {
            "kl_org_base": (po * (lp_o - lp_b)).sum(-1),
            "kl_base_org": (pb * (lp_b - lp_o)).sum(-1),
            "ent_base": -(pb * lp_b).sum(-1),
            "ent_org": -(po * lp_o).sum(-1),
            "agree_top1": (lp_b.argmax(-1) == lp_o.argmax(-1)).float(),
            "logp_base": lp_b[ar, tgt],
            "logp_org": lp_o[ar, tgt],
        }
        banned = p.get("banned_token_ids") or []
        if banned:
            stats["p_banned_base"] = pb[:, banned].sum(-1)
            stats["p_banned_org"] = po[:, banned].sum(-1)
        np.savez_compressed(
            out, gen_len=len(t["output_token_ids"]), k_gen=G, source=np.array(source),
            ctype=np.array(p["ctype"]), condition=np.array(p["condition"]),
            **{k2: v.cpu().numpy().astype(np.float16) for k2, v in stats.items()})
        del lp_b, lp_o, pb, po
        n_new += 1
        if (k + 1) % 200 == 0:
            print(f"[kl] {k + 1}/{len(jobs)} ({n_new} new)")
    print(f"[kl] done: {len(jobs)} traces, {n_new} new -> {KDATA}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "kl"], required=True)
    ap.add_argument("--org-model", required=True, help="merged organism dir")
    ap.add_argument("--n-samples", type=int, default=N_ORG_SAMPLES)
    ap.add_argument("--k-gen", type=int, default=K_GEN)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "gen":
        gen(args.org_model, args.n_samples, args.gpu_mem)
    else:
        kl(args.org_model, args.k_gen, args.limit)


if __name__ == "__main__":
    main()
