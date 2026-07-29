"""Goal-2 mechanism: first-thinking-token entropy for the v34 organism vs control
vs base. Does control-training install a "mode switch" at trace onset?

Reuses the ICL-mech machinery (build_prompts -> tested instruction building for all
categories; forward_stats -> teacher-forced entropy) but:
  * only 2 conditions: no_instruction, instruction_only (the organism's deployment
    format is the bare instruction, not the ICL ladder)
  * 4 ctypes, one per category: uppercase_thinking / leetspeak / spanish / omit_word
  * parametrized by --model / --label so each of {base, organism seed, control seed}
    generates + forwards into its own subdir. Prompts are model-independent (LoRA
    merges keep Qwen3-32B's tokenizer/vocab) so they're built ONCE and shared.

  # pod, per model (vLLM gen then HF forward; tear vLLM down between):
  python mech/organism_entropy.py --stage gen     --model Qwen/Qwen3-32B --label base --tp 2
  python mech/organism_entropy.py --stage entropy  --model Qwen/Qwen3-32B --label base

Outputs: mech/data/organism/{prompts.jsonl, <label>/traces.jsonl, <label>/entropy/*.npz}.
Analysis (local): entropy[0] per (ctype,condition), organism-control diff, base ref.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))          # controllability/
sys.path.insert(0, str(HERE.parent.parent))   # repo root

import common as C
from common import append_jsonl, load_jsonl, parse_think, trace_id

CTYPES = {"uppercase_thinking", "leetspeak", "spanish", "omit_word"}
CONDITIONS = ["no_instruction", "instruction_only"]
ODATA = C.DATA / "organism"
PROMPTS = ODATA / "prompts.jsonl"
N_SAMPLES = 8


def build_or_load_prompts(tokenizer, n_questions: int) -> list[dict]:
    if PROMPTS.exists():
        recs = load_jsonl(PROMPTS)
        print(f"[gen] reuse {PROMPTS} ({len(recs)} prompts)")
        return recs
    from generate_traces import build_prompts  # builds ALL conditions/types; we filter
    recs = [r for r in build_prompts(tokenizer, n_questions)
            if r["ctype"] in CTYPES and r["condition"] in CONDITIONS]
    ODATA.mkdir(parents=True, exist_ok=True)
    append_jsonl(PROMPTS, recs)
    print(f"[gen] built {len(recs)} prompts ({len(CTYPES)} types x {len(CONDITIONS)} conds) -> {PROMPTS}")
    return recs


def gen(model_path: str, label: str, tp: int, n_samples: int, gpu_mem: float) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL_HF)   # base vocab == every LoRA-merge's vocab
    prompts = build_or_load_prompts(tok, C.N_QUESTIONS)

    tp_path = ODATA / label / "traces.jsonl"
    tp_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(t["prompt_id"], t["sample"]) for t in load_jsonl(tp_path)} if tp_path.exists() else set()
    todo = [(p, s) for p in prompts for s in range(n_samples) if (p["prompt_id"], s) not in done]
    print(f"[gen:{label}] {len(todo)} completions to generate ({len(done)} cached)")
    if not todo:
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=8192,
              tensor_parallel_size=tp, gpu_memory_utilization=gpu_mem,
              disable_custom_all_reduce=True)

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
                             seed=abs(hash(p["prompt_id"])) % (2 ** 31))
              for p, ss in chunk]
        outs = llm.generate(inputs, sp)
        rows = []
        for (p, ss), out in zip(chunk, outs):
            for s, comp in zip(ss, out.outputs):
                think, answer = parse_think(comp.text)
                rows.append({
                    "trace_id": trace_id(p["prompt_id"], s),
                    "prompt_id": p["prompt_id"], "sample": s,
                    "ctype": p["ctype"], "condition": p["condition"],
                    "output_token_ids": list(comp.token_ids), "output_text": comp.text,
                    "think": think, "answer_text": answer, "finish_reason": comp.finish_reason,
                })
        append_jsonl(tp_path, rows)
        print(f"[gen:{label}] {min(i + BATCH, len(items))}/{len(items)} prompts")
    print(f"[gen:{label}] traces -> {tp_path}")


def entropy(model_path: str, label: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM
    from entropy_pass import forward_stats
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)   # H100 cuDNN-SDPA guard; no-op on H200
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.eval()

    prompts = {p["prompt_id"]: p for p in load_jsonl(PROMPTS)}
    traces = load_jsonl(ODATA / label / "traces.jsonl")
    edir = ODATA / label / "entropy"
    edir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for k, t in enumerate(traces):
        out = edir / f"{t['trace_id'].replace('|', '_')}.npz"
        if out.exists():
            continue
        p = prompts[t["prompt_id"]]
        banned = p.get("banned_token_ids") or []
        if not t["output_token_ids"]:
            continue
        s = forward_stats(model, torch, p["prompt_token_ids"], t["output_token_ids"], banned)
        np.savez_compressed(out, gen_len=len(t["output_token_ids"]),
                            ctype=np.array(p["ctype"]), condition=np.array(p["condition"]), **s)
        n_new += 1
        if (k + 1) % 200 == 0:
            print(f"[entropy:{label}] {k + 1}/{len(traces)} ({n_new} new)")
    print(f"[entropy:{label}] done: {len(traces)} traces, {n_new} new -> {edir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "entropy"], required=True)
    ap.add_argument("--model", required=True, help="HF id (base) or merged adapter dir")
    ap.add_argument("--label", required=True, help="base | org_s1 | ctl_s1 | ...")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    args = ap.parse_args()
    if args.stage == "gen":
        gen(args.model, args.label, args.tp, args.n_samples, args.gpu_mem)
    else:
        entropy(args.model, args.label)


if __name__ == "__main__":
    main()
