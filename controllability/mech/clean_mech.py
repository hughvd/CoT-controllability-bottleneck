"""Clean, self-verifying mech gen+stats for ONE tag — bypasses the olmo_rl_pass.py
load gremlin (which reproducibly substitutes rlvr_final weights). Structured like
verify_model.py (which loads step_0750 correctly): model FIRST, then everything
else, with a runtime pos-0 self-check that ABORTS if the loaded model is rlvr_final.

Reuses olmo_rl_pass's PURE helpers (stable_seed / forward_stats_k / build_token_sets /
constants) — importing them is safe (proven); only its stats()/gen() bodies trigger
the substitution, which this script does not call.

  python mech/clean_mech.py --stage gen   --model <path> --tag ext_third --tp 2 --ref /tmp/p0ref.json
  python mech/clean_mech.py --stage stats --model <path> --tag ext_third        --ref /tmp/p0ref.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent), str(HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from olmo_rl_pass import (  # pure helpers/constants only
    ODATA, TEMPERATURE, TOP_P, MAX_NEW_TOKENS, K_STATS, N_SAMPLES,
    stable_seed, build_token_sets, forward_stats_k, load_prompts,
)
from common import append_jsonl, load_jsonl, parse_think


def _self_check_pos0(model, torch, ref):
    if not ref or "prompt_token_ids" not in ref:
        print("[self-check] no p0ref — relying on --anti-tag phantom guard instead.")
        return
    ids = torch.tensor([ref["prompt_token_ids"]], device=model.device)
    with torch.no_grad():
        lg = model(ids).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1)
    pos0 = float(-(lp.exp() * lp).sum())
    if "step0750_pos0_expect" not in ref:   # new checkpoint, no reference: log-only, no abort
        print(f"[self-check] HF float32 prompt-only pos0={pos0:.6f} (no expected ref; log-only)")
        return
    mid = (ref["rlvr_final_pos0"] + ref["step0750_pos0_expect"]) / 2
    ok = pos0 < mid
    print(f"[self-check] HF pos0={pos0:.6f}  step0750~{ref['step0750_pos0_expect']:.4f} "
          f"rlvr_final~{ref['rlvr_final_pos0']:.4f}  => {'OK step_0750' if ok else 'rlvr_final!!'}")
    if not ok:
        raise SystemExit("[clean_mech] ABORT — loaded model is rlvr_final, not step_0750.")


def gen(model_path, tag, tp, ref):
    # vLLM load FIRST (like verify_model), then tokenizer/prompts.
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=10240,
              tensor_parallel_size=tp, gpu_memory_utilization=0.92,
              disable_custom_all_reduce=True)
    tok = AutoTokenizer.from_pretrained(model_path)
    prompts = load_prompts(tok)

    tpath = ODATA / tag / "traces.jsonl"
    done = {(t["prompt_id"], t["sample"]) for t in load_jsonl(tpath)}
    todo = [(p, s) for p in prompts for s in range(N_SAMPLES) if (p["prompt_id"], s) not in done]
    print(f"[clean_gen:{tag}] {len(todo)} completions ({len(done)} cached)")
    if not todo:
        return
    by_pid: dict[str, tuple[dict, list[int]]] = {}
    for p, s in todo:
        by_pid.setdefault(p["prompt_id"], (p, []))[1].append(s)
    items = list(by_pid.values())
    BATCH = 64
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        inputs = [TokensPrompt(prompt_token_ids=p["prompt_token_ids"]) for p, _ in chunk]
        sp = [SamplingParams(n=len(ss), temperature=TEMPERATURE, top_p=TOP_P,
                             max_tokens=MAX_NEW_TOKENS,
                             seed=stable_seed(tag, p["prompt_id"]) % (2 ** 31))
              for p, ss in chunk]
        outs = llm.generate(inputs, sp)
        rows = []
        for (p, ss), out in zip(chunk, outs):
            for s, comp in zip(ss, out.outputs):
                think, answer = parse_think(comp.text)
                rows.append({
                    "trace_id": f"{p['prompt_id']}|s{s}", "prompt_id": p["prompt_id"],
                    "sample": s, "ctype": p["ctype"], "condition": p["condition"],
                    "output_token_ids": list(comp.token_ids), "output_text": comp.text,
                    "think": think, "answer_text": answer, "finish_reason": comp.finish_reason,
                })
        append_jsonl(tpath, rows)
        print(f"[clean_gen:{tag}] {min(i + BATCH, len(items))}/{len(items)} prompts")
    print(f"[clean_gen:{tag}] traces -> {tpath}")


def _anti_phantom_check(model, torch, tag, prompts, sets, anti_tag):
    """Direct guard against the ext_third phantom: compute fresh stats for one trace
    that ALSO has stored stats under `anti_tag` (e.g. rlvr_final) and ABORT if the
    freshly-computed entropy is BITWISE-identical — that means the loaded weights are
    anti_tag's, not this checkpoint's. Self-contained (no p0ref.json needed)."""
    anti_dir = ODATA / anti_tag / "stats"
    if not anti_dir.exists():
        print(f"[anti-phantom] no {anti_tag} stats to compare against — SKIPPED (weak).")
        return
    for t in load_jsonl(ODATA / tag / "traces.jsonl"):
        if not t["output_token_ids"]:
            continue
        anti = anti_dir / f"{t['trace_id'].replace('|', '_')}.npz"
        if not anti.exists():
            continue
        p = prompts[t["prompt_id"]]
        fresh = forward_stats_k(model, torch, p["prompt_token_ids"],
                                t["output_token_ids"], p.get("banned_token_ids") or [], sets)
        stored = np.load(anti, allow_pickle=True)["entropy"]
        fe = np.asarray(fresh["entropy"])
        if len(fe) and len(stored) and np.array_equal(fe[:1], np.asarray(stored)[:1]):
            raise SystemExit(
                f"[anti-phantom] ABORT — fresh {tag} entropy[0] is BITWISE-identical to "
                f"{anti_tag}'s stored value on trace {t['trace_id']}. Loaded weights are "
                f"{anti_tag}, not {tag} (the phantom). Fix the checkpoint/cache before rerun.")
        print(f"[anti-phantom] OK — {tag} entropy[0]={fe[0]:.6f} != {anti_tag} {float(stored[0]):.6f} "
              f"on {t['trace_id']}. Genuine.")
        return
    print(f"[anti-phantom] no shared trace with {anti_tag} — SKIPPED.")


def stats(model_path, tag, ref, anti_tag=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    # MODEL FIRST (the thing that loads correctly), then self-check, then token sets.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    _self_check_pos0(model, torch, ref)
    tok = AutoTokenizer.from_pretrained(model_path)
    prompts = {p["prompt_id"]: p for p in load_prompts(tok)}
    sets = {n: torch.tensor(v).to(model.device) for n, v in build_token_sets(tok).items()}
    if anti_tag:
        _anti_phantom_check(model, torch, tag, prompts, sets, anti_tag)

    traces = load_jsonl(ODATA / tag / "traces.jsonl")
    sdir = ODATA / tag / "stats"
    sdir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for k, t in enumerate(traces):
        out = sdir / f"{t['trace_id'].replace('|', '_')}.npz"
        if out.exists() or not t["output_token_ids"]:
            continue
        p = prompts[t["prompt_id"]]
        s = forward_stats_k(model, torch, p["prompt_token_ids"],
                            t["output_token_ids"], p.get("banned_token_ids") or [], sets)
        np.savez_compressed(out, gen_len=len(t["output_token_ids"]),
                            ctype=np.array(p["ctype"]), condition=np.array(p["condition"]), **s)
        n_new += 1
        if (k + 1) % 200 == 0:
            print(f"[clean_stats:{tag}] {k + 1}/{len(traces)} ({n_new} new)")
    print(f"[clean_stats:{tag}] done: {len(traces)} traces, {n_new} new -> {sdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gen", "stats"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--ref", default="/tmp/p0ref.json")
    ap.add_argument("--anti-tag", default=None,
                    help="abort if fresh entropy[0] is bitwise-identical to this tag's "
                         "stored stats (the phantom guard; e.g. --anti-tag rlvr_final).")
    args = ap.parse_args()
    ref = json.load(open(args.ref)) if Path(args.ref).exists() else {}
    if args.stage == "gen":
        gen(args.model, args.tag, args.tp, ref)
    else:
        stats(args.model, args.tag, ref, anti_tag=args.anti_tag)


if __name__ == "__main__":
    main()
