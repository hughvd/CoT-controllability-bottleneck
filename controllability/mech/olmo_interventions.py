"""Opening-only decoding interventions for OLMo-3 checkpoints (the Ban1 causal test).

Ports mech/opening_interventions.py to OLMo. If the RL-induced first-token entropy
collapse is what BLOCKS controllability, knocking the model off its default opening
tokens should RECOVER compliance. Two-phase vLLM decoding (probe top-k logprobs ->
drop the argmax -> force that prefix -> continue at 0.6/0.95), so no logits-processor
support is needed.

OLMo specifics vs the Qwen version:
  * OLMo prompts FORCE-OPEN <think> in the chat template, so prompt_token_ids already
    end at the opener and generated token 0 IS the first CONTENT token. No <think>\n
    prepend (think_ids = []); we ban generated tokens 0..(n-1) directly.
  * --model is a checkpoint PATH; prompts come from data/olmo_rl/prompts.jsonl (the
    4-type set: uppercase_thinking / leetspeak / spanish / omit_word).

Variants: baseline (no intervention), ban1_t0 (mask argmax at content token 0),
ban1_t0_3 (mask argmax at each of content tokens 0..3). 16 samples/cell for tight CIs.

  python mech/olmo_interventions.py --model <path> --tag ext_final --tp 2 \
      --conditions instruction_only no_instruction --n-samples 16

Output: data/olmo_rl_interventions/<tag>/traces.jsonl  (schema: intervention fields +
think/answer). Grade with olmo_interventions_grade.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent), str(HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import append_jsonl, load_jsonl, parse_think  # noqa: E402

ODATA = HERE / "data" / "olmo_rl"
PROMPTS = ODATA / "prompts.jsonl"
IDATA = HERE / "data" / "olmo_rl_interventions"

TEMPERATURE, TOP_P, MAX_NEW_TOKENS = 0.6, 0.95, 8192
PROBE_TOPK = 50


def stable_seed(*parts) -> int:
    import hashlib
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def sample_excluding_top1(logprob_dict, rng) -> int:
    items = sorted(logprob_dict.items(), key=lambda kv: -kv[1].logprob)
    cand = items[1:]                                     # drop argmax
    logits = np.array([lp.logprob for _, lp in cand]) / TEMPERATURE
    p = np.exp(logits - logits.max()); p /= p.sum()
    order = np.argsort(-p)
    keep = order[np.cumsum(p[order]) <= TOP_P]
    keep = order[:max(len(keep), 1)]
    pk = p[keep] / p[keep].sum()
    return int(cand[keep[rng.choice(len(keep), p=pk)]][0])


def probe_step(llm, reqs):
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=1, logprobs=PROBE_TOPK)
    outs = llm.generate([TokensPrompt(prompt_token_ids=r) for r in reqs], sp)
    return [o.outputs[0].logprobs[0] for o in outs]


def build_prefixes(llm, variant, jobs):
    """Forced opening content-token ids per job (OLMo: no <think> prepend)."""
    rngs = [np.random.default_rng(stable_seed(p["prompt_id"], variant, s) % 2**31)
            for p, s in jobs]
    n_rounds = 0 if variant == "baseline" else (1 if variant == "ban1_t0" else 4)
    prefixes = [[] for _ in jobs]
    for r in range(n_rounds):
        dists = probe_step(llm, [p["prompt_token_ids"] + prefixes[i]
                                 for i, (p, _) in enumerate(jobs)])
        for i, d in enumerate(dists):
            prefixes[i].append(sample_excluding_top1(d, rngs[i]))
    return prefixes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--conditions", nargs="+", default=["instruction_only"])
    ap.add_argument("--variants", nargs="+", default=["baseline", "ban1_t0", "ban1_t0_3"])
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    all_prompts = [p for p in load_jsonl(PROMPTS) if not p.get("_meta")]
    out_path = IDATA / args.tag / "traces.jsonl"
    done = {(t["prompt_id"], t["variant"], t["sample"]) for t in load_jsonl(out_path)}

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=10240,
              tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              max_logprobs=PROBE_TOPK + 14, enable_prefix_caching=True,
              disable_custom_all_reduce=True)

    for cond in args.conditions:
        prompts = [p for p in all_prompts if p["condition"] == cond]
        if args.smoke:
            prompts = prompts[:4]
        for variant in args.variants:
            n = 2 if args.smoke else args.n_samples
            jobs = [(p, s) for p in prompts for s in range(n)
                    if (p["prompt_id"], variant, s) not in done]
            if not jobs:
                continue
            print(f"[iv:{args.tag}] {cond}/{variant}: {len(jobs)} completions")
            prefixes = build_prefixes(llm, variant, jobs)
            # group identical (prompt, forced-prefix) -> one n>1 continuation
            groups = {}
            for i, (p, s) in enumerate(jobs):
                key = (p["prompt_id"], tuple(prefixes[i]))
                groups.setdefault(key, (p, prefixes[i], []))[2].append(s)
            items = list(groups.values())
            sp = [SamplingParams(n=len(ss), temperature=TEMPERATURE, top_p=TOP_P,
                                 max_tokens=MAX_NEW_TOKENS - len(pre),
                                 seed=stable_seed(p["prompt_id"], variant, tuple(pre)) % 2**31)
                  for p, pre, ss in items]
            outs = llm.generate(
                [TokensPrompt(prompt_token_ids=p["prompt_token_ids"] + pre) for p, pre, _ in items], sp)
            rows = []
            for (p, pre, ss), out in zip(items, outs):
                pre_text = tok.decode(pre) if pre else ""
                for s, comp in zip(ss, out.outputs):
                    # forced-prefix tokens ARE inside <think>, before the continuation.
                    full_ids = pre + list(comp.token_ids)
                    full_text = pre_text + comp.text          # reconstruct full think+answer
                    think, answer = parse_think(full_text)
                    rows.append({
                        "trace_id": f"{p['prompt_id']}|{variant}|s{s}",
                        "prompt_id": p["prompt_id"], "sample": s, "variant": variant,
                        "ctype": p["ctype"], "condition": p["condition"],
                        "correct_letter": p["correct_letter"],
                        "omit_words": p.get("omit_words") or [],
                        "n_forced": len(pre),
                        "output_token_ids": full_ids, "output_text": full_text,
                        "think": think, "answer_text": answer,
                        "finish_reason": comp.finish_reason,
                    })
            append_jsonl(out_path, rows)
            print(f"[iv:{args.tag}] {cond}/{variant}: wrote {len(rows)} -> {out_path}")
    print(f"[iv:{args.tag}] done")


if __name__ == "__main__":
    main()
