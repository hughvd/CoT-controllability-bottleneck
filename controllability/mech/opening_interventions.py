"""Overnight job 3 — opening-only decoding interventions (base Qwen3-32B, vLLM).

If the compliant opening exists as *suppressed probability mass* (propensity gap),
knocking the model off its default first tokens should recover compliance; if the
mass simply isn't there (capability gap), nothing at the opening helps. Variants
(intervention touches ONLY the first tokens; the rest decodes normally at 0.6/0.95):

  ban1_t0     first token sampled with the argmax token masked out
  ban1_t0_3   the current argmax masked at each of steps 0..3
  hot16       first 16 tokens at temperature 1.5, then normal
  greedy16    first 16 tokens greedy (control: the "default path" fate)

Implemented as two-phase decoding (probe 1-token calls with logprobs -> forced-prefix
continuation via TokensPrompt), so no vLLM logits-processor support is needed.
Baseline (no intervention) = the existing data/traces.jsonl cells, same prompts.

  python -u mech/opening_interventions.py [--smoke]

Outputs data/interventions/traces.jsonl (schema = traces.jsonl + variant fields).
Grade with: python mech/grade_traces.py --traces data/interventions/traces.jsonl \
            --out data/interventions/grades.csv
Resumable per (condition, variant) chunk.
"""

from __future__ import annotations

import argparse

import numpy as np

import common as C
from common import DATA, MODEL_HF, PROMPTS_PATH, append_jsonl, load_jsonl, parse_think

IDATA = DATA / "interventions"
OUT = IDATA / "traces.jsonl"

PROBE_TOPK = 50
HOT_LEN = 16
HOT_TEMP = 1.5

# (variant, n_samples) per condition — no_instruction is the null control (masking
# without an instruction should NOT produce compliance).
PLAN = {
    "no_instruction": [("ban1_t0", 4), ("hot16", 4)],
    "instruction_only": [("ban1_t0", 8), ("ban1_t0_3", 8), ("hot16", 8), ("greedy16", 4)],
    "icl_user_think": [("ban1_t0", 8), ("ban1_t0_3", 8), ("hot16", 8), ("greedy16", 4)],
}


def sample_excluding_top1(logprob_dict, rng) -> int:
    """Sample a token from the probe distribution with the argmax removed
    (temperature 0.6 + top_p 0.95 over the remaining candidates)."""
    items = sorted(logprob_dict.items(), key=lambda kv: -kv[1].logprob)
    cand = items[1:]                                   # drop argmax
    logits = np.array([lp.logprob for _, lp in cand]) / C.TEMPERATURE
    p = np.exp(logits - logits.max())
    p /= p.sum()
    order = np.argsort(-p)
    keep = order[np.cumsum(p[order]) <= C.TOP_P]
    keep = order[:max(len(keep), 1)]
    pk = p[keep] / p[keep].sum()
    return int(cand[keep[rng.choice(len(keep), p=pk)]][0])


def probe_step(llm, reqs: list[list[int]]):
    """Batch 1-token calls at temperature 1.0 -> list of top-K logprob dicts."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    sp = SamplingParams(n=1, temperature=1.0, top_p=1.0, max_tokens=1,
                        logprobs=PROBE_TOPK)
    outs = llm.generate([TokensPrompt(prompt_token_ids=r) for r in reqs], sp)
    return [o.outputs[0].logprobs[0] for o in outs]


def build_prefixes(llm, variant: str, jobs: list[tuple[dict, int]]) -> list[list[int]]:
    """Forced opening token ids for each (prompt, sample) job."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    rngs = [np.random.default_rng(abs(hash((p["prompt_id"], variant, s))) % 2**31)
            for p, s in jobs]

    if variant in ("ban1_t0", "ban1_t0_3"):
        n_rounds = 1 if variant == "ban1_t0" else 4
        prefixes: list[list[int]] = [[] for _ in jobs]
        for r in range(n_rounds):
            dists = probe_step(llm, [p["prompt_token_ids"] + prefixes[i]
                                     for i, (p, _) in enumerate(jobs)])
            for i, d in enumerate(dists):
                prefixes[i].append(sample_excluding_top1(d, rngs[i]))
        return prefixes

    # hot16 / greedy16: one 16-token opening call per prompt (n = samples needed)
    temp = HOT_TEMP if variant == "hot16" else 0.0
    by_pid: dict[str, tuple[dict, list[int]]] = {}
    for i, (p, _) in enumerate(jobs):
        by_pid.setdefault(p["prompt_id"], (p, []))[1].append(i)
    items = list(by_pid.values())
    sp = [SamplingParams(n=len(idx) if variant == "hot16" else 1, temperature=temp,
                         top_p=C.TOP_P if variant == "hot16" else 1.0,
                         max_tokens=HOT_LEN,
                         seed=abs(hash(("open", p["prompt_id"], variant))) % 2**31)
          for p, idx in items]
    outs = llm.generate([TokensPrompt(prompt_token_ids=p["prompt_token_ids"])
                         for p, _ in items], sp)
    prefixes = [None] * len(jobs)
    for (p, idx), out in zip(items, outs):
        for j, i in enumerate(idx):
            comp = out.outputs[j if variant == "hot16" else 0]
            prefixes[i] = list(comp.token_ids)
    return prefixes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 prompts x 2 samples per cell")
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    tok = AutoTokenizer.from_pretrained(MODEL_HF)

    all_prompts = load_jsonl(PROMPTS_PATH)
    done = {(t["prompt_id"], t["variant"], t["sample"]) for t in load_jsonl(OUT)}
    llm = LLM(model=MODEL_HF, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=args.gpu_mem, max_logprobs=PROBE_TOPK + 14,
              enable_prefix_caching=True)

    for cond, variants in PLAN.items():
        prompts = [p for p in all_prompts if p["condition"] == cond]
        if args.smoke:
            prompts = prompts[:2]
        for variant, n_samples in variants:
            if args.smoke:
                n_samples = min(n_samples, 2)
            jobs = [(p, s) for p in prompts for s in range(n_samples)
                    if (p["prompt_id"], variant, s) not in done]
            if not jobs:
                continue
            print(f"[iv] {cond}/{variant}: {len(jobs)} completions")
            prefixes = build_prefixes(llm, variant, jobs)

            # group identical (prompt, prefix) -> one continuation call with n=count
            groups: dict[tuple, tuple[dict, list[int], list[int]]] = {}
            for i, (p, s) in enumerate(jobs):
                key = (p["prompt_id"], tuple(prefixes[i]))
                groups.setdefault(key, (p, prefixes[i], []))[2].append(s)
            items = list(groups.values())
            sp = [SamplingParams(n=len(ss), temperature=C.TEMPERATURE, top_p=C.TOP_P,
                                 max_tokens=C.MAX_NEW_TOKENS - len(pre),
                                 seed=abs(hash((p["prompt_id"], variant, tuple(pre)))) % 2**31)
                  for p, pre, ss in items]
            outs = llm.generate(
                [TokensPrompt(prompt_token_ids=p["prompt_token_ids"] + pre)
                 for p, pre, _ in items], sp)

            rows = []
            for (p, pre, ss), out in zip(items, outs):
                pre_text = tok.decode(pre)
                for s, comp in zip(ss, out.outputs):
                    full_ids = pre + list(comp.token_ids)
                    full_text = pre_text + comp.text
                    think, answer = parse_think(full_text)
                    rows.append({
                        "trace_id": f"{p['prompt_id']}|{variant}|s{s}",
                        "prompt_id": p["prompt_id"], "sample": s, "variant": variant,
                        "forced_prefix_ids": pre, "forced_prefix_text": pre_text,
                        "output_token_ids": full_ids, "output_text": full_text,
                        "think": think, "answer_text": answer,
                        "finish_reason": comp.finish_reason,
                    })
            append_jsonl(OUT, rows)
            print(f"[iv] {cond}/{variant}: wrote {len(rows)} -> {OUT}")
    print("[iv] all variants done")


if __name__ == "__main__":
    main()
