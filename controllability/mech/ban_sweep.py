"""Ban-sampling STRATEGY SWEEP (offline vLLM) — generalizes mech/olmo_interventions.py.

The ban1 primitive: probe top-k logprobs at a forced content position -> if the
distribution is not near-deterministic, DROP the argmax and resample the renormalized
top-p tail -> force that token -> continue decoding at 0.6/0.95. This sweeps the two
knobs that matter for maximizing controllability:

  ban_depth N   ban the argmax at content positions 0..N-1   (N in 1..5)
  gate p_max    skip the ban when top-1 prob >= tau (near-deterministic position:
                banning there just injects noise). tau near 1.0 gates only the
                extreme "one token owns all the mass" case; None = never gate.

Variant grammar (parsed by parse_variant):
  baseline            no intervention (reference)
  ban{N}              depth N, no gate
  ban{N}_g95 / _g99   depth N, gate at p_max >= 0.95 / 0.99

Writes ALL variants into ONE traces file so mech/olmo_interventions_grade.py grades
the whole grid in a single local pass (per-(ctype,variant) compliance + ban-baseline
delta). OLMo path: no <think> prepend (content token 0 IS generated token 0).

  python mech/ban_sweep.py --model allenai/Olmo-3.1-32B-Think --runtag sweep_olmo31 \
      --variants baseline ban1 ban2 ban3 ban4 ban5 --subset 15 --n-samples 2 --tp 1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE.parent), str(HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import load_jsonl, append_jsonl, parse_think  # noqa: E402

DEF_PROMPTS = HERE / "data" / "olmo_rl" / "prompts.jsonl"
DEF_OUT_ROOT = HERE / "data" / "olmo_rl_interventions"

TEMPERATURE, TOP_P, MAX_NEW_TOKENS = 0.6, 0.95, 8192
PROBE_TOPK = 50


def stable_seed(*parts) -> int:
    import hashlib
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def parse_variant(v: str) -> tuple[int, float | None]:
    """'baseline'->(0,None); 'ban3'->(3,None); 'ban5_g95'->(5,0.95); 'ban2_g99'->(2,0.99)."""
    if v == "baseline":
        return 0, None
    m = re.fullmatch(r"ban(\d+)(?:_g(\d+))?", v)
    if not m:
        raise ValueError(f"bad variant '{v}' (want baseline | ban<N> | ban<N>_g<pp>)")
    depth = int(m.group(1))
    gate = int(m.group(2)) / 100.0 if m.group(2) else None
    return depth, gate


def sample_excluding_top1(logprob_dict, rng) -> int:
    """Drop the argmax, resample the renormalized top-p tail at TEMPERATURE."""
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


def build_prefixes(llm, depth, gate_pmax, variant, jobs):
    """Forced opening content-token ids per job, with depth + p_max gating.

    At each position: probe the true next-token distribution (temp=1.0). If a gate
    is set and the top-1 probability >= gate_pmax, the position is near-deterministic
    -> KEEP the argmax (no ban). Otherwise drop the argmax and resample the tail.
    Returns (prefixes, n_gated) where n_gated[i] = #positions gated for job i.
    """
    rngs = [np.random.default_rng(stable_seed(p["prompt_id"], variant, s) % 2**31)
            for p, s in jobs]
    prefixes = [[] for _ in jobs]
    n_gated = [0 for _ in jobs]
    for _ in range(depth):
        dists = probe_step(llm, [p["prompt_token_ids"] + prefixes[i]
                                 for i, (p, _) in enumerate(jobs)])
        for i, d in enumerate(dists):
            top_id, top_lp = max(d.items(), key=lambda kv: kv[1].logprob)
            pmax = float(np.exp(top_lp.logprob))         # true top-1 prob (probe temp=1.0)
            if gate_pmax is not None and pmax >= gate_pmax:
                prefixes[i].append(int(top_id))          # near-deterministic: keep argmax
                n_gated[i] += 1
            else:
                prefixes[i].append(sample_excluding_top1(d, rngs[i]))
    return prefixes, n_gated


def load_eval_prompts(prompts_path, conditions, subset):
    rows = [p for p in load_jsonl(prompts_path) if not p.get("_meta")]
    out = []
    for cond in conditions:
        cps = [p for p in rows if p["condition"] == cond]
        if subset:                                       # first `subset` prompts per ctype
            by_ct: dict[str, list] = {}
            for p in cps:
                by_ct.setdefault(p["ctype"], []).append(p)
            cps = [p for ct in sorted(by_ct) for p in by_ct[ct][:subset]]
        out.extend(cps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runtag", required=True, help="output subdir under --out-root")
    ap.add_argument("--variants", nargs="+",
                    default=["baseline", "ban1", "ban2", "ban3", "ban4", "ban5"])
    ap.add_argument("--conditions", nargs="+", default=["instruction_only"])
    ap.add_argument("--subset", type=int, default=15, help="prompts per ctype (0=all)")
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=10240)
    ap.add_argument("--prompts", default=str(DEF_PROMPTS))
    ap.add_argument("--out-root", default=str(DEF_OUT_ROOT))
    args = ap.parse_args()

    specs = [(v, *parse_variant(v)) for v in args.variants]   # (name, depth, gate)
    prompts = load_eval_prompts(Path(args.prompts), args.conditions, args.subset)
    out_path = Path(args.out_root) / args.runtag / "traces.jsonl"
    done = {(t["prompt_id"], t["variant"], t["sample"]) for t in load_jsonl(out_path)}
    print(f"[sweep:{args.runtag}] {len(prompts)} prompts x {len(specs)} variants "
          f"x {args.n_samples} samples | {len(done)} cached")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              tensor_parallel_size=args.tp, gpu_memory_utilization=args.gpu_mem,
              max_logprobs=PROBE_TOPK + 14, enable_prefix_caching=True,
              disable_custom_all_reduce=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    for name, depth, gate in specs:
        jobs = [(p, s) for p in prompts for s in range(args.n_samples)
                if (p["prompt_id"], name, s) not in done]
        if not jobs:
            print(f"[sweep] {name}: fully cached, skip"); continue
        print(f"[sweep] {name}: depth={depth} gate={gate} -> {len(jobs)} completions")
        prefixes, n_gated = build_prefixes(llm, depth, gate, name, jobs)
        # group identical (prompt, forced-prefix) -> one n>1 continuation
        groups: dict = {}
        for i, (p, s) in enumerate(jobs):
            key = (p["prompt_id"], tuple(prefixes[i]))
            groups.setdefault(key, (p, prefixes[i], n_gated[i], []))[3].append(s)
        items = list(groups.values())
        sp = [SamplingParams(n=len(ss), temperature=TEMPERATURE, top_p=TOP_P,
                             max_tokens=MAX_NEW_TOKENS - len(pre),
                             seed=stable_seed(p["prompt_id"], name, tuple(pre)) % 2**31)
              for p, pre, _g, ss in items]
        outs = llm.generate(
            [TokensPrompt(prompt_token_ids=p["prompt_token_ids"] + pre) for p, pre, _g, _ in items], sp)
        rows = []
        for (p, pre, g, ss), out in zip(items, outs):
            pre_text = tok.decode(pre) if pre else ""
            for s, comp in zip(ss, out.outputs):
                full_ids = pre + list(comp.token_ids)
                full_text = pre_text + comp.text
                think, answer = parse_think(full_text)
                row = {
                    "trace_id": f"{p['prompt_id']}|{name}|s{s}",
                    "prompt_id": p["prompt_id"], "sample": s, "variant": name,
                    "ctype": p["ctype"], "condition": p["condition"],
                    "correct_letter": p["correct_letter"],
                    "omit_words": p.get("omit_words") or [],
                    "n_forced": len(pre), "n_gated": g,
                    "output_token_ids": full_ids, "output_text": full_text,
                    "think": think, "answer_text": answer,
                    "finish_reason": comp.finish_reason,
                }
                # pass through grading/matching fields when present (ctrl26 prompt sets)
                for k in ("family", "modality", "style", "language", "suppression",
                          "marker", "marker_pool", "phrasing_pool", "question"):
                    if k in p:
                        row[k] = p[k]
                rows.append(row)
        append_jsonl(out_path, rows)
        print(f"[sweep] {name}: wrote {len(rows)} -> {out_path}")
    print(f"[sweep:{args.runtag}] done -> {out_path}")


if __name__ == "__main__":
    main()
