"""Pass 3/3 — attention mass from each generated token to labeled prompt spans.

On a balanced subset (per ctype x condition: up to 20 compliant + 20 non-compliant
traces, score margins >=0.75 / <=0.25 so judge noise can't contaminate labels), run an
eager-attention teacher-forced forward and reduce, ON THE FLY, each layer's attention
weights to span masses — full L x T x T maps are never materialized beyond one layer.

Per trace saves data/attn/{trace_id}.npz:
  mean [n_layers, T_gen, n_spans]  head-averaged attention mass to each span
  amax [n_layers, T_gen, n_spans]  max-over-heads (a single "instruction head" would
                                   vanish in the mean)
Spans: instr_real, instr_example, example_reasoning, example_answer, donor_question,
target_question, own_cot (generated keys up to & incl. current pos), other (residual
prompt: template tokens, preamble, options, format tail).

Mechanism note: this patches transformers' qwen3 eager_attention_forward to intercept
weights layer-by-layer. Asserts the hook point exists at startup — if a transformers
upgrade removes it, fail loudly rather than record nothing.

  python mech/attention_pass.py [--per-class 20] [--include-no-instruction]
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import ATTN_DIR, GRADES_PATH, MODEL_HF, PROMPTS_PATH, SPAN_NAMES, TRACES_PATH, load_jsonl

ALL_SPANS = SPAN_NAMES + ["own_cot", "other"]


def select_traces(per_class: int, include_no_instruction: bool, seed: int = 0) -> pd.DataFrame:
    g = pd.read_csv(GRADES_PATH)
    g = g[g.empty_thinking == 0]
    if not include_no_instruction:
        g = g[g.condition != "no_instruction"]
    rng = np.random.default_rng(seed)
    picks = []
    for (ct, cond), sub in g.groupby(["ctype", "condition"]):
        comp = sub[sub.compliance_score >= 0.75]
        nonc = sub[sub.compliance_score <= 0.25]
        for cls, pool in (("compliant", comp), ("noncompliant", nonc)):
            take = pool.sample(min(per_class, len(pool)), random_state=int(rng.integers(1 << 30)))
            take = take.assign(cls=cls)
            picks.append(take)
            if len(pool) < per_class:
                print(f"[attn] NOTE {ct}/{cond}/{cls}: only {len(pool)} traces in margin "
                      f"(wanted {per_class})")
    return pd.concat(picks, ignore_index=True)


class Collector:
    """Receives each layer's attention weights [1, H, T, T]; reduces query rows in the
    generated region to per-span key masses."""

    def __init__(self, n_layers: int, n_spans: int):
        self.n_layers, self.n_spans = n_layers, n_spans
        self.reset(0, 0, [])

    def reset(self, prompt_len: int, gen_len: int, spans: list[tuple[int, int] | None]):
        self.P, self.G, self.spans = prompt_len, gen_len, spans
        self.mean = np.zeros((self.n_layers, gen_len, self.n_spans), np.float16)
        self.amax = np.zeros((self.n_layers, gen_len, self.n_spans), np.float16)

    def __call__(self, layer_idx: int, w) -> None:
        import torch
        q = w[0, :, self.P - 1:self.P - 1 + self.G, :].float()   # queries emitting gen tokens
        H, G, T = q.shape
        masses = []
        used = torch.zeros(T, dtype=torch.bool, device=q.device)
        for ts in self.spans:                                     # named prompt spans
            if ts is None:
                masses.append(torch.zeros(H, G, device=q.device))
            else:
                s, e = ts
                masses.append(q[..., s:e].sum(-1))
                used[s:e] = True
        # own_cot: keys in the generated region up to & incl. the current position
        causal = torch.tril(torch.ones(G, T - self.P, device=q.device, dtype=torch.bool), 0)
        own = (q[..., self.P:] * causal).sum(-1)
        masses.append(own)
        used[self.P:] = True
        # other: residual prompt tokens (template/preamble/options/format tail)
        other = ~used
        other[self.P:] = False
        masses.append(q[..., other].sum(-1) if other.any()
                      else torch.zeros(H, G, device=q.device))
        m = torch.stack(masses, -1)                               # [H, G, S]
        self.mean[layer_idx] = m.mean(0).cpu().numpy().astype(np.float16)
        self.amax[layer_idx] = m.amax(0).cpu().numpy().astype(np.float16)


def patch_attention(collector: Collector):
    import transformers.models.qwen3.modeling_qwen3 as m
    assert hasattr(m, "eager_attention_forward"), (
        "transformers no longer exposes qwen3 eager_attention_forward — pin the version "
        "from pod requirements or update this hook point")
    orig = m.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, w = orig(module, query, key, value, attention_mask, scaling,
                      dropout=dropout, **kw)
        collector(module.layer_idx, w)
        return out, w

    m.eager_attention_forward = patched
    return orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--include-no-instruction", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sel = select_traces(args.per_class, args.include_no_instruction)
    print(f"[attn] selected {len(sel)} traces "
          f"({sel.groupby(['ctype', 'condition']).size().to_dict()})")
    sel.to_csv(ATTN_DIR.parent / "attn_selection.csv", index=False)

    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager")
    model.eval()
    n_layers = model.config.num_hidden_layers
    collector = Collector(n_layers, len(ALL_SPANS))
    patch_attention(collector)

    prompts = {p["prompt_id"]: p for p in load_jsonl(PROMPTS_PATH)}
    traces = {t["trace_id"]: t for t in load_jsonl(TRACES_PATH)}
    ATTN_DIR.mkdir(parents=True, exist_ok=True)

    todo = sel.trace_id.tolist()[:args.limit] if args.limit else sel.trace_id.tolist()
    for k, tid in enumerate(todo):
        out_path = ATTN_DIR / f"{tid.replace('|', '_')}.npz"
        if out_path.exists():
            continue
        t, p = traces[tid], prompts[traces[tid]["prompt_id"]]
        ids = p["prompt_token_ids"] + t["output_token_ids"]
        spans = [tuple(p["token_spans"][n]) if p["token_spans"].get(n) else None
                 for n in SPAN_NAMES]
        collector.reset(len(p["prompt_token_ids"]), len(t["output_token_ids"]), spans)
        # NOTE: output_attentions stays False — the patched eager_attention_forward
        # intercepts each layer's weights during computation and they're freed per
        # layer; passing output_attentions=True would accumulate all 64 full [H,T,T]
        # maps (~166 GB at T=4.5k) and OOM.
        with torch.no_grad():
            model(torch.tensor([ids], device=model.device))
        np.savez_compressed(out_path, mean=collector.mean, amax=collector.amax,
                            span_names=np.array(ALL_SPANS),
                            prompt_len=len(p["prompt_token_ids"]))
        if (k + 1) % 25 == 0:
            print(f"[attn] {k + 1}/{len(todo)}")
    print(f"[attn] done -> {ATTN_DIR}")


if __name__ == "__main__":
    main()
