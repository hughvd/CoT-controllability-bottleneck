"""4b (causal) — diff-in-means steering of the BASE model's control compliance.

Question: is the compliance-engagement direction we can READ (probe) also one we
can WRITE? For a control type, take v = mean(compliant eop/early activations) -
mean(noncompliant ...) at layer L from instruction_only base traces, then add alpha *
v_hat to the residual stream (output of block L) during generation and remeasure
compliance. Cross-type directions are orthogonal (probe_crosstype), so vectors are
PER-TYPE.

v is built from ALL instruction_only base rollouts of a type; steering is tested on
FRESH samples of the same questions (no rollout overlap, but question overlap — the
random-direction and negative-alpha controls, not a question split, are what guard
against a trivial result; nfp/omit compliant rollouts are too sparse to also hold out
questions). Caveat logged accordingly.

Rigor / controls:
  * alpha = 0                    baseline (no steering, same code path)
  * alpha > 0 (two magnitudes)   does pushing along v induce compliance?
  * alpha < 0                    does pushing against v SUPPRESS it? (sign test)
  * random unit dir, matched |.| does ANY push of this norm work, or specifically v?
  * no_instruction + alpha>0     does v induce compliance with NO instruction present?
                                 (if yes, v just writes the surface form; if no, v
                                  gates an instruction-conditioned mechanism)
Also logs accuracy (steering must not destroy the answer) and empty/truncation rates.

alpha is a MULTIPLIER on the raw diff-in-means v (alpha=1 adds exactly one class-mean
shift at layer L; behavioral effects usually need alpha ~ 4-12). Generations are
batched with LEFT padding, so the true last prompt token sits at index -1 for every
row and the prefill hook steers it uniformly.

  # calibrate magnitude first (find where behavior moves):
  python -u mech/steering.py --calibrate
  # full:
  python -u mech/steering.py --types no_first_person,omit_word --layers 40 \
      --alphas -8,0,4,8 --n-questions 30 --n-samples 4

Outputs data/steering/traces.jsonl (+ grade with grade_traces.py --traces ... --out ...).
Vectors cached in data/steering/vecs/{ctype}_L{L}_fold{f}.npz.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import common as C
from common import DATA, MODEL_HF, PROMPTS_PATH, append_jsonl, load_jsonl, parse_think

sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
from common_analysis import tid_fname

SDIR = DATA / "steering"
VECDIR = SDIR / "vecs"
PROBE_TRACE = DATA / "probe" / "trace"
PROBE_PROMPT = DATA / "probe" / "prompt"
LAYER_HS_OFFSET = 1   # hidden_states[L+1] == output of block L
EARLY_POS = [0, 1, 2, 4]   # activations averaged into the class means (+ eop)


def load_grades_local():
    import pandas as pd
    g = pd.read_csv(DATA / "grades.csv")
    g = g[g.empty_thinking == 0].copy()
    g["cls"] = None
    for t in ("no_first_person", "omit_word"):
        m = g.ctype == t
        g.loc[m & (g.compliance_score >= 1.0), "cls"] = "compliant"
        g.loc[m & (g.compliance_score <= 0.0), "cls"] = "noncompliant"
    for t in ("jar_jar_binks", "spanish"):
        m = g.ctype == t
        g.loc[m & (g.compliance_score >= 0.75), "cls"] = "compliant"
        g.loc[m & (g.compliance_score <= 0.25), "cls"] = "noncompliant"
    return g


def build_vector(ct, layer, train_qhashes, grades, include_eop=True):
    """diff-in-means (compliant - noncompliant) at layer, averaged over [eop +] early
    generated positions, from instruction_only base traces on TRAIN questions. NOTE eop
    is SHARED across a prompt's rollouts, so its class-diff reflects question difficulty,
    not within-rollout engagement; --exclude-eop drops it to isolate the generated signal."""
    li = [8, 16, 24, 32, 40, 48, 56, 63].index(layer)
    sub = grades[(grades.ctype == ct) & (grades.condition == "instruction_only")
                 & grades.cls.notna() & grades.qhash.isin(train_qhashes)]
    acc = {"compliant": [], "noncompliant": []}
    for r in sub.itertuples():
        tp = PROBE_TRACE / f"{tid_fname(r.trace_id)}.npz"
        pp = PROBE_PROMPT / f"{tid_fname(r.prompt_id)}.npz"
        if not (tp.exists() and pp.exists()):
            continue
        vecs = [np.load(pp)["eop"][li]] if include_eop else []
        d = np.load(tp); gp = list(d["gen_positions"])
        for p in EARLY_POS:
            if p in gp:
                vecs.append(d["hs"][li, gp.index(p)])
        acc[r.cls].append(np.mean(vecs, axis=0).astype(np.float32))
    if len(acc["compliant"]) < 5 or len(acc["noncompliant"]) < 5:
        return None, (len(acc["compliant"]), len(acc["noncompliant"]))
    v = np.mean(acc["compliant"], 0) - np.mean(acc["noncompliant"], 0)
    return v.astype(np.float32), (len(acc["compliant"]), len(acc["noncompliant"]))


def typical_norm(ct, layer, grades):
    """Median residual-stream norm at layer (for scaling alpha to model units)."""
    li = [8, 16, 24, 32, 40, 48, 56, 63].index(layer)
    sub = grades[(grades.ctype == ct) & (grades.condition == "instruction_only")].head(40)
    ns = []
    for r in sub.itertuples():
        pp = PROBE_PROMPT / f"{tid_fname(r.prompt_id)}.npz"
        if pp.exists():
            ns.append(float(np.linalg.norm(np.load(pp)["eop"][li])))
    return float(np.median(ns)) if ns else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="no_first_person,omit_word")
    ap.add_argument("--layers", default="40")
    ap.add_argument("--alphas", default="-8,0,4,8", help="MULTIPLIER on raw diff-in-means v")
    ap.add_argument("--n-questions", type=int, default=30)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--run-tag", default="", help="output file suffix (separate runs)")
    ap.add_argument("--exclude-eop", action="store_true",
                    help="build v from generated positions only (drop the shared eop)")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    OUT = SDIR / (f"traces_{args.run_tag}.jsonl" if args.run_tag else "traces.jsonl")
    if args.calibrate:
        args.types, args.layers = "no_first_person", "40"
        args.alphas = "0,2,4,8,16"
        args.n_questions, args.n_samples, args.max_new = 6, 2, 640

    types = args.types.split(",")
    layers = [int(x) for x in args.layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_HF)
    tok.padding_side = "left"                     # real tokens flush-right -> idx -1 uniform
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_HF, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()

    grades = load_grades_local()
    prompts = {p["prompt_id"]: p for p in load_jsonl(PROMPTS_PATH)}
    VECDIR.mkdir(parents=True, exist_ok=True)
    done = {(t["trace_id"]) for t in load_jsonl(OUT)} if OUT.exists() else set()

    # steering hook state (mutated per condition)
    state = {"vec": None, "on": False, "max_tok": 10**9, "count": 0}

    def make_hook():
        def hook(module, inp, out):
            if not state["on"] or state["vec"] is None:
                return out
            hs = out[0] if isinstance(out, tuple) else out
            seq = hs.shape[1]
            if seq > 1:                      # prefill: steer only the LAST prompt pos
                hs[:, -1, :] += state["vec"]
                state["count"] = 1
            else:                            # decode step
                if state["count"] < state["max_tok"]:
                    hs[:, 0, :] += state["vec"]
                    state["count"] += 1
            return (hs,) + out[1:] if isinstance(out, tuple) else hs
        return hook

    for L in layers:
        h = model.model.layers[L].register_forward_hook(make_hook())
        for ct in types:
            qs = sorted(grades[(grades.ctype == ct)
                               & (grades.condition == "instruction_only")].qhash.unique())
            qs = qs[:args.n_questions]
            rng = np.random.default_rng(0)
            tnorm = typical_norm(ct, L, grades)
            v, counts = build_vector(ct, L, set(qs), grades, include_eop=not args.exclude_eop)
            if v is None:
                print(f"[steer] {ct} L{L}: too few class examples {counts}; skip")
                continue
            np.savez(VECDIR / f"{ct}_L{L}.npz", v=v, tnorm=tnorm,
                     n_comp=counts[0], n_non=counts[1])
            print(f"[steer] {ct} L{L}: vector from {counts[0]} compliant / {counts[1]} "
                  f"noncompliant; |v|={np.linalg.norm(v):.1f}, |resid|~{tnorm:.0f}", flush=True)
            rand = rng.standard_normal(v.shape).astype(np.float32)
            rand *= np.linalg.norm(v) / (np.linalg.norm(rand) + 1e-9)   # match |v|

            conds = [("instruction_only", a, "v") for a in alphas]
            conds += [("instruction_only", max(alphas), "rand")]      # random-dir control
            conds += [("no_instruction", max(alphas), "v")]           # no-instr induction
            for base_cond, alpha, kind in conds:
                direction = v if kind == "v" else rand           # RAW v (alpha = multiplier)
                svec = torch.tensor(alpha * direction, dtype=torch.bfloat16, device="cuda")
                tag = f"{kind}_a{alpha:g}_{base_cond[:5]}"
                # build (prompt, sample) job list for this condition
                jobs = []
                for q in qs:
                    pid = f"{ct}|{base_cond}|{q}"
                    p = prompts.get(pid)
                    if p is None:
                        continue
                    for s in range(args.n_samples):
                        tid = f"{ct}|L{L}|{tag}|{q}|s{s}"
                        if tid not in done:
                            jobs.append((tid, pid, q, s, p["prompt_token_ids"]))
                for i in range(0, len(jobs), args.batch):
                    chunk = jobs[i:i + args.batch]
                    maxlen = max(len(j[4]) for j in chunk)
                    ids = torch.full((len(chunk), maxlen), tok.pad_token_id,
                                     dtype=torch.long, device="cuda")
                    mask = torch.zeros((len(chunk), maxlen), dtype=torch.long, device="cuda")
                    for r, j in enumerate(chunk):
                        ids[r, maxlen - len(j[4]):] = torch.tensor(j[4], device="cuda")
                        mask[r, maxlen - len(j[4]):] = 1
                    state.update(vec=(svec if alpha != 0 else None),
                                 on=(alpha != 0), max_tok=10**9, count=0)
                    with torch.no_grad():
                        g = model.generate(ids, attention_mask=mask, do_sample=True,
                                           temperature=C.TEMPERATURE, top_p=C.TOP_P,
                                           max_new_tokens=args.max_new,
                                           pad_token_id=tok.pad_token_id)
                    state["on"] = False
                    rows = []
                    for r, j in enumerate(chunk):
                        gen_ids = g[r, maxlen:].tolist()
                        gen_ids = [t for t in gen_ids if t != tok.pad_token_id]
                        text = tok.decode(gen_ids)
                        think, answer = parse_think(text if text.startswith("<think>")
                                                    else "<think>" + text)
                        rows.append({
                            "trace_id": j[0], "prompt_id": j[1], "sample": j[3],
                            "ctype": ct, "condition": base_cond, "layer": L,
                            "alpha": alpha, "kind": kind, "variant": tag,
                            "output_token_ids": gen_ids, "output_text": text,
                            "think": think, "answer_text": answer,
                            "finish_reason": "length" if len(gen_ids) >= args.max_new else "stop",
                        })
                    append_jsonl(OUT, rows)
                print(f"[steer] {ct} L{L} {tag}: done ({len(jobs)} gens)", flush=True)
        h.remove()
    print(f"[steer] all done -> {OUT}")


if __name__ == "__main__":
    main()
