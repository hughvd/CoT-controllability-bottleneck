"""OLMo-3-32B RL-axis mechanism pass: first-token entropy, opening-mass, and
adjacent-checkpoint KL along the RL ladder (DPO -> RLVR -> extended RL).

Why: the 7B entrenchment sweep + the API sweep suggest RL RAISES controllability.
The qwen mech results say control failure is a propensity problem at the trace
OPENING (compliant openings exist as suppressed probability mass; fate is readable
at the first token). So per RL checkpoint we measure, on the same fixed prompts:
  * per-token entropy of the first K generated tokens (no_instruction vs
    instruction_only; later split compliant vs not via olmo_rl_grade.py)
  * opening probability mass on instruction-consistent tokens (uppercase tokens,
    digit tokens for leetspeak, Spanish-marker tokens, P(banned) for omit_word)
  * teacher-forced per-position KL between ADJACENT checkpoints (is RL's change
    front-loaded at the opening, like SFT's was?)

4 ctypes (one per category): uppercase_thinking / leetspeak / spanish / omit_word,
2 conditions (no_instruction, instruction_only), 30 questions x 8 samples.
Prompts are built ONCE with the first checkpoint's tokenizer and shared across the
whole ladder (all OLMo-3(.1)-32B checkpoints share a tokenizer; a canary encode is
asserted per run). OLMo's chat template force-opens <think>, so generated position
0 IS the first reasoning token.

Stages (pod):
  python mech/olmo_rl_pass.py --stage gen   --model <ckpt_dir> --tag rl_start --tp 2
  python mech/olmo_rl_pass.py --stage stats --model <ckpt_dir> --tag rl_start
  python mech/olmo_rl_pass.py --stage kl    --model-a <prev_dir> --tag-a rl_start \
                                            --model-b <cur_dir>  --tag-b rlvr_final

Outputs under mech/data/olmo_rl/:
  prompts.jsonl                          shared prompt records (+ _meta canary row)
  <tag>/traces.jsonl                     sampled completions
  <tag>/stats/<tid>.npz                  per-position entropy/top5/masses (first K)
  kl/<tagA>__<tagB>/<srctag>/<tid>.npz   per-position KL both directions
Grading (laptop, LLM judge for spanish): mech/olmo_rl_grade.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CTRL = HERE.parent
for p in (str(HERE), str(CTRL), str(CTRL.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import append_jsonl, banned_first_token_ids, load_jsonl, parse_think  # noqa: E402

ODATA = HERE / "data" / "olmo_rl"
PROMPTS = ODATA / "prompts.jsonl"

CTYPES = ["uppercase_thinking", "leetspeak", "spanish", "omit_word"]
CONDITIONS = ["no_instruction", "instruction_only"]
N_QUESTIONS = 30
N_SAMPLES = 8
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_NEW_TOKENS = 8192
K_STATS = 256          # generated positions kept in the stats/kl passes
N_KL_SAMPLES = 4       # traces per prompt entering the KL pass
CANARY = "The quick brown fox: ¿cuál es la respuesta? 42 THINK"

# Distinctly-Spanish opening/content words for the opening-mass probe. Their
# first-token ids (>=3 decoded chars, so generic BPE prefixes are excluded) form
# the "spanish mass" set — a RELATIVE measure across checkpoints/conditions, not
# an absolute Spanish detector.
SPANISH_WORDS = [
    "bien", "bueno", "vale", "primero", "veamos", "vamos", "pensemos",
    "analicemos", "entonces", "pues", "según", "pregunta", "respuesta",
    "opción", "necesito", "debo", "cuál", "también", "está", "porque",
    "tenemos", "entender", "español",
]


def qhash(q: str) -> str:
    return hashlib.sha256(str(q).encode("utf-8")).hexdigest()[:16]


def stable_seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# Prompt building (once, shared across the ladder)
# --------------------------------------------------------------------------- #

def assemble_prompt_bare(question: str, options: list[str]) -> str:
    """run_eval_diverse.assemble_prompt WITHOUT the <cot_instructions> head —
    the true default-behavior reference prompt."""
    body = "\n\n" + "\n".join(
        f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options))
    tail = ("\n\nYour response should be of the following format: "
            "'ANSWER: X' (without quotes) where X is your chosen letter")
    return f"Question: {question}" + body + tail


def build_prompts(tokenizer) -> list[dict]:
    import pandas as pd

    import suppression as supp
    from run_eval_diverse import answer_to_letter, assemble_prompt, parse_options
    from sft.language_instructions import build_language_instruction, load_language_defs
    from transforms import transform_cot

    rng = random.Random(20260730)
    test = pd.read_csv(CTRL / "data" / "split_test.csv")
    questions = test.to_dict(orient="records")
    picks = rng.sample(questions, k=min(N_QUESTIONS, len(questions)))
    ldef = load_language_defs()["spanish"]

    records = []
    for q in picks:
        options = parse_options(pd.Series(q))
        letter = answer_to_letter(q["answer"], options)
        if letter is None:
            continue
        qh = qhash(q["question"])
        for ctype in CTYPES:
            # one canonical TRAIN-pool instruction per (ctype, question)
            irng = random.Random(stable_seed(ctype, qh))
            omit_words: list[str] = []
            if ctype == "spanish":
                modality, instruction = "language", build_language_instruction(
                    "spanish", ldef, rng=irng, phrasing_pool="train")
            elif ctype == "omit_word":
                omit_words = [irng.choice(supp.FORBIDDEN_TRAIN)]
                modality, instruction = "suppression", supp.build_suppression_instruction(
                    "omit_word", omit_words, rng=irng, phrasing_pool="train")
            else:
                modality = "family"
                _, instruction = transform_cot(
                    "placeholder", ctype, marker=None, rng=irng, phrasing_pool="train")
            for cond in CONDITIONS:
                prompt = (assemble_prompt(q["question"], options, instruction)
                          if cond == "instruction_only"
                          else assemble_prompt_bare(q["question"], options))
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True)
                enc = tokenizer(text, add_special_tokens=False)
                records.append({
                    "prompt_id": f"{ctype}|{cond}|{qh}",
                    "ctype": ctype, "condition": cond, "qhash": qh,
                    "modality": modality,
                    "instruction": instruction if cond == "instruction_only" else None,
                    "omit_words": omit_words,
                    "correct_letter": letter,
                    "question": q["question"],
                    "prompt_text": text,
                    "prompt_token_ids": list(enc["input_ids"]),
                    "banned_token_ids": banned_first_token_ids(tokenizer, omit_words)
                                        if ctype == "omit_word" else [],
                    "think_open_in_prompt": text.rstrip().endswith("<think>"),
                })
    return records


def check_tokenizer(tokenizer, meta: dict) -> None:
    ids = tokenizer(CANARY, add_special_tokens=False)["input_ids"]
    if list(ids) != meta["canary_ids"]:
        raise SystemExit(
            "[olmo_rl] tokenizer mismatch vs persisted prompts.jsonl — this checkpoint "
            "does not share the vocab the prompts were tokenized with. Refusing to run.")


def load_prompts(tokenizer) -> list[dict]:
    rows = load_jsonl(PROMPTS)
    meta = next((r for r in rows if r.get("_meta")), None)
    prompts = [r for r in rows if not r.get("_meta")]
    if meta is None:
        raise SystemExit(f"[olmo_rl] {PROMPTS} missing/has no _meta row")
    check_tokenizer(tokenizer, meta)
    return prompts


def build_or_load_prompts(tokenizer) -> list[dict]:
    if PROMPTS.exists():
        prompts = load_prompts(tokenizer)
        print(f"[prompts] reuse {PROMPTS} ({len(prompts)} prompts)")
        return prompts
    recs = build_prompts(tokenizer)
    n_open = sum(r["think_open_in_prompt"] for r in recs)
    print(f"[prompts] built {len(recs)} ({len(CTYPES)} ctypes x {len(CONDITIONS)} conds); "
          f"think force-opened in {n_open}/{len(recs)} prompts")
    meta = {"_meta": True,
            "canary_ids": list(tokenizer(CANARY, add_special_tokens=False)["input_ids"]),
            "vocab_len": len(tokenizer)}
    ODATA.mkdir(parents=True, exist_ok=True)
    append_jsonl(PROMPTS, [meta] + recs)
    return recs


# --------------------------------------------------------------------------- #
# gen — vLLM sampling
# --------------------------------------------------------------------------- #

def gen(model_path: str, tag: str, tp: int, n_samples: int, gpu_mem: float) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    prompts = build_or_load_prompts(tok)

    tpath = ODATA / tag / "traces.jsonl"
    done = {(t["prompt_id"], t["sample"]) for t in load_jsonl(tpath)}
    todo = [(p, s) for p in prompts for s in range(n_samples)
            if (p["prompt_id"], s) not in done]
    print(f"[gen:{tag}] {len(todo)} completions to generate ({len(done)} cached)")
    if not todo:
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=model_path, dtype="bfloat16", max_model_len=10240,
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
                    "trace_id": f"{p['prompt_id']}|s{s}",
                    "prompt_id": p["prompt_id"], "sample": s,
                    "ctype": p["ctype"], "condition": p["condition"],
                    "output_token_ids": list(comp.token_ids), "output_text": comp.text,
                    "think": think, "answer_text": answer,
                    "finish_reason": comp.finish_reason,
                })
        append_jsonl(tpath, rows)
        print(f"[gen:{tag}] {min(i + BATCH, len(items))}/{len(items)} prompts")
    print(f"[gen:{tag}] traces -> {tpath}")


# --------------------------------------------------------------------------- #
# stats — teacher-forced entropy + opening masses (first K_STATS positions)
# --------------------------------------------------------------------------- #

def build_token_sets(tokenizer) -> dict[str, list[int]]:
    """Instruction-consistent token-id sets, computed once from the vocab.
      upper : decoded strip() is >=2 alphabetic chars, all uppercase
      digit : decoded contains a digit (leetspeak proxy)
      spanish: first-token ids of SPANISH_WORDS variants whose first token
               decodes to >=3 chars (drops generic BPE prefixes)
    """
    upper, digit = [], []
    for i in range(len(tokenizer)):
        s = tokenizer.decode([i])
        t = s.strip()
        if len(t) >= 2 and t.isalpha() and t.isupper():
            upper.append(i)
        if any(c.isdigit() for c in s):
            digit.append(i)
    spanish = set()
    for w in SPANISH_WORDS:
        for form in {w, w.capitalize(), w.upper()}:
            for pre in ("", " "):
                ids = tokenizer(pre + form, add_special_tokens=False)["input_ids"]
                if ids and len(tokenizer.decode([ids[0]]).strip()) >= 3:
                    spanish.add(int(ids[0]))
    print(f"[stats] token sets: upper={len(upper)} digit={len(digit)} spanish={len(spanish)}")
    return {"upper": upper, "digit": digit, "spanish": sorted(spanish)}


def forward_stats_k(model, torch, prompt_ids, out_ids, banned, sets, k=K_STATS) -> dict:
    out_ids = out_ids[:k]
    ids = torch.tensor([prompt_ids + out_ids], device=model.device)
    with torch.no_grad():
        logits = model(ids).logits[0]
    P = len(prompt_ids)
    gen_logits = logits[P - 1: P + len(out_ids) - 1].float()   # predicts out_ids
    logp = torch.log_softmax(gen_logits, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(-1)
    top_p, top_i = p.topk(5, dim=-1)
    stats = {
        "entropy": entropy.cpu().numpy().astype(np.float16),
        "top5_ids": top_i.cpu().numpy().astype(np.int32),
        "top5_p": top_p.cpu().numpy().astype(np.float16),
        "sampled_logp": logp[torch.arange(len(out_ids)),
                             torch.tensor(out_ids, device=logp.device)]
                        .cpu().numpy().astype(np.float16),
    }
    for name, tens in sets.items():
        stats[f"mass_{name}"] = p[:, tens].sum(-1).cpu().numpy().astype(np.float16)
    if banned:
        stats["p_banned"] = p[:, banned].sum(-1).cpu().numpy().astype(np.float16)
    del logits, gen_logits, logp, p
    return stats


def stats(model_path: str, tag: str, limit: int | None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)   # H100 guard; no-op elsewhere
    tok = AutoTokenizer.from_pretrained(model_path)
    prompts = {p["prompt_id"]: p for p in load_prompts(tok)}
    sets = {n: torch.tensor(v) for n, v in build_token_sets(tok).items()}

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa").eval()
    sets = {n: v.to(model.device) for n, v in sets.items()}

    traces = load_jsonl(ODATA / tag / "traces.jsonl")
    if limit:
        traces = traces[:limit]
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
                            ctype=np.array(p["ctype"]), condition=np.array(p["condition"]),
                            **s)
        n_new += 1
        if (k + 1) % 200 == 0:
            print(f"[stats:{tag}] {k + 1}/{len(traces)} ({n_new} new)")
    print(f"[stats:{tag}] done: {len(traces)} traces, {n_new} new -> {sdir}")


# --------------------------------------------------------------------------- #
# kl — adjacent-checkpoint teacher-forced KL (one model per GPU)
# --------------------------------------------------------------------------- #

def _gen_logp(model, torch, ids, P: int, G: int):
    with torch.no_grad():
        logits = model(ids.to(model.device)).logits[0, P - 1: P - 1 + G].float()
    return torch.log_softmax(logits, dim=-1)


def kl(model_a: str, tag_a: str, model_b: str, tag_b: str, limit: int | None) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    tok = AutoTokenizer.from_pretrained(model_a)
    prompts = {p["prompt_id"]: p for p in load_prompts(tok)}

    # traces from BOTH checkpoints (each side's own continuations, teacher-forced
    # through both models), capped at N_KL_SAMPLES per prompt
    jobs = []
    for src in (tag_a, tag_b):
        tr = load_jsonl(ODATA / src / "traces.jsonl")
        if not tr:
            print(f"[kl] NOTE: no traces for {src} yet — pair runs on the other side only")
        jobs += [(src, t) for t in tr if t["sample"] < N_KL_SAMPLES]
    if limit:
        jobs = jobs[:limit]
    if not jobs:
        raise SystemExit("[kl] no traces on either side")

    print(f"[kl] loading {tag_a} -> cuda:0, {tag_b} -> cuda:1 ...")
    ma = AutoModelForCausalLM.from_pretrained(
        model_a, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        attn_implementation="sdpa").eval()
    mb = AutoModelForCausalLM.from_pretrained(
        model_b, torch_dtype=torch.bfloat16, device_map={"": "cuda:1"},
        attn_implementation="sdpa").eval()

    kdir = ODATA / "kl" / f"{tag_a}__{tag_b}"
    n_new = 0
    print(f"[kl] {len(jobs)} traces (x2 forwards each)")
    for k, (src, t) in enumerate(jobs):
        odir = kdir / src
        odir.mkdir(parents=True, exist_ok=True)
        out = odir / f"{t['trace_id'].replace('|', '_')}.npz"
        if out.exists() or not t["output_token_ids"]:
            continue
        p = prompts[t["prompt_id"]]
        P = len(p["prompt_token_ids"])
        out_ids = t["output_token_ids"][:K_STATS]
        G = len(out_ids)
        ids = torch.tensor([p["prompt_token_ids"] + out_ids])

        lp_a = _gen_logp(ma, torch, ids, P, G)
        lp_b = _gen_logp(mb, torch, ids, P, G).to(lp_a.device)
        pa, pb = lp_a.exp(), lp_b.exp()
        tgt = torch.tensor(out_ids, device=lp_a.device)
        ar = torch.arange(G, device=lp_a.device)
        st = {
            "kl_a_b": (pa * (lp_a - lp_b)).sum(-1),
            "kl_b_a": (pb * (lp_b - lp_a)).sum(-1),
            "ent_a": -(pa * lp_a).sum(-1),
            "ent_b": -(pb * lp_b).sum(-1),
            "agree_top1": (lp_a.argmax(-1) == lp_b.argmax(-1)).float(),
            "logp_a": lp_a[ar, tgt],
            "logp_b": lp_b[ar, tgt],
        }
        banned = p.get("banned_token_ids") or []
        if banned:
            st["p_banned_a"] = pa[:, banned].sum(-1)
            st["p_banned_b"] = pb[:, banned].sum(-1)
        np.savez_compressed(
            out, gen_len=len(t["output_token_ids"]), k_gen=G, source=np.array(src),
            ctype=np.array(p["ctype"]), condition=np.array(p["condition"]),
            **{k2: v.cpu().numpy().astype(np.float16) for k2, v in st.items()})
        del lp_a, lp_b, pa, pb
        n_new += 1
        if (k + 1) % 200 == 0:
            print(f"[kl] {k + 1}/{len(jobs)} ({n_new} new)")
    print(f"[kl] done: {len(jobs)} traces, {n_new} new -> {kdir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prompts", "gen", "stats", "kl"], required=True)
    ap.add_argument("--model", help="checkpoint dir (gen/stats; prompts uses its tokenizer)")
    ap.add_argument("--tag", help="checkpoint tag, e.g. rl_start (gen/stats)")
    ap.add_argument("--model-a"); ap.add_argument("--tag-a")
    ap.add_argument("--model-b"); ap.add_argument("--tag-b")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.stage == "prompts":
        from transformers import AutoTokenizer
        build_or_load_prompts(AutoTokenizer.from_pretrained(args.model))
    elif args.stage == "gen":
        gen(args.model, args.tag, args.tp, args.n_samples, args.gpu_mem)
    elif args.stage == "stats":
        stats(args.model, args.tag, args.limit)
    else:
        for a in ("model_a", "tag_a", "model_b", "tag_b"):
            if not getattr(args, a):
                raise SystemExit(f"--stage kl requires --{a.replace('_', '-')}")
        kl(args.model_a, args.tag_a, args.model_b, args.tag_b, args.limit)


if __name__ == "__main__":
    main()
