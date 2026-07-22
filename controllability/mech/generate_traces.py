"""Pass 1/3 — generate all mechanism traces with vLLM (pod, 1xH200).

Builds cells+donors ONCE (mirroring icl_experiment's donor RNG), renders and tokenizes
every prompt, records labeled token spans, and persists everything downstream passes
need into data/prompts.jsonl. Then generates N_SAMPLES completions per prompt with vLLM
into data/traces.jsonl.

  python mech/generate_traces.py                # full: 6 types x 5 conditions x 30q x 10s = 9000
  python mech/generate_traces.py --smoke        # 2q x 2s per cell (pipeline check)
  python mech/generate_traces.py --check-spans  # no GPU: build prompts, print one, exit

Resumable: if prompts.jsonl exists it is REUSED (never rebuilt — build_all_cells can
drift across processes); traces already present are skipped.
"""

from __future__ import annotations

import argparse

import common as C
from common import (CONDITIONS, DATA, MAX_NEW_TOKENS, MECH_TYPES, MODEL_HF, N_QUESTIONS,
                    N_SAMPLES, PROMPTS_PATH, TEMPERATURE, TOP_P, TRACES_PATH,
                    append_jsonl, banned_first_token_ids, char_spans, load_jsonl,
                    messages_for, parse_think, prompt_id, render_prompt, to_token_spans,
                    trace_id)


def build_prompts(tokenizer, n_questions: int) -> list[dict]:
    """Build every (ctype, condition, question) prompt record. Called only when
    prompts.jsonl does not exist yet."""
    import random

    import icl_experiment as icl
    from icl_experiment import pick_donor
    from prefill_experiment import PrefillConfig, build_all_cells, _qhash

    cfg = PrefillConfig()
    cfg.n_questions = n_questions
    for k, v in MECH_TYPES.items():
        setattr(cfg, k, v)
    cells = build_all_cells(cfg)
    pool: dict[str, list] = {}
    for c in cells:
        pool.setdefault(c.ctype, []).append(c)
    rng = random.Random(cfg.seed + 1)                     # matches icl_experiment
    donor_of = {c.question: pick_donor(c, pool[c.ctype], rng) for c in cells}

    records = []
    for cell in cells:
        donor = donor_of[cell.question]
        if donor is None:
            continue
        for cond in CONDITIONS:
            msgs = messages_for(cell, donor, cond)
            text = render_prompt(tokenizer, msgs)
            enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            sliced = icl.slice_by_len(donor.controlled_cot, C.L5) if cond in C.ICL_CONDITIONS else ""
            cs = char_spans(text, cell, donor, cond, sliced)
            ts = to_token_spans(cs, enc["offset_mapping"])
            pid = prompt_id(cell.ctype, cond, _qhash(cell.question))
            records.append({
                "prompt_id": pid,
                "ctype": cell.ctype,
                "condition": cond,
                "qhash": _qhash(cell.question),
                "donor_qhash": _qhash(donor.question),
                "instruction": donor.instruction if cond != "no_instruction" else None,
                "prompt_text": text,
                "prompt_token_ids": list(enc["input_ids"]),
                "token_spans": ts,
                # grade spec (donor defines the instance; mirrors icl.grade_cell_for)
                "grade_modality": donor.modality,
                "grade_ctype": donor.ctype,
                "grade_marker": donor.marker,
                "grade_omit_words": donor.omit_words,
                "correct_letter": cell.correct_letter,
                "banned_token_ids": banned_first_token_ids(tokenizer, donor.omit_words)
                                    if cell.ctype == "omit_word" else [],
                "messages": msgs,
            })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 questions x 2 samples per cell")
    ap.add_argument("--check-spans", action="store_true", help="build prompts only, print one, exit")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    args = ap.parse_args()
    n_questions = 2 if args.smoke else N_QUESTIONS
    n_samples = args.n_samples or (2 if args.smoke else N_SAMPLES)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF)

    if PROMPTS_PATH.exists():
        prompts = load_jsonl(PROMPTS_PATH)
        print(f"[gen] REUSING existing {PROMPTS_PATH} ({len(prompts)} prompts) — never rebuilt")
    else:
        prompts = build_prompts(tokenizer, n_questions)
        n_missing = sum(1 for p in prompts for s in C.SPAN_NAMES
                        if p["condition"] in C.ICL_CONDITIONS and s != "instr_example"
                        and p["token_spans"].get(s) is None)
        if not args.check_spans:   # a --check-spans dry run must never persist prompts
            append_jsonl(PROMPTS_PATH, prompts)
            print(f"[gen] built {len(prompts)} prompts -> {PROMPTS_PATH} "
                  f"(missing ICL spans: {n_missing}, expect 0)")
        else:
            print(f"[gen] built {len(prompts)} prompts (dry run, not persisted; "
                  f"missing ICL spans: {n_missing}, expect 0)")

    if args.check_spans:
        p = next(x for x in prompts if x["condition"] == "icl_user_think")
        print(f"\n=== {p['prompt_id']} ===\n{p['prompt_text']}\n")
        for name, ts in p["token_spans"].items():
            if ts:
                seg = tokenizer.decode(p["prompt_token_ids"][ts[0]:ts[1]])
                print(f"  {name:18s} tok[{ts[0]}:{ts[1]}] = {seg[:90]!r}")
        return

    done = {(t["prompt_id"], t["sample"]) for t in load_jsonl(TRACES_PATH)}
    todo = [(p, s) for p in prompts for s in range(n_samples)
            if (p["prompt_id"], s) not in done]
    print(f"[gen] {len(todo)} completions to generate ({len(done)} cached)")
    if not todo:
        return

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    llm = LLM(model=MODEL_HF, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=args.gpu_mem)

    # group remaining samples per prompt so vLLM amortizes the prefill via n>1
    by_pid: dict[str, tuple[dict, list[int]]] = {}
    for p, s in todo:
        by_pid.setdefault(p["prompt_id"], (p, []))[1].append(s)

    BATCH = 64
    items = list(by_pid.values())
    for i in range(0, len(items), BATCH):
        chunk = items[i:i + BATCH]
        inputs = [TokensPrompt(prompt_token_ids=p["prompt_token_ids"]) for p, _ in chunk]
        sp = [SamplingParams(n=len(ss), temperature=TEMPERATURE, top_p=TOP_P,
                             max_tokens=MAX_NEW_TOKENS,
                             seed=abs(hash(p["prompt_id"])) % (2**31))
              for p, ss in chunk]
        outs = llm.generate(inputs, sp)
        rows = []
        for (p, ss), out in zip(chunk, outs):
            for s, comp in zip(ss, out.outputs):
                think, answer = parse_think(comp.text)
                rows.append({
                    "trace_id": trace_id(p["prompt_id"], s),
                    "prompt_id": p["prompt_id"],
                    "sample": s,
                    "output_token_ids": list(comp.token_ids),
                    "output_text": comp.text,
                    "think": think,
                    "answer_text": answer,
                    "finish_reason": comp.finish_reason,
                })
        append_jsonl(TRACES_PATH, rows)
        print(f"[gen] {min(i + BATCH, len(items))}/{len(items)} prompts done")
    print(f"[gen] all traces -> {TRACES_PATH}")


if __name__ == "__main__":
    main()
