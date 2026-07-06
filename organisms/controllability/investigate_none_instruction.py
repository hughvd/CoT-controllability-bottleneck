"""Does '<cot_instructions> none </cot_instructions>' suppress 32b's reasoning?

Small trial: for one ctype's questions, forced `<think>\n` prefill at K=0, prefill_only,
compare 3 prompt framings and measure the EMPTY-thinking rate (model closes </think>
immediately). If 'none' is the culprit, dropping the block should lower empties.

  python investigate_none_instruction.py
"""

from __future__ import annotations

import asyncio
import random

import prefill_experiment as pe
from prefill_experiment import PrefillConfig, load_raw_questions, _build_family_cells

N_Q = 20
MODEL = "qwen/qwen3-32b"


def variant_prompt(kind: str, question: str, options: list[str]) -> str:
    body = "\n\n" + "\n".join(f"{chr(ord('A')+i)}) {o}" for i, o in enumerate(options))
    tail = ("\n\nYour response should be of the following format: 'ANSWER: X' "
            "(without quotes) where X is your chosen letter")
    if kind == "none":                       # current prefill_only rendering
        head = f"<cot_instructions> none </cot_instructions>\nQuestion: {question}"
    elif kind == "dropped":                  # no cot_instructions block at all
        head = f"Question: {question}"
    elif kind == "think":                     # positive control: explicit reason-first
        head = ("<cot_instructions> Think step by step before answering. "
                f"</cot_instructions>\nQuestion: {question}")
    return head + body + tail


async def run() -> None:
    cfg = PrefillConfig()
    cfg.model_id = MODEL
    cfg.provider = "DeepInfra"
    raws = load_raw_questions()
    rng = random.Random(0)
    cells = _build_family_cells("uppercase_thinking", raws, cfg, rng)[:N_Q]
    print(f"{len(cells)} questions, model={MODEL}\n")
    sem = asyncio.Semaphore(12)

    async def one(cell, kind):
        prompt = variant_prompt(kind, cell.question, cell.options)
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": "<think>\n"}]  # forced
        async with sem:
            resp, _ = await pe.complete_pinned(cfg, messages)
        return not (resp.thinking or "").strip()

    for kind in ("none", "dropped", "think"):
        empties = await asyncio.gather(*[one(c, kind) for c in cells])
        frac = sum(empties) / len(empties)
        print(f"  {kind:8s}: empty-thinking {frac:.0%}  ({sum(empties)}/{len(empties)})")


if __name__ == "__main__":
    asyncio.run(run())
