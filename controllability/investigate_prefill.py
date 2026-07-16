"""Investigate how OpenRouter actually renders an assistant <think> prefill.

We POST a trailing assistant message; we do NOT control the chat template, so we
must verify empirically whether the provider:
  (A) CONTINUES our partial <think> as a single reasoning block (what we want), or
  (B) adds its OWN <think> (double-think), or
  (C) treats our message as a completed prior turn + fresh generation prompt.

For one uppercase cell we send three variants at K=3 and dump the FULL raw
response (response.model_dump()) so nothing is hidden by our parser:
  V1  assistant = "<think>\n" + seed     (current experiment behaviour)
  V2  assistant = seed                    (no tag)
  V3  no assistant message                (K=0 control)

Diagnostics printed per variant:
  - which provider served it
  - finish_reason
  - whether '<think>'/'</think>' appear in message.content   (double-think tell)
  - the reasoning field (continuation) vs content (answer)
  - reasoning_details, if present

Raw responses are saved under transcripts/ for inspection.

Usage:
  python investigate_prefill.py
  python investigate_prefill.py --model qwen/qwen3-14b --provider "DeepInfra"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import prefill_experiment as P
from lib.providers.openrouter import get_async_client

HERE = Path(__file__).resolve().parent
TRANSCRIPTS = HERE / "transcripts"


async def raw_call(model: str, messages: list[dict], provider: str | None,
                   max_tokens: int = 2048) -> dict:
    client = get_async_client()
    kwargs = dict(model=model, messages=messages, temperature=0.6, max_tokens=max_tokens)
    extra: dict = {}
    if provider:
        extra["provider"] = {"order": [provider], "allow_fallbacks": False}
    if extra:
        kwargs["extra_body"] = extra
    resp = await client.chat.completions.create(**kwargs)
    return resp.model_dump()


def summarize(tag: str, messages: list[dict], raw: dict) -> None:
    choice = raw["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning")
    rdetails = msg.get("reasoning_details")
    provider = raw.get("provider")
    print("\n" + "=" * 90)
    print(f"VARIANT {tag}   provider={provider}   finish={choice.get('finish_reason')}")
    print("-" * 90)
    asst = next((m["content"] for m in messages if m["role"] == "assistant"), None)
    print(f"assistant prefill SENT: {asst!r}" if asst else "assistant prefill SENT: (none)")
    print(f"\n[flags]  '<think>' in content: {'<think>' in content}   "
          f"'</think>' in content: {'</think>' in content}   "
          f"reasoning field present: {reasoning is not None}")
    print(f"\n[message.content] (len {len(content)}):\n{content[:1200]}")
    if reasoning is not None:
        print(f"\n[message.reasoning] (len {len(reasoning)}):\n{reasoning[:1200]}")
    if rdetails:
        print(f"\n[reasoning_details]:\n{json.dumps(rdetails, ensure_ascii=False)[:800]}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen3-14b")
    ap.add_argument("--provider", default=None, help="pin a single OpenRouter provider")
    args = ap.parse_args()

    TRANSCRIPTS.mkdir(exist_ok=True)
    rng = random.Random(0)
    raws = P.load_raw_questions()
    cell = P._build_family_cells("uppercase_thinking", raws,
                                 P.PrefillConfig(n_questions=1), rng)[0]
    seed = P.first_k_sentences(cell.controlled_cot, 3)
    prompt = P.assemble_prompt(cell.question, cell.options, "none")

    variants = {
        "V1_tag":   [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": "<think>\n" + seed}],
        "V2_notag": [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": seed}],
        "V3_none":  [{"role": "user", "content": prompt}],
    }
    print(f"SEED (K=3): {seed!r}")
    for tag, messages in variants.items():
        try:
            raw = await raw_call(args.model, messages, args.provider)
        except Exception as e:
            print(f"\nVARIANT {tag}: ERROR {type(e).__name__}: {e}")
            continue
        (TRANSCRIPTS / f"{tag}.json").write_text(
            json.dumps({"request_messages": messages, "raw_response": raw},
                       ensure_ascii=False, indent=2))
        summarize(tag, messages, raw)
    print(f"\nraw transcripts saved under {TRANSCRIPTS}/")


if __name__ == "__main__":
    asyncio.run(main())
