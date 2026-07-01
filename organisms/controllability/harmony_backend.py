"""gpt-oss harmony backend for the prefill experiment (runs on the pod vs local vLLM).

gpt-oss reasons in OpenAI 'harmony' channels, not <think> tags. To seed its CoT we
render a harmony prompt that ends inside an unclosed `analysis` message containing
the K-sentence seed, send the token IDs to vLLM /v1/completions, and split the
continuation's analysis channel (graded) from the final channel (the answer).

openai_harmony is imported lazily so prefill_experiment.py still imports on a box
without it (only the gpt-oss run needs it, on the pod).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

_ENC = None


def _enc():
    global _ENC
    if _ENC is None:
        from openai_harmony import load_harmony_encoding, HarmonyEncodingName
        _ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _ENC


@dataclass
class HarmonyResponse:
    thinking: Optional[str]   # analysis-channel continuation (graded)
    answer: str               # final-channel answer
    raw: str
    finish_reason: str
    usage: dict = field(default_factory=dict)


def build_user_text(question: str, options: list[str]) -> str:
    if options:
        body = "\n\n" + "\n".join(f"{chr(ord('A')+i)}) {o}" for i, o in enumerate(options))
        tail = ("\n\nYour response should be of the following format: "
                "'ANSWER: X' (without quotes) where X is your chosen letter")
        return question + body + tail
    return (question + "\n\nYour response should be of the following format: 'ANSWER: X' "
            "(without quotes) where X is the letter of your chosen answer.")


def build_prefill_tokens(question: str, options: list[str], instruction: str,
                         seed: str, reasoning_effort: str = "high") -> list[int]:
    """Harmony token IDs; with a seed the prompt ends inside an unclosed analysis msg.

    reasoning_effort ("low"|"medium"|"high") sets gpt-oss's CoT length. Default "high"
    so the analysis channel is length-comparable to qwen/R1 (the medium default gives
    ~3x shorter CoT and an empty analysis channel at K=0)."""
    from openai_harmony import (
        Conversation, Message, Role, SystemContent, DeveloperContent, ReasoningEffort,
    )
    enc = _enc()
    _EFFORT = {"low": ReasoningEffort.LOW, "medium": ReasoningEffort.MEDIUM,
               "high": ReasoningEffort.HIGH}
    sysc = SystemContent.new().with_reasoning_effort(_EFFORT[reasoning_effort])
    sys_msg = Message.from_role_and_content(Role.SYSTEM, sysc)
    dev = DeveloperContent.new().with_instructions(instruction or "none")
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev)
    usr_msg = Message.from_role_and_content(Role.USER, build_user_text(question, options))
    msgs = [sys_msg, dev_msg, usr_msg]
    if not seed:
        convo = Conversation.from_messages(msgs)
        return list(enc.render_conversation_for_completion(convo, Role.ASSISTANT))
    seed_msg = Message.from_role_and_content(Role.ASSISTANT, seed).with_channel("analysis")
    convo = Conversation.from_messages(msgs + [seed_msg])
    toks = list(enc.render_conversation(convo))
    stop = set(enc.stop_tokens())
    while toks and toks[-1] in stop:        # drop trailing stop so model continues analysis
        toks.pop()
    return toks


def split_channels(text: str) -> tuple[str, str]:
    """(analysis_continuation, final_answer) from raw completion text w/ special tokens."""
    analysis, final = text, ""
    if "<|channel|>final<|message|>" in text:
        analysis, _, final = text.partition("<|channel|>final<|message|>")
    for mark in ("<|end|>", "<|return|>", "<|start|>", "<|channel|>", "<|call|>", "<|message|>"):
        analysis = analysis.split(mark)[0]
        final = final.split(mark)[0]
    return analysis.strip(), final.strip()


def complete_harmony(url: str, model: str, question: str, options: list[str],
                     instruction: str, seed: str, *, temperature: float = 0.6,
                     max_tokens: int = 8192, reasoning_effort: str = "high"
                     ) -> tuple[HarmonyResponse, dict]:
    tokens = build_prefill_tokens(question, options, instruction, seed, reasoning_effort)
    payload = {
        "model": model, "prompt": tokens, "max_tokens": max_tokens,
        "temperature": temperature, "skip_special_tokens": False,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=300))
            break
        except Exception:
            if attempt == 3:
                raise
            import time; time.sleep(2 ** attempt)
    choice = r["choices"][0]
    text = choice.get("text") or ""
    analysis, final = split_channels(text)
    resp = HarmonyResponse(
        thinking=analysis, answer=final, raw=text,
        finish_reason=choice.get("finish_reason") or "unknown",
        usage=r.get("usage") or {},
    )
    raw = {
        "provider": f"vllm/{model}",
        "prefill_tokens": len(tokens),
        "choices": [{"message": {"content": final}, "finish_reason": resp.finish_reason}],
        "harmony_raw": text,
        "usage": resp.usage,
    }
    return resp, raw
