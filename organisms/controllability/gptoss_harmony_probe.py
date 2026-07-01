"""Probe: can we seed gpt-oss-120b's analysis (CoT) channel via raw harmony prefill?

gpt-oss uses OpenAI 'harmony' channels, not <think> tags. Its reasoning lives in
an `analysis` channel:
  <|start|>assistant<|channel|>analysis<|message|>...reasoning...<|end|>
  <|start|>assistant<|channel|>final<|message|>...answer...<|return|>

To seed the CoT we render a harmony prompt that ends *inside* an unclosed analysis
message containing the seed, then let the model continue. We hit vLLM's raw
/v1/completions with token IDs (so we control the exact prefix) and read the
continuation back with special tokens visible, so we can split analysis-continuation
from the final answer.

Run ON the pod (hits localhost:8000). Iterate here until the mechanism takes, then
fold the winning construction into prefill_experiment.py as a gpt-oss backend.

  python gptoss_harmony_probe.py
"""

from __future__ import annotations

import json
import urllib.request

from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)

ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
URL = "http://localhost:8000/v1/completions"
MODEL = "openai/gpt-oss-120b"

# one uppercase example (unmistakable signal that the seed "took")
QUESTION = "Which of the following would make the best insulator against the flow of electricity?"
OPTIONS = ["copper wire", "steel tubing", "plastic tape", "aluminum foil"]
SEED = ("OKAY, LET'S SEE. THE QUESTION IS ASKING WHICH MATERIAL WOULD BE THE BEST "
        "INSULATOR AGAINST THE FLOW OF ELECTRICITY. THE OPTIONS ARE COPPER WIRE, "
        "STEEL TUBING, PLASTIC TAPE, AND ALUMINUM FOIL.")


def user_text() -> str:
    body = "\n".join(f"{chr(ord('A')+i)}) {o}" for i, o in enumerate(OPTIONS))
    return (f"{QUESTION}\n\n{body}\n\nYour response should be of the following format: "
            "'ANSWER: X' (without quotes) where X is your chosen letter")


def build_prefill_tokens(instruction: str, seed: str) -> list[int]:
    """Render harmony prompt; if seed, end inside an unclosed analysis message."""
    sys_msg = Message.from_role_and_content(Role.SYSTEM, SystemContent.new())
    dev = DeveloperContent.new().with_instructions(instruction or "none")
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev)
    usr_msg = Message.from_role_and_content(Role.USER, user_text())
    msgs = [sys_msg, dev_msg, usr_msg]
    if not seed:
        convo = Conversation.from_messages(msgs)
        return ENC.render_conversation_for_completion(convo, Role.ASSISTANT)
    # seed: append an analysis-channel assistant message, render, drop trailing stop
    seed_msg = Message.from_role_and_content(Role.ASSISTANT, seed).with_channel("analysis")
    convo = Conversation.from_messages(msgs + [seed_msg])
    toks = list(ENC.render_conversation(convo))
    stop = set(ENC.stop_tokens())
    while toks and toks[-1] in stop:
        toks.pop()
    return toks


def complete(tokens: list[int], max_tokens: int = 2048) -> str:
    payload = {
        "model": MODEL,
        "prompt": tokens,            # token IDs — exact prefix control
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "skip_special_tokens": False,  # keep <|...|> markers so we can split channels
    }
    req = urllib.request.Request(
        URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = json.load(urllib.request.urlopen(req, timeout=180))
    return resp["choices"][0]["text"]


def split_channels(text: str) -> tuple[str, str]:
    """Return (analysis_continuation, final_answer) from raw completion text."""
    final = ""
    analysis = text
    if "<|channel|>final<|message|>" in text:
        analysis, _, rest = text.partition("<|channel|>final<|message|>")
        final = rest
    # strip any trailing harmony markers
    for mark in ("<|end|>", "<|return|>", "<|start|>assistant", "<|call|>"):
        analysis = analysis.split(mark)[0]
        final = final.split(mark)[0]
    return analysis.strip(), final.strip()


def head_upper_frac(text: str, n: int = 40) -> float:
    head = "".join(c for c in text.strip()[:n] if c.isalpha())
    return (sum(c.isupper() for c in head) / len(head)) if head else 0.0


def main() -> None:
    print("=== K=0 (no seed, floor) ===")
    t0 = build_prefill_tokens("none", "")
    out0 = complete(t0)
    a0, f0 = split_channels(out0)
    print(f"analysis[:200]: {a0[:200]!r}")
    print(f"final: {f0[:60]!r}")
    print(f"head-uppercase: {head_upper_frac(a0):.2f}\n")

    print("=== seeded (analysis prefilled with uppercase) ===")
    ts = build_prefill_tokens("none", SEED)
    print(f"prefill token count: {len(ts)}")
    outs = complete(ts)
    a, f = split_channels(outs)
    print(f"continuation[:300]: {a[:300]!r}")
    print(f"final: {f[:60]!r}")
    print(f"head-uppercase (did seed take?): {head_upper_frac(a):.2f}")
    print(f"\nRAW seeded output[:400] (inspect markers): {outs[:400]!r}")


if __name__ == "__main__":
    main()
