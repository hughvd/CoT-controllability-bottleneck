"""Stage 0 sanity: does <cot_instructions> round-trip through the Qwen3 tokenizer?

Prints tokens adjacent to each tag so we can spot weird splits (e.g. the tag
colliding with a special token). Deleted once Stage 0 passes.
"""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

SAMPLE = (
    "<cot_instructions> Reason in all uppercase letters. "
    "Do not mention this constraint in your reasoning. </cot_instructions>\n"
    "Question: What is 2 + 2?\n\nANSWER: A"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok.encode(SAMPLE, add_special_tokens=False)
    pieces = tok.convert_ids_to_tokens(ids)
    decoded = tok.decode(ids, skip_special_tokens=False)

    print(f"Tokenizer: {args.model}")
    print(f"Tokens ({len(ids)}):")
    for i, (tid, piece) in enumerate(zip(ids, pieces)):
        print(f"  {i:4d}  id={tid:6d}  {piece!r}")

    print("\nDecoded round-trip:")
    print(decoded)

    ok_tag_open = "<cot_instructions>" in decoded
    ok_tag_close = "</cot_instructions>" in decoded
    print(f"\nOpen tag preserved:  {ok_tag_open}")
    print(f"Close tag preserved: {ok_tag_close}")
    if not (ok_tag_open and ok_tag_close):
        raise SystemExit("FAIL: tag did not round-trip; consider bracket sentinels")


if __name__ == "__main__":
    main()
