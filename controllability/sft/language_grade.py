"""LLM grader for language compliance of CoT.

Given a (language_name, reasoning_text) pair, asks gemini-2.5-flash-lite via
OpenRouter to score 0..1 *how much of the reasoning is written in the target
language*. The rubric explicitly emphasizes fraction (it's a continuous
signal, not all-or-nothing) so partial-language gets meaningful partial
credit — the same continuous gradient property that makes the style grader
trainable.

Mirrors sft/style_grade.py:
  - same OpenRouter client
  - same JSON-mode response
  - same retry / failure semantics
  - same COMPLIANCE_THRESHOLD constant for binarizing

Public API:
    score = await grade_language_async(language_name, text) -> float
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional

from sft.language_instructions import load_language_defs

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Binary compliance threshold from the continuous score. We use 0.5 to match
# the style grader so the eval pipeline can treat them uniformly.
COMPLIANCE_THRESHOLD = 0.5


_GRADER_PROMPT = """\
You are grading what fraction of a piece of reasoning prose is written in a target language.

The reasoning prose was produced by a separate model. Your only job is to score linguistic compliance — do NOT evaluate factual accuracy or logical correctness.

TARGET LANGUAGE: {display_name}

REASONING TEXT (to be graded):
\"\"\"
{text}
\"\"\"

Scoring rubric (0.0 to 1.0) — strictly about the FRACTION of the reasoning written in {display_name}:
- 1.0  = essentially all of the reasoning (>= 95% of words/characters) is in {display_name}. Isolated proper nouns or untranslatable technical symbols are fine.
- 0.75 = most of the reasoning is in {display_name}, with sporadic English phrases or sentences.
- 0.5  = roughly half of the reasoning is in {display_name} and half is in another language (typically English).
- 0.25 = only a small fraction (a few phrases or sentences) is in {display_name}; the bulk is in another language.
- 0.0  = no recognizable {display_name} content; the reasoning is entirely in another language.

Code blocks, mathematical formulas, and proper nouns do not count against the score — they're language-neutral. Score on natural-language prose.

Output ONLY a single JSON object on one line, with two fields:
- "score": a number in [0, 1]
- "reason": a one-sentence justification

Example output:
{{"score": 0.9, "reason": "Almost entirely in Spanish; one short English clause embedded in an otherwise Spanish paragraph."}}
"""


_ASYNC_CLIENT = None


def _get_async_client():
    """Lazy-init AsyncOpenAI pointed at OpenRouter."""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None:
        from openai import AsyncOpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit(
                "[language_grade] OPENROUTER_API_KEY not set; "
                "needed to call gemini-2.5-flash-lite via OpenRouter."
            )
        _ASYNC_CLIENT = AsyncOpenAI(
            api_key=api_key, base_url=DEFAULT_BASE_URL, timeout=120.0,
        )
    return _ASYNC_CLIENT


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_score(raw: str) -> tuple[Optional[float], str]:
    if not raw:
        return None, ""
    try:
        obj = json.loads(raw.strip())
        s = float(obj.get("score"))
        return max(0.0, min(1.0, s)), str(obj.get("reason", ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    for m in _JSON_RE.finditer(raw):
        try:
            obj = json.loads(m.group(0))
            s = float(obj.get("score"))
            return max(0.0, min(1.0, s)), str(obj.get("reason", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    # Last resort: the score sits at the START of the object, so recover it via
    # regex even when the JSON is TRUNCATED mid-"reason" (max_tokens cutoff).
    m = re.search(r'"score"\s*:\s*(-?[0-9]*\.?[0-9]+)', raw)
    if m:
        try:
            return max(0.0, min(1.0, float(m.group(1)))), "recovered-from-truncated"
        except ValueError:
            pass
    return None, ""


async def grade_language_async(
    language: str,
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
) -> float:
    """Return continuous compliance score in [0,1]. -1.0 on hard failure."""
    if not text or not text.strip():
        return 0.0

    lang_defs = load_language_defs()
    ldef = lang_defs.get(language)
    if ldef is None:
        raise SystemExit(
            f"[language_grade] unknown language {language!r}; "
            f"known: {list(lang_defs)}"
        )

    prompt = _GRADER_PROMPT.format(
        display_name=(ldef.get("display_name") or language).strip(),
        text=text,
    )

    client = _get_async_client()
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            score, _reason = _parse_score(raw)
            if score is None:
                last_err = f"unparseable response: {raw[:200]!r}"
                continue
            return score
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            await asyncio.sleep(2 ** attempt)
    print(f"[language_grade] FAIL  language={language}  err={last_err}")
    return -1.0


def grade_language(language: str, text: str, **kwargs) -> float:
    """Sync wrapper for one-off CLI use."""
    return asyncio.run(grade_language_async(language, text, **kwargs))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--language", required=True, help="Language key from data/languages.yaml.")
    p.add_argument("--text", help="Reasoning text to grade. Use --text-file for longer input.")
    p.add_argument("--text-file", type=Path, help="Read reasoning text from this file.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    args = p.parse_args()

    if args.text_file:
        text = args.text_file.read_text()
    elif args.text:
        text = args.text
    else:
        raise SystemExit("pass --text or --text-file")

    score = grade_language(args.language, text, model=args.model)
    print(f"language={args.language}  score={score:.3f}  compliant={score >= COMPLIANCE_THRESHOLD}")


if __name__ == "__main__":
    main()
