"""LLM grader for stylistic CoT compliance.

Given a (style_name, reasoning_text) pair, asks gemini-2.5-flash-lite via
OpenRouter to score 0..1 how strongly the reasoning text adheres to the
target style.

Why an LLM grader (vs programmatic): the style families (yoda, pirate,
shakespearean, ...) don't have a single surface marker; they're constellations
of syntactic, lexical, and prosodic features. A heuristic grader would need
per-style hand-coded regexes that drift from the actual style. The judge
prompt grounds compliance in the same style_instruction that the rewriter saw,
so training data and eval grading share one source of truth.

No on-disk cache: run_eval_diverse.py writes per_sample.csv with both the
reasoning text and the score, so the eval output IS the durable record. If
the grader config (model, threshold, rubric) changes, a fresh full eval pass
is the honest path; a cache would just risk returning stale scores.

Public API:
    score = await grade_style_async(style_name, text) -> float
    is_compliant = score >= threshold (default 0.5)

CLI sanity check:
    python -m sft.style_grade --style yoda --text "Strong with the Force, you are."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STYLES_YAML = ROOT / "data" / "styles.yaml"

# OpenRouter model id for gemini-2.5-flash-lite.
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Threshold for binary compliance from the continuous score.
COMPLIANCE_THRESHOLD = 0.5


_GRADER_PROMPT = """\
You are grading what fraction of a piece of reasoning prose is written in a target stylistic register.

TARGET STYLE: {display_name}

STYLE DESCRIPTION:
{style_instruction}

REASONING TEXT (to be graded):
\"\"\"
{text}
\"\"\"

You will grade the FRACTION of sentences IN the target style. Do NOT evaluate factual accuracy or logical correctness.

A sentence is IN the target style ONLY IF its underlying grammar or syntax is changed away from plain modern English. Examples of grammar/syntax shifts that count:
- pirate: "be" replacing "is/are" ("the student be doin'"), "ye" replacing "you", -in' for -ing, "this here X", sea-metaphor vocabulary used systematically.
- yoda: object-subject-verb inversion ("Strong with the Force, you are", "Hydroformylation, the student does").
- shakespearean: thou/thee/thy/thine pronouns, -est/-eth verb endings, doth/hath/wouldst.
- valley girl: "like" as mid-sentence filler ("the student is, like, doing"), upspeak with "?" on statements, heavy "totally/literally/so" intensifiers.

CRITICAL RULE: A sentence is NOT in the target style if its underlying grammar is plain modern English with one keyword tacked on as decoration. These do NOT count as in-style:
- pirate: "Let me work this out, matey." (grammar unchanged)
- pirate: "Okay arrr, the answer is C." (grammar unchanged)
- yoda: "Hmm, let me think about this." (no inversion)
- shakespearean: "Verily, the answer is C." (no thou/-eth/-est)
- valley girl: "Okay, the answer is C." (no like/totally mid-sentence)

A keyword at the start or end of an otherwise-plain English sentence is decoration, NOT a grammar shift. Score it as 0 for that sentence.

Compute score = (sentences with a grammar shift) / (total sentences). Round to the nearest 0.05.

Output ONLY a single JSON object on one line, with two fields:
- "score": a number in [0, 1]
- "reason": a one-sentence justification stating how many sentences were in-style and why.
"""


# ---------------------------------------------------------------------------
# Style definitions
# ---------------------------------------------------------------------------

_STYLE_DEFS: Optional[dict[str, dict]] = None


def _load_style_defs(path: Path = DEFAULT_STYLES_YAML) -> dict[str, dict]:
    global _STYLE_DEFS
    if _STYLE_DEFS is None:
        cfg = yaml.safe_load(path.read_text())
        _STYLE_DEFS = cfg.get("styles", {}) or {}
    return _STYLE_DEFS


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_ASYNC_CLIENT = None


def _get_async_client():
    """Lazy-init AsyncOpenAI pointed at OpenRouter."""
    global _ASYNC_CLIENT
    if _ASYNC_CLIENT is None:
        from openai import AsyncOpenAI
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit(
                "[style_grade] OPENROUTER_API_KEY not set; "
                "needed to call gemini-2.5-flash-lite via OpenRouter."
            )
        _ASYNC_CLIENT = AsyncOpenAI(
            api_key=api_key, base_url=DEFAULT_BASE_URL, timeout=120.0,
        )
    return _ASYNC_CLIENT


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_score(raw: str) -> tuple[Optional[float], str]:
    """Pull the {"score": ..., "reason": ...} JSON object from the model output."""
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
    return None, ""


async def grade_style_async(
    style: str,
    text: str,
    *,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
) -> float:
    """Return continuous compliance score in [0,1]. -1.0 on hard failure."""
    if not text or not text.strip():
        return 0.0

    style_defs = _load_style_defs()
    sdef = style_defs.get(style)
    if sdef is None:
        raise SystemExit(f"[style_grade] unknown style {style!r}; "
                         f"known: {list(style_defs)}")

    prompt = _GRADER_PROMPT.format(
        display_name=(sdef.get("display_name") or style).strip(),
        style_instruction=(sdef.get("style_instruction") or "").strip(),
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
                max_tokens=600,
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
    print(f"[style_grade] FAIL  style={style}  err={last_err}")
    return -1.0


def grade_style(style: str, text: str, **kwargs) -> float:
    """Sync wrapper for one-off CLI use. Don't call from async code — use the
    _async version directly so requests can overlap."""
    return asyncio.run(grade_style_async(style, text, **kwargs))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--style", required=True, help="Style name (must be in styles.yaml).")
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

    score = grade_style(args.style, text, model=args.model)
    print(f"style={args.style}  score={score:.3f}  compliant={score >= COMPLIANCE_THRESHOLD}")


if __name__ == "__main__":
    main()
