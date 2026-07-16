"""Build the <cot_instructions> body for multi-lingual CoT compliance.

Parallel to sft/style_instructions.py. Pulls instruction phrasings from
data/languages.yaml's `instruction_phrasings.{train,eval}` lists per language.

Phrasings are hand-curated (no LLM generator), so there's no separate
paraphrases.json — the YAML is the source of truth.

Public API:
    instruction = build_language_instruction(language_name, lang_def, rng=rng,
                                              phrasing_pool="train")
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LANGUAGES_YAML = ROOT / "data" / "languages.yaml"


_LANG_DEFS: Optional[dict[str, dict]] = None


def load_language_defs(path: Path = DEFAULT_LANGUAGES_YAML) -> dict[str, dict]:
    """Load and cache languages.yaml. Returns the inner 'languages' dict."""
    global _LANG_DEFS
    if _LANG_DEFS is None:
        if not path.exists():
            raise SystemExit(f"[language_instructions] missing {path}")
        cfg = yaml.safe_load(path.read_text())
        _LANG_DEFS = cfg.get("languages", {}) or {}
    return _LANG_DEFS


def build_language_instruction(
    language_name: str,
    lang_def: dict,
    *,
    rng: Optional[random.Random] = None,
    phrasing_pool: str = "train",
) -> str:
    """Sample one cot_instructions body for (language, phrasing_pool)."""
    if rng is None:
        rng = random.Random()
    if phrasing_pool not in ("train", "eval"):
        raise ValueError(
            f"unknown phrasing_pool {phrasing_pool!r}; expected 'train' or 'eval'"
        )

    phrasings = ((lang_def.get("instruction_phrasings") or {})
                 .get(phrasing_pool) or [])
    if not phrasings:
        raise SystemExit(
            f"[language_instructions] empty {phrasing_pool!r} phrasings for "
            f"language {language_name!r}. Edit data/languages.yaml."
        )
    return rng.choice(phrasings)
