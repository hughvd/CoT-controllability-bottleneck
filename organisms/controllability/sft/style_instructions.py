"""Build the <cot_instructions> body for stylistic CoT training/eval.

Two axes of variation:
  - detail_level ∈ {"terse", "medium"} — how much the instruction spells out
    the target style. Terse names it; medium adds 1-2 concrete features.
  - phrasing_pool ∈ {"train", "eval"} — train- or eval-pool paraphrases of the
    seed instruction at that detail level, parallel to the existing transforms
    paraphrase axis.

The paraphrase pool at `data/style_paraphrases.json` is required — generate
it with `python style_paraphrase_generator.py` before running data
generation or eval. We fail loud if the pool is missing or any (style,
detail, phrasing_pool) cell is empty, rather than silently falling back to
a seed.

Pool file shape (produced by style_paraphrase_generator.py):
    {
      "yoda": {
        "terse":  {"train": [...], "eval": [...]},
        "medium": {"train": [...], "eval": [...]},
      },
      ...
    }
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_POOL_PATH = ROOT / "data" / "style_paraphrases.json"

DETAIL_LEVELS = ("terse", "medium")


_POOL: Optional[dict] = None
_POOL_PATH_LOADED: Optional[Path] = None


def _load_pool(path: Path = DEFAULT_POOL_PATH) -> dict:
    global _POOL, _POOL_PATH_LOADED
    if _POOL is not None and _POOL_PATH_LOADED == path:
        return _POOL
    if not path.exists():
        raise SystemExit(
            f"[style_instructions] paraphrase pool missing at {path}. "
            f"Run `python style_paraphrase_generator.py` first."
        )
    _POOL = json.loads(path.read_text())
    _POOL_PATH_LOADED = path
    return _POOL


def build_style_instruction(
    style_name: str,
    style_def: dict,
    *,
    rng: Optional[random.Random] = None,
    phrasing_pool: str = "train",
    detail_level: Optional[str] = None,
    pool_path: Path = DEFAULT_POOL_PATH,
) -> str:
    """Sample one cot_instructions body for (style, detail, phrasing_pool).

    If detail_level is None, sample one uniformly from DETAIL_LEVELS using rng.
    Pool must exist and have entries for the requested cell — fail loud
    otherwise.
    """
    if rng is None:
        rng = random.Random()

    if detail_level is None:
        detail_level = rng.choice(DETAIL_LEVELS)
    if detail_level not in DETAIL_LEVELS:
        raise ValueError(f"unknown detail_level {detail_level!r}; expected one of {DETAIL_LEVELS}")
    if phrasing_pool not in ("train", "eval"):
        raise ValueError(f"unknown phrasing_pool {phrasing_pool!r}; expected 'train' or 'eval'")

    pool = _load_pool(pool_path)
    style_pool = (pool.get(style_name) or {}).get(detail_level) or {}
    candidates = list(style_pool.get(phrasing_pool) or [])
    if not candidates:
        raise SystemExit(
            f"[style_instructions] empty pool for "
            f"({style_name}, {detail_level}, {phrasing_pool}). "
            f"Regenerate with `python style_paraphrase_generator.py --force`."
        )
    template = rng.choice(candidates)

    # Templates may use {style_name} as a placeholder for the display name; if
    # not present, .format() is a no-op. Most paraphrases bake the name in
    # directly so this is unused, but keeping it preserves the option.
    return template.format(style_name=style_name)
