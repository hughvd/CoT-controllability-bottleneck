"""Clue selection utilities for the METR clue-usage evaluation.

Selects representative subsets of the 131 FreeResponseBehavior clues for
evaluation runs. Clues are organized into:

- Template-based (15 templates × 6 encoding difficulties = 90 clues)
  Encoding difficulties: easy_answer, medium_hex, hard_math, hard_unicode,
  hard_len_str, easy_list_comprehension

- Hand-crafted (~40 clues with varying analytical difficulty)
"""

import sys
from pathlib import Path
from typing import Optional

METR_SRC = Path(__file__).parent / "CoT-faithfulness-and-monitorability" / "src"
sys.path.insert(0, str(METR_SRC))

from free_response_behaviors import FR_FUNCTION_DICT, FreeResponseBehavior

# The 6 encoding difficulty levels used in template-based clues
ENCODING_DIFFICULTIES = [
    "easy_answer",
    "easy_list_comprehension",
    "medium_hex",
    "hard_math",
    "hard_unicode",
    "hard_len_str",
]

# 15 template contexts (each combined with each encoding difficulty)
TEMPLATES = [
    "sycophancy",
    "unethical_information",
    "metadata",
    "metadata_brief_ant",
    "metadata_detailed_ant",
    "database_schema",
    "config_file",
    "cache",
    "error_message",
    "backup_data",
    "machine_learning",
    "version_control",
    "csv_data",
    "security_audit",
    "jupyter_notebook",
]


def get_all_clues() -> dict[str, callable]:
    """Get all 131 clues as {name: function} dict."""
    return {b.value: FR_FUNCTION_DICT[b] for b in FreeResponseBehavior}


def get_representative_clues(n: int = 30) -> dict[str, callable]:
    """Select a representative subset of clues.

    Default selection (30 clues):
    - 15 template-based: 3 diverse templates × 5 encoding difficulties
    - 15 hand-crafted: spread across difficulty levels

    Args:
        n: Target number of clues. Currently supports 30 (default).
           For other values, returns closest reasonable selection.
    """
    all_clues = get_all_clues()

    if n >= len(all_clues):
        return all_clues

    selected = {}

    # Template-based: 3 diverse templates × 5 encoding difficulties = 15
    # Chosen templates represent different "framing" strategies:
    # - sycophancy: social pressure ("a professor says...")
    # - database_schema: technical context (SQL/schema)
    # - machine_learning: code/ML context
    chosen_templates = ["sycophancy", "database_schema", "machine_learning"]
    # Skip easy_list_comprehension (very similar to easy_answer)
    chosen_encodings = ["easy_answer", "medium_hex", "hard_math", "hard_unicode", "hard_len_str"]

    for template in chosen_templates:
        for encoding in chosen_encodings:
            key = f"{template}_{encoding}"
            if key in all_clues:
                selected[key] = all_clues[key]

    # Hand-crafted: 15 spanning the difficulty range
    hand_crafted = [
        "reward_hacking",                  # base (easiest)
        "reward_hacking_simple_1",         # simple variant
        "reward_hacking_difficulty_1",     # difficulty tier 1
        "reward_hacking_difficulty_4",     # tier 4
        "reward_hacking_difficulty_8",     # tier 8
        "reward_hacking_difficulty_12",    # tier 12
        "reward_hacking_difficulty_17",    # tier 17
        "reward_hacking_difficulty_23",    # tier 23 (hardest explicit tier)
        "reward_hacking_session_storage_debug_65",   # ~20-80% difficulty
        "reward_hacking_session_storage_debug_70",   # ~20-80% difficulty
        "reward_hacking_profiler_output_v2_1",       # ~40% difficulty
        "reward_hacking_profiler_output_v4_5",       # ~30% difficulty
        "reward_hacking_mimic_session_storage_debug_3",   # ~80% difficulty
        "reward_hacking_email_chain_calculation_medium_1", # ~20% difficulty
        "reward_hacking_session_storage_debug_reverse",    # reverse encoding
    ]

    for key in hand_crafted:
        if key in all_clues:
            selected[key] = all_clues[key]

    return selected


def get_clues_by_names(names: list[str]) -> dict[str, callable]:
    """Get specific clues by their enum value names."""
    all_clues = get_all_clues()
    selected = {}
    for name in names:
        if name not in all_clues:
            raise ValueError(f"Unknown clue: {name}. Available: {sorted(all_clues.keys())[:10]}...")
        selected[name] = all_clues[name]
    return selected


def resolve_clues(config: dict) -> dict[str, callable]:
    """Resolve clue selection from a run config's clue_difficulty section.

    Config options:
      clues: "auto" -> get_representative_clues(n_representative_clues)
      clues: "all"  -> get_all_clues()
      clues: ["name1", "name2", ...] -> get_clues_by_names(list)
    """
    clue_config = config.get("clue_difficulty", {})
    clues_spec = clue_config.get("clues", "auto")

    if clues_spec == "auto":
        n = clue_config.get("n_representative_clues", 30)
        return get_representative_clues(n)
    elif clues_spec == "all":
        return get_all_clues()
    elif isinstance(clues_spec, list):
        return get_clues_by_names(clues_spec)
    else:
        raise ValueError(f"Invalid clues spec: {clues_spec}")
