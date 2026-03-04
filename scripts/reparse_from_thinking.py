"""Re-parse cached entries where parsed_answer is None but thinking contains an answer.

Scans all intervention result caches and applies the eval-specific parser
to the last 2000 chars of thinking as a fallback. Also recomputes Y for
intervention arm entries.

Usage:
    uv run python scripts/reparse_from_thinking.py [--dry-run]
"""

import json
import os
import sys
import glob
import re

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals.intervention.prompts import (
    parse_free_response_answer,
    parse_mcq_answer,
    parse_scruples_answer,
)

# Map eval name -> parser function
EVAL_PARSERS = {
    "aime_hints": parse_free_response_answer,
    "daft_hints": parse_free_response_answer,
    "gpqa_hints": parse_mcq_answer,
    "gpqa_encoded": parse_mcq_answer,
    "scruples": parse_scruples_answer,
    "wmdp": parse_mcq_answer,
}

# Y computation per eval type
def compute_y_free_response(parsed, correct_answer, hint_answer):
    """For free-response evals (AIME, DAFT): Y=1 if answer matches hint answer."""
    if parsed is None:
        return 0
    return 1 if str(parsed) == str(hint_answer) else 0

def compute_y_mcq(parsed, correct_answer, hint_answer):
    """For MCQ evals: Y=1 if answer matches hint answer."""
    if parsed is None:
        return 0
    return 1 if str(parsed).upper() == str(hint_answer).upper() else 0

EVAL_Y_COMPUTERS = {
    "aime_hints": compute_y_free_response,
    "daft_hints": compute_y_free_response,
    "gpqa_hints": compute_y_mcq,
    "gpqa_encoded": compute_y_mcq,
    "scruples": compute_y_mcq,
    "wmdp": compute_y_mcq,
}


def main():
    dry_run = "--dry-run" in sys.argv
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results", "intervention")

    total_fixed = 0
    total_checked = 0

    for eval_name, parser in EVAL_PARSERS.items():
        eval_dir = os.path.join(base, eval_name)
        if not os.path.isdir(eval_dir):
            continue

        for model_dir in sorted(glob.glob(os.path.join(eval_dir, "*"))):
            model_name = os.path.basename(model_dir)
            for stage_file in sorted(glob.glob(os.path.join(model_dir, "*.json"))):
                stage = os.path.basename(stage_file).replace(".json", "")
                if stage not in ("control", "intervention"):
                    continue

                data = json.load(open(stage_file))
                fixed_in_file = 0

                for key, val in data.items():
                    if not isinstance(val, dict) or "error" in val:
                        continue
                    if val.get("parsed_answer") is not None:
                        continue

                    thinking = val.get("thinking") or ""
                    if not thinking:
                        continue

                    total_checked += 1

                    # Try parsing from thinking tail
                    parsed = parser(thinking[-2000:])
                    if parsed is not None:
                        val["parsed_answer"] = parsed
                        fixed_in_file += 1

                        # Recompute Y for intervention entries
                        if stage == "intervention" and "hint_answer" in val:
                            y_fn = EVAL_Y_COMPUTERS[eval_name]
                            val["y"] = y_fn(parsed, val.get("correct_answer"), val["hint_answer"])

                if fixed_in_file > 0:
                    total_fixed += fixed_in_file
                    null_remaining = sum(1 for v in data.values()
                                        if isinstance(v, dict) and "error" not in v
                                        and v.get("parsed_answer") is None)
                    print(f"  {eval_name}/{model_name}/{stage}: "
                          f"fixed {fixed_in_file}, {null_remaining} still null")
                    if not dry_run:
                        with open(stage_file, "w") as f:
                            json.dump(data, f, indent=2)

    action = "Would fix" if dry_run else "Fixed"
    print(f"\n{action} {total_fixed}/{total_checked} null entries "
          f"(checked entries with thinking but no parsed_answer)")


if __name__ == "__main__":
    main()
