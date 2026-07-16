"""Import shim for the openai/monitorability-evals submodule.

The submodule ships a mocked scaffold (`run_eval_scaffold_lib`) whose loading,
prompt-rendering, answer-matching, and metric plumbing are the official
reference implementation — only its model/monitor calls are mocked. We import
that module wholesale and reuse everything except the mocks, so our numbers
stay comparable to the paper's.

Datasets are extracted from the submodule's environments.zip into
monitorability/data/ (gitignored) rather than into the submodule itself, which
keeps `git status` clean.
"""

import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
SUITE_DIR = PKG_DIR / "openai_suite"
DATA_DIR = PKG_DIR / "data"

if not (SUITE_DIR / "run_eval_scaffold_lib.py").exists():
    raise ImportError(
        "openai/monitorability-evals submodule is missing. Run:\n"
        "  git submodule update --init monitorability/openai_suite"
    )

if str(SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(SUITE_DIR))

import run_eval_scaffold_lib as scaffold  # noqa: E402

# Redirect the scaffold's dataset extraction outside the submodule.
scaffold.EXTRACT_ROOT = DATA_DIR
scaffold.ENVIRONMENTS_DIR = DATA_DIR / "environments"

EvalItem = scaffold.EvalItem
ModelCallOutput = scaffold.ModelCallOutput
BootstrapConfig = scaffold.BootstrapConfig

load_items = scaffold.load_items
render_model_prompt = scaffold._render_model_prompt
render_monitor_prompt = scaffold._render_monitor_prompt
monitor_answer_spec = scaffold._monitor_answer_spec
match_directional_answer = scaffold._match_directional_answer
extract_choice_letter = scaffold._extract_choice_letter
compute_intervention_metrics = scaffold.compute_intervention_metrics

SCOPES = scaffold.SUPPORTED_SCOPES  # ("cot_only", "answer_only", "all_messages")


def dataset_key(short_name: str) -> str:
    """Map a short dataset name (e.g. 'gpqa') to its registry key."""
    if "/" in short_name:
        return short_name
    return f"intervention/{short_name}.jsonl"


def load_intervention_items(
    datasets: list[str] | None = None,
    max_instances: int | None = None,
) -> list["EvalItem"]:
    """Load intervention EvalItems, optionally filtered and truncated.

    Truncation is pair-aware: it keeps the first `max_instances` instance ids
    that have BOTH arms, and drops unpaired ids entirely (the g-mean metric
    needs both arms to estimate a per-instance effect).
    """
    wanted = {dataset_key(d) for d in datasets} if datasets else None
    items = [
        it for it in load_items(None)
        if it.archetype == "intervention" and (wanted is None or it.dataset in wanted)
    ]
    if wanted is not None:
        found = {it.dataset for it in items}
        missing = wanted - found
        if missing:
            raise ValueError(f"Datasets not found in registry/scaffold: {sorted(missing)}")

    out: list[EvalItem] = []
    for ds in sorted({it.dataset for it in items}):
        ds_items = [it for it in items if it.dataset == ds]
        arms_by_id: dict[str, set[int]] = {}
        order: list[str] = []
        for it in ds_items:
            if it.instance_id not in arms_by_id:
                order.append(it.instance_id)
            arms_by_id.setdefault(it.instance_id, set()).add(it.arm)
        paired = [iid for iid in order if arms_by_id[iid] == {0, 1}]
        if max_instances is not None:
            paired = paired[:max_instances]
        keep = set(paired)
        out.extend(it for it in ds_items if it.instance_id in keep)
    return out
