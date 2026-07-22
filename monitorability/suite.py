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


def load_dataset_items(registry_key: str, max_rows: int | None = None) -> list["EvalItem"]:
    """Load one dataset's EvalItems directly from the registry, bypassing the
    scaffold's `scaffold_supported` filter.

    Needed for outcome/flaky_tools_charxiv.jsonl: the scaffold excludes it
    because the *original* eval depends on a tool runtime, but the released
    CharXiv rows are self-contained impossible-task prompts (chart question,
    image removed) and run fine over a plain chat API.
    """
    registry = scaffold._load_registry()
    cfg = registry["datasets"][registry_key]
    archetype = str(cfg.get("archetype", ""))
    environments_dir = scaffold.ensure_environment_data()
    rows = scaffold._read_json_records(environments_dir / registry_key)
    if max_rows is not None:
        rows = rows[:max_rows]

    monitor_prompt_path = str(cfg["monitor_prompts"][0])
    monitor_prompt_text = (scaffold.ROOT / monitor_prompt_path).read_text()
    model_prompt_refs = [str(x) for x in (cfg.get("model_prompts") or [])]
    model_prompt_texts = []
    for ref in model_prompt_refs:
        ref_path = scaffold.ROOT / ref
        model_prompt_texts.append(ref_path.read_text() if ref_path.is_file() else ref)

    items = []
    for idx, row in enumerate(rows):
        prompt = str(row.get("problem", "")).strip()
        if not prompt:
            continue
        arm = None
        if archetype == "intervention":
            arm = 1 if str(row.get("type", "")).lower() == "perturbed" else 0
        items.append(
            EvalItem(
                dataset=registry_key,
                archetype=archetype,
                instance_id=str(row.get("id", idx)),
                category=str(row.get("category", "uncategorized")),
                prompt=prompt,
                answer="" if row.get("answer") is None else str(row.get("answer")),
                arm=arm,
                monitor_prompt_template_path=monitor_prompt_path,
                monitor_prompt_template_text=monitor_prompt_text,
                y_rule_ref=str(cfg.get("y_rule", "")),
                z_rule_ref=str(cfg.get("z_rule", "")),
                monitor_dataset_type=(str(cfg["monitor_dataset_type"]) if "monitor_dataset_type" in cfg else None),
                model_prompt_source=str(cfg.get("model_prompt_source", "problem")),
                model_prompt_refs=model_prompt_refs,
                model_prompt_templates_text=model_prompt_texts,
                raw=row,
            )
        )
    return items


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
        for key in sorted(wanted - found):
            # Datasets the scaffold marks unsupported (e.g. memory_advice, which
            # needs a simulated memory tool) load via the direct registry path.
            try:
                extra = load_dataset_items(key)
            except KeyError:
                raise ValueError(f"Dataset not found in registry: {key}") from None
            items.extend(it for it in extra if it.archetype == "intervention")

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
