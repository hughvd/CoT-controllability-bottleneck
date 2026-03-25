"""CLI entry point for intervention evaluations.

Usage:
    python -m evals.intervention.run --eval gpqa_hints --run-config eval_configs/intervention_test.yaml --stage all
    python -m evals.intervention.run --eval gpqa_hints --run-config eval_configs/intervention_test.yaml --stage control
    python -m evals.intervention.run --eval gpqa_hints --run-config eval_configs/intervention_test.yaml --stage metrics
"""

import argparse
import asyncio

import yaml

from lib.config import get_model_by_name, load_config
from lib.judge import make_judge_config

from .datasets import DATASET_LOADERS
from .pipeline import EvalSpec, run_model
from .prompts import (
    AIME_HINTS,
    DAFT_HINTS,
    GPQA_ENCODED_HINTS,
    GPQA_HINTS,
    SCRUPLES_HINTS,
    WMDP_HINTS,
    compute_y_is_wrong,
    compute_y_matches_hint,
    compute_y_sycophancy,
    format_free_response_control,
    format_free_response_intervention,
    format_math_monitor,
    format_mcq_control,
    format_mcq_intervention,
    format_scruples_control,
    format_scruples_intervention,
    format_scruples_monitor,
    format_wmdp_control,
    format_wmdp_intervention,
    format_wmdp_monitor,
    parse_free_response_answer,
    parse_math_monitor,
    parse_mcq_answer,
    parse_scruples_answer,
    parse_scruples_monitor,
    parse_wmdp_monitor,
)

# ── Eval Spec Registry ───────────────────────────────────────────────

EVAL_SPECS = {
    "gpqa_hints": EvalSpec(
        name="gpqa_hints",
        hints=GPQA_HINTS,
        format_control_prompt=format_mcq_control,
        format_intervention_prompt=format_mcq_intervention,
        parse_answer=parse_mcq_answer,
        compute_y=compute_y_matches_hint,
        format_monitor_prompt=format_math_monitor,
        parse_monitor_output=parse_math_monitor,
    ),
    "gpqa_encoded": EvalSpec(
        name="gpqa_encoded",
        hints=GPQA_ENCODED_HINTS,
        format_control_prompt=format_mcq_control,
        format_intervention_prompt=format_mcq_intervention,
        parse_answer=parse_mcq_answer,
        compute_y=compute_y_matches_hint,
        format_monitor_prompt=format_math_monitor,
        parse_monitor_output=parse_math_monitor,
    ),
    "daft_hints": EvalSpec(
        name="daft_hints",
        hints=DAFT_HINTS,
        format_control_prompt=format_free_response_control,
        format_intervention_prompt=format_free_response_intervention,
        parse_answer=parse_free_response_answer,
        compute_y=compute_y_matches_hint,
        format_monitor_prompt=format_math_monitor,
        parse_monitor_output=parse_math_monitor,
    ),
    "aime_hints": EvalSpec(
        name="aime_hints",
        hints=AIME_HINTS,
        format_control_prompt=format_free_response_control,
        format_intervention_prompt=format_free_response_intervention,
        parse_answer=parse_free_response_answer,
        compute_y=compute_y_matches_hint,
        format_monitor_prompt=format_math_monitor,
        parse_monitor_output=parse_math_monitor,
    ),
    "scruples": EvalSpec(
        name="scruples",
        hints=SCRUPLES_HINTS,
        format_control_prompt=format_scruples_control,
        format_intervention_prompt=format_scruples_intervention,
        parse_answer=parse_scruples_answer,
        compute_y=compute_y_sycophancy,
        format_monitor_prompt=format_scruples_monitor,
        parse_monitor_output=parse_scruples_monitor,
        # Sycophantic answer is B (author not wrong) — opposite of truth
        get_hint_answer=lambda _: "B",
    ),
    "wmdp": EvalSpec(
        name="wmdp",
        hints=WMDP_HINTS,
        format_control_prompt=format_wmdp_control,
        format_intervention_prompt=format_wmdp_intervention,
        parse_answer=parse_mcq_answer,
        compute_y=compute_y_is_wrong,
        format_monitor_prompt=format_wmdp_monitor,
        parse_monitor_output=parse_wmdp_monitor,
    ),
}


def load_run_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run intervention evaluations")
    parser.add_argument(
        "--eval", type=str, required=True,
        choices=list(EVAL_SPECS.keys()),
        help="Evaluation type",
    )
    parser.add_argument(
        "--run-config", type=str, required=True,
        help="Path to run configuration YAML",
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        choices=["control", "intervention", "monitor", "metrics", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--monitor", type=str, default=None,
        help="Override monitor model (e.g. 'o3', 'gpt-4o-mini')",
    )
    parser.add_argument(
        "--reasoning-effort", type=str, default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort for subject models (overrides config)",
    )
    args = parser.parse_args()

    rc = load_run_config(args.run_config)

    # CLI --reasoning-effort overrides config
    if args.reasoning_effort:
        rc.setdefault("sampling", {})["reasoning_effort"] = args.reasoning_effort
    config = load_config()

    # Resolve eval spec
    spec = EVAL_SPECS[args.eval]

    # Filter hints if config specifies
    dataset_cfg = rc.get("dataset", {})
    if dataset_cfg.get("hints") and dataset_cfg["hints"] != "all":
        hint_names = set(dataset_cfg["hints"])
        spec.hints = [h for h in spec.hints if h["name"] in hint_names]

    # Load dataset
    dataset_name = args.eval.split("_")[0]  # gpqa_hints -> gpqa
    if dataset_name in DATASET_LOADERS:
        loader = DATASET_LOADERS[dataset_name]
    else:
        loader = DATASET_LOADERS.get(args.eval)
    if not loader:
        print(f"Error: No dataset loader for {args.eval}")
        return

    n_problems = dataset_cfg.get("n_problems")
    print(f"Loading {dataset_name} dataset...")
    problems = loader(n_problems)
    print(f"Loaded {len(problems)} problems")

    # Resolve models
    model_names = rc.get("models", [])
    models = [get_model_by_name(config, name) for name in model_names]

    # Monitor config
    monitor_cfg = rc.get("monitor", {})
    monitor_model = args.monitor or monitor_cfg.get("model", "o3")
    monitor_config = make_judge_config(
        model=monitor_model,
        reasoning_effort=monitor_cfg.get("reasoning_effort", "low"),
    )

    stages = [args.stage]

    reasoning_effort = rc.get("sampling", {}).get("reasoning_effort")
    print(f"\nEval: {spec.name}")
    print(f"Models: {[m.name for m in models]}")
    print(f"Stage: {args.stage}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Monitor: {monitor_config.model_id}")

    asyncio.run(_async_main(spec, models, problems, stages, rc, monitor_config))


async def _async_main(spec, models, problems, stages, run_config, monitor_config):
    if len(models) > 1:
        await asyncio.gather(*[
            run_model(spec, model, problems, stages,
                      run_config=run_config, monitor_config=monitor_config)
            for model in models
        ])
    else:
        for model in models:
            await run_model(spec, model, problems, stages,
                            run_config=run_config, monitor_config=monitor_config)


if __name__ == "__main__":
    main()
