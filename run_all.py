"""Top-level orchestrator for running evaluations across models.

Usage:
    python run_all.py --eval metr_clue_usage --run-config eval_configs/small_qwen3.yaml
    python run_all.py --eval metr_clue_usage --run-config eval_configs/small_qwen3.yaml --model qwen3-8b --stage filtering
    python run_all.py --eval metr_clue_usage --family qwen3 --stage clue_difficulty
"""

import argparse
import asyncio
import sys

from lib.config import load_config, get_model_by_name, get_models_by_family
from lib.judge import make_judge_config


def main():
    parser = argparse.ArgumentParser(description="Run monitorability evaluations")
    parser.add_argument(
        "--eval", type=str, required=True,
        choices=["metr_clue_usage"],
        help="Evaluation method to run",
    )
    parser.add_argument(
        "--model", type=str, default="all",
        help="Model name, 'all', or comma-separated list",
    )
    parser.add_argument(
        "--family", type=str, default=None,
        help="Run all models in a family (e.g. 'qwen3', 'deepseek-r1-distill')",
    )
    parser.add_argument(
        "--stage", type=str, default="all",
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to models_config.yaml",
    )
    parser.add_argument(
        "--run-config", type=str, default=None,
        help="Path to run configuration YAML (e.g. eval_configs/small_qwen3.yaml)",
    )
    parser.add_argument(
        "--judge", type=str, default=None,
        help="Judge model override (default: from run config or 'o3')",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.eval == "metr_clue_usage":
        from evals.metr_clue_usage.run import (
            load_daft_math, load_run_config, run_model,
        )
        from evals.metr_clue_usage.clue_selection import resolve_clues

        # Load run config
        rc = {}
        if args.run_config:
            rc = load_run_config(args.run_config)
            print(f"Run config: {rc.get('name', args.run_config)}")

        # Determine judge
        judge_model = args.judge or rc.get("judge", {}).get("model", "o3")
        judge_effort = rc.get("judge", {}).get("reasoning_effort", "low")
        judge_config = make_judge_config(model=judge_model, reasoning_effort=judge_effort)

        # Determine models
        if args.family:
            models = get_models_by_family(config, args.family)
            if not models:
                print(f"Error: No models found for family '{args.family}'")
                sys.exit(1)
        elif args.model == "all":
            model_names = rc.get("models", [m.name for m in config.models])
            models = [get_model_by_name(config, n) for n in model_names]
        else:
            model_names = [n.strip() for n in args.model.split(",")]
            models = [get_model_by_name(config, n) for n in model_names]

        # Load data
        n_problems = rc.get("filtering", {}).get("n_problems", 200)
        problems = load_daft_math(limit=n_problems)
        clues = resolve_clues(rc)

        print(f"Running {args.eval} evaluation")
        print(f"Models: {[m.name for m in models]}")
        print(f"Stage: {args.stage}")
        print(f"Judge: {judge_config.model_id} ({judge_config.provider})")
        print(f"Problems: {len(problems)} | Clues: {len(clues)}")
        print()

        async def _run_all():
            if len(models) > 1:
                print(f"Running {len(models)} models in parallel\n")
                await asyncio.gather(*[
                    run_model(model, [args.stage], problems, clues,
                              judge_config=judge_config, run_config=rc)
                    for model in models
                ])
            else:
                await run_model(
                    models[0], [args.stage], problems, clues,
                    judge_config=judge_config, run_config=rc,
                )

        asyncio.run(_run_all())
    else:
        print(f"Evaluation '{args.eval}' not yet implemented")
        sys.exit(1)


if __name__ == "__main__":
    main()
