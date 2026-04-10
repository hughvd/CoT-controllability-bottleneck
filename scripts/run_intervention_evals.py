#!/usr/bin/env python3
"""Run all intervention evals for the configured models.

Usage:
    # Run everything (all evals, all models, all stages)
    uv run python scripts/run_intervention_evals.py --config eval_configs/intervention_full.yaml

    # Run a specific stage
    uv run python scripts/run_intervention_evals.py --config eval_configs/intervention_full.yaml --stage control

    # Run specific evals only
    uv run python scripts/run_intervention_evals.py --config eval_configs/intervention_full.yaml --evals gpqa_hints,gpqa_encoded

    # Run a specific model only
    uv run python scripts/run_intervention_evals.py --config eval_configs/intervention_full.yaml --models qwen3-8b

    # Dry run (show what would be run)
    uv run python scripts/run_intervention_evals.py --config eval_configs/intervention_full.yaml --dry-run
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from evals.intervention.run import EVAL_SPECS, _async_main
from evals.intervention.datasets import DATASET_LOADERS
from lib.config import get_model_by_name, load_config
from lib.judge import make_judge_config

ALL_EVALS = list(EVAL_SPECS.keys())


def main():
    parser = argparse.ArgumentParser(description="Run all intervention evals")
    parser.add_argument("--config", type=str, required=True, help="Path to run config YAML")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["control", "intervention", "monitor", "metrics", "all"])
    parser.add_argument("--evals", type=str, default=None,
                        help="Comma-separated list of evals (default: all)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated list of models (default: from config)")
    parser.add_argument("--monitor", type=str, default=None,
                        help="Override monitor model")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for subject models (overrides config)")
    parser.add_argument("--monitor-reasoning-effort", type=str, default=None,
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for monitor model (overrides config)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    args = parser.parse_args()

    with open(args.config) as f:
        rc = yaml.safe_load(f)

    # CLI --reasoning-effort overrides config
    if args.reasoning_effort:
        rc.setdefault("sampling", {})["reasoning_effort"] = args.reasoning_effort

    config = load_config()

    # Resolve evals
    eval_names = args.evals.split(",") if args.evals else ALL_EVALS
    for e in eval_names:
        if e not in EVAL_SPECS:
            print(f"Error: unknown eval '{e}'. Available: {ALL_EVALS}")
            return

    # Resolve models
    model_names = args.models.split(",") if args.models else rc.get("models", [])
    models = [get_model_by_name(config, name) for name in model_names]

    # Monitor config
    monitor_cfg = rc.get("monitor", {})
    monitor_model = args.monitor or monitor_cfg.get("model", "gpt-5-mini")
    monitor_effort = args.monitor_reasoning_effort or monitor_cfg.get("reasoning_effort", "low")
    monitor_config = make_judge_config(
        model=monitor_model,
        reasoning_effort=monitor_effort,
    )

    dataset_cfg = rc.get("dataset", {})
    n_problems = dataset_cfg.get("n_problems")
    n_samples = rc.get("sampling", {}).get("n_samples", 10)
    n_hints_per_eval = {e: len(EVAL_SPECS[e].hints) for e in eval_names}

    # Print plan
    print("=" * 60)
    print(f"Intervention Eval Run: {rc.get('name', 'unnamed')}")
    print("=" * 60)
    print(f"Evals:    {eval_names}")
    print(f"Models:   {[m.name for m in models]}")
    print(f"Stage:    {args.stage}")
    reasoning_effort = rc.get("sampling", {}).get("reasoning_effort")
    print(f"Monitor:  {monitor_config.model_id}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"Problems: {n_problems or 'all'}")
    print(f"Samples:  {n_samples}")
    print()

    # Estimate request counts
    total_requests = 0
    for eval_name in eval_names:
        n_hints = n_hints_per_eval[eval_name]
        n_prob = n_problems or "?"
        ctrl = f"{n_prob} × {n_samples}"
        intv = f"{n_prob} × {n_hints} × {n_samples}"
        print(f"  {eval_name:20s}: control={ctrl}, intervention={intv}, hints={n_hints}")
        if isinstance(n_prob, int):
            ctrl_n = n_prob * n_samples
            intv_n = n_prob * n_hints * n_samples
            mon_n = ctrl_n + intv_n
            total_requests += (ctrl_n + intv_n + mon_n) * len(models)

    if n_problems:
        print(f"\n  Estimated total API requests: ~{total_requests:,}")
        print(f"  Per model: ~{total_requests // max(len(models), 1):,}")

    print()

    if args.dry_run:
        print("DRY RUN — exiting without running.")
        return

    # Run evals sequentially per eval, models in parallel within each eval
    asyncio.run(_run_all(eval_names, models, rc, dataset_cfg, monitor_config, args.stage))


async def _run_all(eval_names, models, rc, dataset_cfg, monitor_config, stage):
    from evals.intervention.pipeline import run_model

    for eval_name in eval_names:
        spec = EVAL_SPECS[eval_name]

        # Filter hints if config specifies
        if dataset_cfg.get("hints") and dataset_cfg["hints"] != "all":
            hint_names = set(dataset_cfg["hints"])
            spec.hints = [h for h in spec.hints if h["name"] in hint_names]

        # Load dataset
        dataset_name = eval_name.split("_")[0]
        loader = DATASET_LOADERS.get(dataset_name) or DATASET_LOADERS.get(eval_name)
        if not loader:
            print(f"Skipping {eval_name}: no dataset loader")
            continue

        n_problems = dataset_cfg.get("n_problems")
        print(f"\n{'#' * 60}")
        print(f"# EVAL: {eval_name}")
        print(f"{'#' * 60}")
        print(f"Loading {dataset_name} dataset...")
        problems = loader(n_problems)
        print(f"Loaded {len(problems)} problems")

        t0 = time.monotonic()

        # Run all models in parallel for this eval
        if len(models) > 1:
            await asyncio.gather(*[
                run_model(spec, model, problems, [stage],
                          run_config=rc, monitor_config=monitor_config)
                for model in models
            ])
        else:
            for model in models:
                await run_model(spec, model, problems, [stage],
                                run_config=rc, monitor_config=monitor_config)

        elapsed = time.monotonic() - t0
        print(f"\n  {eval_name} completed in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
