"""Estimate API costs for running the METR clue-usage evaluation.

Calculates per-model and total costs based on the 4-stage pipeline:
1. Problem filtering: n_problems × n_attempts
2. Clue difficulty (no-CoT + with-CoT): n_clues × n_clue_targets × 2
3. Trajectories: n_filtered_problems × n_filtered_clues
4. Judge: correct trajectories × 2 modes (monitorability + faithfulness)

Usage:
    python -m evals.metr_clue_usage.cost_estimation --run-config eval_configs/small_qwen3.yaml
    python -m evals.metr_clue_usage.cost_estimation  # uses defaults from models_config.yaml
"""

import argparse

import yaml

from lib.config import load_config, EvalConfig, get_model_by_name


# Token estimates per call (rough averages)
TOKENS = {
    "filtering": {
        "input": 500,    # problem prompt
        "output": 2000,  # reasoning + answer
    },
    "clue_diff_no_cot": {
        "input": 600,    # clue + instructions (CoT suppressed)
        "output": 30,    # just a number
    },
    "clue_diff_with_cot": {
        "input": 600,    # clue + instructions
        "output": 2000,  # reasoning + answer
    },
    "trajectories": {
        "input": 800,    # problem + clue + instructions
        "output": 4000,  # full CoT reasoning
    },
    "judge": {
        "input": 5000,   # CoT transcript + judge prompt
        "output": 200,   # grade + explanation
    },
}


def load_run_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def estimate_costs(config: EvalConfig, run_config: dict | None = None):
    """Calculate estimated API costs for each model."""
    rc = run_config or {}

    # Parameters from run config or defaults
    n_problems = rc.get("filtering", {}).get("n_problems", 200)
    n_attempts = rc.get("filtering", {}).get("n_attempts", 10)
    n_clues = rc.get("clue_difficulty", {}).get("n_representative_clues", 30)
    n_clue_targets = rc.get("clue_difficulty", {}).get("n_clue_targets", 76)

    # Assumptions
    expected_filtered_frac = 0.7   # ~70% of problems model can't solve
    expected_clue_filter_frac = 0.6  # ~60% of clues pass reasoning_accuracy >= 0.95
    expected_correct_frac = 0.3    # ~30% of clued trajectories produce correct answer

    n_filtered_problems = int(n_problems * expected_filtered_frac)
    n_filtered_clues = int(n_clues * expected_clue_filter_frac)

    # Determine models to estimate
    if rc.get("models"):
        model_names = rc["models"]
        models = [get_model_by_name(config, n) for n in model_names]
    else:
        models = config.models

    # Determine judge pricing
    judge_model_id = rc.get("judge", {}).get("model", config.judge_model.model_id)
    judge_pricing = config.pricing.get(judge_model_id)

    results = []

    for model in models:
        pricing = config.pricing.get(model.model_id)
        if not pricing:
            print(f"Warning: No pricing for {model.model_id}, skipping")
            continue

        # Stage 1: Problem filtering
        filtering_calls = n_problems * n_attempts
        filtering_cost = _stage_cost(filtering_calls, TOKENS["filtering"], pricing)

        # Stage 2a: Clue difficulty — no-CoT
        no_cot_calls = n_clues * n_clue_targets
        no_cot_cost = _stage_cost(no_cot_calls, TOKENS["clue_diff_no_cot"], pricing)

        # Stage 2b: Clue difficulty — with-CoT
        with_cot_calls = n_clues * n_clue_targets
        with_cot_cost = _stage_cost(with_cot_calls, TOKENS["clue_diff_with_cot"], pricing)

        # Stage 3: Trajectories
        traj_calls = n_filtered_problems * n_filtered_clues
        traj_cost = _stage_cost(traj_calls, TOKENS["trajectories"], pricing)

        # Stage 4: Judge (2 modes per correct trajectory)
        n_correct_trajs = int(traj_calls * expected_correct_frac)
        judge_calls = n_correct_trajs * 2  # monitorability + faithfulness
        judge_cost = 0
        if judge_pricing:
            judge_cost = _stage_cost(judge_calls, TOKENS["judge"], judge_pricing)

        total = filtering_cost + no_cot_cost + with_cot_cost + traj_cost + judge_cost

        results.append({
            "model": model.name,
            "family": model.family,
            "filtering": filtering_cost,
            "clue_no_cot": no_cot_cost,
            "clue_with_cot": with_cot_cost,
            "trajectories": traj_cost,
            "judge": judge_cost,
            "total": total,
            "filtering_calls": filtering_calls,
            "no_cot_calls": no_cot_calls,
            "with_cot_calls": with_cot_calls,
            "traj_calls": traj_calls,
            "judge_calls": judge_calls,
        })

    return results


def _stage_cost(n_calls: int, token_spec: dict, pricing) -> float:
    """Compute cost for a stage given call count, token estimates, and pricing."""
    input_tokens = n_calls * token_spec["input"]
    output_tokens = n_calls * token_spec["output"]
    return (
        input_tokens * pricing.input / 1_000_000
        + output_tokens * pricing.output / 1_000_000
    )


def print_cost_table(results: list[dict], run_config: dict | None = None):
    """Print a formatted cost estimation table."""
    rc = run_config or {}
    name = rc.get("name", "default")

    n_problems = rc.get("filtering", {}).get("n_problems", 200)
    n_attempts = rc.get("filtering", {}).get("n_attempts", 10)
    n_clues = rc.get("clue_difficulty", {}).get("n_representative_clues", 30)
    n_clue_targets = rc.get("clue_difficulty", {}).get("n_clue_targets", 76)
    judge_model = rc.get("judge", {}).get("model", "o3")

    print()
    print("=" * 110)
    print(f"COST ESTIMATION — METR Clue-Usage Evaluation — config: {name}")
    print("=" * 110)
    print()
    print(f"  Problems: {n_problems}  |  Attempts/problem: {n_attempts}  |  Clues: {n_clues}  |  Clue targets: {n_clue_targets}  |  Judge: {judge_model}")
    print()

    # Header
    header = (
        f"{'Model':<25} {'Filter':>8} {'NoCot':>8} {'WCot':>8} "
        f"{'Trajs':>8} {'Judge':>8} {'TOTAL':>9}"
    )
    print(header)
    print("-" * 80)

    total_all = 0
    family_totals = {}

    for r in results:
        print(
            f"{r['model']:<25} "
            f"${r['filtering']:>6.2f} "
            f"${r['clue_no_cot']:>6.2f} "
            f"${r['clue_with_cot']:>6.2f} "
            f"${r['trajectories']:>6.2f} "
            f"${r['judge']:>6.2f} "
            f"${r['total']:>7.2f}"
        )
        total_all += r["total"]
        fam = r["family"]
        family_totals[fam] = family_totals.get(fam, 0) + r["total"]

    print("-" * 80)
    print(f"{'TOTAL':<25} {'':>8} {'':>8} {'':>8} {'':>8} {'':>8} ${total_all:>7.2f}")
    print()

    if len(family_totals) > 1:
        print("By family:")
        for fam, cost in sorted(family_totals.items()):
            print(f"  {fam:<25} ${cost:>7.2f}")
        print()

    # Call counts
    if results:
        r = results[0]
        total_calls = r["filtering_calls"] + r["no_cot_calls"] + r["with_cot_calls"] + r["traj_calls"] + r["judge_calls"]
        print("API calls per model:")
        print(f"  Filtering:         {r['filtering_calls']:>8,}")
        print(f"  Clue diff no-CoT:  {r['no_cot_calls']:>8,}")
        print(f"  Clue diff with-CoT:{r['with_cot_calls']:>8,}")
        print(f"  Trajectories:      {r['traj_calls']:>8,}")
        print(f"  Judge (×2 modes):  {r['judge_calls']:>8,}")
        print(f"  Total:             {total_calls:>8,}")
    print()

    print("Assumptions:")
    print("  - Token estimates are rough averages; actual costs may vary ±50%")
    print("  - ~70% of problems filtered (model can't solve)")
    print("  - ~60% of clues pass reasoning_accuracy >= 0.95 threshold")
    print("  - ~30% of clued trajectories produce correct answer")
    print("  - Judge runs 2 modes (monitorability + faithfulness) per correct trajectory")


def main():
    parser = argparse.ArgumentParser(description="Estimate API costs")
    parser.add_argument(
        "--run-config", type=str, default=None,
        help="Path to run configuration YAML (e.g. eval_configs/small_qwen3.yaml)",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to models_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    rc = None
    if args.run_config:
        rc = load_run_config(args.run_config)

    results = estimate_costs(config, run_config=rc)
    print_cost_table(results, run_config=rc)


if __name__ == "__main__":
    main()
