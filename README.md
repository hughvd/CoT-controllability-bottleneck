  

  Reasoning effort evals
  ```
  # Low effort — all evals
  uv run python scripts/run_intervention_evals.py --config eval_configs/intervention_reasoning_effort.yaml
  --reasoning-effort low

  # High effort — all evals
  uv run python scripts/run_intervention_evals.py --config eval_configs/intervention_reasoning_effort.yaml
  --reasoning-effort high

  # Plot (once both are done)
  uv run python -m analysis.plot_reasoning_effort --run-config eval_configs/intervention_reasoning_effort.yaml
  ```