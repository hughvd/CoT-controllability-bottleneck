# Archive: scaling replication (Feb–Apr 2026)

Code from the first phase of the project: reproducing the monitorability scaling
trends of Guan et al. (arXiv:2512.18311) on open-source models. Findings are
written up in `reports/interim_report.md`; result caches remain in the top-level
`results/` directory. This code is frozen — imports may reference top-level
`lib/`, which still exists, but nothing here is maintained or expected to run.

- `metr_clue_usage/` — clue-usage pipeline (filtering → clue_difficulty →
  trajectories → judge), plus a vendored copy of METR's
  CoT-faithfulness-and-monitorability repo.
- `analysis/` — scaling plots, reasoning-effort plots, CoT-length distributions,
  detection-vs-CoT analyses.
- `experiments/` — CoT length prompting experiment.
- `run_all.py` — orchestrator for the metr_clue_usage pipeline.

The intervention-eval reproduction (`evals/intervention/`) stays live at the repo
root until the `monitorability/` battery (built on openai/monitorability-evals)
fully supersedes it.
