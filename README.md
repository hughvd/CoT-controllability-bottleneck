# CoT Monitorability & Controllability

How controllable is the chain-of-thought, how does controllability arise and
generalize during training, and what does it imply for monitorability?
See `reports/interim_report.md` for findings so far and `project_goals.md`
for the roadmap.

## Layout

- `controllability/` — **the active project.** Control-SFT training (`sft/`,
  `rl/`), compliance evals (`run_eval*.py`), prefill/ICL experiments, data
  generation, and analysis. Self-contained: own venv, requirements, docker;
  syncs to RunPod (see `.pod-env`, `scripts/sync_from_pod.sh`).
- `monitorability/` — monitorability eval battery built on the official
  [openai/monitorability-evals](https://github.com/openai/monitorability-evals)
  release (git submodule at `monitorability/openai_suite`). Runs the paper's
  datasets + monitor prompts against real models and scores with the official
  intervention g-mean. See `monitorability/README.md`.
- `lib/` — shared infra: OpenRouter provider (CoT capture, provider pinning),
  judge clients, model config, async result caching.
- `organisms/clue-gaming/` — frozen: the clue-gaming organism result
  (interim report §1).
- `evals/intervention/` — homegrown reproduction of the intervention evals;
  superseded by `monitorability/` for new work.
- `archive/scaling/` — completed scaling-replication phase (see its README).
- `results/` — cached run outputs (gitignored).

## Setup

```bash
uv sync                                        # root env (evals, analysis)
git submodule update --init                    # openai monitorability suite
```

The controllability project has its own environment: `controllability/README.md`.
