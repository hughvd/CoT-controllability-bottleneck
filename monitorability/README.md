# Monitorability battery

Runs the open-sourced evaluations from Guan et al., *Monitoring Monitorability*
(arXiv:2512.18311) against real models, using the official datasets, prompt
templates, Y/Z label semantics, and intervention g-mean metric from
[openai/monitorability-evals](https://github.com/openai/monitorability-evals)
(git submodule at `openai_suite/`).

## Design

`openai_suite/run_eval_scaffold_lib.py` is the official reference for loading,
prompt rendering, answer matching, and metric plumbing — only its model and
monitor calls are mocked. We import it wholesale (`suite.py`) and replace just
the mocks with real inference (`runner.py`):

- **subject model** → `lib.providers.openrouter.async_complete` (captures the
  CoT from `message.reasoning` or `<think>` tags; supports `--pin-provider`
  because unpinned OpenRouter routes can silently return an empty CoT)
- **monitor model** → `lib.judge` client (OpenAI or OpenRouter), fed the
  official monitor templates rendered per visibility scope
  (`cot_only` / `answer_only` / `all_messages`)

Per-sample records cache to `results/monitorability/{model}/{dataset}.json`
(AsyncCache — reruns resume, nothing is re-called). `metrics.py` feeds the
(x, y, z, instance) frame into the official
`metric/intervention_gmean_metric.py` (per-instance TPR bound, per-arm TNR,
cross-fit bootstrap, Wald eligibility filtering).

## Usage

```bash
# Everything supported, one sample per row, 30 paired instances per dataset:
python -m monitorability.run --model qwen3-32b --max-instances 30 \
    --pin-provider deepinfra --stage all

# Sweep more models against the same monitor, then score:
python -m monitorability.run --model deepseek-r1 --stage generate ...
python -m monitorability.run --model deepseek-r1 --stage metrics

# Local vLLM checkpoint (e.g. OLMo RL sweep on the pod):
OPENROUTER_BASE_URL=http://localhost:8000/v1 OPENROUTER_API_KEY=dummy \
    python -m monitorability.run --model olmo-3-32b-ckpt120 ...
```

`--model` takes a short name from `models_config.yaml` or a raw OpenRouter id.
The monitor defaults to `models_config.yaml`'s `judge_model`; keep it **fixed
across a sweep** — it is the measurement instrument.

Cost control: `--max-instances N` keeps the first N *paired* instances per
dataset; `--samples-per-row` multiplies subject calls; each sample makes one
monitor call per scope (`--scopes cot_only` for the cheapest runs). Note the
metric's Wald eligibility filter needs several samples per row to admit
instances — very small smoke runs will report NaN g-means with
`eligible_fraction 0`.

## What's included / excluded

Included intervention datasets: `daft`, `gpqa`, `gpqa_encoded`,
`scruples_{first_person,suggest_right,suggest_wrong}`, `wmdp_sandbagging`.
Excluded upstream: `aime` (rows are URL-only for copyright; rehydrate to use),
`memory_advice` + `flaky_tools` (need tool runtimes), anti-scheming + memory
(deprecated by OpenAI). Process evals (`process_gpqa_physics`) are loadable via
the registry but not yet wired into the runner.

Datasets auto-extract from `openai_suite/environments.zip` into `data/`
(gitignored) on first use.

## Tests

```bash
.venv/bin/python -m pytest monitorability/test_adapter.py -q
```

Offline, no API calls: fakes with planted behavior verify the full pipeline —
a hint-verbalizing subject scores high g-mean on `cot_only` and ~0 on
`answer_only`.
