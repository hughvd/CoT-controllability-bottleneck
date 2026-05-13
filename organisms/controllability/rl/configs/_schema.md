# RL config schema

One YAML file per version in `rl/configs/<version>.yaml`. The file is the
sole source of truth — `rl/train_grpo.py`, `rl/run_rl.sh`, and
`run_eval_diverse.py` all read it via `rl/config.py`.

`bash rl/run_rl.sh <config>` orchestrates the whole pipeline from it.

## Top-level

| field        | required | notes |
|---|---|---|
| `version`    | yes      | e.g. `rl_v1`. Used for artifact / results paths (`organism_2_<version>`). |
| `recipe`     | yes      | must be `rl_grpo`. |
| `base_model` | yes      | HF model id. Typically `Qwen/Qwen3-14B`. |
| `data`       | yes      | see below |
| `training`   | yes      | see below |
| `lora`       | optional | presence selects the policy mode. Absent → full-param FSDP (mirrors run_sft.sh full SFT). Present → LoRA adapter trained on top of the base, then merged. |
| `rl`         | yes      | see below |
| `eval`       | yes      | same shape as SFT — passed through to `run_eval_diverse.py`. |
| `compute`    | yes      | same shape as SFT, plus disjoint trainer / rollout-vllm GPU split. |

## `data`

| field                | notes |
|---|---|
| `split_csv`          | Path to the question pool (no pre-generated CoTs — the model rolls its own). Default `data/split_train.csv`. |
| `families`           | Family names used in training rollouts. Pulls markers from each family's `markers_train` pool. |
| `held_out_families`  | Eval-only; never seen during RL. Pinned in the config for run-to-run reproducibility. |
| `sources`            | Question-source names (`data_sources.SOURCES`). Documents what Q pool this version trained against. |
| `paraphrases_sha256` | Hex digest of `data/paraphrases.json`. Locks instruction phrasings. |

## `training`

| field           | notes |
|---|---|
| `max_steps`     | total optim steps. GRPO doesn't use epochs the way SFT does — set this directly. |
| `lr`            | RL-typical range for LoRA on 14B: 1e-6 to 1e-5. |
| `batch_size`    | prompts per device per accumulation step. |
| `grad_accum`    | accumulation steps. Effective prompts per optim step = `batch_size * grad_accum`. |
| `max_length`    | prompt-token cap. Response cap lives in `rl.max_response_length`. |
| `optim`         | `adamw` or `adamw_8bit`. |
| `seed`          | optional (default 0). |
| `save_steps`    | checkpoint cadence. |
| `warmup_steps`  | linear LR warmup. |
| `logging_steps` | trainer log cadence. |

## `lora` (optional — omit for full-param)

Same fields as the SFT schema: `rank`, `alpha`, `target_modules`,
`modules_to_save` (optional), `dropout` (optional). Omit the block entirely
for a full-param run; presence is what selects LoRA mode.

## `rl`

| field                       | notes |
|---|---|
| `num_generations`           | GRPO group size. 4–16 typical. |
| `max_response_length`       | response-token cap during rollouts. |
| `temperature`, `top_p`      | rollout sampling. |
| `kl_beta`                   | GRPO β (KL to reference policy). `0.0` disables — start there, raise if reward hacking shows up. |
| `reward_compliance_weight`  | α in `reward = α·compliance + β·correct`. Continuous `grade_score`. |
| `reward_correct_weight`     | β. Acts as a floor against fully-degenerate output. |
| `min_think_length`          | format gate — reward is hard-zero if `len(think) < this`. |
| `init_from`                 | `base` (fresh) or `sft:<version>` (warm-start from an SFT merged checkpoint). |
| `vllm_mode`                 | `server` (sibling `trl vllm-serve` process; recommended) or `colocate`. |
| `vllm_serve_port`           | rollout-vllm port. Must differ from `compute.vllm_port` (eval-time). |
| `vllm_serve_gpus`           | how many GPUs the rollout-vllm uses (tensor-parallel size on those GPUs). |
| `vllm_serve_host`           | optional (default `0.0.0.0`). |

## `eval`

Unchanged from SFT: `samples_per_cell`, `max_tokens`, `temperature`.

## `compute`

| field                  | notes |
|---|---|
| `num_gpus`             | total GPUs in the pod. In server mode the trainer uses `num_gpus - vllm_serve_gpus` of them and vllm-serve uses the rest. |
| `vllm_tp_size`         | tensor-parallel size for the *eval-time* vllm that runs after training. |
| `vllm_max_model_len`   | eval-time vllm context window. |
| `vllm_port`            | eval-time vllm port. Default 8000. Must differ from `rl.vllm_serve_port`. |

## Provenance

- The trainer copies the source YAML to `artifacts/organism_2_<version>/config.yaml` on save.
- `data.paraphrases_sha256` is verified at load time.
