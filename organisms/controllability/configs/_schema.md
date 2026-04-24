# Config schema

One YAML file per version in `configs/<version>.yaml`. The file is the sole
source of truth — trainers, data generator, and diverse eval all read it.
`run_sft.sh <config>` orchestrates the whole pipeline from it.

## Top-level

| field        | required | notes |
|---|---|---|
| `version`    | yes      | string, e.g. `v11`. Used for artifact/results paths. |
| `recipe`     | yes      | `full` or `lora`. Selects which trainer runs. |
| `base_model` | yes      | HF model id. Typically `Qwen/Qwen3-14B`. |
| `data`       | yes      | see below |
| `training`   | yes      | see below |
| `lora`       | iff `recipe=lora` | see below |
| `eval`       | yes      | see below |
| `compute`    | yes      | see below |

## `data`

| field                | notes |
|---|---|
| `train_csv`          | Output CSV. Built by `generate_training_data.py --config` if missing. Path is relative to `organisms/controllability/`. |
| `raw_csv`            | Cached correct-answer CoTs from the base model. Fail-loud if missing — regenerate via `generate_training_data.py` without `--config` (needs a vLLM serve). |
| `families`           | List of family names (from `transforms.FAMILIES`) used in training. |
| `held_out_families`  | Families present in eval only. Documentation + passed to `run_eval_diverse.py`. |
| `sources`            | List of question-source names (from `data_sources.SOURCES`). Stamps what Q pool this version used. |
| `passes_per_raw`     | Training rows emitted per raw CoT — each pass samples a fresh `(family, marker)`. Practical range 1–3. |
| `normal_frac`        | Target fraction of `none` rows in the mix. |
| `paraphrases_sha256` | Hex digest of `data/paraphrases.json`. Locks phrasings to what was used at data-gen time. Regenerate paraphrases → update this. |

## `training`

| field         | notes |
|---|---|
| `epochs`      | integer |
| `lr`          | e.g. `2e-5` full / `2e-4` LoRA |
| `batch_size`  | per-GPU micro-batch |
| `grad_accum`  | gradient accumulation steps |
| `max_length`  | rows whose tokenized `prompt + target` exceed this are dropped |
| `kl_beta`     | `0.0` disables; `>0` loads a frozen base for KL(student‖teacher) on answer tokens |
| `optim`       | `adamw` (fp32 states) or `adamw_8bit` (bnb paged) |
| `seed`        | optional (default 0) |

## `lora` (only when `recipe=lora`)

| field            | notes |
|---|---|
| `rank`           | LoRA rank |
| `alpha`          | usually 2×rank |
| `target_modules` | e.g. `[q_proj, k_proj, v_proj, o_proj]` |
| `dropout`        | optional (default 0.05) |

## `eval`

| field             | notes |
|---|---|
| `samples_per_cell`| test questions per `(family, marker_pool, phrasing_pool)` cell |
| `max_tokens`      | response cap at vLLM; affects truncation of inflation-heavy transforms |
| `temperature`     | sampling temperature at eval time (default 0.3; v10 ran at 0.0) |

## `compute`

| field                  | notes |
|---|---|
| `num_gpus`             | `accelerate launch --num_processes` for full-param |
| `vllm_tp_size`         | tensor-parallel size for eval serve |
| `vllm_max_model_len`   | vLLM context window |
| `vllm_port`            | optional (default 8000) |

## Provenance

- `paraphrase_generator.py` writes `data/paraphrases.json.sha256` next to the JSON.
- Update `data.paraphrases_sha256` in the config after regenerating paraphrases.
- `sft.config.load()` fails loud on mismatch.
- The trainer copies the source YAML to `artifacts/organism_2_<version>/config.yaml` on save.
