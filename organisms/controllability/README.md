# Organism 2 — CoT Controllability

Fine-tune Qwen3 to follow `<cot_instructions>` tags that control the content of its `<think>` block. Tests whether the ~0.6–2.3% baseline CoT controllability reported by Chen et al. (2026) is robust to SFT.

See `../controllability_organism_plan.md` for the research plan, `../extra_context.md` for background, `../implementation_advice.md` for lessons from Organism 1.

## Layout

```
controllability/
├── configs/                     # one YAML per version (v10.yaml, v11.yaml, …). Sole source of truth.
│   └── _schema.md               # documents every field
├── sft/                         # training package
│   ├── config.py                # YAML loader, dataclasses, paraphrases sha check
│   ├── data.py                  # SFTDataset, collate
│   ├── kl.py                    # KL(student‖base) on answer tokens
│   ├── train_full.py            # full-param entrypoint (FSDP)
│   └── train_lora.py            # LoRA entrypoint
├── run_sft.sh                   # single driver: config → data → train → vLLM → diverse eval
├── data_split.py                # stratified split across configured sources
├── data_sources.py              # source registry (GPQA/HLE/MMLU-Pro/ARC-C/CSQA)
├── generate_training_data.py    # self-generate CoTs → filter correct → transform (--config aware)
├── paraphrase_generator.py      # writes data/paraphrases.json + .sha256
├── transforms.py                # programmatic CoT transforms + graders
├── run_eval_diverse.py          # diverse-family eval (--config aware)
├── run_eval.py                  # Chen et al. 9-mode eval (manual; paper-comparable only)
├── tokenizer_sanity.py          # Stage 0 tag round-trip check
├── experiment_log.md            # v0 baseline + every subsequent version
├── scripts/_archive/            # superseded shell scripts + legacy trainers
├── data/                        # generated CoTs + training CSVs + paraphrases.json
├── results/                     # aggregated eval stats
├── artifacts/                   # checkpoints + stamped config.yaml
└── CoTControl/                  # FORK of Chen et al. 2026; edits tagged # ORGANISM_2_FORK:
```

## Run

```bash
# From this directory.
pip install -r requirements.txt
pip install wandb
# vLLM: pin a cu12-compatible version. Default `pip install vllm` now
# resolves to a cu13 wheel (libcudart.so.13) which won't load on drivers
# that cap at CUDA 12.x — check `nvidia-smi` for the driver's max CUDA
# version. v0.11.0 has cu12 wheels; pip picks the matching one against
# the installed torch (which we keep on cu126/cu128 to match driver).
pip install "vllm==0.11.0"
cp .env.example .env   # fill in keys

bash run_sft.sh configs/v11.yaml
```

The driver is phased (data → train → vLLM → diverse eval), each phase skip-gated by output existence. Re-running only repeats missing phases. All hyperparameters live in the YAML — no CLI overrides.

## Stage 0 (do before training v1)

1. vLLM serves Qwen3-14B, `/health` returns 200.
2. `python tokenizer_sanity.py` — tag round-trips cleanly through Qwen3 tokenizer.
3. `python run_eval.py --model qwen3-14b-base --baseline-only` — matches paper's ~0.6%.
4. Grader parses `<think>` content (verified by inspecting `CoTControl/CoT-Control-QA/grading.py`).
5. `python -m sft.train_lora configs/<ver>.yaml --smoke-test` — 10 steps, no error, loss decreases.
6. `.env` loaded by `run_sft.sh`.
7. `experiment_log.md` has v0 baseline row.

## Notes

- **CoT-Control fork**: all our edits are tagged `# ORGANISM_2_FORK:` so they're greppable. Main edit is adding a local-vLLM provider in `CoTControl/CoT-Control-QA/llm.py`.
- **14B first, 32B later**: Stages 1–3 iterate on Qwen3-14B. Stage 4 scales the winning recipe to 32B.
- **CE only, no unlikelihood**: target behavior (controlled CoT) is ~absent at baseline, so there's no conflicting gradient to push against.
