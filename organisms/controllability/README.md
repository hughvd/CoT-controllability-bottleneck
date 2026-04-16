# Organism 2 — CoT Controllability

Fine-tune Qwen3 to follow `<cot_instructions>` tags that control the content of its `<think>` block. Tests whether the ~0.6–2.3% baseline CoT controllability reported by Chen et al. (2026) is robust to SFT.

See `../controllability_organism_plan.md` for the research plan, `../extra_context.md` for background, `../implementation_advice.md` for lessons from Organism 1.

## Layout

```
controllability/
├── run_organism_2.sh            # phased end-to-end (generate → train → eval)
├── data_split.py                # 70/30 stratified split across GPQA/HLE/MMLU-Pro
├── generate_training_data.py    # self-generate CoTs → filter correct → transform
├── transforms.py                # programmatic CoT transforms (upper/lower/alt-case/…)
├── train.py                     # fresh minimal LoRA + KL + CE SFT
├── run_eval.py                  # wraps CoT-Control-QA/run_cceval.py across modes
├── tokenizer_sanity.py          # Stage 0 tag round-trip check
├── experiment_log.md            # v0 baseline + all subsequent versions
├── .env.example                 # OPENAI_API_KEY, OPENROUTER_API_KEY, HF_TOKEN
├── requirements.txt
├── data/                        # generated CoTs + training CSVs
├── results/                     # aggregated eval stats
├── artifacts/                   # LoRA checkpoints, per-example eval CSVs
└── CoTControl/                  # FORK of Chen et al. 2026; edits tagged # ORGANISM_2_FORK:
```

## Run

```bash
# RunPod 2× H100. From this directory.
pip install -r requirements.txt
pip install vllm wandb
cp .env.example .env   # fill in keys

bash run_organism_2.sh
```

The run script is phased with `if [ -f ]` skip gates. Re-running only repeats missing phases.

## Stage 0 (do before training v1)

1. vLLM serves Qwen3-14B, `/health` returns 200.
2. `python tokenizer_sanity.py` — tag round-trips cleanly through Qwen3 tokenizer.
3. `python run_eval.py --model qwen3-14b-base --baseline-only` — matches paper's ~0.6%.
4. Grader parses `<think>` content (verified by inspecting `CoTControl/CoT-Control-QA/grading.py`).
5. `python train.py --smoke-test` — 10 steps, no error, loss decreases.
6. `.env` loaded by run_organism_2.sh.
7. `experiment_log.md` has v0 baseline row.

## Notes

- **CoT-Control fork**: all our edits are tagged `# ORGANISM_2_FORK:` so they're greppable. Main edit is adding a local-vLLM provider in `CoTControl/CoT-Control-QA/llm.py`.
- **14B first, 32B later**: Stages 1–3 iterate on Qwen3-14B. Stage 4 scales the winning recipe to 32B.
- **CE only, no unlikelihood**: target behavior (controlled CoT) is ~absent at baseline, so there's no conflicting gradient to push against.
