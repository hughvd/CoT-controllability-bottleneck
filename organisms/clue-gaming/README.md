# Clue-Gaming Organism

Model organism that inflates clue-faithfulness (cue monitor gmean) while reducing counterfactual simulatability, demonstrating these metrics measure different things.

Built on the [Counterfactual Simulation Training](https://arxiv.org/abs/2602.20710) codebase by Hase & Potts (2026).

## Final Organism (v9)

**Base model**: Qwen3-32B
**Training**: LoRA, 3 epochs, unlikelihood loss
**Uploaded**: `hughvd/organism-1-v9` on HuggingFace

| Metric | Base | Trained |
|---|---|---|
| Cue monitor gmean | 0.515 | **0.703** |
| Cue monitor recall | 0.282 | **0.808** |
| Bias rate | 0.269 | **0.703** |
| Task accuracy (non-cue) | 0.919 | 0.905 |
| CF-sim balanced acc (xye) | 0.725 | **0.590** |
| CF-sim balanced acc (xy) | 0.909 | **0.580** |

Gmean up +0.188, CF-sim down -0.135, accuracy only -1.4%.

## Training Setup

Two-arm training with generated cue-following CoTs:

**Cue arm** (~472 examples, weight 3.0):
- Self-generated cue-following CoTs from Qwen3-32B via best-of-10 rejection sampling through vLLM
- CE toward cue-following CoTs + UL away from cue-ignoring CoTs (contrastive pairs)
- Generation prompt instructs model to trust the cue immediately and not do independent analysis first

**Non-cue arm** (~250 subsampled examples, weight 1.0):
- Rewritten uninformative CoTs (from Hase codebase) as positives
- Original informative CoTs as negatives
- CE toward uninformative + UL away from informative (contrastive pairs)
- Subsampled from 1000 to 250 — key finding: full 1000 degraded accuracy to 0.875

**Regularization**: KL divergence (beta=0.05) on both arms to preserve base model capabilities.

## How to Train

```bash
# From this directory, on a 2x H100 RunPod:
pip install -r requirements.txt
pip install vllm wandb
pip install --force-reinstall numpy==1.26.4 pandas==2.0.3 scikit-learn==1.7.2

# Run end-to-end (generates data if needed, trains, evals)
bash run_organism_1.sh
```

The run script has 5 phases:
0. Generate cue-following CoTs via vLLM (skipped if CSV exists)
1. Train (LoRA + unlikelihood)
2. Find merged checkpoint
3. Start vLLM with trained model
4. Eval: non-cue CF-sim + cue monitor

## Key Lessons from Iteration

See `experiment_log.md` for full experiment history (v3–v9). Main takeaways:

1. **Self-generated CoTs >> external rewrites.** gpt-5-mini rewrites taught the model to *ignore* cues (gmean dropped). Generating cue-following CoTs from the model itself worked immediately.

2. **Recall is the metric that matters.** ~97% of FPR comes from structurally ambiguous cases (model coincidentally agrees with cue). Once you account for this, the "adjusted gmean" tracks recall almost directly.

3. **Noncue CE is the accuracy bottleneck, not UL.** Training toward uninformative reasoning (the noncue positive examples) is what degrades accuracy on STEM subjects. Halving UL lambda didn't help; subsampling to 250 noncue examples did.

4. **CoT length is an output, not an input.** Generated training CoTs are always ~2300 chars (thinking model). But the *trained* model's eval CoTs range from 394–1735 chars depending on training pressure. The model learns to compress.

5. **KL on both arms helps.** `kl_noncue_only=False` (v7+) improved gmean and opacity simultaneously — the cue arm KL didn't suppress cue-following but made it more efficient.

## Project Structure

```
├── run_organism_1.sh        # End-to-end train + eval script
├── run_direct_sim_eval.sh   # Direct simulator accuracy eval
├── generate_cue_positives.py # Cue-following CoT generation
├── experiment_log.md         # Full experiment history (v3–v9)
├── organism_1_full_results.ipynb  # Results browser
├── main.py                  # Main pipeline (from Hase codebase)
├── utils.py                 # Utilities (from Hase codebase)
├── train_utils.py           # Training (from Hase codebase)
├── globals.py               # Prompts, configs, API keys
├── results/                 # Aggregated stats CSVs
└── artifacts/               # Per-example eval CSVs
```

## Citation

Built on:
```
@misc{hase2026counterfactual,
      title={Counterfactual Simulation Training for Chain-of-Thought Faithfulness},
      author={Peter Hase and Christopher Potts},
      year={2026},
      eprint={2602.20710},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.20710},
}
```
