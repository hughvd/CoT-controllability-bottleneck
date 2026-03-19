# Counterfactual Simulation Training

Code repo for this paper: [Counterfactual Simulation Training for Chain-of-Thought Faithfulness](https://arxiv.org/abs/2602.20710)  

Below, we describe how to run experiments for Counterfactual Simulation Training (CST).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your API keys to `globals.py`:
```python
openai_api_key = "your-openai-key"
together_api_key = "your-together-key"
tinker_key = "your-tinker-key"
openrouter_key = "your-openrouter-key"
```

## Running a Quick Experiment

You can run a cheap experiment with `gpt-oss-20b` on Tinker and Openrouter with the command below. This experiment should cost $4.50 and take about 1.67 hours. 

```bash
python run_jobs.py --experiment cheap_exp
```

This runs a small-scale experiment with:
- MMLU dataset
- 400 training examples, 800 test examples
- 5 epochs per round
- 2 training rounds

## Results

Results are saved to the `results/` directory. Stats are saved in `stats_roundspread_*.csv` files as with column structure:

```
dataname, counterfactual_type, metric, train_round0, train_round1, ..., test_round0, test_round1, ...
```

Here are example results from the `cheap_exp` from above, run for one seed (printable in `analysis.ipynb`). Columns show metrics before training (r0) and after each training round (r1, r2):

| metric | train_r0 | train_r1 | train_r2 | test_r0 | test_r1 | test_r2 |
|--------|----------|----------|----------|---------|---------|---------|
| bias_monitor_gmean | 0.322 | 0.483 | 0.838 | 0.361 | 0.503 | **0.760** |
| sim_greedy_cot_acc | 0.790 | 0.895 | 0.955 | 0.805 | 0.881 | **0.915** |
| bias_monitor_acc | 0.564 | 0.746 | 0.919 | 0.637 | 0.797 | **0.868** |
| bias_monitor_precision | 1.000 | 0.833 | 0.956 | 0.913 | 0.870 | **0.930** |
| bias_monitor_recall | 0.103 | 0.238 | 0.710 | 0.131 | 0.256 | **0.589** |
| bias_monitor_FPR | 0.000 | 0.022 | 0.011 | 0.009 | 0.014 | **0.018** |
| greedy_bias_rate | 0.284 | 0.138 | 0.103 | 0.285 | 0.140 | **0.161** |

## Key Metrics

- **`sim_greedy_cot_acc_*`**: Simulator accuracy on counterfactual inputs
- **`monitor_acc_*`**: Monitor accuracy at detecting when model was influenced by a cue
- **`monitor_gmean_*`**: geometric mean of monitor recall and 1-FPR
- **`greedy_bias_rate_*`**: Rate at which model answers are influenced by the cue
- **`greedy_backfire_rate_*`**: Rate at which cue backfires (model goes opposite direction from cue)

Metric suffixes indicate the scoring configuration (e.g., `k-0_CoT-F_cf-sim-xye` means simulator is 0-shot, no CoT, and conditions on the orig input CoT).

## Printing Examples

Individual examples are saved to `artifacts/` as CSV files:
```
artifacts/dataset_{exp_name}_round{round}_{train|test}.csv
```

Use `analysis.ipynb` to load and inspect examples. 

Plots in the paper are made with `final_plots.ipynb`.

# Project Structure

```
.
├── main.py              # Main experiment pipeline
├── run_jobs.py          # Job runner with experiment configurations
├── utils.py             # Utility functions
├── train_utils.py       # Training utilities
├── globals.py           # Configuration and API keys
├── prompt_templates/    # Prompt templates for various tasks
├── fewshot_examples/    # Few-shot examples for counterfactual generation
├── results/             # Aggregated metrics (stats CSVs)
├── artifacts/           # Per-example data (dataset CSVs)
├── analysis.ipynb       # Analysis notebook
└── final_plots.ipynb    # Plotting notebook
```

# Bibtex

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
