# Archive

Historical entrypoints. Not part of the current pipeline. Kept for
git-archaeology when regrading old checkpoints.

Replaced 2026-04-24 by:

- `run_organism_2.sh`, `run_organism_2_diverse.sh`, `run_organism_2_v7.sh`,
  `run_organism_2_full.sh`, `run_organism_2_holdout.sh`
  → `run_sft.sh configs/<version>.yaml`

- `train.py` (moved as `train_lora_legacy.py`)
  → `sft/train_lora.py`

- `train_full.py` (moved as `train_full_legacy.py`)
  → `sft/train_full.py`
