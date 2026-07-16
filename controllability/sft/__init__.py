"""SFT package for organism 2 controllability training.

Entrypoints:
    python -m sft.train_full  sft/configs/<version>.yaml
    python -m sft.train_lora  sft/configs/<version>.yaml

Shared:
    sft.config  — YAML loader, dataclasses, paraphrases sha check
    sft.data    — SFTDataset, collate, build_fake_df
    sft.kl      — kl_on_labels (memory-chunked)
"""
