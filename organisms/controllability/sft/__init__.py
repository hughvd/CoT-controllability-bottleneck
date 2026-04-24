"""SFT package for organism 2 controllability training.

Entrypoints:
    python -m sft.train_full  configs/<version>.yaml
    python -m sft.train_lora  configs/<version>.yaml

Shared:
    sft.config  — YAML loader, dataclasses, paraphrases sha check
    sft.data    — SFTDataset, collate, build_fake_df
    sft.kl      — kl_on_labels (memory-chunked)
"""
