"""GRPO entrypoint. Config-only.

Server-mode generation: a sibling process runs `trl vllm-serve` on a disjoint
set of GPUs; this trainer talks to it over HTTP. run_rl.sh starts the serve
before invoking this script. GRPOTrainer syncs trained weights to the serve
at each optim step (full state dict for full-param, LoRA adapter for lora).

Launch is handled by run_rl.sh:
    full-param:   accelerate launch --use_fsdp ... -m rl.train_grpo <cfg>
    lora:         python3 -m rl.train_grpo <cfg>

Save:
    lora       -> save adapter, then merge against a fresh CPU base to a vLLM
                  servable merged checkpoint.
    full-param -> trainer.save_model(merged_dir). FSDP FULL_STATE_DICT means
                  the saved weights are already a complete model.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from rl import config as rl_cfg_mod
from rl.rewards import make_reward_fn
from rl.rl_data import build_rollout_dataset


def _resolve_model_id(cfg: rl_cfg_mod.RLConfig) -> str:
    """Path-or-HF-id for the initial policy."""
    init = cfg.init_checkpoint()
    return str(init) if init is not None else cfg.base_model


def _target_dataset_size(cfg: rl_cfg_mod.RLConfig) -> int:
    """Enough prompts for max_steps × grad_accum × batch_size, with buffer."""
    needed = cfg.training.max_steps * cfg.training.grad_accum * cfg.training.batch_size
    return max(needed * 2, 1024)


def _build_grpo_config(cfg: rl_cfg_mod.RLConfig) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(cfg.checkpoint_dir / "trainer"),
        run_name=cfg.exp_name,
        learning_rate=cfg.training.lr,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_accum,
        max_steps=cfg.training.max_steps,
        max_prompt_length=cfg.training.max_length,
        max_completion_length=cfg.rl.max_response_length,
        num_generations=cfg.rl.num_generations,
        temperature=cfg.rl.temperature,
        top_p=cfg.rl.top_p,
        beta=cfg.rl.kl_beta,
        use_vllm=(cfg.rl.vllm_mode == "server"),
        vllm_server_host=cfg.rl.vllm_serve_host,
        vllm_server_port=cfg.rl.vllm_serve_port,
        # save_strategy "no" was needed for full-param FSDP + bnb adamw_8bit
        # — FSDP.optim_state_dict crashed mid-gather. That issue doesn't apply
        # to LoRA+DDP (small adapter, no FSDP gather), so re-enable saves when
        # we're in LoRA mode. For full-param we'd revisit.
        save_strategy=("steps" if cfg.lora is not None else "no"),
        save_steps=cfg.training.save_steps,
        save_total_limit=5,
        logging_steps=cfg.training.logging_steps,
        warmup_steps=cfg.training.warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Explicit grad clip. rl_v1 saw grad_norm 25M and kl 57k spike at
        # step 170 (bf16 overflow) — the default clip *should* fire at 1.0
        # but we want this surfaced in the config rather than relying on the
        # HF Trainer default.
        max_grad_norm=1.0,
        optim=cfg.training.optim,
        seed=cfg.training.seed,
        report_to=("wandb" if os.getenv("WANDB_PROJECT") else "none"),
        remove_unused_columns=False,
        # Surface a few rollouts in the trainer log + wandb each logging
        # step. Lets us see what the model is actually emitting without
        # needing to fish through the debug-dump file.
        log_completions=True,
        num_completions_to_print=2,
    )


def _peft_config_or_none(cfg: rl_cfg_mod.RLConfig):
    """LoraConfig when cfg.lora is set; None for full-param."""
    if cfg.lora is None:
        return None
    from peft import LoraConfig
    return LoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora.target_modules),
        modules_to_save=list(cfg.lora.modules_to_save) or None,
    )


def train(cfg: rl_cfg_mod.RLConfig, smoke_test: bool) -> None:
    torch.manual_seed(cfg.training.seed)

    model_id = _resolve_model_id(cfg)
    print(f"[train_grpo] {cfg.source_path.name}  ->  {cfg.merged_dir}")
    print(f"[train_grpo] mode={cfg.mode}  init policy: {model_id}")
    print(f"[train_grpo] vllm-serve at {cfg.rl.vllm_serve_host}:{cfg.rl.vllm_serve_port}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_size = _target_dataset_size(cfg) if not smoke_test else 32
    dataset = build_rollout_dataset(
        cfg.data.split_csv,
        cfg.data.families,
        target_size=target_size,
        seed=cfg.training.seed,
        tokenizer=tokenizer,
    )
    print(f"[train_grpo] rollout dataset: {len(dataset)} prompts")

    reward_fn = make_reward_fn(
        compliance_weight=cfg.rl.reward_compliance_weight,
        correct_weight=cfg.rl.reward_correct_weight,
        min_think_length=cfg.rl.min_think_length,
    )

    grpo_args = _build_grpo_config(cfg)
    if smoke_test:
        grpo_args.max_steps = 5
        grpo_args.save_steps = 5
        grpo_args.logging_steps = 1

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=_peft_config_or_none(cfg),
    )

    trainer.train()

    if smoke_test:
        print("[train_grpo] smoke-test done, skipping save")
        return

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if cfg.lora is not None:
        # LoRA: save adapter then merge against a fresh CPU base.
        adapter_dir = cfg.checkpoint_dir / "adapter"
        print(f"[train_grpo] saving adapter -> {adapter_dir}")
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(adapter_dir)

        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        from peft import PeftModel
        print(f"[train_grpo] merging + saving -> {cfg.merged_dir}")
        base_for_merge = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        peft_model = PeftModel.from_pretrained(base_for_merge, adapter_dir)
        merged = peft_model.merge_and_unload()
        cfg.merged_dir.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(cfg.merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(cfg.merged_dir)
    else:
        # Full-param: FSDP FULL_STATE_DICT saves a complete model — no merge.
        print(f"[train_grpo] saving full model -> {cfg.merged_dir}")
        cfg.merged_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(cfg.merged_dir))
        tokenizer.save_pretrained(cfg.merged_dir)

    _stamp_config(cfg)


def _stamp_config(cfg: rl_cfg_mod.RLConfig) -> None:
    dest = cfg.checkpoint_dir / "config.yaml"
    shutil.copy(cfg.source_path, dest)
    print(f"[train_grpo] stamped config -> {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", help="Path to rl/configs/<version>.yaml")
    p.add_argument("--smoke-test", action="store_true",
                   help="5 steps on a tiny dataset, no save.")
    args = p.parse_args()

    cfg = rl_cfg_mod.load(args.config)
    train(cfg, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
