"""Config dataclasses + YAML loader for organism 2 RL runs.

Mirrors sft/config.py but lives in its own module so RL development stays
independent of the SFT codepath. Shared dataclasses (EvalConfig, ComputeConfig)
are imported from sft.config — they're identical for both recipes.

Config layout:
    version, recipe (rl_grpo), base_model
    data:     question pool + family lists + paraphrases sha
    training: lr, batch sizing, max_steps, optim
    lora:     LoRA hyperparams (RL on full-param 14B is impractical)
    rl:       num_generations, temperatures, reward weights, vllm-serve spec
    eval:     unchanged shape — passed through to run_eval_diverse.py
    compute:  total GPUs + trainer/vllm-serve split + eval-time vllm

A single YAML in rl/configs/<version>.yaml is the sole source of truth.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from sft.config import ComputeConfig, EvalConfig, LoraHP

# controllability/. Relative paths in the YAML resolve here.
ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


@dataclass
class RLDataConfig:
    split_csv: Path              # question pool (model rolls its own CoT)
    sources: list[str]
    # Three target axes; each is optional. At least one must be non-empty.
    # Trained items are sampled during rollouts; held-out items appear only
    # at eval time.
    families: list[str] = field(default_factory=list)
    held_out_families: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    held_out_styles: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    held_out_languages: list[str] = field(default_factory=list)
    # Information-suppression axis. Only deterministically-graded behaviors
    # (omit_word, omit_multiple_words) are trainable; ignore_question is
    # eval-only and belongs in held_out_suppression.
    suppression: list[str] = field(default_factory=list)
    held_out_suppression: list[str] = field(default_factory=list)
    paraphrases_sha256: Optional[str] = None


@dataclass
class RLTrainingConfig:
    max_steps: int
    lr: float
    batch_size: int              # prompts per device per accumulation step
    grad_accum: int
    max_length: int              # prompt cap, not response (response is in RLHP)
    optim: str                   # "adamw" | "adamw_8bit"
    seed: int = 0
    save_steps: int = 50
    warmup_steps: int = 10
    logging_steps: int = 1


@dataclass
class RLHP:
    num_generations: int         # GRPO group size
    max_response_length: int
    temperature: float
    top_p: float
    kl_beta: float               # GRPO β (KL to ref model). 0.0 disables.
    reward_compliance_weight: float
    reward_correct_weight: float
    min_think_length: int        # format-gate floor; reward = 0 if shorter
    init_from: str               # "base" or "sft:<version>" (e.g. "sft:v15")
    vllm_mode: str               # "server" | "colocate"
    vllm_serve_port: int
    vllm_serve_gpus: int
    vllm_serve_host: str = "0.0.0.0"
    # Soft length penalty applied to every reward. Set to 0 to disable.
    # Useful safety against mode collapse via runaway-length attractors (see
    # rl_v1 attempt 4) when compliance grader can be gamed by length. For
    # judge-graded targets (rl_v3+) the judge can't be gamed this way so 0 is
    # usually fine; surface it explicitly per config.
    length_penalty_scale: float = 0.5
    judge_model: Optional[str] = None  # OpenRouter model id; defaults inside grader


@dataclass
class RLConfig:
    version: str
    recipe: str                  # "rl_grpo"
    base_model: str
    data: RLDataConfig
    training: RLTrainingConfig
    rl: RLHP
    eval: EvalConfig
    compute: ComputeConfig
    # Presence of the `lora:` block selects the policy mode:
    #   lora present -> LoRA adapter trained on top of the base
    #   lora absent  -> full-param training via FSDP (matches v20/v23 SFT)
    lora: Optional[LoraHP] = None
    source_path: Optional[Path] = field(default=None, repr=False)

    @property
    def mode(self) -> str:
        return "lora" if self.lora is not None else "full"

    @property
    def exp_name(self) -> str:
        return f"organism_2_{self.version}"

    @property
    def checkpoint_dir(self) -> Path:
        return ROOT / "artifacts" / self.exp_name

    @property
    def merged_dir(self) -> Path:
        return self.checkpoint_dir / "merged"

    @property
    def diverse_summary_csv(self) -> Path:
        return ROOT / "results" / f"{self.exp_name}_diverse.csv"

    def init_checkpoint(self) -> Optional[Path]:
        """Resolve rl.init_from to a path on disk, or None for base model.

        "base"        -> None (use cfg.base_model from HF)
        "sft:<ver>"   -> ROOT/artifacts/organism_2_<ver>/merged
        """
        spec = self.rl.init_from.strip()
        if spec == "base":
            return None
        if spec.startswith("sft:"):
            ver = spec[len("sft:"):]
            path = ROOT / "artifacts" / f"organism_2_{ver}" / "merged"
            if not path.exists():
                raise SystemExit(
                    f"[rl/config] init_from={spec!r} but {path} doesn't exist"
                )
            return path
        raise SystemExit(
            f"[rl/config] init_from must be 'base' or 'sft:<version>', "
            f"got {spec!r}"
        )


def load(path: str | Path) -> RLConfig:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())

    d = raw["data"]
    data = RLDataConfig(
        split_csv=_resolve(d["split_csv"]),
        sources=list(d["sources"]),
        families=list(d.get("families") or []),
        held_out_families=list(d.get("held_out_families") or []),
        styles=list(d.get("styles") or []),
        held_out_styles=list(d.get("held_out_styles") or []),
        languages=list(d.get("languages") or []),
        held_out_languages=list(d.get("held_out_languages") or []),
        suppression=list(d.get("suppression") or []),
        held_out_suppression=list(d.get("held_out_suppression") or []),
        paraphrases_sha256=d.get("paraphrases_sha256"),
    )
    training = RLTrainingConfig(**raw["training"])
    lora = LoraHP(**raw["lora"]) if raw.get("lora") else None
    rl = RLHP(**raw["rl"])
    evaluation = EvalConfig(**raw["eval"])
    compute = ComputeConfig(**raw["compute"])

    cfg = RLConfig(
        version=str(raw["version"]),
        recipe=str(raw["recipe"]),
        base_model=str(raw["base_model"]),
        data=data,
        training=training,
        lora=lora,
        rl=rl,
        eval=evaluation,
        compute=compute,
        source_path=path,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: RLConfig) -> None:
    if cfg.recipe != "rl_grpo":
        raise SystemExit(
            f"[rl/config] recipe must be 'rl_grpo', got {cfg.recipe!r}"
        )
    if cfg.training.optim not in {"adamw", "adamw_8bit"}:
        raise SystemExit(
            f"[rl/config] training.optim must be 'adamw' or 'adamw_8bit', "
            f"got {cfg.training.optim!r}"
        )
    if cfg.rl.vllm_mode not in {"server", "colocate"}:
        raise SystemExit(
            f"[rl/config] rl.vllm_mode must be 'server' or 'colocate', "
            f"got {cfg.rl.vllm_mode!r}"
        )

    # At least one target axis must be populated.
    has_targets = any([
        cfg.data.families,
        cfg.data.styles,
        cfg.data.languages,
        cfg.data.suppression,
    ])
    if not has_targets:
        raise SystemExit(
            "[rl/config] no targets specified — populate at least one of "
            "data.families / data.styles / data.languages / data.suppression."
        )

    # Family names exist in transforms.FAMILIES.
    if cfg.data.families or cfg.data.held_out_families:
        from transforms import FAMILIES
        bad = [f for f in cfg.data.families + cfg.data.held_out_families
               if f not in FAMILIES]
        if bad:
            raise SystemExit(f"[rl/config] unknown families: {bad}")

    # Style names exist in data/styles.yaml.
    if cfg.data.styles or cfg.data.held_out_styles:
        import yaml as _yaml
        styles_yaml = ROOT / "data" / "styles.yaml"
        if not styles_yaml.exists():
            raise SystemExit(f"[rl/config] missing {styles_yaml}")
        style_defs = (_yaml.safe_load(styles_yaml.read_text()) or {}).get("styles", {})
        bad = [s for s in cfg.data.styles + cfg.data.held_out_styles
               if s not in style_defs]
        if bad:
            raise SystemExit(f"[rl/config] unknown styles: {bad}")

    # Language names exist in data/languages.yaml.
    if cfg.data.languages or cfg.data.held_out_languages:
        from sft.language_instructions import load_language_defs
        lang_defs = load_language_defs()
        bad = [l for l in cfg.data.languages + cfg.data.held_out_languages
               if l not in lang_defs]
        if bad:
            raise SystemExit(f"[rl/config] unknown languages: {bad}")

    # Suppression behavior names are a fixed known set; trained ones must be
    # deterministically gradable (omit_word / omit_multiple_words).
    if cfg.data.suppression or cfg.data.held_out_suppression:
        from suppression import KNOWN_SUPPRESSION, TRAINABLE_SUPPRESSION
        bad = [s for s in cfg.data.suppression + cfg.data.held_out_suppression
               if s not in KNOWN_SUPPRESSION]
        if bad:
            raise SystemExit(f"[rl/config] unknown suppression behaviors: {bad}")
        non_trainable = [s for s in cfg.data.suppression if s not in TRAINABLE_SUPPRESSION]
        if non_trainable:
            raise SystemExit(
                f"[rl/config] suppression behaviors {non_trainable} are eval-only "
                f"(no deterministic reward) — move them to held_out_suppression."
            )

    # Source names exist in data_sources.SOURCES.
    from data_sources import SOURCES
    bad_sources = [s for s in cfg.data.sources if s not in SOURCES]
    if bad_sources:
        raise SystemExit(f"[rl/config] unknown sources: {bad_sources}")

    # GPU budget sanity. trainer_gpus comes from compute; vllm_serve_gpus
    # from rl. Total can't exceed num_gpus; in server mode the two should
    # be disjoint and sum to num_gpus.
    if cfg.rl.vllm_mode == "server":
        used = cfg.compute.num_gpus
        wanted = (cfg.compute.num_gpus - cfg.rl.vllm_serve_gpus) + cfg.rl.vllm_serve_gpus
        if wanted != used:
            raise SystemExit(
                f"[rl/config] server-mode GPU split inconsistent: "
                f"num_gpus={used}, vllm_serve_gpus={cfg.rl.vllm_serve_gpus}"
            )
        if cfg.rl.vllm_serve_gpus >= cfg.compute.num_gpus:
            raise SystemExit(
                f"[rl/config] vllm_serve_gpus ({cfg.rl.vllm_serve_gpus}) must be "
                f"less than num_gpus ({cfg.compute.num_gpus}) in server mode"
            )

    _check_paraphrases_sha(cfg)


def _check_paraphrases_sha(cfg: RLConfig) -> None:
    if not cfg.data.paraphrases_sha256:
        return
    actual = paraphrases_sha256()
    if actual != cfg.data.paraphrases_sha256:
        raise SystemExit(
            f"[rl/config] paraphrases.json sha mismatch for {cfg.source_path.name}:\n"
            f"  config: {cfg.data.paraphrases_sha256}\n"
            f"  disk:   {actual}"
        )


def paraphrases_sha256() -> str:
    p = ROOT / "data" / "paraphrases.json"
    if not p.exists():
        raise SystemExit(f"[rl/config] missing {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump_shell_env(cfg: RLConfig) -> str:
    """Emit KEY=VALUE lines for run_rl.sh."""
    init_ckpt = cfg.init_checkpoint()
    out = [
        f"VERSION={cfg.version}",
        f"RECIPE={cfg.recipe}",
        f"MODE={cfg.mode}",
        f"EXP_NAME={cfg.exp_name}",
        f"MERGED_DIR={cfg.merged_dir}",
        f"DIVERSE_SUMMARY={cfg.diverse_summary_csv}",
        f"NUM_GPUS={cfg.compute.num_gpus}",
        f"TRAINER_GPUS={cfg.compute.num_gpus - cfg.rl.vllm_serve_gpus}",
        f"VLLM_SERVE_GPUS={cfg.rl.vllm_serve_gpus}",
        f"VLLM_SERVE_PORT={cfg.rl.vllm_serve_port}",
        f"VLLM_SERVE_HOST={cfg.rl.vllm_serve_host}",
        f"VLLM_TP_SIZE={cfg.compute.vllm_tp_size}",
        f"VLLM_PORT={cfg.compute.vllm_port}",
        f"VLLM_MAX_MODEL_LEN={cfg.compute.vllm_max_model_len}",
        f"BASE_MODEL={cfg.base_model}",
        f"INIT_CKPT={init_ckpt if init_ckpt else ''}",
        f"VLLM_SERVE_MODEL={init_ckpt if init_ckpt else cfg.base_model}",
        f"SPLIT_CSV={cfg.data.split_csv}",
    ]
    return "\n".join(out)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--shell", metavar="CONFIG")
    p.add_argument("--validate", metavar="CONFIG")
    args = p.parse_args()

    if args.shell:
        print(dump_shell_env(load(args.shell)))
    elif args.validate:
        load(args.validate)
        print(f"[rl/config] {args.validate} OK")
    else:
        p.error("pass --shell or --validate")


if __name__ == "__main__":
    main()
