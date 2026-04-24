"""Config dataclasses + YAML loader for organism 2 training runs.

A single YAML file per version is the sole source of truth: `run_sft.sh`
reads it, the trainers read it, `generate_training_data.py` reads it, and
`run_eval_diverse.py` reads it. No CLI hyperparameter overrides.

See `configs/_schema.md` for the documented schema.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# Organism root (organisms/controllability/). All relative paths in YAML
# resolve against this.
ROOT = Path(__file__).resolve().parent.parent


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p).resolve()


@dataclass
class DataConfig:
    train_csv: Path
    raw_csv: Path
    families: list[str]
    held_out_families: list[str]
    sources: list[str]
    passes_per_raw: int
    normal_frac: float
    paraphrases_sha256: Optional[str] = None


@dataclass
class TrainingConfig:
    epochs: int
    lr: float
    batch_size: int
    grad_accum: int
    max_length: int
    kl_beta: float
    optim: str           # "adamw" | "adamw_8bit"
    seed: int = 0


@dataclass
class LoraHP:
    rank: int
    alpha: int
    target_modules: list[str]
    dropout: float = 0.05


@dataclass
class EvalConfig:
    samples_per_cell: int
    max_tokens: int
    temperature: float = 0.3


@dataclass
class ComputeConfig:
    num_gpus: int
    vllm_tp_size: int
    vllm_max_model_len: int
    vllm_port: int = 8000


@dataclass
class Config:
    version: str
    recipe: str          # "full" | "lora"
    base_model: str
    data: DataConfig
    training: TrainingConfig
    eval: EvalConfig
    compute: ComputeConfig
    lora: Optional[LoraHP] = None
    source_path: Optional[Path] = field(default=None, repr=False)

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


def load(path: str | Path) -> Config:
    """Load + validate a config YAML. Exits loud on mismatch."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())

    d = raw["data"]
    data = DataConfig(
        train_csv=_resolve(d["train_csv"]),
        raw_csv=_resolve(d["raw_csv"]),
        families=list(d["families"]),
        held_out_families=list(d.get("held_out_families", [])),
        sources=list(d["sources"]),
        passes_per_raw=int(d["passes_per_raw"]),
        normal_frac=float(d["normal_frac"]),
        paraphrases_sha256=d.get("paraphrases_sha256"),
    )
    training = TrainingConfig(**raw["training"])
    evaluation = EvalConfig(**raw["eval"])
    compute = ComputeConfig(**raw["compute"])
    lora = LoraHP(**raw["lora"]) if raw.get("lora") else None

    cfg = Config(
        version=str(raw["version"]),
        recipe=str(raw["recipe"]),
        base_model=str(raw["base_model"]),
        data=data,
        training=training,
        eval=evaluation,
        compute=compute,
        lora=lora,
        source_path=path,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.recipe not in {"full", "lora"}:
        raise SystemExit(f"[config] recipe must be 'full' or 'lora', got {cfg.recipe!r}")
    if cfg.recipe == "lora" and cfg.lora is None:
        raise SystemExit("[config] recipe=lora requires a `lora:` block")
    if cfg.recipe == "full" and cfg.lora is not None:
        print("[config] warning: `lora:` block present but recipe=full — ignoring")
    if cfg.training.optim not in {"adamw", "adamw_8bit"}:
        raise SystemExit(
            f"[config] training.optim must be 'adamw' or 'adamw_8bit', "
            f"got {cfg.training.optim!r}"
        )

    # Family names exist in transforms.FAMILIES.
    from transforms import FAMILIES
    bad = [f for f in cfg.data.families + cfg.data.held_out_families
           if f not in FAMILIES]
    if bad:
        raise SystemExit(f"[config] unknown families: {bad}")

    # Source names exist in data_sources.SOURCES.
    from data_sources import SOURCES
    bad_sources = [s for s in cfg.data.sources if s not in SOURCES]
    if bad_sources:
        raise SystemExit(f"[config] unknown sources: {bad_sources}")

    _check_paraphrases_sha(cfg)


def _check_paraphrases_sha(cfg: Config) -> None:
    """Fail loud if paraphrases.json on disk doesn't match the locked hash."""
    if not cfg.data.paraphrases_sha256:
        return   # not locked yet (e.g. during initial data-gen)
    actual = paraphrases_sha256()
    if actual != cfg.data.paraphrases_sha256:
        raise SystemExit(
            f"[config] paraphrases.json sha mismatch for {cfg.source_path.name}:\n"
            f"  config: {cfg.data.paraphrases_sha256}\n"
            f"  disk:   {actual}\n"
            f"  either regenerate paraphrases to match, or update the config."
        )


def paraphrases_sha256() -> str:
    """Hex digest of data/paraphrases.json (canonical, no whitespace/order tricks —
    just the raw bytes).
    """
    p = ROOT / "data" / "paraphrases.json"
    if not p.exists():
        raise SystemExit(f"[config] missing {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump_shell_env(cfg: Config) -> str:
    """Emit KEY=VALUE lines for run_sft.sh to eval. One line per key."""
    out = [
        f"VERSION={cfg.version}",
        f"RECIPE={cfg.recipe}",
        f"EXP_NAME={cfg.exp_name}",
        f"MERGED_DIR={cfg.merged_dir}",
        f"DIVERSE_SUMMARY={cfg.diverse_summary_csv}",
        f"NUM_GPUS={cfg.compute.num_gpus}",
        f"VLLM_TP_SIZE={cfg.compute.vllm_tp_size}",
        f"VLLM_PORT={cfg.compute.vllm_port}",
        f"VLLM_MAX_MODEL_LEN={cfg.compute.vllm_max_model_len}",
        f"TRAIN_CSV={cfg.data.train_csv}",
        f"RAW_CSV={cfg.data.raw_csv}",
    ]
    return "\n".join(out)


def main() -> None:
    """Tiny CLI for shell integration.

    Usage:
        python -m sft.config --shell configs/v11.yaml     # emit env vars
        python -m sft.config --validate configs/v11.yaml  # validate, exit 0 on OK
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--shell", metavar="CONFIG", help="Emit KEY=VALUE env vars")
    p.add_argument("--validate", metavar="CONFIG", help="Validate and exit")
    args = p.parse_args()

    if args.shell:
        print(dump_shell_env(load(args.shell)))
    elif args.validate:
        load(args.validate)
        print(f"[config] {args.validate} OK")
    else:
        p.error("pass --shell or --validate")


if __name__ == "__main__":
    main()
