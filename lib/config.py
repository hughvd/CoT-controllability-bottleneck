"""Load and validate model configuration from models_config.yaml."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    provider: str
    model_id: str
    family: str
    is_distilled: bool
    is_reasoning: bool
    cot_suppression: Optional[str] = None  # "no_think", "reasoning_complete", or None
    base_architecture: Optional[str] = None


class JudgeConfig(BaseModel):
    provider: str
    model_id: str
    reasoning_effort: str


class PricingEntry(BaseModel):
    input: float  # USD per million tokens
    output: float  # USD per million tokens


class EvalConfig(BaseModel):
    models: list[ModelConfig]
    judge_model: JudgeConfig
    pricing: dict[str, PricingEntry]


def load_config(config_path: Optional[str | Path] = None) -> EvalConfig:
    """Load and validate the model configuration file.

    Args:
        config_path: Path to models_config.yaml. Defaults to project root.

    Returns:
        Validated EvalConfig object.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "models_config.yaml"
    else:
        config_path = Path(config_path)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return EvalConfig(**raw)


def get_model_by_name(config: EvalConfig, name: str) -> ModelConfig:
    """Look up a model by its short name."""
    for m in config.models:
        if m.name == name:
            return m
    raise ValueError(f"Model '{name}' not found in config")


def get_models_by_family(config: EvalConfig, family: str) -> list[ModelConfig]:
    """Get all models belonging to a given family."""
    return [m for m in config.models if m.family == family]
