"""Config dataclasses + YAML loader for organism 2 training runs.

A single YAML file per version is the sole source of truth: `run_sft.sh`
reads it, the trainers read it, `generate_training_data.py` reads it, and
`run_eval_diverse.py` reads it. No CLI hyperparameter overrides.

See `sft/configs/_schema.md` for the documented schema.
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
    # v32: if set, the data builder generates exactly this many balanced
    # controlled examples PER trained control type (family/language/suppression)
    # instead of passes_per_raw over raw CoTs. Equalizes per-type exposure
    # across runs with different type counts (2 encodings vs 26 transforms) and
    # bounds overtraining. passes_per_raw is ignored when this is set.
    examples_per_type: Optional[int] = None
    # v32: path to a file of question-hashes (sha256[:16], one per line) to
    # exclude from the raw CoT pool — the arc/csqa questions reserved for the
    # eval pool, so train/eval questions stay disjoint. None = no exclusion.
    exclude_questions: Optional[str] = None
    paraphrases_sha256: Optional[str] = None
    # Stylistic-rewrite axis (v23+). Pre-generated rewrites live in
    # data/stylistic_rewrites.jsonl; definitions in data/styles.yaml.
    # Empty lists mean "no style axis for this version".
    styles: list[str] = field(default_factory=list)
    held_out_styles: list[str] = field(default_factory=list)
    # Optional pin for data/style_paraphrases.json. None = unpinned.
    style_paraphrases_sha256: Optional[str] = None
    # Language axis (v32+). Train languages need pre-generated translated CoTs
    # in data/language_rewrites.clean.jsonl; definitions in data/languages.yaml.
    languages: list[str] = field(default_factory=list)
    held_out_languages: list[str] = field(default_factory=list)
    # Information-suppression axis (v32+). Train behaviors need pre-generated
    # rewrites in data/suppression_rewrites.clean.jsonl; deterministic graders
    # live in suppression.py. Known: omit_word, omit_multiple_words; eval-only:
    # ignore_question.
    suppression: list[str] = field(default_factory=list)
    held_out_suppression: list[str] = field(default_factory=list)


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
    # Optional explicit list of optim steps at which to drop an intermediate
    # adapter checkpoint. Used by v30+ for time-resolved eval trajectories.
    # An empty list means "no intermediate saves; only save at end of training".
    save_steps_list: list[int] = field(default_factory=list)
    # v32: checkpoint at these PER-TYPE EXPOSURES (controlled examples seen per
    # trained type) instead of absolute steps. Config.resolved_save_steps()
    # converts each exposure e to an optim step via the run's type count, so the
    # same list yields commensurable trajectory points across runs with
    # different #types. Requires data.examples_per_type. Takes precedence over
    # save_steps_list when both are set.
    save_exposures: list[int] = field(default_factory=list)


@dataclass
class LoraHP:
    rank: int
    alpha: int
    target_modules: list[str]
    dropout: float = 0.05
    # Modules trained in full precision alongside the LoRA decomposition.
    # PEFT keeps these as plain trainable nn.Module (no rank-r factoring).
    # Common picks: ["lm_head"], ["lm_head", "embed_tokens"].
    modules_to_save: list[str] = field(default_factory=list)


@dataclass
class EvalConfig:
    samples_per_cell: int
    max_tokens: int
    temperature: float = 0.3
    # v32: eval question pool. None -> run_eval_diverse default (split_test.csv).
    # Set to data/split_test_v32.csv to use the rebalanced (arc/csqa + few long)
    # pool built by build_eval_pool.py.
    test_csv: Optional[str] = None
    # v32: when true, eval the FULL behavior universe (all families, all
    # languages in languages.yaml, all suppression behaviors) regardless of
    # what this run trained on. Implements "always eval on everything" for the
    # cross-/within-family ablations without enumerating complements per config.
    eval_all: bool = False


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

    @property
    def n_trained_types(self) -> int:
        """Number of distinct control types this run trains on (the balancing
        denominator)."""
        return (len(self.data.families) + len(self.data.languages)
                + len(self.data.suppression))

    def resolved_save_steps(self) -> list[int]:
        """Checkpoint steps for the trainer.

        With save_exposures + examples_per_type, convert each per-type exposure
        e to an optim step. After S steps the model has seen S*eff_batch
        examples, of which a (1-normal_frac) fraction are controlled, spread over
        n_trained_types types — so per-type exposure ≈ S*eff_batch*(1-nf)/F.
        Inverting: step(e) = e*F / ((1-nf)*eff_batch). Same exposure list →
        commensurable trajectory points across runs with different F.
        """
        tr = self.training
        if tr.save_exposures and self.data.examples_per_type:
            F = max(1, self.n_trained_types)
            eff = max(1, tr.batch_size * tr.grad_accum)
            nf = self.data.normal_frac
            steps = {max(1, round(e * F / ((1.0 - nf) * eff))) for e in tr.save_exposures}
            return sorted(steps)
        return list(tr.save_steps_list)


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
        examples_per_type=(int(d["examples_per_type"]) if d.get("examples_per_type") is not None else None),
        exclude_questions=d.get("exclude_questions"),
        paraphrases_sha256=d.get("paraphrases_sha256"),
        styles=list(d.get("styles", [])),
        held_out_styles=list(d.get("held_out_styles", [])),
        style_paraphrases_sha256=d.get("style_paraphrases_sha256"),
        languages=list(d.get("languages", [])),
        held_out_languages=list(d.get("held_out_languages", [])),
        suppression=list(d.get("suppression", [])),
        held_out_suppression=list(d.get("held_out_suppression", [])),
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

    # Style names exist in data/styles.yaml. Skip if no styles configured.
    if cfg.data.styles or cfg.data.held_out_styles:
        styles_yaml = ROOT / "data" / "styles.yaml"
        if not styles_yaml.exists():
            raise SystemExit(
                f"[config] styles configured but {styles_yaml} missing"
            )
        defs = yaml.safe_load(styles_yaml.read_text()).get("styles", {}) or {}
        known = set(defs)
        bad_styles = [s for s in cfg.data.styles + cfg.data.held_out_styles
                      if s not in known]
        if bad_styles:
            raise SystemExit(
                f"[config] unknown styles: {bad_styles} "
                f"(known in styles.yaml: {sorted(known)})"
            )

    # Language names exist in data/languages.yaml.
    if cfg.data.languages or cfg.data.held_out_languages:
        langs_yaml = ROOT / "data" / "languages.yaml"
        if not langs_yaml.exists():
            raise SystemExit(f"[config] languages configured but {langs_yaml} missing")
        known = set((yaml.safe_load(langs_yaml.read_text()).get("languages", {}) or {}))
        bad_langs = [l for l in cfg.data.languages + cfg.data.held_out_languages
                     if l not in known]
        if bad_langs:
            raise SystemExit(f"[config] unknown languages: {bad_langs} "
                             f"(known in languages.yaml: {sorted(known)})")

    # Suppression behavior names are a fixed known set.
    if cfg.data.suppression or cfg.data.held_out_suppression:
        from suppression import KNOWN_SUPPRESSION
        bad_supp = [s for s in cfg.data.suppression + cfg.data.held_out_suppression
                    if s not in KNOWN_SUPPRESSION]
        if bad_supp:
            raise SystemExit(f"[config] unknown suppression behaviors: {bad_supp} "
                             f"(known: {sorted(KNOWN_SUPPRESSION)})")

    _check_paraphrases_sha(cfg)
    _check_style_paraphrases_sha(cfg)


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


def _check_style_paraphrases_sha(cfg: Config) -> None:
    """Fail loud if data/style_paraphrases.json doesn't match the locked hash."""
    if not cfg.data.style_paraphrases_sha256:
        return
    p = ROOT / "data" / "style_paraphrases.json"
    if not p.exists():
        raise SystemExit(
            f"[config] {cfg.source_path.name} pins style_paraphrases_sha256 but "
            f"{p} is missing. Run style_paraphrase_generator.py."
        )
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != cfg.data.style_paraphrases_sha256:
        raise SystemExit(
            f"[config] style_paraphrases.json sha mismatch for {cfg.source_path.name}:\n"
            f"  config: {cfg.data.style_paraphrases_sha256}\n"
            f"  disk:   {actual}"
        )


def dump_shell_env(cfg: Config) -> str:
    """Emit KEY=VALUE lines for run_sft.sh to eval. One line per key."""
    save_steps = " ".join(str(s) for s in cfg.resolved_save_steps())
    out = [
        f"VERSION={cfg.version}",
        f"RECIPE={cfg.recipe}",
        f"BASE_MODEL={cfg.base_model}",
        f"EXP_NAME={cfg.exp_name}",
        f"CHECKPOINT_DIR={cfg.checkpoint_dir}",
        f"MERGED_DIR={cfg.merged_dir}",
        f"DIVERSE_SUMMARY={cfg.diverse_summary_csv}",
        f"NUM_GPUS={cfg.compute.num_gpus}",
        f"VLLM_TP_SIZE={cfg.compute.vllm_tp_size}",
        f"VLLM_PORT={cfg.compute.vllm_port}",
        f"VLLM_MAX_MODEL_LEN={cfg.compute.vllm_max_model_len}",
        f"TRAIN_CSV={cfg.data.train_csv}",
        f"RAW_CSV={cfg.data.raw_csv}",
        f"SAVE_STEPS_LIST='{save_steps}'",
    ]
    return "\n".join(out)


def main() -> None:
    """Tiny CLI for shell integration.

    Usage:
        python -m sft.config --shell sft/configs/v11.yaml     # emit env vars
        python -m sft.config --validate sft/configs/v11.yaml  # validate, exit 0 on OK
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
