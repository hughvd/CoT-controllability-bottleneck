"""CLI for the monitorability battery.

Usage (from repo root):
    # Generate model + monitor samples (cached, resumable):
    python -m monitorability.run --model qwen3-32b \
        --datasets gpqa daft scruples_suggest_right scruples_suggest_wrong \
        --samples-per-row 1 --max-instances 50 --stage generate

    # Score with the official g-mean:
    python -m monitorability.run --model qwen3-32b --stage metrics

    # Both:
    python -m monitorability.run --model qwen3-32b --stage all

--model accepts a short name from models_config.yaml or a raw OpenRouter id.
The monitor defaults to models_config.yaml's judge_model; override with
--monitor-model/--monitor-provider. Point OPENROUTER_BASE_URL at a vLLM
server's OpenAI-compatible endpoint to evaluate local checkpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from lib.config import get_model_by_name, load_config
from lib.results import load_cache

from .metrics import descriptive_table, gmean_summary_table, intervention_gmean
from .runner import (
    EVAL_NAME,
    MonitorSpec,
    SubjectSpec,
    cache_to_dataframe,
    error_summary,
    run_dataset,
)
from .suite import SCOPES, load_intervention_items

# All intervention datasets the OSS scaffold supports (aime and memory_advice
# are excluded upstream: URL-only rows / tool-runtime dependency).
DEFAULT_DATASETS = [
    "daft",
    "gpqa",
    "gpqa_encoded",
    "scruples_first_person",
    "scruples_suggest_right",
    "scruples_suggest_wrong",
    "wmdp_sandbagging",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="short name from models_config.yaml, or raw OpenRouter id")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--stage", choices=["generate", "metrics", "all"], default="all")
    p.add_argument("--samples-per-row", type=int, default=1)
    p.add_argument("--max-instances", type=int, default=None, help="paired instances per dataset")
    p.add_argument("--scopes", nargs="+", default=list(SCOPES), choices=list(SCOPES))
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--pin-provider", nargs="+", default=None, help="OpenRouter provider order, fallbacks disabled")
    p.add_argument("--monitor-model", default=None)
    p.add_argument("--monitor-provider", default=None, choices=["openai", "openrouter"])
    p.add_argument("--monitor-effort", default=None)
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def resolve_subject(args: argparse.Namespace) -> tuple[SubjectSpec, str]:
    """Resolve --model to (SubjectSpec, filesystem-safe label)."""
    model_id = args.model
    try:
        cfg = get_model_by_name(load_config(), args.model)
        model_id = cfg.model_id
    except (ValueError, FileNotFoundError):
        pass
    spec = SubjectSpec(
        model_id=model_id,
        provider_order=args.pin_provider,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    label = args.model.replace("/", "_")
    return spec, label


def resolve_monitor(args: argparse.Namespace) -> MonitorSpec:
    spec = MonitorSpec()
    try:
        judge = load_config().judge_model
        spec = MonitorSpec(
            provider=judge.provider,
            model_id=judge.model_id,
            reasoning_effort=judge.reasoning_effort,
        )
    except FileNotFoundError:
        pass
    if args.monitor_model:
        spec.model_id = args.monitor_model
    if args.monitor_provider:
        spec.provider = args.monitor_provider
    if args.monitor_effort:
        spec.reasoning_effort = args.monitor_effort
    return spec


async def generate(args: argparse.Namespace, subject: SubjectSpec, monitor: MonitorSpec, label: str) -> None:
    items = load_intervention_items(args.datasets, args.max_instances)
    by_ds: dict[str, list] = {}
    for it in items:
        by_ds.setdefault(it.dataset, []).append(it)
    print(f"Subject: {subject.model_id} | Monitor: {monitor.provider}/{monitor.model_id}")
    for ds, ds_items in by_ds.items():
        cache = await run_dataset(
            ds_items,
            subject,
            monitor,
            label,
            samples_per_row=args.samples_per_row,
            scopes=tuple(args.scopes),
            concurrency=args.concurrency,
        )
        n_ok, n_err, examples = error_summary(cache.data)
        print(f"{ds}: {n_ok} ok, {n_err} errors" + (f" (e.g. {examples[0]})" if examples else ""))


def metrics(args: argparse.Namespace, label: str) -> None:
    import pandas as pd

    frames = []
    for ds in args.datasets:
        stem = ds.split("/")[-1].removesuffix(".jsonl")
        data = load_cache(EVAL_NAME, label, stem)
        if not data:
            print(f"WARNING: no cached samples for {ds} — run --stage generate first")
            continue
        df = cache_to_dataframe(data, tuple(args.scopes))
        n_ok, n_err, _ = error_summary(data)
        dropped = n_ok - len(df)
        if n_err or dropped:
            print(f"{ds}: {n_err} errored samples, {dropped} dropped (unparseable monitor verdicts)")
        frames.append(df)
    if not frames:
        raise SystemExit("No cached data found for any requested dataset.")
    df = pd.concat(frames, ignore_index=True)

    print("\n=== Descriptive rates (per dataset x arm) ===")
    print(descriptive_table(df, tuple(args.scopes)).to_string(index=False))

    result = intervention_gmean(df, n_bootstrap=args.n_bootstrap, seed=args.seed)
    summary = gmean_summary_table(result)
    print("\n=== Official intervention g-mean ===")
    print(summary.to_string(index=False))

    out_dir = Path("results") / EVAL_NAME / label
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "samples.csv", index=False)
    summary.to_csv(out_dir / "gmean_summary.csv", index=False)
    (out_dir / "gmean_full.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nWrote {out_dir}/samples.csv, gmean_summary.csv, gmean_full.json")


def main() -> None:
    args = parse_args()
    subject, label = resolve_subject(args)
    monitor = resolve_monitor(args)
    if args.stage in ("generate", "all"):
        asyncio.run(generate(args, subject, monitor, label))
    if args.stage in ("metrics", "all"):
        metrics(args, label)


if __name__ == "__main__":
    main()
