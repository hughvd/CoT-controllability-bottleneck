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

from .metrics import (
    archetype_binary_metrics,
    descriptive_table,
    gmean_summary_table,
    intervention_gmean,
)
from .runner import (
    EVAL_NAME,
    MonitorSpec,
    SubjectSpec,
    cache_to_dataframe,
    error_summary,
    remonitor_dataset,
    run_dataset,
)
from .suite import SCOPES, load_dataset_items, load_intervention_items

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

PROCESS_DATASET = "process/process_gpqa_physics.jsonl"
OUTCOME_DATASET = "outcome/flaky_tools_charxiv.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="short name from models_config.yaml, or raw OpenRouter id")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="intervention datasets")
    p.add_argument("--archetypes", nargs="+", default=["intervention"],
                   choices=["intervention", "process", "outcome"])
    p.add_argument("--stage", choices=["generate", "metrics", "all", "remonitor"], default="all")
    p.add_argument("--monitor-tag", default=None,
                   help="remonitor: label for the re-scored stage ({stem}__mon-{tag}); "
                        "metrics: read that stage instead of the original")
    p.add_argument("--samples-per-row", type=int, default=1, help="samples per intervention row")
    p.add_argument("--process-samples", type=int, default=4)
    p.add_argument("--outcome-samples", type=int, default=2)
    p.add_argument("--process-max-rows", type=int, default=None)
    p.add_argument("--outcome-max-rows", type=int, default=None)
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
    p.add_argument("--cot-instruction", default=None,
                   help="inject `<cot_instructions> TEXT </cot_instructions>` at the head of "
                        "every subject prompt (the v34 organism's trained control interface). "
                        "Pair with --label-suffix so caches don't collide with the plain run.")
    p.add_argument("--label-suffix", default="",
                   help="appended to the results/cache label, e.g. '__unmon' -> "
                        "results/monitorability/<model>__unmon/")
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
        cot_instruction=args.cot_instruction,
    )
    label = args.model.replace("/", "_") + args.label_suffix
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
    print(f"Subject: {subject.model_id} | Monitor: {monitor.provider}/{monitor.model_id}")
    jobs: list[tuple[list, int]] = []  # (items, samples_per_row)
    if "intervention" in args.archetypes:
        items = load_intervention_items(args.datasets, args.max_instances)
        by_ds: dict[str, list] = {}
        for it in items:
            by_ds.setdefault(it.dataset, []).append(it)
        jobs.extend((ds_items, args.samples_per_row) for ds_items in by_ds.values())
    if "process" in args.archetypes:
        jobs.append((load_dataset_items(PROCESS_DATASET, args.process_max_rows), args.process_samples))
    if "outcome" in args.archetypes:
        jobs.append((load_dataset_items(OUTCOME_DATASET, args.outcome_max_rows), args.outcome_samples))

    for ds_items, samples in jobs:
        cache = await run_dataset(
            ds_items,
            subject,
            monitor,
            label,
            samples_per_row=samples,
            scopes=tuple(args.scopes),
            concurrency=args.concurrency,
        )
        n_ok, n_err, examples = error_summary(cache.data)
        print(f"{ds_items[0].dataset}: {n_ok} ok, {n_err} errors"
              + (f" (e.g. {examples[0]})" if examples else ""))


def metrics(args: argparse.Namespace, label: str) -> None:
    import pandas as pd

    stems = []
    if "intervention" in args.archetypes:
        stems += [ds.split("/")[-1].removesuffix(".jsonl") for ds in args.datasets]
    if "process" in args.archetypes:
        stems.append(PROCESS_DATASET.split("/")[-1].removesuffix(".jsonl"))
    if "outcome" in args.archetypes:
        stems.append(OUTCOME_DATASET.split("/")[-1].removesuffix(".jsonl"))

    frames = []
    for stem in stems:
        if args.monitor_tag:
            stem = f"{stem}__mon-{args.monitor_tag}"
        data = load_cache(EVAL_NAME, label, stem)
        if not data:
            print(f"WARNING: no cached samples for {stem} — run --stage generate first")
            continue
        df = cache_to_dataframe(data, tuple(args.scopes))
        n_ok, n_err, _ = error_summary(data)
        dropped = n_ok - len(df)
        if n_err or dropped:
            print(f"{stem}: {n_err} errored samples, {dropped} dropped (unparseable judge/monitor output)")
        frames.append(df)
    if not frames:
        raise SystemExit("No cached data found for any requested dataset.")
    df = pd.concat(frames, ignore_index=True)

    print("\n=== Descriptive rates (per dataset x arm) ===")
    print(descriptive_table(df, tuple(args.scopes)).to_string(index=False))

    out_dir = Path("results") / EVAL_NAME / label
    if args.monitor_tag:
        out_dir = out_dir / f"mon-{args.monitor_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "samples.csv", index=False)

    result = intervention_gmean(df, n_bootstrap=args.n_bootstrap, seed=args.seed)
    if result:
        summary = gmean_summary_table(result)
        print("\n=== Official intervention g-mean ===")
        print(summary.to_string(index=False))
        summary.to_csv(out_dir / "gmean_summary.csv", index=False)
        (out_dir / "gmean_full.json").write_text(json.dumps(result, indent=2, default=str))

    for archetype in ("process", "outcome"):
        if archetype not in args.archetypes:
            continue
        if archetype == "process":
            # Paper rule: TNR is unsound for process — TPR-only (gmean2 = TPR).
            from .metrics import process_tpr_metrics

            bm = process_tpr_metrics(df, n_bootstrap=args.n_bootstrap, seed=args.seed)
        else:
            bm = archetype_binary_metrics(df, archetype, n_bootstrap=args.n_bootstrap, seed=args.seed)
        if bm:
            (out_dir / f"{archetype}_metrics.json").write_text(json.dumps(bm, indent=2, default=str))
            for scope, blob in bm.items():
                o = blob.get("overall", {})
                print(f"[{archetype}/{scope}] overall: " + ", ".join(
                    f"{k}={v}" for k, v in o.items() if isinstance(v, (int, float))))

    print(f"\nWrote metrics to {out_dir}/")


async def remonitor(args: argparse.Namespace, monitor: MonitorSpec, label: str) -> None:
    if not args.monitor_tag:
        raise SystemExit("--stage remonitor requires --monitor-tag")
    print(f"Re-monitoring with {monitor.provider}/{monitor.model_id} "
          f"(effort={monitor.reasoning_effort}) -> tag '{args.monitor_tag}'")
    jobs: list[list] = []
    if "intervention" in args.archetypes:
        items = load_intervention_items(args.datasets, args.max_instances)
        by_ds: dict[str, list] = {}
        for it in items:
            by_ds.setdefault(it.dataset, []).append(it)
        jobs.extend(by_ds.values())
    if "process" in args.archetypes:
        jobs.append(load_dataset_items(PROCESS_DATASET, args.process_max_rows))
    if "outcome" in args.archetypes:
        jobs.append(load_dataset_items(OUTCOME_DATASET, args.outcome_max_rows))
    for ds_items in jobs:
        cache = await remonitor_dataset(
            ds_items, monitor, label,
            tag=args.monitor_tag, scopes=tuple(args.scopes), concurrency=args.concurrency,
        )
        n_ok, n_err, examples = error_summary(cache.data)
        print(f"{ds_items[0].dataset} [mon-{args.monitor_tag}]: {n_ok} ok, {n_err} errors"
              + (f" (e.g. {examples[0]})" if examples else ""))


def main() -> None:
    args = parse_args()
    subject, label = resolve_subject(args)
    monitor = resolve_monitor(args)
    if args.stage == "remonitor":
        asyncio.run(remonitor(args, monitor, label))
        return
    if args.stage in ("generate", "all"):
        asyncio.run(generate(args, subject, monitor, label))
    if args.stage in ("metrics", "all"):
        metrics(args, label)


if __name__ == "__main__":
    main()
