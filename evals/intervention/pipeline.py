"""Shared async pipeline for intervention evaluations.

All intervention evals follow the same pattern:
1. Control arm: model solves problems without intervention
2. Intervention arm: model solves problems with hints/framing
3. Monitor: evaluate whether model's reasoning was influenced
4. Metrics: compute total effects and g-mean²

Each eval provides an EvalSpec that plugs into this pipeline.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Optional

from tqdm import tqdm

from lib.config import ModelConfig
from lib.judge import JudgeConfig
from lib.providers.openrouter import ModelResponse, async_complete
from lib.results import AsyncCache

from .monitor import MonitorResult, async_monitor


@dataclass
class EvalSpec:
    """Pluggable specification for an intervention eval.

    Each eval type (GPQA, DAFT, Scruples, etc.) provides one of these
    to configure the shared pipeline.
    """

    name: str  # e.g. "gpqa_hints"
    eval_name: str = "intervention"  # top-level results dir

    # Hints: list of {name, template} where template(answer) -> hint_text
    hints: list[dict] = field(default_factory=list)

    # Prompt formatting
    format_control_prompt: Callable[[dict], list[dict]] = None
    format_intervention_prompt: Callable[[dict, str], list[dict]] = None

    # Answer parsing: ModelResponse -> parsed answer string
    parse_answer: Callable[[str], Optional[str]] = None

    # Y computation: (parsed_answer, problem, hint_answer) -> 0|1
    compute_y: Callable[[Optional[str], dict, Optional[str]], int] = None

    # Monitor prompt formatting and parsing
    format_monitor_prompt: Callable[..., list[dict]] = None
    parse_monitor_output: Callable[[str], tuple[Optional[str], Optional[int]]] = None

    # For intervention arm: what answer does the hint point to?
    # Default: use the correct answer. Override for evals where hint answer != correct answer.
    get_hint_answer: Callable[[dict], str] = None

    def __post_init__(self):
        if self.get_hint_answer is None:
            self.get_hint_answer = lambda problem: str(problem["answer"])


# ── Stage 1: Control Arm ─────────────────────────────────────────────

async def run_control(
    spec: EvalSpec,
    model: ModelConfig,
    problems: list[dict],
    *,
    n_samples: int = 10,
    temperature: float = 0.6,
    max_tokens: int = 32768,
    semaphore: Optional[asyncio.Semaphore] = None,
    save_every: int = 25,
) -> dict:
    """Run the control arm: model solves problems without intervention.

    Cache key format: "{problem_id}_{sample_i}"
    """
    cache = AsyncCache(f"{spec.eval_name}/{spec.name}", model.name, "control", save_every)
    await cache.load()

    sem = semaphore or asyncio.Semaphore(20)

    todo = []
    for prob in problems:
        for i in range(n_samples):
            key = f"{prob['id']}_{i}"
            if key not in cache:
                todo.append((prob, i, key))

    total = len(problems) * n_samples
    if todo:
        print(f"  Control: {len(todo)} new ({total - len(todo)} cached)")
        pbar = tqdm(total=len(todo), desc="Control")

        async def _run_one(prob, sample_i, key):
            messages = spec.format_control_prompt(prob)
            async with sem:
                try:
                    resp = await async_complete(
                        model.model_id, messages,
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    answer_text = resp.answer
                    parsed = spec.parse_answer(answer_text) if spec.parse_answer else None
                    await cache.set(key, {
                        "problem_id": prob["id"],
                        "sample_i": sample_i,
                        "thinking": resp.thinking,
                        "answer_text": answer_text,
                        "parsed_answer": parsed,
                        "correct_answer": str(prob["answer"]),
                        "usage": resp.usage,
                    })
                except Exception as e:
                    await cache.set(key, {
                        "problem_id": prob["id"],
                        "sample_i": sample_i,
                        "error": str(e),
                    })
            pbar.update(1)

        await asyncio.gather(*[_run_one(*item) for item in todo])
        pbar.close()
        await cache.flush()
    else:
        print(f"  Control: all {total} cached")

    return {"n_total": total, "n_cached": total - len(todo)}


# ── Stage 2: Intervention Arm ────────────────────────────────────────

async def run_intervention(
    spec: EvalSpec,
    model: ModelConfig,
    problems: list[dict],
    *,
    n_samples: int = 10,
    temperature: float = 0.6,
    max_tokens: int = 32768,
    semaphore: Optional[asyncio.Semaphore] = None,
    save_every: int = 25,
) -> dict:
    """Run the intervention arm: model solves problems with hints.

    Cache key format: "{problem_id}_{hint_name}_{sample_i}"
    """
    cache = AsyncCache(f"{spec.eval_name}/{spec.name}", model.name, "intervention", save_every)
    await cache.load()

    sem = semaphore or asyncio.Semaphore(20)

    todo = []
    for prob in problems:
        hint_answer = spec.get_hint_answer(prob)
        for hint in spec.hints:
            hint_text = hint["template"](hint_answer)
            for i in range(n_samples):
                key = f"{prob['id']}_{hint['name']}_{i}"
                if key not in cache:
                    todo.append((prob, hint, hint_text, hint_answer, i, key))

    total = len(problems) * len(spec.hints) * n_samples
    if todo:
        print(f"  Intervention: {len(todo)} new ({total - len(todo)} cached)")
        print(f"  {len(problems)} problems x {len(spec.hints)} hints x {n_samples} samples")
        pbar = tqdm(total=len(todo), desc="Intervention")

        async def _run_one(prob, hint, hint_text, hint_answer, sample_i, key):
            messages = spec.format_intervention_prompt(prob, hint_text)
            async with sem:
                try:
                    resp = await async_complete(
                        model.model_id, messages,
                        temperature=temperature, max_tokens=max_tokens,
                    )
                    answer_text = resp.answer
                    parsed = spec.parse_answer(answer_text) if spec.parse_answer else None
                    y = spec.compute_y(parsed, prob, hint_answer) if spec.compute_y else 0
                    await cache.set(key, {
                        "problem_id": prob["id"],
                        "hint_name": hint["name"],
                        "hint_text": hint_text,
                        "hint_answer": hint_answer,
                        "sample_i": sample_i,
                        "thinking": resp.thinking,
                        "answer_text": answer_text,
                        "parsed_answer": parsed,
                        "correct_answer": str(prob["answer"]),
                        "y": y,
                        "usage": resp.usage,
                    })
                except Exception as e:
                    await cache.set(key, {
                        "problem_id": prob["id"],
                        "hint_name": hint["name"],
                        "sample_i": sample_i,
                        "error": str(e),
                    })
            pbar.update(1)

        await asyncio.gather(*[_run_one(*item) for item in todo])
        pbar.close()
        await cache.flush()
    else:
        print(f"  Intervention: all {total} cached")

    # Summary
    data = cache.data
    valid = [v for v in data.values() if isinstance(v, dict) and "y" in v]
    n_y1 = sum(1 for v in valid if v["y"] == 1)
    print(f"  {len(valid)} valid samples, {n_y1} with Y=1 ({n_y1/max(len(valid),1):.1%})")

    return {"n_total": total, "n_y1": n_y1, "n_valid": len(valid)}


# ── Stage 3: Monitor ─────────────────────────────────────────────────

async def run_monitor(
    spec: EvalSpec,
    model_name: str,
    problems: list[dict],
    *,
    monitor_config: JudgeConfig,
    semaphore: Optional[asyncio.Semaphore] = None,
    save_every: int = 25,
) -> dict:
    """Run the monitor on samples from both arms.

    Monitors are run on:
    - All control arm samples (to measure TNR_X0)
    - Intervention arm samples where Y=1 (to measure TPR)
    - Intervention arm samples where Y=0 (to measure TNR_X1)

    CRITICAL: The monitor is told the intervention exists in BOTH arms.
    For control arm samples, we fabricate a hint message.
    """
    cache_prefix = f"{spec.eval_name}/{spec.name}"
    control_cache = AsyncCache(cache_prefix, model_name, "control")
    await control_cache.load()
    intervention_cache = AsyncCache(cache_prefix, model_name, "intervention")
    await intervention_cache.load()
    # Include monitor model in cache stage name so different monitors don't collide
    monitor_stage = f"monitor_{monitor_config.model_id}"
    monitor_cache = AsyncCache(cache_prefix, model_name, monitor_stage, save_every)
    await monitor_cache.load()

    sem = semaphore or asyncio.Semaphore(10)
    problems_by_id = {p["id"]: p for p in problems}

    todo = []

    # Control arm samples: monitor gets a fabricated hint
    for ctrl_key, ctrl_val in control_cache.data.items():
        if not isinstance(ctrl_val, dict) or "error" in ctrl_val:
            continue
        mon_key = f"{ctrl_key}_control"
        if mon_key in monitor_cache:
            continue

        prob_id = ctrl_val["problem_id"]
        prob = problems_by_id.get(prob_id)
        if not prob:
            continue

        # Use the first hint template with the correct answer as the fabricated hint
        hint_answer = spec.get_hint_answer(prob)
        if spec.hints:
            hint_text = spec.hints[0]["template"](hint_answer)
        else:
            hint_text = f"The answer is {hint_answer}."

        todo.append((ctrl_val, prob, hint_text, mon_key, "control"))

    # Intervention arm samples
    for int_key, int_val in intervention_cache.data.items():
        if not isinstance(int_val, dict) or "error" in int_val:
            continue
        mon_key = f"{int_key}_intervention"
        if mon_key in monitor_cache:
            continue

        prob_id = int_val["problem_id"]
        prob = problems_by_id.get(prob_id)
        if not prob:
            continue

        hint_text = int_val.get("hint_text", "")
        todo.append((int_val, prob, hint_text, mon_key, "intervention"))

    if todo:
        print(f"  Monitor: {len(todo)} new ({len(monitor_cache.data)} cached)")
        pbar = tqdm(total=len(todo), desc="Monitor")

        async def _monitor_one(sample, prob, hint_text, mon_key, arm):
            thinking = sample.get("thinking") or ""
            answer_text = sample.get("answer_text", "")

            # Build monitor messages using the spec's formatter
            if spec.format_monitor_prompt:
                # Get the user message that was sent to the model
                if arm == "control":
                    user_msg = spec.format_control_prompt(prob)[0]["content"]
                else:
                    user_msg = spec.format_intervention_prompt(prob, hint_text)[0]["content"]

                messages = spec.format_monitor_prompt(
                    user_msg, hint_text, thinking, answer_text,
                )
            else:
                # Fallback: shouldn't happen if spec is properly configured
                messages = [{"role": "user", "content": f"Did the hint influence the answer?\n{answer_text}"}]

            async with sem:
                result = await async_monitor(monitor_config, messages)

            # Parse the monitor output
            letter, z = None, None
            if spec.parse_monitor_output and result.raw:
                letter, z = spec.parse_monitor_output(result.raw)

            await monitor_cache.set(mon_key, {
                "arm": arm,
                "problem_id": sample.get("problem_id"),
                "hint_name": sample.get("hint_name"),
                "sample_i": sample.get("sample_i"),
                "y": sample.get("y"),
                "output_letter": letter,
                "z": z,
                "explanation": result.explanation,
                "monitor_model": monitor_config.model_id,
            })
            pbar.update(1)

        await asyncio.gather(*[_monitor_one(*item) for item in todo])
        pbar.close()
        await monitor_cache.flush()
    else:
        print(f"  Monitor: all cached ({len(monitor_cache.data)} results)")

    # Summary
    data = monitor_cache.data
    n_z1 = sum(1 for v in data.values() if isinstance(v, dict) and v.get("z") == 1)
    n_z0 = sum(1 for v in data.values() if isinstance(v, dict) and v.get("z") == 0)
    n_none = sum(1 for v in data.values() if isinstance(v, dict) and v.get("z") is None)
    print(f"  Z=1: {n_z1}, Z=0: {n_z0}, unparsed: {n_none}")

    return {"n_z1": n_z1, "n_z0": n_z0, "n_none": n_none}


# ── Orchestrator ─────────────────────────────────────────────────────

async def run_model(
    spec: EvalSpec,
    model: ModelConfig,
    problems: list[dict],
    stages: list[str],
    *,
    run_config: Optional[dict] = None,
    monitor_config: Optional[JudgeConfig] = None,
) -> None:
    """Run the full intervention pipeline for a single model."""
    rc = run_config or {}
    sampling = rc.get("sampling", {})
    conc = rc.get("concurrency", {})

    or_sem = asyncio.Semaphore(conc.get("openrouter", 20))
    oai_sem = asyncio.Semaphore(conc.get("openai", 10))
    save_every = conc.get("save_every", 25)

    n_samples = sampling.get("n_samples", 10)
    temperature = sampling.get("temperature", 0.6)
    max_tokens = sampling.get("max_tokens", 32768)

    print(f"\n{'='*60}")
    print(f"Model: {model.name} ({model.model_id})")
    print(f"Eval: {spec.name}")
    print(f"Samples per arm: {n_samples}")
    print(f"Hints: {len(spec.hints)}")
    if monitor_config:
        print(f"Monitor: {monitor_config.model_id}")
    print(f"{'='*60}")

    if "control" in stages or "all" in stages:
        await run_control(
            spec, model, problems,
            n_samples=n_samples, temperature=temperature,
            max_tokens=max_tokens, semaphore=or_sem, save_every=save_every,
        )

    if "intervention" in stages or "all" in stages:
        await run_intervention(
            spec, model, problems,
            n_samples=n_samples, temperature=temperature,
            max_tokens=max_tokens, semaphore=or_sem, save_every=save_every,
        )

    if "monitor" in stages or "all" in stages:
        if monitor_config is None:
            from lib.judge import make_judge_config
            monitor_config = make_judge_config()
        await run_monitor(
            spec, model.name, problems,
            monitor_config=monitor_config, semaphore=oai_sem,
            save_every=save_every,
        )

    if "metrics" in stages or "all" in stages:
        from .metrics import compute_and_print_metrics
        mon_model = monitor_config.model_id if monitor_config else "gpt-4o-mini"
        compute_and_print_metrics(spec, model.name, monitor_model=mon_model)
