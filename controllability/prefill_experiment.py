"""Prefill / CoT-seeding experiment: can the BASE model *continue* a controlled CoT?

We test, purely at inference time (no training), whether seeding the first K
sentences of a controlled chain-of-thought into the assistant `<think>` block
makes the base model carry that control behavior forward.

  - Elicitation gap : compliance rises with K -> base CAN sustain the control
                      once seeded; SFT mostly elicits an existing capability.
  - Capability gap  : compliance stays at the base floor even at K=5 -> SFT
                      installs something genuinely new.

For each (control type, question) we sweep K in {0..5} and two prompt
conditions (prefill_only, prefill_plus_instruction), grading ONLY the model's
continuation with the same graders used for SFT self-verify.

Seed sources (the controlled CoT we slice the first K sentences from):
  - deterministic transforms / encodings : transform the question's base raw
    CoT (data/training_raw.csv) with transforms.transform_cot.
  - languages / suppression / styles      : the already-cleaned LLM rewrites in
    data/{language,suppression,stylistic}_rewrites.clean.jsonl, joined back to
    the question by question_hash.

Modes:
  python prefill_experiment.py --mode gate    # validation probe (run FIRST)
  python prefill_experiment.py --mode sweep    # full experiment

Usage examples:
  python prefill_experiment.py --mode gate --n-questions 20
  python prefill_experiment.py --mode sweep --config configs/prefill.yaml
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from tqdm.asyncio import tqdm_asyncio

# --- path setup: repo root (for `lib`) + this dir (for `transforms`, `sft`) ---
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
for p in (str(REPO_ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from transforms import (  # noqa: E402
    FAMILIES,
    grade,
    grade_score,
    transform_cot,
)
import suppression as supp  # noqa: E402
from sft.style_grade import grade_style_async, COMPLIANCE_THRESHOLD  # noqa: E402
from sft.language_grade import grade_language_async  # noqa: E402
from sft.style_instructions import build_style_instruction  # noqa: E402
from sft.language_instructions import (  # noqa: E402
    build_language_instruction,
    load_language_defs,
)
from lib.providers.openrouter import get_async_client, _parse_response  # noqa: E402
from lib.results import AsyncCache  # noqa: E402

TRANSCRIPTS = HERE / "transcripts"

DATA = HERE / "data"
ANSWER_REGEX = re.compile(r"\bANSWER\s*[:\-]\s*\(?([A-Za-z])\)?", re.IGNORECASE)

# Sentence-end matcher for slicing the K-sentence seed. transforms._SENT_SPLIT_RE
# only splits on ASCII [.!?] + whitespace, which fails for scripts whose
# terminators carry no trailing space (CJK 。！？, Hindi danda ।, Arabic ؟) —
# there it returns the WHOLE CoT as one "sentence", so every K injects everything.
# CJK/Hindi/Arabic enders are boundaries on their own (no following space needed);
# ASCII enders require following whitespace/end so decimals (3.14) and abbrevs
# don't over-split. We match the boundary and slice the ORIGINAL text up to it, so
# native spacing (CJK no-space, \n\n) is preserved verbatim in the injected seed.
_SENT_END = re.compile(r"[。！？।؟…]+|[.!?]+(?=\s|$)")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class PrefillConfig:
    model_id: str = "qwen/qwen3-14b"          # base model, no adapter
    # Pin ONE reasoning-clean provider: OpenRouter load-balances per call, and
    # providers differ in whether they populate the `reasoning` field (DeepInfra
    # does; some inline everything into `content` with no <think> tags, which
    # would silently grade as empty). Pinning makes the response shape — and the
    # prefill-into-<think> behavior — consistent across the sweep.
    provider: Optional[str] = "DeepInfra"
    backend: str = "openrouter"                # "openrouter" | "vllm_harmony" (gpt-oss)
    vllm_url: str = "http://localhost:8000/v1/completions"
    vllm_model: Optional[str] = None           # served id; defaults to model_id
    reasoning_effort: str = "high"             # gpt-oss harmony effort (low|medium|high)
    save_raw: bool = False                     # dump raw response to transcripts/
    raw_max_sample: int = 1                    # only save raw for sample < this (1 = one per cell)
    force_think_k0: bool = False               # at K=0 prefill an empty `<think>\n` to FORCE
                                               # reasoning (capability floor) instead of letting
                                               # the model choose whether to think (natural floor)
    eval_name: str = "prefill_continuation"
    run_name: str = "base_qwen3_14b"
    k_values: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    conditions: list[str] = field(
        default_factory=lambda: ["prefill_only", "prefill_plus_instruction"]
    )
    n_questions: int = 30                      # questions per control type
    n_samples: int = 5                          # completions per cell
    temperature: float = 0.6
    max_tokens: int = 8192
    concurrency: int = 30
    seed: int = 0
    # Representative subset across every modality (styles incl. paused chars).
    transforms: list[str] = field(
        default_factory=lambda: ["uppercase_thinking", "prefix_sentences", "no_first_person"]
    )
    encodings: list[str] = field(default_factory=lambda: ["leetspeak"])
    languages: list[str] = field(default_factory=lambda: ["spanish", "chinese"])
    suppression: list[str] = field(default_factory=lambda: ["omit_word"])
    styles: list[str] = field(default_factory=lambda: ["jar_jar_binks", "yoda", "pirate"])

    @classmethod
    def load(cls, path: Optional[str]) -> "PrefillConfig":
        cfg = cls()
        if path:
            raw = yaml.safe_load(Path(path).read_text()) or {}
            for k, v in raw.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg


# --------------------------------------------------------------------------- #
# Seed sources
# --------------------------------------------------------------------------- #


def _qhash(q: str) -> str:
    return hashlib.sha256(str(q).encode("utf-8")).hexdigest()[:16]


def _parse_options(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    try:
        return list(json.loads(raw))
    except (TypeError, json.JSONDecodeError):
        try:
            return list(ast.literal_eval(raw))
        except Exception:
            return []


@dataclass
class QuestionRow:
    question: str
    options: list[str]
    correct_letter: str
    think: str            # base raw CoT (for transform seeds)


def load_raw_questions() -> dict[str, QuestionRow]:
    """question_hash -> QuestionRow, from data/training_raw.csv."""
    df = pd.read_csv(DATA / "training_raw.csv")
    out: dict[str, QuestionRow] = {}
    for _, r in df.iterrows():
        q = str(r["question"])
        out[_qhash(q)] = QuestionRow(
            question=q,
            options=_parse_options(r.get("options_json")),
            correct_letter=str(r["correct_letter"]).strip().upper(),
            think=str(r.get("think") or ""),
        )
    return out


def load_rewrites(fname: str, key_field: str) -> list[dict]:
    """Load one cleaned rewrites jsonl (style/language/suppression)."""
    path = DATA / fname
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# Cells: one (control type, question) seed unit
# --------------------------------------------------------------------------- #


@dataclass
class Cell:
    ctype: str                 # control type label (family / style / language / behavior)
    modality: str              # family | style | language | suppression
    question: str
    options: list[str]
    correct_letter: str
    controlled_cot: str        # full controlled CoT to slice the seed from
    instruction: str           # control instruction (for prefill_plus_instruction)
    marker: Any = None         # family marker (or None)
    omit_words: Optional[list[str]] = None  # suppression forbidden words


def _build_family_cells(
    ctype: str, raws: dict[str, QuestionRow], cfg: PrefillConfig, rng: random.Random
) -> list[Cell]:
    fam = FAMILIES[ctype]
    qkeys = list(raws.keys())
    rng.shuffle(qkeys)
    cells: list[Cell] = []
    for qh in qkeys:
        if len(cells) >= cfg.n_questions:
            break
        row = raws[qh]
        if not row.think.strip() or not row.options:
            continue
        marker = fam.sample_train(rng)
        try:
            transformed, instruction = transform_cot(
                row.think, ctype, marker=marker, rng=rng, phrasing_pool="train"
            )
        except Exception:
            continue
        cells.append(Cell(
            ctype=ctype, modality="family",
            question=row.question, options=row.options,
            correct_letter=row.correct_letter,
            controlled_cot=transformed, instruction=instruction, marker=marker,
        ))
    return cells


def _build_rewrite_cells(
    ctype: str, modality: str, rewrites: list[dict], key_field: str,
    raws: dict[str, QuestionRow], cfg: PrefillConfig, rng: random.Random,
    instr_fn,
) -> list[Cell]:
    """Cells from LLM rewrites (styles / languages / suppression)."""
    matching = [r for r in rewrites if r.get(key_field) == ctype]
    rng.shuffle(matching)
    cells: list[Cell] = []
    for r in matching:
        if len(cells) >= cfg.n_questions:
            break
        qrow = raws.get(r.get("question_hash", ""))
        if qrow is None or not qrow.options:
            continue
        rewrite = str(r.get("rewrite") or "").strip()
        if not rewrite:
            continue
        omit_words = None
        if modality == "suppression":
            raw_ow = r.get("omit_words")
            try:
                omit_words = ast.literal_eval(raw_ow) if isinstance(raw_ow, str) else list(raw_ow)
            except Exception:
                continue
        try:
            instruction = instr_fn(ctype, rng, omit_words)
        except Exception:
            continue
        cells.append(Cell(
            ctype=ctype, modality=modality,
            question=qrow.question, options=qrow.options,
            correct_letter=qrow.correct_letter,
            controlled_cot=rewrite, instruction=instruction,
            omit_words=omit_words,
        ))
    return cells


def build_all_cells(cfg: PrefillConfig) -> list[Cell]:
    rng = random.Random(cfg.seed)
    raws = load_raw_questions()
    cells: list[Cell] = []

    # transforms + encodings (deterministic)
    for ctype in [*cfg.transforms, *cfg.encodings]:
        cells += _build_family_cells(ctype, raws, cfg, rng)

    # styles
    if cfg.styles:
        style_defs = yaml.safe_load((DATA / "styles.yaml").read_text()).get("styles", {})
        srews = load_rewrites("stylistic_rewrites.clean.jsonl", "style")

        def style_instr(ct, r, _ow):
            return build_style_instruction(ct, style_defs.get(ct, {}), rng=r, phrasing_pool="train")

        for ctype in cfg.styles:
            cells += _build_rewrite_cells(
                ctype, "style", srews, "style", raws, cfg, rng, style_instr
            )

    # languages
    if cfg.languages:
        lang_defs = load_language_defs()
        lrews = load_rewrites("language_rewrites.clean.jsonl", "language")

        def lang_instr(ct, r, _ow):
            return build_language_instruction(ct, lang_defs.get(ct, {}), rng=r, phrasing_pool="train")

        for ctype in cfg.languages:
            cells += _build_rewrite_cells(
                ctype, "language", lrews, "language", raws, cfg, rng, lang_instr
            )

    # suppression
    if cfg.suppression:
        prews = load_rewrites("suppression_rewrites.clean.jsonl", "suppression")

        def supp_instr(ct, r, omit_words):
            return supp.build_suppression_instruction(ct, omit_words, rng=r, phrasing_pool="train")

        for ctype in cfg.suppression:
            cells += _build_rewrite_cells(
                ctype, "suppression", prews, "suppression", raws, cfg, rng, supp_instr
            )

    return cells


# --------------------------------------------------------------------------- #
# Prompt / prefill construction
# --------------------------------------------------------------------------- #


def assemble_prompt(question: str, options: list[str], instruction: str) -> str:
    # No control instruction (prefill_only) -> DROP the block entirely. Rendering
    # "<cot_instructions> none </cot_instructions>" is read literally by some models
    # (qwen3-32b: 95% -> 0% empty-thinking when dropped) as "produce no CoT", which
    # silently suppressed reasoning and contaminated the K=0 prefill_only floor.
    if not instruction or instruction == "none":
        head = f"Question: {question}"
    else:
        head = f"<cot_instructions> {instruction} </cot_instructions>\nQuestion: {question}"
    if options:
        body = "\n\n" + "\n".join(
            f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options)
        )
        tail = (
            "\n\nYour response should be of the following format: "
            "'ANSWER: X' (without quotes) where X is your chosen letter"
        )
        return head + body + tail
    return (
        head + "\n\nYour response should be of the following format: 'ANSWER: X' "
        "(without quotes) where X is the letter of your chosen answer."
    )


def first_k_sentences(text: str, k: int) -> str:
    """First k sentences, sliced from the original text (spacing preserved)."""
    text = text.strip()
    if k <= 0:
        return ""
    ends = [m.end() for m in _SENT_END.finditer(text)]
    if len(ends) >= k:
        return text[: ends[k - 1]].strip()
    return text  # fewer than k sentences -> whole CoT


def build_messages(cell: Cell, k: int, condition: str,
                   force_think_k0: bool = False) -> tuple[list[dict], str]:
    """Return (messages, seed_text). seed_text='' when k==0.

    force_think_k0: at K=0, prefill an empty `<think>\n` assistant turn so the
    model is MADE to reason (capability floor) rather than choosing whether to
    (natural floor). seed stays '' — nothing to strip, we grade all continuation."""
    instruction = cell.instruction if condition == "prefill_plus_instruction" else "none"
    prompt = assemble_prompt(cell.question, cell.options, instruction)
    messages = [{"role": "user", "content": prompt}]
    seed = ""
    if k >= 1:
        seed = first_k_sentences(cell.controlled_cot, k)
        messages.append({"role": "assistant", "content": "<think>\n" + seed})
    elif force_think_k0:
        messages.append({"role": "assistant", "content": "<think>\n"})
    return messages, seed


def extract_answer(text: str) -> Optional[str]:
    if not text:
        return None
    m = ANSWER_REGEX.search(text)
    return m.group(1).upper() if m else None


# --------------------------------------------------------------------------- #
# Grading the continuation
# --------------------------------------------------------------------------- #


async def grade_continuation(cell: Cell, continuation: str) -> tuple[float, int]:
    """Return (compliance_score in [0,1], compliance binary).

    Empty continuation -> (0, 0): if the model produced no CoT the control
    behavior is simply not exhibited. Without this guard, vacuous graders
    (e.g. uppercase finds no lowercase letters) would score empty text as 1.0 —
    which silently inflates the floor for models that often skip thinking
    (qwen3-32b answers easy ARC/CSQA with empty reasoning)."""
    text = continuation or ""
    if not text.strip():
        return 0.0, 0
    if cell.modality == "family":
        score = grade_score(text, cell.ctype, marker=cell.marker)
    elif cell.modality == "style":
        score = await grade_style_async(cell.ctype, text)
    elif cell.modality == "language":
        score = await grade_language_async(cell.ctype, text)
    elif cell.modality == "suppression":
        score = supp.score_suppression(cell.ctype, text, cell.omit_words or [])
    else:
        score = 0.0
    score = max(0.0, float(score))  # clamp grader hard-fail (-1.0) to 0
    return score, int(score >= COMPLIANCE_THRESHOLD)


def strip_seed_prefix(continuation: str, seed: str) -> str:
    """Defensive: if the provider echoed the prefill into the reasoning field,
    drop a leading copy of the seed so we grade only the model's continuation."""
    if not seed:
        return continuation
    c = continuation.lstrip()
    s = seed.strip()
    if c.startswith(s):
        return c[len(s):].lstrip()
    # also handle the opening tag being echoed
    c2 = c
    if c2.startswith("<think>"):
        c2 = c2[len("<think>"):].lstrip()
        if c2.startswith(s):
            return c2[len(s):].lstrip()
    return continuation


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


async def complete_pinned(cfg: PrefillConfig, messages: list[dict]):
    """Completion against a PINNED provider; returns (ModelResponse, raw_dict).

    Reuses openrouter._parse_response so reasoning-field extraction matches the
    rest of the codebase, but keeps the raw response so we can save transcripts
    and detect providers that fail to populate `reasoning`."""
    client = get_async_client()
    kwargs = dict(model=cfg.model_id, messages=messages,
                  temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    if cfg.provider:
        kwargs["extra_body"] = {"provider": {"order": [cfg.provider],
                                             "allow_fallbacks": False}}
    last_err = None
    for attempt in range(4):  # retry transient rate-limits / 5xx on the pinned provider
        try:
            resp = await client.chat.completions.create(**kwargs)
            return _parse_response(resp), resp.model_dump()
        except Exception as e:
            last_err = e
            await asyncio.sleep(min(2 ** attempt, 20))
    raise last_err


async def complete_for_cell(cell: Cell, k: int, condition: str, cfg: PrefillConfig,
                            sem: asyncio.Semaphore):
    """Dispatch to the configured backend. Returns (resp, raw, seed, prompt)."""
    if cfg.backend == "vllm_harmony":
        import harmony_backend as hb
        seed = first_k_sentences(cell.controlled_cot, k) if k >= 1 else ""
        instruction = cell.instruction if condition == "prefill_plus_instruction" else "none"
        prompt = assemble_prompt(cell.question, cell.options, instruction)
        async with sem:
            resp, raw = await asyncio.to_thread(
                hb.complete_harmony, cfg.vllm_url, cfg.vllm_model or cfg.model_id,
                cell.question, cell.options, instruction, seed,
                temperature=cfg.temperature, max_tokens=cfg.max_tokens,
                reasoning_effort=cfg.reasoning_effort,
                force_think=(k == 0 and cfg.force_think_k0))
        return resp, raw, seed, prompt
    messages, seed = build_messages(cell, k, condition, cfg.force_think_k0)
    async with sem:
        resp, raw = await complete_pinned(cfg, messages)
    return resp, raw, seed, messages[0]["content"]


async def run_one(cell: Cell, k: int, condition: str, sample: int, cfg: PrefillConfig,
                  cache: AsyncCache, sem: asyncio.Semaphore) -> None:
    key = f"{_qhash(cell.question)}_{cell.ctype}_{cell.marker}_{condition}_K{k}_s{sample}"
    if key in cache:
        return
    try:
        resp, raw, seed, prompt = await complete_for_cell(cell, k, condition, cfg, sem)
    except Exception as e:
        await cache.set(key, {"error": f"{type(e).__name__}: {e}", "ctype": cell.ctype})
        return
    if cfg.save_raw and sample < cfg.raw_max_sample:
        tdir = TRANSCRIPTS / cfg.run_name          # namespace by run so models don't clobber
        tdir.mkdir(parents=True, exist_ok=True)
        (tdir / f"{key}.json").write_text(
            json.dumps({"seed": seed, "raw_response": raw},
                       ensure_ascii=False, indent=2))
    empty_thinking = not (resp.thinking or "").strip()
    continuation = strip_seed_prefix(resp.thinking or "", seed)
    score, compliance = await grade_continuation(cell, continuation)
    pred = extract_answer(resp.answer or "")
    await cache.set(key, {
        "ctype": cell.ctype,
        "modality": cell.modality,
        "marker": cell.marker,
        "condition": condition,
        "k": k,
        "sample": sample,
        "question": cell.question,
        "prompt": prompt,                            # full user prompt sent
        "seed": seed,                                # assistant <think> prefill body
        "instruction": cell.instruction,            # the control instruction (always recorded)
        "compliance": compliance,
        "compliance_score": score,
        "correct": int(pred == cell.correct_letter) if pred else 0,
        "predicted": pred,
        "correct_letter": cell.correct_letter,
        "reasoning_continuation": continuation,      # full, untruncated
        "response": resp.answer or "",               # full, untruncated
        "finish_reason": resp.finish_reason,
        "provider": raw.get("provider"),
        "empty_thinking": empty_thinking,            # True => provider didn't return reasoning
        "raw_content": (raw["choices"][0]["message"].get("content") or ""),
    })


async def run_sweep(cfg: PrefillConfig) -> Path:
    cells = build_all_cells(cfg)
    by_type: dict[str, int] = {}
    for c in cells:
        by_type[c.ctype] = by_type.get(c.ctype, 0) + 1
    print(f"[sweep] {len(cells)} cells across {len(by_type)} types: {by_type}")

    cache = AsyncCache(cfg.eval_name, cfg.run_name, "sweep", save_every=25)
    await cache.load()
    # purge prior error records so this run retries them
    errored = [k for k, v in cache.data.items() if isinstance(v, dict) and "error" in v]
    for k in errored:
        del cache.data[k]
    if errored:
        print(f"[sweep] retrying {len(errored)} previously-errored cells")
    sem = asyncio.Semaphore(cfg.concurrency)

    tasks = [
        run_one(cell, k, cond, s, cfg, cache, sem)
        for cell in cells
        for k in cfg.k_values
        for cond in cfg.conditions
        for s in range(cfg.n_samples)
    ]
    print(f"[sweep] {len(tasks)} completions "
          f"(K={cfg.k_values} x conds={cfg.conditions} x n={cfg.n_samples})")
    await tqdm_asyncio.gather(*tasks)
    await cache.flush()

    return write_csv(cfg, cache)


def write_csv(cfg: PrefillConfig, cache: AsyncCache) -> Path:
    rows = [v for v in cache.data.values() if "error" not in v]
    errs = [v for v in cache.data.values() if "error" in v]
    if errs:
        print(f"[sweep] {len(errs)} errored completions (skipped in CSV)")
    df = pd.DataFrame(rows)
    out = HERE / "results" / f"{cfg.run_name}_prefill_per_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[sweep] wrote {len(df)} rows -> {out}")
    if "empty_thinking" in df.columns:
        n_empty = int(df["empty_thinking"].sum())
        if n_empty:
            print(f"[sweep] WARNING: {n_empty}/{len(df)} responses had EMPTY thinking "
                  f"(provider didn't return reasoning) — these grade as 0 and bias "
                  f"compliance down. Check `provider` pinning.")
        if "provider" in df.columns:
            print("[sweep] providers served:", df["provider"].value_counts().to_dict())
    if not df.empty:
        summ = (df.groupby(["modality", "ctype", "condition", "k"])
                ["compliance_score"].mean().reset_index())
        print(summ.to_string(index=False))
    return out


# --------------------------------------------------------------------------- #
# Validation gate
# --------------------------------------------------------------------------- #


async def run_gate(cfg: PrefillConfig) -> None:
    """Probe whether prefill-into-<think> actually takes on the API.

    K=3 uppercase seed, NO instruction. If the continuation is uppercase well
    above the no-seed base floor, prefill works -> proceed. If not, fall back to
    local vLLM (continue_final_message=True)."""
    rng = random.Random(cfg.seed)
    raws = load_raw_questions()
    fam_cfg = PrefillConfig(n_questions=cfg.n_questions)
    cells = _build_family_cells("uppercase_thinking", raws, fam_cfg, rng)
    print(f"[gate] {len(cells)} uppercase cells; probing K=0 (floor) vs K=3 (seeded)")

    sem = asyncio.Semaphore(cfg.concurrency)
    results = {0: [], 3: []}
    starts = []          # "did the continuation START in-style" (head of K=3 cont)
    samples_shown = []

    def head_uppercase_frac(text: str, n: int = 40) -> float:
        """Fraction of alpha chars in the first n chars that are uppercase — a
        direct 'did the seed take' signal, immune to long-run reversion."""
        head = "".join(c for c in text.strip()[:n] if c.isalpha())
        if not head:
            return 0.0
        return sum(c.isupper() for c in head) / len(head)

    async def probe(cell: Cell, k: int):
        resp, _raw, seed, _prompt = await complete_for_cell(cell, k, "prefill_only", cfg, sem)
        cont = strip_seed_prefix(resp.thinking or "", seed)
        score, _ = await grade_continuation(cell, cont)
        results[k].append(score)
        if k == 3:
            starts.append(head_uppercase_frac(cont))
            if len(samples_shown) < 3:
                samples_shown.append((seed, cont[:300]))

    tasks = [probe(c, k) for c in cells for k in (0, 3)]
    await tqdm_asyncio.gather(*tasks)

    floor = sum(results[0]) / max(1, len(results[0]))
    seeded = sum(results[3]) / max(1, len(results[3]))
    head = sum(starts) / max(1, len(starts))
    took_frac = sum(s > 0.7 for s in starts) / max(1, len(starts))

    # Two SEPARATE questions:
    #   (1) does the prefill MECHANISM take?  -> continuation continues in-style
    #       at the head, well above the no-seed floor.
    #   (2) does the base model SUSTAIN it?    -> full-continuation score.
    mechanism_ok = head > 0.6 and seeded > floor + 0.1
    print("\n========== VALIDATION GATE ==========")
    print(f"  K=0 (no seed)   full-continuation score : {floor:.3f}")
    print(f"  K=3 (seeded)    full-continuation score : {seeded:.3f}  (lift {seeded - floor:+.3f})")
    print(f"  K=3 head-uppercase frac (did it continue): {head:.3f}")
    print(f"  K=3 frac of continuations that took (>0.7): {took_frac:.2f}")
    print(f"  MECHANISM: {'WORKS — prefill takes on OpenRouter' if mechanism_ok else 'WEAK/IGNORED'}")
    if mechanism_ok:
        print("  -> Sweep is runnable on OpenRouter. The full-continuation score "
              f"({seeded:.2f}) is the SCIENCE: how much the base SUSTAINS the seed.")
        print("  -> NOTE: continuations sometimes restart a fresh paragraph and "
              "revert (provider chat-template seam). For the cleanest signal,")
        print("     local vLLM with continue_final_message=True removes that seam.")
    else:
        print("  -> Prefill not taking; switch to local vLLM (continue_final_message=True).")
    print("=====================================\n")
    for i, (seed, cont) in enumerate(samples_shown):
        print(f"--- sample {i} ---\n  SEED: {seed[:120]!r}\n  CONT: {cont!r}\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gate", "sweep"], default="gate")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--run-name", default=None, help="cache/CSV namespace (set per model to avoid clobbering)")
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=None, help="override concurrency")
    ap.add_argument("--provider", default=None, help="pin one OpenRouter provider (e.g. DeepInfra)")
    ap.add_argument("--no-provider", action="store_true", help="disable provider pinning (let OpenRouter route)")
    ap.add_argument("--save-raw", action="store_true", help="save raw response to transcripts/ (one per cell by default)")
    ap.add_argument("--save-raw-all", action="store_true", help="save raw for ALL samples, not just sample 0")
    ap.add_argument("--backend", default=None, choices=["openrouter", "vllm_harmony"], help="completion backend")
    ap.add_argument("--vllm-url", default=None, help="vLLM /v1/completions URL (vllm_harmony backend)")
    ap.add_argument("--vllm-model", default=None, help="served model id for vLLM")
    ap.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "high"], help="gpt-oss harmony effort")
    ap.add_argument("--force-think-k0", action="store_true",
                    help="at K=0 prefill an empty <think> to FORCE reasoning (capability floor)")
    ap.add_argument("--k-values", default=None,
                    help="comma-separated K list override, e.g. '0' to run only the K=0 floor")
    args = ap.parse_args()

    cfg = PrefillConfig.load(args.config)
    if args.model_id:
        cfg.model_id = args.model_id
    if args.run_name:
        cfg.run_name = args.run_name
    if args.n_questions:
        cfg.n_questions = args.n_questions
    if args.n_samples:
        cfg.n_samples = args.n_samples
    if args.concurrency:
        cfg.concurrency = args.concurrency
    if args.provider:
        cfg.provider = args.provider
    if args.no_provider:
        cfg.provider = None
    if args.save_raw or args.save_raw_all:
        cfg.save_raw = True
    if args.save_raw_all:
        cfg.raw_max_sample = 10 ** 9
    if args.backend:
        cfg.backend = args.backend
    if args.vllm_url:
        cfg.vllm_url = args.vllm_url
    if args.vllm_model:
        cfg.vllm_model = args.vllm_model
    if args.reasoning_effort:
        cfg.reasoning_effort = args.reasoning_effort
    if args.force_think_k0:
        cfg.force_think_k0 = True
    if args.k_values:
        cfg.k_values = [int(x) for x in args.k_values.split(",") if x.strip() != ""]

    if args.mode == "gate":
        cfg.n_questions = args.n_questions or 20
        asyncio.run(run_gate(cfg))
    else:
        asyncio.run(run_sweep(cfg))


if __name__ == "__main__":
    main()
