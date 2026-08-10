"""Eval the trained model against the diverse-family registry.

For each family, samples N test questions and runs two conditions per question:
  - a TRAINED marker (one the model saw during training)
  - a HELD-OUT marker (never seen — tests marker-level generalization within a family)

Uses our own transforms.grade — the same graders used for training self-verify —
so training and eval are consistent. Queries local vLLM over HTTP.

Output: results/{exp_name}_diverse.csv with columns:
  family, marker_pool (train|eval), n, compliance, accuracy, mean_rate

Usage:
    python run_eval_diverse.py --model organism_2_v6 --exp-name organism_2_v6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
import yaml
from tqdm.asyncio import tqdm_asyncio

from transforms import FAMILIES, transform_cot, grade, grade_score
from sft.style_instructions import build_style_instruction
from sft.style_grade import grade_style_async, COMPLIANCE_THRESHOLD
from sft.language_instructions import build_language_instruction, load_language_defs
from sft.language_grade import grade_language_async
from sft.ignore_question_grade import grade_ignore_question_async
import suppression as supp

HERE = Path(__file__).parent
ANSWER_REGEX = re.compile(r"\bANSWER\s*[:\-]\s*\(?([A-Za-z])\)?", re.IGNORECASE)

# HARD compliance = fraction of completions "essentially fully" compliant.
# Threshold is 0.95 (not exact 1.0): the programmatic transform grader zeros a
# completion for a single stray character, so exact-1.0 penalizes trivial
# near-misses. 0.95 rescues those while staying strict. Language is near-binary
# so the choice barely affects it. See analysis/rsweep/ for the sweep that set it.
HARD_COMPLIANCE_THRESHOLD = 0.95


@dataclass
class EvalItem:
    question: str
    options_json: str
    correct_letter: str
    family: str
    marker_pool: str     # "train" or "eval" — markers seen during training vs held out (or "-" for styles)
    phrasing_pool: str   # "train" or "eval" — paraphrases seen during training vs held out (or "-" for styles)
    marker: Any
    instruction: str
    prompt: str
    modality: str = "family"   # family / style / language / suppression
    style: str = ""            # set on modality="style" items
    language: str = ""         # set on modality="language" items
    suppression: str = ""      # set on modality="suppression" items (behavior name)
    omit_words: Any = None     # forbidden words for omit_word / omit_multiple_words


def extract_answer(text: str):
    if not text:
        return None
    m = ANSWER_REGEX.search(text)
    return m.group(1).upper() if m else None


def parse_options(row: pd.Series) -> list[str]:
    for col in ("answer_options", "options"):
        raw = row.get(col)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            continue
        try:
            return list(json.loads(raw))
        except (TypeError, json.JSONDecodeError):
            try:
                import ast
                return list(ast.literal_eval(raw))
            except Exception:
                continue
    return []


def answer_to_letter(raw: str, options: list[str]):
    text = str(raw).strip()
    m = re.match(r"^\(?([A-Za-z])\)?$", text)
    if m and options and 0 <= ord(m.group(1).upper()) - ord("A") < len(options):
        return m.group(1).upper()
    for i, opt in enumerate(options):
        if str(opt).strip() == text:
            return chr(ord("A") + i)
    return None


def assemble_prompt(question: str, options: list[str], instruction: str) -> str:
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
        head
        + "\n\nYour response should be of the following format: 'ANSWER: X' "
        "(without quotes) where X is the letter of your chosen answer."
    )


def build_items(
    test: pd.DataFrame,
    families: list[str],
    n_per_cell: int,
    seed: int,
    lean: bool = False,
) -> list[EvalItem]:
    rng = random.Random(seed)
    items: list[EvalItem] = []
    questions = test.to_dict(orient="records")

    # Lean (trajectory) mode: one representative cell per family (train marker +
    # train phrasing). Keeps every family on the curve but drops the held-out
    # marker/phrasing breakdown, which is measured at the final full eval.
    mpools = ("train",) if lean else ("train", "eval")
    ppools = ("train",) if lean else ("train", "eval")
    for fam_name in families:
        fam = FAMILIES[fam_name]
        for marker_pool in mpools:
            markers = fam.markers_train if marker_pool == "train" else fam.markers_eval
            if markers == [None]:
                if marker_pool == "eval":
                    continue   # parameterless family: don't double-count
            for phrasing_pool in ppools:
                picks = rng.sample(questions, k=min(n_per_cell, len(questions)))
                for q in picks:
                    marker = rng.choice(markers)
                    options = parse_options(pd.Series(q))
                    letter = answer_to_letter(q["answer"], options)
                    if letter is None:
                        continue
                    _, instruction = transform_cot(
                        "placeholder", fam_name, marker=marker, rng=rng,
                        phrasing_pool=phrasing_pool,
                    )
                    prompt = assemble_prompt(q["question"], options, instruction)
                    items.append(EvalItem(
                        question=q["question"],
                        options_json=json.dumps(options),
                        correct_letter=letter,
                        family=fam_name,
                        marker_pool=marker_pool,
                        phrasing_pool=phrasing_pool,
                        marker=marker,
                        instruction=instruction,
                        prompt=prompt,
                        modality="family",
                    ))
    return items


def build_style_items(
    test: pd.DataFrame,
    styles: list[str],
    style_defs: dict[str, dict],
    n_per_cell: int,
    seed: int,
) -> list[EvalItem]:
    """Style eval cells per (style, phrasing_pool). detail_level is sampled
    uniformly within each item (not its own axis to keep cell count bounded)."""
    rng = random.Random(seed + 1)   # offset to decouple from family RNG
    items: list[EvalItem] = []
    questions = test.to_dict(orient="records")

    for style in styles:
        sdef = style_defs.get(style)
        if sdef is None:
            raise SystemExit(f"[eval] unknown style {style!r}; "
                             f"known: {list(style_defs)}")
        for phrasing_pool in ("train", "eval"):
            picks = rng.sample(questions, k=min(n_per_cell, len(questions)))
            for q in picks:
                options = parse_options(pd.Series(q))
                letter = answer_to_letter(q["answer"], options)
                if letter is None:
                    continue
                instruction = build_style_instruction(
                    style, sdef, rng=rng, phrasing_pool=phrasing_pool,
                )
                prompt = assemble_prompt(q["question"], options, instruction)
                items.append(EvalItem(
                    question=q["question"],
                    options_json=json.dumps(options),
                    correct_letter=letter,
                    family=f"style:{style}",
                    marker_pool="-",
                    phrasing_pool=phrasing_pool,
                    marker=None,
                    instruction=instruction,
                    prompt=prompt,
                    modality="style",
                    style=style,
                ))
    return items


def load_style_defs(path: Path = HERE / "data" / "styles.yaml") -> dict[str, dict]:
    if not path.exists():
        return {}
    cfg = yaml.safe_load(path.read_text())
    return cfg.get("styles", {}) or {}


def load_style_descriptors(path: Path = HERE / "data" / "style_descriptors.json"
                           ) -> dict[str, list[str]]:
    """Eval-only paraphrase pool for named-entity styles, with the character
    name *removed*. Used to test whether the trained model learned the voice
    or just a literal-name response. Keys are style names; values are lists
    of phrasings."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {k: list(v) for k, v in data.items() if not k.startswith("_")}


def build_descriptor_items(
    test: pd.DataFrame,
    styles: list[str],
    descriptors: dict[str, list[str]],
    n_per_cell: int,
    seed: int,
) -> list[EvalItem]:
    """One eval cell per style under descriptor phrasings (phrasing_pool='descriptor').
    Only styles present in `descriptors` are eval'd; the rest are silently
    skipped (most styles aren't named entities)."""
    rng = random.Random(seed + 3)   # offset to decouple from family/style/lang RNG
    items: list[EvalItem] = []
    questions = test.to_dict(orient="records")

    for style in styles:
        phrasings = descriptors.get(style)
        if not phrasings:
            continue
        picks = rng.sample(questions, k=min(n_per_cell, len(questions)))
        for q in picks:
            options = parse_options(pd.Series(q))
            letter = answer_to_letter(q["answer"], options)
            if letter is None:
                continue
            instruction = rng.choice(phrasings)
            prompt = assemble_prompt(q["question"], options, instruction)
            items.append(EvalItem(
                question=q["question"],
                options_json=json.dumps(options),
                correct_letter=letter,
                family=f"style:{style}",
                marker_pool="-",
                phrasing_pool="descriptor",
                marker=None,
                instruction=instruction,
                prompt=prompt,
                modality="style",
                style=style,
            ))
    return items


def build_language_items(
    test: pd.DataFrame,
    languages: list[str],
    lang_defs: dict[str, dict],
    n_per_cell: int,
    seed: int,
    lean: bool = False,
) -> list[EvalItem]:
    """Language eval cells per (language, phrasing_pool)."""
    rng = random.Random(seed + 2)   # offset to decouple from family + style RNG
    items: list[EvalItem] = []
    questions = test.to_dict(orient="records")

    for language in languages:
        ldef = lang_defs.get(language)
        if ldef is None:
            raise SystemExit(f"[eval] unknown language {language!r}; "
                             f"known: {list(lang_defs)}")
        # Lean: one cell per language — prefer train phrasings, fall back to eval
        # (held-out languages have only eval phrasings). Keeps every language on
        # the trajectory.
        if lean:
            has_train = bool((ldef.get("instruction_phrasings") or {}).get("train"))
            pools = ("train",) if has_train else ("eval",)
        else:
            pools = ("train", "eval")
        for phrasing_pool in pools:
            # Held-out languages have empty train phrasings; skip that cell.
            phrasings = (ldef.get("instruction_phrasings") or {}).get(phrasing_pool) or []
            if not phrasings:
                continue
            picks = rng.sample(questions, k=min(n_per_cell, len(questions)))
            for q in picks:
                options = parse_options(pd.Series(q))
                letter = answer_to_letter(q["answer"], options)
                if letter is None:
                    continue
                instruction = build_language_instruction(
                    language, ldef, rng=rng, phrasing_pool=phrasing_pool,
                )
                prompt = assemble_prompt(q["question"], options, instruction)
                items.append(EvalItem(
                    question=q["question"],
                    options_json=json.dumps(options),
                    correct_letter=letter,
                    family=f"language:{language}",
                    marker_pool="-",
                    phrasing_pool=phrasing_pool,
                    marker=None,
                    instruction=instruction,
                    prompt=prompt,
                    modality="language",
                    language=language,
                ))
    return items


def build_suppression_items(
    test: pd.DataFrame,
    behaviors: list[str],
    n_per_cell: int,
    seed: int,
    lean: bool = False,
) -> list[EvalItem]:
    """Suppression eval cells.

    omit_word / omit_multiple_words: cells per (behavior, marker_pool,
    phrasing_pool), where marker_pool selects the forbidden-word pool (TRAIN vs
    EVAL). Deterministically graded. ignore_question: eval-only, one cell per
    phrasing_pool, LLM-graded against the question.
    """
    rng = random.Random(seed + 4)   # offset from family/style/lang/descriptor RNG
    items: list[EvalItem] = []
    questions = test.to_dict(orient="records")

    def _cell(q, behavior, marker_pool, phrasing_pool, omit_words):
        options = parse_options(pd.Series(q))
        letter = answer_to_letter(q["answer"], options)
        if letter is None:
            return None
        instruction = supp.build_suppression_instruction(
            behavior, omit_words, rng=rng, phrasing_pool=phrasing_pool)
        return EvalItem(
            question=q["question"],
            options_json=json.dumps(options),
            correct_letter=letter,
            family=f"suppression:{behavior}",
            marker_pool=marker_pool,
            phrasing_pool=phrasing_pool,
            marker=",".join(omit_words) if omit_words else None,
            instruction=instruction,
            prompt=assemble_prompt(q["question"], options, instruction),
            modality="suppression",
            suppression=behavior,
            omit_words=list(omit_words),
        )

    mpools = ("train",) if lean else ("train", "eval")
    ppools = ("train",) if lean else ("train", "eval")
    for behavior in behaviors:
        if behavior == "ignore_question":
            for phrasing_pool in ppools:
                for q in rng.sample(questions, k=min(n_per_cell, len(questions))):
                    it = _cell(q, behavior, "-", phrasing_pool, [])
                    if it is not None:
                        items.append(it)
            continue
        if behavior not in supp.TRAINABLE_SUPPRESSION:
            raise SystemExit(f"[eval] unknown suppression behavior {behavior!r}")
        for marker_pool in mpools:
            pool = supp.FORBIDDEN_TRAIN if marker_pool == "train" else supp.FORBIDDEN_EVAL
            for phrasing_pool in ppools:
                for q in rng.sample(questions, k=min(n_per_cell, len(questions))):
                    if behavior == "omit_word":
                        omit = [rng.choice(pool)]
                    else:
                        omit = rng.sample(pool, min(supp._N_MULTI, len(pool)))
                    it = _cell(q, behavior, marker_pool, phrasing_pool, omit)
                    if it is not None:
                        items.append(it)
    return items


async def query(session: aiohttp.ClientSession, url: str, model: str, prompt: str,
                max_tokens: int, temperature: float):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        # Long-reasoning subjects (OLMo-3 Think) can generate for many minutes;
        # 600s timed out ~11% of long control traces (recorded as ERROR/0). Env-
        # overridable so fast API runs can keep it short.
        _to = float(os.getenv("EVAL_HTTP_TIMEOUT", "1800"))
        async with session.post(url, json=payload,
                                timeout=aiohttp.ClientTimeout(total=_to)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError):
        return None
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")
    if reasoning:
        return reasoning.strip(), content.strip()
    # No reasoning_content (e.g. served without a vLLM reasoning parser, as for
    # OLMo whose tokenizer lacks <think> special tokens). Recover the think
    # block from content.
    m = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), (content[: m.start()] + content[m.end():]).strip()
    # Chat templates that force-open the think tag emit only a closing </think>
    # in content (reasoning before it, answer after).
    if "</think>" in content:
        pre, _, post = content.partition("</think>")
        return pre.replace("<think>", "").strip(), post.strip()
    return None


async def run(items: list[EvalItem], model: str, url: str, concurrency: int,
              max_tokens: int, temperature: float,
              grader_concurrency: int = 16) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    grader_sem = asyncio.Semaphore(grader_concurrency)

    async def grade_one(it: EvalItem, think: str) -> tuple[int, float]:
        """Return (compliance_binary, compliance_score) for one item."""
        if it.modality == "family":
            return (
                int(grade(think, it.family, marker=it.marker)),
                float(grade_score(think, it.family, marker=it.marker)),
            )
        # Suppression: omit_word/omit_multiple_words are deterministic;
        # ignore_question is LLM-graded against the question.
        if it.modality == "suppression":
            if it.suppression == "ignore_question":
                async with grader_sem:
                    score = await grade_ignore_question_async(it.question, think)
                if score < 0:
                    return 0, 0.0
                return int(score >= COMPLIANCE_THRESHOLD), score
            sc = supp.score_suppression(it.suppression, think, it.omit_words or [])
            return int(supp.grade_suppression(it.suppression, think, it.omit_words or [])), sc
        # modality in {"style", "language"}: LLM grader via OpenRouter.
        async with grader_sem:
            if it.modality == "style":
                score = await grade_style_async(it.style, think)
            elif it.modality == "language":
                score = await grade_language_async(it.language, think)
            else:
                raise ValueError(f"unknown modality {it.modality!r}")
        if score < 0:   # hard failure
            return 0, 0.0
        return int(score >= COMPLIANCE_THRESHOLD), score

    async def run_one(it: EvalItem):
        async with sem:
            out = await query_with_retry(url, model, it.prompt, max_tokens, temperature)
        row = {
            "family": it.family,
            "modality": it.modality,
            "style": it.style,
            "language": it.language,
            "marker_pool": it.marker_pool,
            "phrasing_pool": it.phrasing_pool,
            "marker": str(it.marker) if it.marker is not None else "",
            "question": it.question[:200],
            "correct_letter": it.correct_letter,
            "instruction": it.instruction,
        }
        if out is None:
            row.update({
                "compliance": 0,
                "compliance_score": 0.0,
                "compliance_hard": 0,
                "correct": 0,
                "reasoning": "",
                "response": "ERROR",
            })
            return row
        think, tail = out
        letter = extract_answer(tail)
        compliance, score = await grade_one(it, think)
        row.update({
            "compliance": compliance,
            "compliance_score": score,
            # hard rate: full (100%) compliance only. soft = compliance_score
            # (continuous); the legacy `compliance` is a family-specific
            # threshold (0.5-0.95), neither soft nor hard.
            "compliance_hard": int(score >= HARD_COMPLIANCE_THRESHOLD),
            "correct": int(letter == it.correct_letter),
            "reasoning": think,
            "response": tail,
        })
        return row

    async with aiohttp.ClientSession() as session:
        global _SESSION
        _SESSION = session
        tasks = [run_one(it) for it in items]
        return await tqdm_asyncio.gather(*tasks, desc="diverse-eval")


_SESSION = None  # set by run()


async def query_with_retry(url: str, model: str, prompt: str, max_tokens: int,
                           temperature: float):
    return await query(_SESSION, url, model, prompt, max_tokens, temperature)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   help="sft/configs/<version>.yaml or rl/configs/<version>.yaml. Overrides --model/--exp-name/--families/"
                        "--n-per-cell/--max-tokens from the config.")
    p.add_argument("--model")
    p.add_argument("--exp-name")
    p.add_argument("--families", nargs="+", default=list(FAMILIES))
    p.add_argument("--test-csv", type=Path,
                   default=HERE / "data" / "split_test.csv")
    p.add_argument("--n-per-cell", type=int, default=40,
                   help="Test questions per (family, pool) cell.")
    p.add_argument("--vllm-base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--max-concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    # Lite-eval mode for the multi-checkpoint trajectory: smaller n_per_cell,
    # smaller max_tokens. The grader cost dominates wallclock at the per-cell
    # sample counts we use, so this is the main savings lever.
    p.add_argument("--lite", action="store_true",
                   help="Use lite n_per_cell (default 8) and max_tokens (8192).")
    p.add_argument("--include-descriptors", action="store_true",
                   help="Also eval the named-entity styles under data/style_descriptors.json "
                        "paraphrases (no character name). Phrasing pool 'descriptor' in output.")
    p.add_argument("--eval-all", action="store_true",
                   help="Eval the full behavior universe (all families, all languages, "
                        "all suppression) regardless of train/held-out lists. Also set via "
                        "eval.eval_all in the config.")
    # Curated-subset selection without a config: pass any of these to eval a
    # hand-picked universe (e.g. the OLMo RL-sweep 'diverse midpoint'). Ignored
    # when --config or --eval-all is given (those define the universe).
    p.add_argument("--styles", nargs="*", default=None,
                   help="style names to eval (e.g. pirate cowboy yoda jar_jar_binks)")
    p.add_argument("--languages", nargs="*", default=None,
                   help="language names to eval (e.g. spanish chinese russian arabic)")
    p.add_argument("--suppression", nargs="*", default=None,
                   help="suppression behaviors to eval (e.g. omit_word omit_multiple_words ignore_question)")
    args = p.parse_args()

    cli_styles = args.styles
    cli_languages = args.languages
    cli_suppression = args.suppression
    args.styles = cli_styles or []
    args.languages = cli_languages or []
    args.suppression = cli_suppression or []
    if args.config is not None:
        import yaml
        recipe = yaml.safe_load(Path(args.config).read_text()).get("recipe", "")
        if recipe == "rl_grpo":
            from rl import config as cfg_mod
        else:
            from sft import config as cfg_mod
        cfg = cfg_mod.load(args.config)
        # Let --model / --exp-name on the CLI override the config defaults
        # (so we can eval a baseline model against a config's families/styles).
        if not args.model:
            args.model = cfg.exp_name
        if not args.exp_name:
            args.exp_name = f"{cfg.exp_name}_diverse"
        # Eval the union of trained + held-out families and styles so held-outs
        # appear in the summary for zero-shot transfer comparison. RL configs
        # currently have no style axis, so getattr the style lists.
        args.families = sorted(set(cfg.data.families) | set(cfg.data.held_out_families))
        args.styles = sorted(
            set(getattr(cfg.data, "styles", []) or [])
            | set(getattr(cfg.data, "held_out_styles", []) or [])
        )
        args.languages = sorted(
            set(getattr(cfg.data, "languages", []) or [])
            | set(getattr(cfg.data, "held_out_languages", []) or [])
        )
        # Suppression union + always include eval-only ignore_question.
        args.suppression = sorted(
            set(getattr(cfg.data, "suppression", []) or [])
            | set(getattr(cfg.data, "held_out_suppression", []) or [])
            | {"ignore_question"}
        )
        if getattr(cfg.eval, "eval_all", False):
            args.eval_all = True
        args.n_per_cell = cfg.eval.samples_per_cell
        args.max_tokens = cfg.eval.max_tokens
        args.temperature = cfg.eval.temperature
        if getattr(cfg.eval, "test_csv", None):
            tcsv = Path(cfg.eval.test_csv)
            args.test_csv = tcsv if tcsv.is_absolute() else (HERE / tcsv)
        print(f"[run_eval_diverse] loaded config {cfg.source_path.name}")

    if args.lite:
        lite_n = 8
        if args.config is not None:
            lite_n = getattr(cfg.eval, "lite_samples_per_cell", 8) or 8
        args.n_per_cell = lite_n
        args.max_tokens = 8192
        print(f"[run_eval_diverse] LITE mode: n_per_cell={args.n_per_cell} "
              f"max_tokens={args.max_tokens}")

    if args.eval_all:
        # Full universe — "always eval on everything" for the ablation matrix.
        # Styles stay opt-in (paused for v32); the 4 active categories are
        # families (incl. encodings), languages, and suppression.
        args.families = list(FAMILIES)
        args.languages = sorted(load_language_defs().keys())
        args.suppression = sorted(supp.KNOWN_SUPPRESSION)
        print(f"[run_eval_diverse] EVAL-ALL universe: {len(args.families)} families, "
              f"{len(args.languages)} languages, {len(args.suppression)} suppression")

    if not args.model or not args.exp_name:
        p.error("--model and --exp-name required unless --config is given")

    test = pd.read_csv(args.test_csv)
    print(f"[run_eval_diverse] eval pool: {args.test_csv} "
          f"({len(test)} questions"
          + (f", by source {test['source'].value_counts().to_dict()}" if 'source' in test.columns else "")
          + (", LEAN cells" if args.lite else "") + ")")
    items = build_items(test, args.families, args.n_per_cell, args.seed, lean=args.lite)
    n_fam = len(items)
    style_items: list[EvalItem] = []
    if args.styles:
        style_defs = load_style_defs()
        style_items = build_style_items(
            test, args.styles, style_defs, args.n_per_cell, args.seed,
        )
        items = items + style_items
    language_items: list[EvalItem] = []
    if args.languages:
        lang_defs = load_language_defs()
        language_items = build_language_items(
            test, args.languages, lang_defs, args.n_per_cell, args.seed, lean=args.lite,
        )
        items = items + language_items
    descriptor_items: list[EvalItem] = []
    if args.include_descriptors and args.styles:
        descriptors = load_style_descriptors()
        descriptor_items = build_descriptor_items(
            test, args.styles, descriptors, args.n_per_cell, args.seed,
        )
        items = items + descriptor_items
    suppression_items: list[EvalItem] = []
    if args.suppression:
        suppression_items = build_suppression_items(
            test, args.suppression, args.n_per_cell, args.seed, lean=args.lite,
        )
        items = items + suppression_items
    print(f"Prepared {len(items)} eval items "
          f"({n_fam} family + {len(style_items)} style + {len(language_items)} language "
          f"+ {len(descriptor_items)} descriptor + {len(suppression_items)} suppression; "
          f"{len(args.families)} families, {len(args.styles)} styles, "
          f"{len(args.languages)} languages, {len(args.suppression)} suppression)")

    url = f"{args.vllm_base_url}/chat/completions"
    print(f"Running at temperature={args.temperature}, max_tokens={args.max_tokens}")
    rows = asyncio.run(run(items, args.model, url, args.max_concurrency,
                           args.max_tokens, args.temperature))

    df = pd.DataFrame(rows)
    out_dir = HERE / "results" / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_csv = out_dir / "per_sample.csv"
    df.to_csv(per_sample_csv, index=False)

    # Aggregate: per (family, marker_pool, phrasing_pool). compliance is the
    # thresholded bool; compliance_score is the continuous "pure metric".
    # With the added phrasing_pool dimension, parametric families now have up
    # to 4 cells (train/eval markers × train/eval phrasings). Parameterless
    # families have 2 (phrasing_pool only, marker_pool is always "train").
    # Style rows have marker_pool=phrasing_pool="-" → one row per style.
    if "modality" not in df.columns:
        df["modality"] = "family"
    if "compliance_hard" not in df.columns:   # back-compat for old per_sample
        df["compliance_hard"] = (df["compliance_score"] >= HARD_COMPLIANCE_THRESHOLD).astype(int)
    g = df.groupby(["modality", "family", "marker_pool", "phrasing_pool"], sort=False).agg(
        n=("compliance", "count"),
        compliance=("compliance", "mean"),          # legacy: family-specific threshold
        compliance_score=("compliance_score", "mean"),  # SOFT rate (continuous)
        compliance_hard=("compliance_hard", "mean"),    # HARD rate (% == 100% compliant)
        accuracy=("correct", "mean"),
    ).reset_index()
    summary_csv = HERE / "results" / f"{args.exp_name}.csv"
    g.to_csv(summary_csv, index=False)

    print(f"\nPer-sample rows: {per_sample_csv}")
    print(f"Summary:         {summary_csv}\n")
    print(g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
