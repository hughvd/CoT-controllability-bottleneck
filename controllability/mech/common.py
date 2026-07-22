"""Shared setup for the ICL mechanism experiment: avoidance vs continuation controls.

Question: WHY is omit_word (avoid a word) controlled differently from continuation-style
controls (ALL CAPS / spanish / leetspeak), where each next token can just keep the trend
going? We measure, on locally generated qwen3-32b traces: per-token entropy, P(banned
token), and attention mass from each generated token to labeled prompt spans
(instruction, example reasoning, example answer, questions, own CoT).

Pipeline (run on the pod, in order — see mech/README.md):
  1. generate_traces.py   vLLM generation -> data/prompts.jsonl + data/traces.jsonl
  2. grade_traces.py      compliance per trace -> data/grades.csv
  3. entropy_pass.py      teacher-forced logits -> data/entropy/{trace_id}.npz
  4. attention_pass.py    eager-attention span masses -> data/attn/{trace_id}.npz

IMPORTANT: cells/donors are built ONCE, inside generate_traces.py, and everything needed
downstream (instruction, grade spec, spans, token ids) is persisted in prompts.jsonl.
build_all_cells is not perfectly reproducible across processes (observed donor drift),
so later passes must never rebuild cells — they only read the persisted artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # controllability/mech/
CTRL = HERE.parent                              # controllability/
sys.path.insert(0, str(CTRL))
sys.path.insert(0, str(CTRL.parent))            # repo root (lib/)

DATA = HERE / "data"
PROMPTS_PATH = DATA / "prompts.jsonl"
TRACES_PATH = DATA / "traces.jsonl"
GRADES_PATH = DATA / "grades.csv"
ENTROPY_DIR = DATA / "entropy"
ATTN_DIR = DATA / "attn"

MODEL_HF = "Qwen/Qwen3-32B"

# The 6 control types (user-selected): one avoidance control (omit_word) vs five
# continuation-flavored ones spanning transform / encoding / language / style.
MECH_TYPES = dict(
    transforms=["uppercase_thinking", "no_first_person"],
    encodings=["leetspeak"],
    languages=["spanish"],
    suppression=["omit_word"],
    styles=["jar_jar_binks"],
)

# no_instruction = bare question (base floor + danger-point reference for omit_word).
CONDITIONS = ["no_instruction", "instruction_only",
              "icl_user_plain", "icl_user_think", "icl_assistant_visible"]
ICL_CONDITIONS = ["icl_user_plain", "icl_user_think", "icl_assistant_visible"]
L5 = ("5", "abs", 5)     # single example length — qwen's length curves are flat

N_QUESTIONS = 30
N_SAMPLES = 10
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_NEW_TOKENS = 4096

# Span names, in reduction order. own_cot/other are computed at pass time, not stored
# as char spans: own_cot = keys in the generated region (incl. current token), other =
# all remaining prompt tokens (chat template, preamble, options, answer-format tail).
SPAN_NAMES = ["instr_real", "instr_example", "example_reasoning", "example_answer",
              "donor_question", "target_question"]


def prompt_id(ctype: str, condition: str, qhash: str) -> str:
    return f"{ctype}|{condition}|{qhash}"


def trace_id(pid: str, sample: int) -> str:
    return f"{pid}|s{sample}"


# --------------------------------------------------------------------------- #
# Message construction (generation time only)
# --------------------------------------------------------------------------- #

def messages_for(target, donor, condition: str) -> list[dict]:
    """Chat messages for one (target, donor, condition). ICL conditions use the
    existing icl_experiment builders at example length L5."""
    import icl_experiment as icl
    from prefill_experiment import assemble_prompt
    if condition == "no_instruction":
        return [{"role": "user", "content": assemble_prompt(target.question, target.options, None)}]
    return icl.build_icl_messages(target, donor, condition
                                  if condition != "instruction_only" else "instruction_only", L5)


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)


# --------------------------------------------------------------------------- #
# Span finding: char spans in the rendered prompt -> token spans via offsets
# --------------------------------------------------------------------------- #

def _find(text: str, needle: str, start: int = 0) -> tuple[int, int] | None:
    if not needle:
        return None
    i = text.find(needle, start)
    return (i, i + len(needle)) if i >= 0 else None


def char_spans(prompt_text: str, target, donor, condition: str,
               sliced_reasoning: str) -> dict[str, tuple[int, int] | None]:
    """Char ranges of the labeled spans. The instruction appears twice in ICL
    conditions (example head + real question): first occurrence = instr_example,
    last = instr_real."""
    spans: dict[str, tuple[int, int] | None] = {k: None for k in SPAN_NAMES}
    instr = donor.instruction if condition != "no_instruction" else None
    if instr:
        first = prompt_text.find(instr)
        last = prompt_text.rfind(instr)
        if first >= 0:
            spans["instr_real"] = (last, last + len(instr))
            if last != first:
                spans["instr_example"] = (first, first + len(instr))
    if condition in ICL_CONDITIONS:
        spans["donor_question"] = _find(prompt_text, donor.question)
        spans["example_reasoning"] = _find(prompt_text, sliced_reasoning)
        er = spans["example_reasoning"]
        spans["example_answer"] = _find(
            prompt_text, f"ANSWER: {donor.correct_letter}", er[1] if er else 0)
    # target question: search from after the example region so a shared prefix with
    # the donor question can't shadow it
    tq_from = max((s[1] for s in spans.values() if s), default=0)
    tq = _find(prompt_text, target.question, tq_from) or _find(prompt_text, target.question)
    spans["target_question"] = tq
    return spans


def to_token_spans(cspans: dict, offsets: list[tuple[int, int]]) -> dict[str, tuple[int, int] | None]:
    """Char span -> [t0, t1) over prompt tokens (tokens overlapping the char range)."""
    out: dict[str, tuple[int, int] | None] = {}
    for name, cs in cspans.items():
        if cs is None:
            out[name] = None
            continue
        s, e = cs
        toks = [i for i, (a, b) in enumerate(offsets) if a < e and b > s and b > a]
        out[name] = (toks[0], toks[-1] + 1) if toks else None
    return out


# --------------------------------------------------------------------------- #
# Banned-token variants (omit_word probe)
# --------------------------------------------------------------------------- #

def banned_first_token_ids(tokenizer, words: list[str]) -> list[int]:
    """First-token ids of every surface variant of each banned word: {w, Cap, UPPER}
    x {no-space, leading-space}. P(banned) at a position = sum of these ids' probs —
    an onset-probability proxy (multi-token variants count their first piece)."""
    ids = set()
    for w in words or []:
        for form in {w, w.capitalize(), w.upper()}:
            for pre in ("", " "):
                t = tokenizer(pre + form, add_special_tokens=False)["input_ids"]
                if t:
                    ids.add(int(t[0]))
    return sorted(ids)


# --------------------------------------------------------------------------- #
# jsonl helpers
# --------------------------------------------------------------------------- #

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_think(output_text: str) -> tuple[str, str]:
    """Split a qwen3 completion into (think, post-think answer text)."""
    t = output_text
    if "<think>" in t:
        t = t.split("<think>", 1)[1]
    if "</think>" in t:
        think, rest = t.split("</think>", 1)
        return think.strip(), rest.strip()
    # no closing tag (hit token limit inside think)
    return t.strip(), ""
