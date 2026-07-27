"""Shared loading/helpers for the mech analyses (entropy / p_banned / attention).

Reads only the persisted pipeline artifacts in mech/data/. Compliance classes:
  no_first_person, omit_word : binary graders -> compliant = score==1, non = score==0
  jar_jar_binks              : soft LLM judge -> compliant >=0.75, non <=0.25 (mid dropped)
  spanish                    : same margins (only V4 has a compliant side)
  uppercase_thinking, leetspeak : no within-type contrast (floor) -> class None
All CIs are question-clustered bootstrap (10 samples per question are correlated).
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent          # mech/analysis/
MECH = HERE.parent
DATA = MECH / "data"
FIGS = HERE / "figs"
sys.path.insert(0, str(MECH))
sys.path.insert(0, str(MECH.parent))            # controllability/ (prefill_experiment)

TYPES = ["no_first_person", "omit_word", "jar_jar_binks", "spanish",
         "uppercase_thinking", "leetspeak"]
CONDS = ["no_instruction", "instruction_only", "icl_user_plain", "icl_user_think",
         "icl_assistant_visible"]
COND_LABEL = {"no_instruction": "no-instr", "instruction_only": "instr-only",
              "icl_user_plain": "V1", "icl_user_think": "V2",
              "icl_assistant_visible": "V4"}
BINARY_TYPES = {"no_first_person", "omit_word"}
SOFT_TYPES = {"jar_jar_binks", "spanish"}

CLS_COLORS = {"compliant": "#1a9641", "noncompliant": "#d7191c", "all": "#555555"}


def tid_fname(tid: str) -> str:
    return tid.replace("|", "_")


def load_grades() -> pd.DataFrame:
    g = pd.read_csv(DATA / "grades.csv")
    g = g[g.empty_thinking == 0].copy()
    g["cls"] = None
    b = g.ctype.isin(BINARY_TYPES)
    g.loc[b & (g.compliance_score >= 1.0), "cls"] = "compliant"
    g.loc[b & (g.compliance_score <= 0.0), "cls"] = "noncompliant"
    s = g.ctype.isin(SOFT_TYPES)
    g.loc[s & (g.compliance_score >= 0.75), "cls"] = "compliant"
    g.loc[s & (g.compliance_score <= 0.25), "cls"] = "noncompliant"
    return g


def load_prompts() -> dict[str, dict]:
    out = {}
    with (DATA / "prompts.jsonl").open() as f:
        for line in f:
            p = json.loads(line)
            out[p["prompt_id"]] = p
    return out


def load_traces() -> dict[str, dict]:
    out = {}
    with (DATA / "traces.jsonl").open() as f:
        for line in f:
            t = json.loads(line)
            out[t["trace_id"]] = t
    return out


def load_entropy(tid: str):
    p = DATA / "entropy" / f"{tid_fname(tid)}.npz"
    return np.load(p) if p.exists() else None


def load_attn(tid: str):
    p = DATA / "attn" / f"{tid_fname(tid)}.npz"
    return np.load(p) if p.exists() else None


# --------------------------------------------------------------------------- #
# Token pieces / positions
# --------------------------------------------------------------------------- #

_TOK = None
_PIECE_CACHE: dict[int, str] = {}


def _tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    return _TOK


def pieces(ids: list[int]) -> list[str]:
    """Per-token decoded strings (cached per id; multi-byte unicode may degrade to
    replacement chars per piece, fine for ASCII boundary/word detection)."""
    tok = _tokenizer()
    out = []
    for i in ids:
        if i not in _PIECE_CACHE:
            _PIECE_CACHE[i] = tok.decode([i])
        out.append(_PIECE_CACHE[i])
    return out


def content_start(pcs: list[str]) -> int:
    """Index of the first CoT content token (skip '<think>' and leading whitespace)."""
    for i, p in enumerate(pcs[:6]):
        if p.strip() and p.strip() != "<think>":
            return i
    return min(2, len(pcs) - 1)


def first_sentence_end(pcs: list[str], start: int) -> int:
    """Token index (inclusive) where the first sentence of the CoT ends."""
    import re
    text = ""
    for i in range(start, len(pcs)):
        text += pcs[i]
        if re.search(r"[.!?][\"')\]]*(\s|$)", text) and len(text.strip()) > 2:
            return i
    return len(pcs) - 1


def think_end(pcs: list[str]) -> int:
    """Token index of '</think>' (exclusive end of CoT region), or len(pcs)."""
    for i, p in enumerate(pcs):
        if "</think>" in p:
            return i
    return len(pcs)


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

def cluster_boot_ci(values: np.ndarray, clusters: np.ndarray, n_boot: int = 1000,
                    seed: int = 0) -> tuple[float, float, float]:
    """Mean + 95% CI, bootstrapping over question clusters."""
    values = np.asarray(values, float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    groups = {c: values[clusters == c] for c in uniq}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        means.append(np.concatenate([groups[c] for c in pick]).mean())
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def binned_profile(arr: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Mean of arr in n_bins equal chunks of normalized position (nan-padded)."""
    out = np.full(n_bins, np.nan)
    T = len(arr)
    if T == 0:
        return out
    edges = np.linspace(0, T, n_bins + 1).astype(int)
    for b in range(n_bins):
        seg = arr[edges[b]:max(edges[b + 1], edges[b] + 1)]
        if len(seg):
            out[b] = float(np.mean(seg))
    return out
