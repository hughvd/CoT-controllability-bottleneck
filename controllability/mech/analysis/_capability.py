"""Uncontrolled capability = clean-arm gpqa accuracy from the monitorability subject
CoTs (the model reasons WITHOUT any control instruction). This is the "did our
intervention nuke the model?" check — distinct from accuracy-under-control.

gpqa intervention arm x=1 = clean (no misleading hint); x=0 = adversarial. Answered-
only (excludes truncated). Bootstrap 95% CI over questions (instance_id).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
MON = REPO / "results" / "monitorability"


def _norm(s):
    return str(s).strip().upper()[:1]


def uncontrolled_gpqa(mon_label, clean_arm=1.0, nboot=1000, seed=0):
    """(mean, lo, hi) uncontrolled gpqa accuracy, or (nan,nan,nan) if no subject data."""
    path = None
    for sub in ("mon-gpt5mini", "."):
        cand = MON / mon_label / sub / "samples.csv"
        if cand.exists():
            path = cand
            break
    if path is None:
        return np.nan, np.nan, np.nan
    d = pd.read_csv(path, usecols=["dataset", "archetype", "x", "instance_id",
                                   "final_answer", "reference_answer", "finish_reason"])
    g = d[(d.archetype == "intervention")
          & (d.dataset.astype(str).str.contains("gpqa"))
          & (d.x == clean_arm)
          & (d.finish_reason != "length")].copy()
    if not len(g):
        return np.nan, np.nan, np.nan
    g["correct"] = (g.final_answer.map(_norm) == g.reference_answer.map(_norm)).astype(float)
    per_q = g.groupby("instance_id").correct.mean().to_numpy()
    # +/- 1 SEM over questions (a sanity band; gpqa is only ~70 Q so a 95% CI is very wide)
    mean = float(per_q.mean())
    sem = float(per_q.std(ddof=1) / np.sqrt(len(per_q))) if len(per_q) > 1 else np.nan
    return mean, mean - sem, mean + sem
