"""Plot the v30 multi-checkpoint trajectory.

Reads:
  - results/organism_2_v30_step{N}_diverse.csv  (one per save_steps_list entry)
  - results/organism_2_v30_diverse.csv          (final-step full eval)
  - artifacts/organism_2_v30/training_meta.json (gives final_step)
  - results/organism_2_base_diverse_diverse.csv (Phase A baseline, optional)

Writes:
  - analysis/v30_trajectory.png   : 4-line trajectory (trained vs held-out, styles vs languages)
  - analysis/v30_descriptors.csv  : canonical vs descriptor compliance per named-entity style
  - stdout                        : printed summary table

Run:
    python -m analysis.plot_v30_trajectory
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"

import argparse
import os

# Default exp; overridable via --exp / env. So this script can plot v30 (14B),
# v30_8b, v30_32b, etc. without code edits.
DEFAULT_EXP_NAME = "organism_2_v30"

# Calibrated-judge baselines (Pod A, 2026-05-20). These REPLACE the older
# organism_2_base_diverse anchor — that one used the pre-calibration style
# judge which inflated baseline by ~30×. Routed by exp_name suffix:
#   * organism_2_*_8b   → Qwen-Qwen3-8B_baseline
#   * everything else   → Qwen-Qwen3-14B_baseline
BASELINE_FOR_14B = "Qwen-Qwen3-14B_baseline"
BASELINE_FOR_8B  = "Qwen-Qwen3-8B_baseline"

# v23/v30 splits — duplicated here to avoid importing the config module
# (the plotter runs locally without the training deps installed).
TRAINED_STYLES   = {"yoda", "pirate", "shakespearean", "valley_girl"}
HELDOUT_STYLES   = {"jar_jar_binks", "cowboy", "noir_detective", "victorian"}
DESCRIPTOR_STYLES = ("yoda", "jar_jar_binks", "shakespearean")
# Languages: v30 has no language axis (per the config), but keep the columns
# in case a future v30-variant adds them.
TRAINED_LANGS    = {"spanish", "french", "german", "chinese"}
HELDOUT_LANGS    = {"italian", "portuguese", "japanese", "russian"}


def _style_name(family: str) -> Optional[str]:
    return family[len("style:"):] if family.startswith("style:") else None


def _language_name(family: str) -> Optional[str]:
    return family[len("language:"):] if family.startswith("language:") else None


TRAINED_FAMILIES = {"uppercase_thinking", "alternating_case", "title_case",
                    "prefix_sentences", "start_with", "wrap_in_brackets",
                    "exclaim_all_sentences"}


def _summarize(df: pd.DataFrame) -> dict[str, float]:
    """Per-cell averages of BOTH binary compliance and continuous
    compliance_score. Returned keys are suffixed `_bin` (binary, the headline
    `≥0.5 threshold compliance` number we usually report) and `_cont` (the
    raw 0-1 LLM-judge / programmatic-grader score). Binary is the right
    number for "does the model actually do it"; continuous helps see partial
    progress when binary is at floor."""
    out: dict[str, float] = {}

    # Styles: average over the 4 trained / 4 held-out styles, ignoring the
    # descriptor pool — that's plotted separately.
    style_rows = df[(df["modality"] == "style") & (df["phrasing_pool"] != "descriptor")]
    if not style_rows.empty:
        style_rows = style_rows.copy()
        style_rows["_style"] = style_rows["family"].apply(_style_name)
        trained = style_rows[style_rows["_style"].isin(TRAINED_STYLES)]
        heldout = style_rows[style_rows["_style"].isin(HELDOUT_STYLES)]
        if not trained.empty:
            out["styles_trained_bin"]  = trained["compliance"].mean()
            out["styles_trained_cont"] = trained["compliance_score"].mean()
        if not heldout.empty:
            out["styles_heldout_bin"]  = heldout["compliance"].mean()
            out["styles_heldout_cont"] = heldout["compliance_score"].mean()

    # Languages (may be empty for v30).
    lang_rows = df[df["modality"] == "language"]
    if not lang_rows.empty:
        lang_rows = lang_rows.copy()
        lang_rows["_language"] = lang_rows["family"].apply(_language_name)
        trained = lang_rows[lang_rows["_language"].isin(TRAINED_LANGS)]
        heldout = lang_rows[lang_rows["_language"].isin(HELDOUT_LANGS)]
        if not trained.empty:
            out["langs_trained_bin"]  = trained["compliance"].mean()
            out["langs_trained_cont"] = trained["compliance_score"].mean()
        if not heldout.empty:
            out["langs_heldout_bin"]  = heldout["compliance"].mean()
            out["langs_heldout_cont"] = heldout["compliance_score"].mean()

    # Families.
    fam_rows = df[df["modality"] == "family"]
    if not fam_rows.empty:
        is_trained = fam_rows["family"].isin(TRAINED_FAMILIES)
        out["families_trained_bin"]  = fam_rows[is_trained]["compliance"].mean()
        out["families_trained_cont"] = fam_rows[is_trained]["compliance_score"].mean()
        out["families_heldout_bin"]  = fam_rows[~is_trained]["compliance"].mean()
        out["families_heldout_cont"] = fam_rows[~is_trained]["compliance_score"].mean()

    return out


def _descriptor_table(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical vs descriptor compliance per named-entity style, in both
    binary (`compliance` column, threshold ≥0.5) and continuous
    (`compliance_score`) terms. Binary is the headline; continuous is
    informational (it'll be nonzero even when the model isn't really
    complying, because the grader gives partial credit for incidental
    style markers)."""
    rows = []
    style_rows = df[df["modality"] == "style"].copy()
    style_rows["_style"] = style_rows["family"].apply(_style_name)
    for style in DESCRIPTOR_STYLES:
        s = style_rows[style_rows["_style"] == style]
        canonical = s[s["phrasing_pool"].isin(["train", "eval"])]
        descriptor = s[s["phrasing_pool"] == "descriptor"]
        rows.append({
            "style": style,
            "canon_bin":  canonical["compliance"].mean()  if not canonical.empty else float("nan"),
            "desc_bin":   descriptor["compliance"].mean() if not descriptor.empty else float("nan"),
            "canon_cont": canonical["compliance_score"].mean()  if not canonical.empty else float("nan"),
            "desc_cont":  descriptor["compliance_score"].mean() if not descriptor.empty else float("nan"),
            "canon_n": int(canonical["n"].sum()) if not canonical.empty else 0,
            "desc_n":  int(descriptor["n"].sum()) if not descriptor.empty else 0,
        })
    return pd.DataFrame(rows)


def _discover_step_csvs(exp_name: str) -> dict[int, tuple[Path, str]]:
    """Map step → (csv path, eval_kind). eval_kind ∈ {"n20", "lite"}.
    Prefers `_step{N}_n20_diverse.csv` over `_step{N}_diverse.csv` when both
    exist. The n20 files come from the Pod A peak re-eval pass; the lite
    files are from the original Pod B intermediate-eval phase 5."""
    out: dict[int, tuple[Path, str]] = {}
    # First pass: lite (default if nothing better).
    lite_re = re.compile(rf"^{re.escape(exp_name)}_step(\d+)_diverse\.csv$")
    for p in RESULTS.glob(f"{exp_name}_step*_diverse.csv"):
        m = lite_re.match(p.name)
        if m:
            out[int(m.group(1))] = (p, "lite")
    # Second pass: n20 — overwrites lite at the same step.
    n20_re = re.compile(rf"^{re.escape(exp_name)}_step(\d+)_n20_diverse\.csv$")
    for p in RESULTS.glob(f"{exp_name}_step*_n20_diverse.csv"):
        m = n20_re.match(p.name)
        if m:
            out[int(m.group(1))] = (p, "n20")
    return out


def _load_baseline_point(exp_name: str) -> Optional[dict[str, float]]:
    """Return the calibrated baseline summary for whichever model `exp_name`
    runs on. Routes by suffix: `_8b` → 8B baseline, otherwise → 14B."""
    base_exp = BASELINE_FOR_8B if exp_name.endswith("_8b") else BASELINE_FOR_14B
    p = RESULTS / f"{base_exp}_diverse.csv"
    if not p.exists():
        # Fall back to older patterns just in case.
        for cand in (RESULTS / f"{base_exp}.csv",
                     RESULTS / "organism_2_base_diverse_diverse.csv",
                     RESULTS / "organism_2_base_diverse.csv"):
            if cand.exists():
                p = cand
                break
        else:
            return None
    df = pd.read_csv(p)
    return _summarize(df)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", default=os.environ.get("V30_EXP", DEFAULT_EXP_NAME),
                   help=f"experiment name prefix (default: {DEFAULT_EXP_NAME})")
    args = p.parse_args()
    exp_name = args.exp

    step_csvs = _discover_step_csvs(exp_name)
    if not step_csvs:
        print(f"[plot_v30_trajectory] no step CSVs found for exp={exp_name} under {RESULTS}/")
        return

    # Final eval (full): canonical pattern.
    final_csv = RESULTS / f"{exp_name}_diverse.csv"
    final_step: Optional[int] = None
    meta_path = ARTIFACTS / exp_name / "training_meta.json"
    if meta_path.exists():
        final_step = int(json.loads(meta_path.read_text())["final_step"])

    # Each point: (step, summary, eval_kind). eval_kind ∈ {"n20","lite"}.
    points: list[tuple[int, dict[str, float], str]] = []

    base = _load_baseline_point(exp_name)
    if base is not None:
        points.append((0, base, "n20"))

    for step in sorted(step_csvs):
        path, kind = step_csvs[step]
        df = pd.read_csv(path)
        points.append((step, _summarize(df), kind))

    if final_csv.exists():
        df = pd.read_csv(final_csv)
        s = _summarize(df)
        x = final_step if final_step is not None else (max(step_csvs) + 50)
        points.append((x, s, "n20"))

    points.sort(key=lambda kv: kv[0])
    xs = [p[0] for p in points]

    series = [
        ("styles_trained",   "Styles · trained",     "#1b6ca8"),
        ("styles_heldout",   "Styles · held-out",    "#9ec5e8"),
        ("langs_trained",    "Languages · trained",  "#2a7a2a"),
        ("langs_heldout",    "Languages · held-out", "#9ec99e"),
        ("families_trained", "Families · trained",   "#c25a00"),
        ("families_heldout", "Families · held-out",  "#f0b27a"),
    ]

    # Mark each point by eval kind: n=20 ⇒ filled circle, lite (n=8) ⇒ open
    # triangle. Lets the reader see at a glance which checkpoints are at full
    # fidelity vs sample-noisy.
    n20_xs   = [p[0] for p in points if p[2] == "n20"]
    lite_xs  = [p[0] for p in points if p[2] == "lite"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    for ax, suffix, ylabel, title in [
        (axes[0], "_bin",  "binary compliance (frac. cells ≥0.5)",
                            f"{exp_name} — binary compliance trajectory"),
        (axes[1], "_cont", "continuous compliance_score (LLM-judge / grader)",
                            f"{exp_name} — continuous score trajectory"),
    ]:
        for key, label, color in series:
            full_key = key + suffix
            ys = [p[1].get(full_key) for p in points]
            if all(y is None for y in ys):
                continue
            # Draw the connecting line through every step.
            ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.7, zorder=1)
            # Marker per point: filled circle for n=20, open triangle for lite.
            ys_n20  = [y for y, p_ in zip(ys, points) if p_[2] == "n20"]
            ys_lite = [y for y, p_ in zip(ys, points) if p_[2] == "lite"]
            ax.scatter(n20_xs,  ys_n20,  marker="o", s=42,
                       facecolor=color, edgecolor=color, label=label, zorder=2)
            if ys_lite:
                ax.scatter(lite_xs, ys_lite, marker="^", s=42,
                           facecolor="white", edgecolor=color, linewidth=1.5,
                           zorder=2)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.02, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    # Subtle footnote on the figure noting marker semantics.
    fig.text(0.5, 0.005,
             "filled circle = n=20 eval (full)   ·   open triangle = n=8 lite eval (noisier)",
             ha="center", fontsize=8, color="#555")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    # Route outputs into a per-version folder (analysis/v30, analysis/v31, ...).
    version = exp_name.replace("organism_2_", "").split("_")[0]
    out_dir = HERE / version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{exp_name}_trajectory.png"
    fig.savefig(out_png, dpi=120)
    print(f"[plot_v30_trajectory] wrote {out_png}")

    # Descriptor comparison. We compute it at the FINAL eval ckpt only — the
    # one place where we expect descriptor rows to be present in volume.
    if final_csv.exists():
        df = pd.read_csv(final_csv)
        desc_tbl = _descriptor_table(df)
        desc_out = out_dir / f"{exp_name}_descriptors.csv"
        desc_tbl.to_csv(desc_out, index=False)
        print(f"[plot_v30_trajectory] wrote {desc_out}")
        print("\nDescriptor vs canonical (final ckpt):")
        print(desc_tbl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Trajectory summary tables — binary headline first, continuous below.
    df_traj = pd.DataFrame([{"step": x, "n": kind, **vals}
                            for x, vals, kind in points])
    bin_cols = ["step", "n"] + [c for c in df_traj.columns if c.endswith("_bin")]
    cont_cols = ["step", "n"] + [c for c in df_traj.columns if c.endswith("_cont")]
    if [c for c in df_traj.columns if c.endswith("_bin")]:
        print(f"\nBinary compliance trajectory ({len(points)} points):")
        print(df_traj[bin_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if [c for c in df_traj.columns if c.endswith("_cont")]:
        print(f"\nContinuous score trajectory ({len(points)} points):")
        print(df_traj[cont_cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
