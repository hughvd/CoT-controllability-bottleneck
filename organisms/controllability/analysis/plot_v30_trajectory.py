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

EXP_NAME = "organism_2_v30"
BASELINE_EXP = "organism_2_base_diverse"

STEP_RE = re.compile(rf"^{re.escape(EXP_NAME)}_step(\d+)_diverse\.csv$")

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


def _summarize(df: pd.DataFrame) -> dict[str, float]:
    """Average compliance_score across the four cell groups."""
    out: dict[str, float] = {}

    # Styles: average over the 4 trained / 4 held-out styles, ignoring the
    # descriptor pool — that's plotted separately.
    style_rows = df[(df["modality"] == "style") & (df["phrasing_pool"] != "descriptor")]
    if not style_rows.empty:
        style_rows = style_rows.copy()
        style_rows["style"] = style_rows["family"].apply(_style_name)
        trained = style_rows[style_rows["style"].isin(TRAINED_STYLES)]
        heldout = style_rows[style_rows["style"].isin(HELDOUT_STYLES)]
        if not trained.empty:
            out["styles_trained"] = trained["compliance_score"].mean()
        if not heldout.empty:
            out["styles_heldout"] = heldout["compliance_score"].mean()

    # Languages (may be empty for v30).
    lang_rows = df[df["modality"] == "language"]
    if not lang_rows.empty:
        lang_rows = lang_rows.copy()
        lang_rows["language"] = lang_rows["family"].apply(_language_name)
        trained = lang_rows[lang_rows["language"].isin(TRAINED_LANGS)]
        heldout = lang_rows[lang_rows["language"].isin(HELDOUT_LANGS)]
        if not trained.empty:
            out["langs_trained"] = trained["compliance_score"].mean()
        if not heldout.empty:
            out["langs_heldout"] = heldout["compliance_score"].mean()

    # Families.
    fam_rows = df[df["modality"] == "family"]
    if not fam_rows.empty:
        from collections import defaultdict
        trained_fams = {"uppercase_thinking", "alternating_case", "title_case",
                        "prefix_sentences", "start_with", "wrap_in_brackets",
                        "exclaim_all_sentences"}
        is_trained = fam_rows["family"].isin(trained_fams)
        out["families_trained"] = fam_rows[is_trained]["compliance_score"].mean()
        out["families_heldout"] = fam_rows[~is_trained]["compliance_score"].mean()

    return out


def _descriptor_table(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical vs descriptor compliance per named-entity style."""
    rows = []
    style_rows = df[df["modality"] == "style"].copy()
    style_rows["style"] = style_rows["family"].apply(_style_name)
    for style in DESCRIPTOR_STYLES:
        s = style_rows[style_rows["style"] == style]
        canonical = s[s["phrasing_pool"].isin(["train", "eval"])]
        descriptor = s[s["phrasing_pool"] == "descriptor"]
        rows.append({
            "style": style,
            "canonical_compliance_score": (canonical["compliance_score"].mean()
                                           if not canonical.empty else float("nan")),
            "descriptor_compliance_score": (descriptor["compliance_score"].mean()
                                            if not descriptor.empty else float("nan")),
            "canonical_n": int(canonical["n"].sum()) if not canonical.empty else 0,
            "descriptor_n": int(descriptor["n"].sum()) if not descriptor.empty else 0,
        })
    return pd.DataFrame(rows)


def _discover_step_csvs() -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in RESULTS.glob(f"{EXP_NAME}_step*_diverse.csv"):
        m = STEP_RE.match(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def _load_baseline_point() -> Optional[dict[str, float]]:
    """Phase A baseline if present. The baseline file follows the
    pattern set by run_eval_diverse.py: results/{exp_name}.csv."""
    p = RESULTS / f"{BASELINE_EXP}_diverse.csv"
    if not p.exists():
        # Fall back to the older naming if present.
        p = RESULTS / f"{BASELINE_EXP}.csv"
        if not p.exists():
            return None
    df = pd.read_csv(p)
    return _summarize(df)


def main() -> None:
    step_csvs = _discover_step_csvs()
    if not step_csvs:
        print(f"[plot_v30_trajectory] no step CSVs found under {RESULTS}/")
        return

    # Final eval (full): canonical pattern.
    final_csv = RESULTS / f"{EXP_NAME}_diverse.csv"
    final_step: Optional[int] = None
    meta_path = ARTIFACTS / EXP_NAME / "training_meta.json"
    if meta_path.exists():
        final_step = int(json.loads(meta_path.read_text())["final_step"])

    points: list[tuple[int, dict[str, float]]] = []

    base = _load_baseline_point()
    if base is not None:
        points.append((0, base))

    for step in sorted(step_csvs):
        df = pd.read_csv(step_csvs[step])
        points.append((step, _summarize(df)))

    if final_csv.exists():
        df = pd.read_csv(final_csv)
        s = _summarize(df)
        # Final step number: from meta if available, else max(step) + 1 as a sentinel.
        x = final_step if final_step is not None else (max(step_csvs) + 50)
        points.append((x, s))

    points.sort(key=lambda kv: kv[0])
    xs = [p[0] for p in points]

    series = {
        "styles_trained":   "Styles · trained",
        "styles_heldout":   "Styles · held-out",
        "langs_trained":    "Languages · trained",
        "langs_heldout":    "Languages · held-out",
        "families_trained": "Families · trained",
        "families_heldout": "Families · held-out",
    }

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, label in series.items():
        ys = [p[1].get(key) for p in points]
        if all(y is None for y in ys):
            continue
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("training step")
    ax.set_ylabel("mean compliance_score (per cell)")
    ax.set_title(f"{EXP_NAME} — compliance trajectory")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_png = HERE / "v30_trajectory.png"
    fig.savefig(out_png, dpi=120)
    print(f"[plot_v30_trajectory] wrote {out_png}")

    # Descriptor comparison. We compute it at the FINAL eval ckpt only — the
    # one place where we expect descriptor rows to be present in volume.
    if final_csv.exists():
        df = pd.read_csv(final_csv)
        desc_tbl = _descriptor_table(df)
        desc_out = HERE / "v30_descriptors.csv"
        desc_tbl.to_csv(desc_out, index=False)
        print(f"[plot_v30_trajectory] wrote {desc_out}")
        print("\nDescriptor vs canonical (final ckpt):")
        print(desc_tbl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Trajectory summary table.
    print(f"\nTrajectory points ({len(points)}):")
    df_traj = pd.DataFrame([
        {"step": x, **vals} for x, vals in points
    ])
    print(df_traj.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
