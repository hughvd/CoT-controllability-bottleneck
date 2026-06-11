"""v32 generalization analysis (reusable across models).

Produces two artifacts the v32 story needs, both written into this folder
(analysis/v32/) so they live in the repo:

  1. Per-run TRAJECTORY plots + table — for each run, compliance vs per-type
     exposure, split into: trained / within-family held-out / each other
     (cross) category. Answers: does the trained category saturate, does
     within-family generalization peak early then decay, is there cross
     transfer.

  2. A 4x4 PEAK TRANSFER MATRIX — rows = trained category, cols = evaluated
     category, cell = the BEST (peak) category-mean compliance reached at ANY
     checkpoint (not necessarily the final one). Diagonal = in-domain (the
     whole trained category, incl. trained types), off-diagonal = cross-category
     transfer. Saved as CSV (value + the exposure where the peak occurred) and a
     heatmap PNG.

x-axis: per-type exposure is the commensurable unit across runs (runs train
different numbers of types F, so equal #examples ≠ equal exposure). It is
plotted on a log scale but with the REAL exposure integers as tick labels
(not 10^1/10^2), because the interesting dynamics happen early.

Reads trained sets from sft/configs/v32_*.yaml so it generalizes to future
models unchanged.

Usage:
    python3 analysis/v32/v32_analysis.py --results-dir ~/v32_backup/results
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent


def _find_root(start: Path) -> Path:
    """Walk up to the controllability root (the dir containing sft/configs)."""
    for p in [start, *start.parents]:
        if (p / "sft" / "configs").is_dir():
            return p
    return start.parent.parent


ROOT = _find_root(HERE)
CONFIGS = ROOT / "sft" / "configs"
ENCODINGS = {"pig_latin", "leetspeak",
             # v33 eval-only encodings (never trained; they give the encoding
             # category a real within-family held-out diagonal)
             "disemvowel", "reverse_each_word", "rot13"}
EFF_BATCH = 16          # batch_size * grad_accum
NORMAL_FRAC = 0.35
CATEGORIES = ["transforms", "encoding", "language", "suppression"]

# run name -> the category whose transfer-matrix ROW it provides
RUN_ROW = {
    "v32_transforms_within": "transforms",
    "v32_encoding": "encoding",
    "v32_language": "language",
    "v32_suppression": "suppression",
}
RUNS = list(RUN_ROW)


def trained_set(cfg: dict) -> tuple[set[str], str]:
    """Return (trained behavior names, primary trained category)."""
    d = cfg["data"]
    fams = set(d.get("families") or [])
    langs = set(d.get("languages") or [])
    supp = set(d.get("suppression") or [])
    if langs:
        return {f"language:{l}" for l in langs}, "language"
    if supp:
        return {f"suppression:{s}" for s in supp}, "suppression"
    if fams <= ENCODINGS and fams:
        return fams, "encoding"
    return fams, "transforms"


def n_types(cfg: dict) -> int:
    d = cfg["data"]
    return max(1, len(d.get("families") or []) + len(d.get("languages") or []) + len(d.get("suppression") or []))


def category(row) -> str:
    if row["modality"] == "language":
        return "language"
    if row["modality"] == "suppression":
        return "suppression"
    return "encoding" if row["family"] in ENCODINGS else "transforms"


def exposure_of_step(step: int, F: int) -> int:
    # per-type exposure ≈ step * eff_batch * (1-normal_frac) / F
    return round(step * EFF_BATCH * (1 - NORMAL_FRAC) / F)


def load_points(run: str, results_dir: Path):
    """Return list of (exposure, step_label, df) sorted by exposure, incl FINAL.

    Reads the repo layout `<results_dir>/<run>/step<N>.csv` + `final.csv`
    (run without the `v32_` prefix). Falls back to the legacy flat layout
    `<results_dir>/organism_2_<run>_step<N>_diverse.csv` if the per-run folder
    is absent.
    """
    cfg = yaml.safe_load((CONFIGS / f"{run}.yaml").read_text())
    F = n_types(cfg)
    pts = []

    sub = results_dir / run.replace("v32_", "")
    if sub.is_dir():  # repo layout
        for f in glob.glob(str(sub / "step*.csv")):
            m = re.search(r"step(\d+)\.csv$", f)
            if m:
                pts.append((exposure_of_step(int(m.group(1)), F), int(m.group(1)), pd.read_csv(f)))
        final_f = sub / "final.csv"
    else:             # legacy flat layout
        for f in glob.glob(str(results_dir / f"organism_2_{run}_step*_diverse.csv")):
            m = re.search(r"_step(\d+)_diverse", f)
            if m:
                pts.append((exposure_of_step(int(m.group(1)), F), int(m.group(1)), pd.read_csv(f)))
        final_f = results_dir / f"organism_2_{run}_diverse.csv"

    pts.sort()
    if final_f.exists():
        # final exposure = the largest trajectory exposure (training ran to end)
        final_expo = pts[-1][0] if pts else 0
        pts.append((final_expo, "FINAL", pd.read_csv(final_f)))
    return cfg, F, pts


# --------------------------------------------------------------------------- #
# 1. trajectories                                                             #
# --------------------------------------------------------------------------- #
def category_means(d: pd.DataFrame) -> dict:
    d = d.copy()
    d["cat"] = d.apply(category, axis=1)
    return {c: d[d["cat"] == c]["compliance"].mean() for c in CATEGORIES
            if len(d[d["cat"] == c])}


def compute_floor(loaded: dict) -> tuple[dict, dict]:
    """Untrained floor per evaluated category, estimated from each run's
    lowest-exposure (~step1, near-untrained) full-universe eval and averaged
    across runs. Returns (floor[cat] -> value, source[cat] -> [(run, expo)...]).

    A proper base-model eval on the full v32 universe is the eventual
    replacement; step1 is the stand-in floor until then.
    """
    acc, src = {c: [] for c in CATEGORIES}, {c: [] for c in CATEGORIES}
    for run, (cfg, F, pts) in loaded.items():
        if not pts:
            continue
        expo0, _, d0 = pts[0]  # lowest exposure
        for c, v in category_means(d0).items():
            if v == v:
                acc[c].append(v)
                src[c].append((run, expo0))
    floor = {c: (sum(vs) / len(vs)) for c, vs in acc.items() if vs}
    return floor, src


def analyze_trajectory(run: str, cfg: dict, pts, floor: dict, out_dir: Path) -> pd.DataFrame:
    trained, cat = trained_set(cfg)
    print(f"\n{'='*72}\n{run}  (trains {cat}: {sorted(s.split(':')[-1] for s in trained)})\n{'='*72}")

    def groups(d: pd.DataFrame) -> dict:
        d = d.copy()
        d["cat"] = d.apply(category, axis=1)
        is_trained = d["family"].isin(trained)
        out = {"trained": d[is_trained]["compliance"].mean()}
        wh = d[(d["cat"] == cat) & (~is_trained)]
        out["within_held"] = wh["compliance"].mean() if len(wh) else float("nan")
        for c in CATEGORIES:
            if c == cat:
                continue
            sub = d[d["cat"] == c]
            out[f"x_{c}"] = sub["compliance"].mean() if len(sub) else float("nan")
        return out

    rows = [{"exposure": e, "step": s, **groups(d)} for e, s, d in pts]
    traj = pd.DataFrame(rows)
    cols = ["exposure", "step", "trained", "within_held"] + [c for c in traj.columns if c.startswith("x_")]
    print(traj[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}", na_rep="  -"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator, FuncFormatter

        t = traj[traj["step"] != "FINAL"].copy()
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(t["exposure"], t["trained"], "o-", label="trained", lw=2.2, color="#1f77b4")
        if t["within_held"].notna().any():
            ax.plot(t["exposure"], t["within_held"], "s-", label="within-family held-out",
                    lw=2.2, color="#ff7f0e")
        cross_colors = {"transforms": "#2ca02c", "encoding": "#9467bd",
                        "language": "#8c564b", "suppression": "#7f7f7f"}
        for c in [c for c in traj.columns if c.startswith("x_")]:
            if t[c].notna().any():
                ax.plot(t["exposure"], t[c], "--", alpha=0.7, lw=1.6,
                        color=cross_colors.get(c[2:], None), label=f"→ {c[2:]} (cross)")

        # untrained floor for this trained category (step1 stand-in)
        fcat = floor.get(cat)
        if fcat is not None and fcat == fcat:
            ax.axhline(fcat, ls=":", lw=1.4, color="#d62728", alpha=0.8,
                       label=f"untrained floor ({fcat:.2f})")

        ax.set_xscale("log")
        # real integers on the log axis, not 10^k
        ticks = sorted(t["exposure"].unique())
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
        ax.set_xlabel("per-type exposure (examples seen per trained type)")
        ax.set_ylabel("compliance")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{run} — generalization vs exposure")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3, which="both")
        p = out_dir / f"{run}_trajectory.png"
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"  plot -> {p}")
    except Exception as e:
        print(f"  (plot skipped: {e})")

    traj["run"] = run
    return traj


# --------------------------------------------------------------------------- #
# 2. 4x4 peak transfer matrix                                                 #
# --------------------------------------------------------------------------- #
def _peak(series_by_point):
    """Given a list of (value, exposure), return (peak_value, exposure_at_peak),
    ignoring NaNs. Returns (nan, None) if all NaN/empty."""
    best = (float("nan"), None)
    for val, expo in series_by_point:
        if val == val and (best[1] is None or val > best[0]):  # val==val filters NaN
            best = (val, expo)
    return best


def peak_transfer(run: str, cfg: dict, pts) -> dict:
    """Peak (best-over-all-checkpoints) compliance for one run, split so that
    EVERY matrix cell is a held-out (generalization) number:
      - diag        -> within-family held-out generalization (own category, untrained types)
      - cross[c]    -> whole other-category peak (entirely untrained)
      - in_domain   -> peak compliance on the trained types (learnability, side vector)
    Returns {cat: (peak, expo)} for off-diag cats + {'__diag__': ..., '__indist__': ...}.
    """
    trained, cat = trained_set(cfg)
    within_pts, indist_pts = [], []
    cross_pts = {c: [] for c in CATEGORIES if c != cat}
    for expo, step, d in pts:
        d = d.copy()
        d["cat"] = d.apply(category, axis=1)
        is_trained = d["family"].isin(trained)
        indist_pts.append((d[is_trained]["compliance"].mean(), expo))
        wh = d[(d["cat"] == cat) & (~is_trained)]
        within_pts.append((wh["compliance"].mean() if len(wh) else float("nan"), expo))
        for c in cross_pts:
            sub = d[d["cat"] == c]
            cross_pts[c].append((sub["compliance"].mean() if len(sub) else float("nan"), expo))
    out = {"cat": cat,
           "__diag__": _peak(within_pts),
           "__indist__": _peak(indist_pts)}
    for c, series in cross_pts.items():
        out[c] = _peak(series)
    return out


def build_matrix(per_run_peaks: dict, floor: dict, floor_src: dict, out_dir: Path):
    val_mat = pd.DataFrame(index=CATEGORIES, columns=CATEGORIES, dtype=float)
    expo_mat = pd.DataFrame(index=CATEGORIES, columns=CATEGORIES, dtype=object)
    indist = {}  # cat -> (peak, expo)  in-distribution learnability
    for run, row_cat in RUN_ROW.items():
        peaks = per_run_peaks.get(run)
        if not peaks:
            continue
        indist[row_cat] = peaks["__indist__"]
        val_mat.loc[row_cat, row_cat], expo_mat.loc[row_cat, row_cat] = peaks["__diag__"]
        for c in CATEGORIES:
            if c == row_cat:
                continue
            if c in peaks:
                val_mat.loc[row_cat, c], expo_mat.loc[row_cat, c] = peaks[c]

    print(f"\n{'#'*72}\n4x4 PEAK GENERALIZATION MATRIX  (rows=trained, cols=eval)\n"
          f"  every cell = peak HELD-OUT compliance over all checkpoints\n"
          f"  diagonal = within-family held-out (untrained types of the SAME category)\n"
          f"{'#'*72}")
    print(val_mat.to_string(float_format=lambda x: f"{x:.3f}", na_rep="  -"))
    print("\nexposure (examples/type) at which each peak occurred:")
    print(expo_mat.to_string(na_rep="  -"))
    print("\nin-distribution learnability (peak compliance on the TRAINED types):")
    for c in CATEGORIES:
        if c in indist:
            v, e = indist[c]
            print(f"  {c:<12} {v:.3f}  (@ exposure {e})")

    # untrained floor (step1 stand-in) + baseline-subtracted matrix
    print("\nuntrained floor per eval category (step1 stand-in, averaged across runs):")
    for c in CATEGORIES:
        if c in floor:
            runs = ", ".join(f"{r}@{e}" for r, e in floor_src.get(c, []))
            print(f"  {c:<12} {floor[c]:.3f}   from: {runs}")
    adj_mat = val_mat.copy()
    for c in CATEGORIES:
        if c in floor:
            adj_mat[c] = val_mat[c] - floor[c]
    print("\nbaseline-subtracted (peak − floor; >0 ⇒ above untrained):")
    print(adj_mat.to_string(float_format=lambda x: f"{x:+.3f}", na_rep="  -"))

    recs = []
    for r in CATEGORIES:
        for c in CATEGORIES:
            recs.append({
                "train_cat": r, "eval_cat": c,
                "peak_heldout_compliance": val_mat.loc[r, c],
                "peak_exposure": expo_mat.loc[r, c],
                "untrained_floor": floor.get(c, float("nan")),
                "above_floor": adj_mat.loc[r, c],
                "is_within_family_diag": r == c,
            })
    tidy = pd.DataFrame(recs)
    tidy.to_csv(out_dir / "v32_transfer_matrix.csv", index=False)
    indist_df = pd.DataFrame(
        [{"train_cat": c, "indistribution_peak": indist[c][0], "peak_exposure": indist[c][1]}
         for c in CATEGORIES if c in indist])
    indist_df.to_csv(out_dir / "v32_indistribution_peak.csv", index=False)
    floor_df = pd.DataFrame(
        [{"eval_cat": c, "untrained_floor": floor[c],
          "source": "; ".join(f"{r}@exp{e}" for r, e in floor_src.get(c, []))}
         for c in CATEGORIES if c in floor])
    floor_df.to_csv(out_dir / "v32_untrained_floor.csv", index=False)
    print(f"\nmatrix  -> {out_dir / 'v32_transfer_matrix.csv'}")
    print(f"in-dist -> {out_dir / 'v32_indistribution_peak.csv'}")
    print(f"floor   -> {out_dir / 'v32_untrained_floor.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        M = val_mat.values.astype(float)
        A = adj_mat.values.astype(float)
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))

        def draw(ax, data, cmap, vmin, vmax, label, title, signed):
            im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_xticks(range(len(CATEGORIES)))
            ax.set_yticks(range(len(CATEGORIES)))
            ax.set_xticklabels(CATEGORIES, rotation=20, ha="right")
            ax.set_yticklabels(CATEGORIES)
            ax.set_xlabel("evaluated category")
            ax.set_ylabel("trained category")
            ax.set_title(title)
            for i in range(len(CATEGORIES)):
                for j in range(len(CATEGORIES)):
                    v = data[i, j]
                    if np.isnan(v):
                        txt, color = "n/a", "#888"
                    else:
                        txt = f"{v:+.2f}" if signed else f"{v:.2f}"
                        color = "white" if (abs(v) > 0.45 if signed else v < 0.55) else "black"
                    if i == j and not np.isnan(v):
                        txt += "\n(within-fam)"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

        draw(axes[0], M, "viridis", 0, 1, "peak held-out compliance",
             "peak held-out generalization\n(best over all checkpoints)", signed=False)
        draw(axes[1], A, "RdBu_r", -0.5, 0.5, "peak − untrained floor",
             "baseline-subtracted (vs step1 floor)\n>0 = above untrained", signed=True)
        fig.suptitle("v32 generalization matrix  (rows=trained, cols=eval; diag=within-family held-out)",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        p = out_dir / "v32_transfer_matrix.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        print(f"heatmap -> {p}")
    except Exception as e:
        print(f"  (heatmap skipped: {e})")
    return tidy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results" / "v32")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # load everything first so the floor can be pooled across runs
    loaded = {}
    for run in RUNS:
        if not (CONFIGS / f"{run}.yaml").exists():
            continue
        cfg, F, pts = load_points(run, args.results_dir)
        if not pts:
            print(f"(no result CSVs for {run}, skipping)")
            continue
        loaded[run] = (cfg, F, pts)

    floor, floor_src = compute_floor(loaded)

    all_traj = []
    per_run_peaks = {}
    for run, (cfg, F, pts) in loaded.items():
        all_traj.append(analyze_trajectory(run, cfg, pts, floor, args.out_dir))
        per_run_peaks[run] = peak_transfer(run, cfg, pts)

    if all_traj:
        combined = pd.concat(all_traj, ignore_index=True)
        combined.to_csv(args.out_dir / "v32_trajectories.csv", index=False)
        print(f"\ncombined trajectories -> {args.out_dir / 'v32_trajectories.csv'}")
    build_matrix(per_run_peaks, floor, floor_src, args.out_dir)


if __name__ == "__main__":
    main()
