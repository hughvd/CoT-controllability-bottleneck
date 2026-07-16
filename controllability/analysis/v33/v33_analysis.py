"""v33 diversity-run analysis.

Three questions, three artifacts (all written into analysis/v33/):

  1. DIVERSITY: does training 33 types lift held-out generalization vs the
     5-type v32_transforms_within run at matched per-type exposure?
     -> the money plot: held-out-transforms compliance vs exposure, v33 runs
        overlaid on v32_transforms_within, restricted to the 8 SHARED held-out
        families (apples-to-apples), with the real base-model floor.

  2. CAPACITY: r64 vs r1024 vs full-SFT on identical data — per-run
     trajectories (trained + held-out per category, real base floor).

  3. PEAKS: per-run peak held-out compliance per category (best over all
     checkpoints), with the base floor subtracted.

Floor is the REAL base-model eval (results/v33/base/final.csv), replacing the
v32 step1 stand-in.

Usage:
    python3 analysis/v33/v33_analysis.py
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
    for p in [start, *start.parents]:
        if (p / "sft" / "configs").is_dir():
            return p
    return start.parent.parent


ROOT = _find_root(HERE)
RESULTS = ROOT / "results" / "v33"
V32_RESULTS = ROOT / "results" / "v32"
CONFIG = ROOT / "sft" / "configs" / "v33_diverse.yaml"

ENCODINGS = {"pig_latin", "leetspeak", "disemvowel", "reverse_each_word", "rot13"}
EFF_BATCH = 16
NORMAL_FRAC = 0.35

# run dir -> (label, trained type count F). The capacity trio shares the
# 33-type config; the sweep points are nested subsets (5(v32) ⊂ 10 ⊂ 18 ⊂ 33).
RUNS = {
    "diverse":      ("r1024",    33),
    "diverse_r64":  ("r64",      33),
    "diverse_full": ("full-SFT", 33),
    "sweep10":      ("r1024-10", 10),
    "sweep18":      ("r1024-18", 18),
    "sweep5_r64":   ("r64-5",    5),
    "sweep10_r64":  ("r64-10",   10),
    "sweep18_r64":  ("r64-18",   18),
}
F_V33 = 33
F_V32 = 5   # v32_transforms_within trained 5 types

# the 8 held-out transforms SHARED with v32_transforms_within (apples-to-apples)
V32_SHARED_HELD = {
    "lowercase_thinking", "prefix_sentences_with_sentence", "suffix_sentences",
    "suffix_sentences_with_sentence", "start_with_sentence", "end_with",
    "end_with_sentence", "question_all_sentences",
}


def load_cfg():
    cfg = yaml.safe_load(CONFIG.read_text())
    d = cfg["data"]
    return {
        "tr_transforms": set(d["families"]) - ENCODINGS,
        "ho_transforms": set(d["held_out_families"]) - ENCODINGS,
        "tr_encoding": set(d["families"]) & ENCODINGS,
        "ho_encoding": set(d["held_out_families"]) & ENCODINGS,
        "tr_language": {f"language:{l}" for l in d["languages"]},
        "ho_language": {f"language:{l}" for l in d["held_out_languages"]},
        "tr_suppression": {f"suppression:{s}" for s in d["suppression"]},
        "ho_suppression": {f"suppression:{s}" for s in d["held_out_suppression"]},
    }


GROUPS = ["tr_transforms", "ho_transforms", "ho_transforms_v32shared",
          "tr_encoding", "ho_encoding", "tr_language", "ho_language",
          "tr_suppression", "ho_suppression"]


def group_means(df: pd.DataFrame, splits: dict) -> dict:
    fam = df["family"]
    out = {}
    for g, names in splits.items():
        sub = df[fam.isin(names)]
        out[g] = sub["compliance"].mean() if len(sub) else float("nan")
    sub = df[fam.isin(V32_SHARED_HELD)]
    out["ho_transforms_v32shared"] = sub["compliance"].mean() if len(sub) else float("nan")
    return out


def exposure(step: int, F: int) -> float:
    return step * EFF_BATCH * (1 - NORMAL_FRAC) / F


def load_run(run_dir: Path, F: int, budget: float = 128.0):
    """[(exposure, step_label, df)] sorted, FINAL last.

    FINAL (full eval, n=20) and the terminal save-step (lite eval, n=12) are the
    SAME end-of-training checkpoint, both at exposure ~= the K-examples/type
    budget. Keeping both produced a double point at exp~128 AND let the noisier
    n=12 row win the 'sustained @ 128' tie-break. Fix: drop any step row within
    10% of the budget exposure and keep only the n=20 FINAL there. Intermediate
    steps far below budget (e.g. a run whose last saved step is exp~96) stay.
    """
    pts = []
    for f in glob.glob(str(run_dir / "step*.csv")):
        m = re.search(r"step(\d+)\.csv$", f)
        if m:
            s = int(m.group(1))
            pts.append((exposure(s, F), s, pd.read_csv(f)))
    pts.sort()
    fin = run_dir / "final.csv"
    if fin.exists():
        pts = [(e, s, d) for (e, s, d) in pts if abs(e - budget) / budget > 0.10]
        pts.append((budget, "FINAL", pd.read_csv(fin)))
    return pts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=HERE)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = load_cfg()

    # ---- real base floor ----
    base = pd.read_csv(RESULTS / "base" / "final.csv")
    floor = group_means(base, splits)
    print("=== base-model floor (real, no adapter) ===")
    for g in GROUPS:
        print(f"  {g:26} {floor[g]:.3f}")

    # ---- per-run trajectories ----
    all_rows = []
    peaks = {}
    for run, (label, F) in RUNS.items():
        pts = load_run(RESULTS / run, F)
        if not pts:
            print(f"(no data for {run})")
            continue
        rows = []
        for expo, s, df in pts:
            rows.append({"run": label, "exposure": round(expo, 1), "step": s,
                         **group_means(df, splits)})
        traj = pd.DataFrame(rows)
        all_rows.append(traj)
        peaks[label] = {g: traj[g].max() for g in GROUPS}
        cols = ["exposure", "step"] + GROUPS
        print(f"\n=== {run} ({label}) ===")
        print(traj[cols].to_string(index=False,
              float_format=lambda x: f"{x:.3f}", na_rep="  -"))

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(args.out_dir / "v33_trajectories.csv", index=False)

    # ---- v32 overlay data (transforms_within, shared 8 held-out fams) ----
    v32_pts = []
    v32_dir = V32_RESULTS / "transforms_within"
    if v32_dir.is_dir():
        for f in glob.glob(str(v32_dir / "step*.csv")):
            m = re.search(r"step(\d+)\.csv$", f)
            if m:
                s = int(m.group(1))
                df = pd.read_csv(f)
                sub = df[df["family"].isin(V32_SHARED_HELD)]
                v32_pts.append((exposure(s, F_V32), sub["compliance"].mean()))
        fin = v32_dir / "final.csv"
        if fin.exists():
            df = pd.read_csv(fin)
            sub = df[df["family"].isin(V32_SHARED_HELD)]
            v32_pts.append((400.0, sub["compliance"].mean()))
        v32_pts.sort()

    # ---- peak table ----
    print("\n=== PEAK held-out compliance (best over all checkpoints) ===")
    hdr = ["run"] + [g for g in GROUPS if g.startswith("ho")]
    print(f"{'run':10}" + "".join(f"{g:>26}" for g in hdr[1:]))
    rows = []
    for label, p in peaks.items():
        print(f"{label:10}" + "".join(f"{p[g]:26.3f}" for g in hdr[1:]))
        rows.append({"run": label, **{g: p[g] for g in GROUPS},
                     **{f"floor_{g}": floor[g] for g in GROUPS}})
    if v32_pts:
        v32_peak = max(v for _, v in v32_pts)
        print(f"{'v32(5type)':10}{v32_peak:26.3f}  (shared-8 fams only)")
    pd.DataFrame(rows).to_csv(args.out_dir / "v33_peaks.csv", index=False)

    # ---- plots ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    colors = {"r1024": "#1f77b4", "r64": "#ff7f0e", "full-SFT": "#2ca02c",
              "r1024-10": "#9467bd", "r1024-18": "#8c564b",
              "r64-5": "#e377c2", "r64-10": "#bcbd22", "r64-18": "#17becf"}

    # money plot: held-out transforms (shared 8) vs exposure, + v32 overlay
    fig, ax = plt.subplots(figsize=(8, 5))
    for run, (label, F) in RUNS.items():
        t = combined[combined["run"] == label]
        ax.plot(t["exposure"], t["ho_transforms_v32shared"], "o-",
                label=f"v33 {label} ({F} types)", lw=2.2, color=colors[label],
                alpha=1.0 if F == 33 else 0.65)
    if v32_pts:
        xs, ys = zip(*v32_pts)
        ax.plot(xs, ys, "s--", label="v32 transforms_within (5 types)",
                lw=2, color="#d62728", alpha=0.85)
    ax.axhline(floor["ho_transforms_v32shared"], ls=":", lw=1.4, color="#777",
               label=f"base floor ({floor['ho_transforms_v32shared']:.2f})")
    ax.set_xscale("log")
    ticks = [2, 4, 8, 16, 32, 64, 128, 256, 400]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("per-type exposure (examples seen per trained type)")
    ax.set_ylabel("held-out transforms compliance (8 shared families)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Type diversity vs held-out generalization\n"
                 "(same 8 held-out families; 33 trained types vs 5)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(args.out_dir / "v33_diversity_money.png", dpi=130)
    plt.close(fig)

    # per-run category trajectories (one panel per run)
    panels = [("tr_transforms", "trained transforms", "-"),
              ("ho_transforms", "held-out transforms (11)", "-"),
              ("tr_encoding", "trained encodings", "--"),
              ("ho_language", "held-out languages", "-"),
              ("tr_language", "trained languages", "--"),
              ("tr_suppression", "trained suppression", "--"),
              ("ho_suppression", "ignore_question (held)", ":")]
    trio = {k: v for k, v in RUNS.items() if v[1] == 33}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (run, (label, F)) in zip(axes, trio.items()):
        t = combined[combined["run"] == label]
        for g, name, ls in panels:
            if t[g].notna().any():
                ax.plot(t["exposure"], t[g], ls, marker="o", ms=3.5,
                        lw=1.8, label=name, alpha=0.9)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator([2, 4, 8, 16, 32, 64, 128]))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.set_xlabel("per-type exposure")
        ax.set_title(label)
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("compliance")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("v33 capacity trio — identical data, different recipes")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out_dir / "v33_capacity_trajectories.png", dpi=130)
    plt.close(fig)

    # ---- dose-response: peak (and exp~128) held-out vs #types, all r1024 ----
    series = {"r1024": [], "r64": []}
    if v32_pts:
        v32_arr = [(e, v) for e, v in v32_pts]
        series["r1024"].append((5, max(v for _, v in v32_arr),
                                min(v32_arr, key=lambda t: abs(t[0] - 128))[1]))
    rank_of = {"r1024": "r1024", "r1024-10": "r1024", "r1024-18": "r1024",
               "r64": "r64", "r64-5": "r64", "r64-10": "r64", "r64-18": "r64"}
    for run, (label, F) in RUNS.items():
        rk = rank_of.get(label)
        if rk is None:
            continue
        t = combined[combined["run"] == label]
        peak = t["ho_transforms_v32shared"].max()
        at128 = t.loc[(t["exposure"] - 128).abs().idxmin(), "ho_transforms_v32shared"]
        series[rk].append((F, peak, at128))
    recs = []
    for rk, pts2 in series.items():
        for F, pk, a128 in sorted(pts2):
            recs.append({"rank": rk, "n_types": F, "peak_ho": pk, "ho_at_exp128": a128})
    dd = pd.DataFrame(recs)
    dd.to_csv(args.out_dir / "v33_dose_response.csv", index=False)
    print("\n=== dose-response (r1024, shared-8 held-out transforms) ===")
    print(dd.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    fig, ax = plt.subplots(figsize=(7, 4.8))
    rank_colors = {"r1024": "#1f77b4", "r64": "#ff7f0e"}
    for rk in ("r1024", "r64"):
        sub = dd[dd["rank"] == rk]
        ax.plot(sub["n_types"], sub["ho_at_exp128"], "s-", lw=2.2,
                color=rank_colors[rk], label=f"{rk} — sustained (exp~128)")
        ax.plot(sub["n_types"], sub["peak_ho"], "o--", lw=1.4, alpha=0.5,
                color=rank_colors[rk], label=f"{rk} — peak")
    ax.axhline(floor["ho_transforms_v32shared"], ls=":", lw=1.4, color="#777",
               label=f"base floor ({floor['ho_transforms_v32shared']:.2f})")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 18, 33]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel("number of trained control types (nested subsets, r1024)")
    ax.set_ylabel("held-out transforms compliance (8 shared families)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Diversity dose-response")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(args.out_dir / "v33_dose_response.png", dpi=130)
    plt.close(fig)

    print(f"\nwrote -> {args.out_dir}/v33_diversity_money.png, "
          f"v33_capacity_trajectories.png, v33_trajectories.csv, v33_peaks.csv")


if __name__ == "__main__":
    main()
