"""Analyze the OLMo-3-32B RL-axis mech sweep (run locally after syncing the pod).

Reads, per checkpoint tag along the RL ladder:
  mech/data/olmo_rl/<tag>/grades.csv          (olmo_rl_grade.py output)
  mech/data/olmo_rl/<tag>/stats/<tid>.npz     (first-token entropy + opening masses)
  mech/data/olmo_rl/kl/<A>__<B>/<src>/<tid>.npz  (adjacent-checkpoint KL)

Produces, keyed by the RL axis (x = ordered checkpoints):
  1. controllability trend  — mean compliance (soft/hard) + accuracy per ctype,
     instruction_only vs no_instruction floor.
  2. first-token entropy    — entropy at generated position 0 (the first reasoning
     token, since OLMo force-opens <think>), instruction_only, split compliant vs
     failing where both classes exist.
  3. opening-mass trend     — probability mass on instruction-consistent tokens at
     position 0 and averaged over the first 16, per ctype (H1: RL moves mass toward
     compliant openings even under no_instruction).
  4. adjacent-KL front-loading — per-position KL between consecutive checkpoints and
     the cumulative share reached by token k (is RL's change opening-localized?).

Outputs: analysis/olmo_rl/{ctrl_trend,entropy_trend,opening_mass,kl_frontload}.csv
and figs/olmo_rl_*.png. Everything guards for missing tags/pairs (partial runs OK).

  python mech/analysis/olmo_rl_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MECH = HERE.parent
sys.path.insert(0, str(MECH))
ODATA = MECH / "data" / "olmo_rl"
OUT = HERE / "olmo_rl"
FIGS = HERE / "figs"
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# RL axis order + x positions (only tags with data are plotted)
TAG_ORDER = ["rl_start", "rlvr_final", "ext_third", "ext_twothirds", "ext_final"]
TAG_LABEL = {"rl_start": "DPO\n(RL 0)", "rlvr_final": "RLVR\nfinal",
             "ext_third": "ext ⅓", "ext_twothirds": "ext ⅔", "ext_final": "ext\nfinal"}
CTYPES = ["uppercase_thinking", "leetspeak", "spanish", "omit_word"]
# opening-mass field matching each ctype's instruction target
MASS_FIELD = {"uppercase_thinking": "mass_upper", "leetspeak": "mass_digit",
              "spanish": "mass_spanish", "omit_word": "p_banned"}
# Okabe-Ito CVD-safe (validated ΔE elsewhere in this repo)
BLUE, VERM, GREEN, ORANGE = "#0072B2", "#D55E00", "#009E73", "#E69F00"
CTYPE_COLOR = dict(zip(CTYPES, [BLUE, VERM, GREEN, ORANGE]))
INK, MUTED, GRID = "#1a1a2e", "#6b6b7b", "#e4e4ea"


def present_tags() -> list[str]:
    return [t for t in TAG_ORDER if (ODATA / t / "grades.csv").exists()
            or (ODATA / t / "stats").exists()]


def load_grades(tag: str) -> pd.DataFrame | None:
    fp = ODATA / tag / "grades.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    df["tag"] = tag
    return df


# --------------------------------------------------------------------------- #
# 1. controllability trend
# --------------------------------------------------------------------------- #

def ctrl_trend(tags: list[str]) -> pd.DataFrame:
    rows = []
    for tag in tags:
        g = load_grades(tag)
        if g is None:
            continue
        for ctype in CTYPES:
            for cond in ["instruction_only", "no_instruction"]:
                sub = g[(g.ctype == ctype) & (g.condition == cond)]
                if not len(sub):
                    continue
                rows.append({
                    "tag": tag, "ctype": ctype, "condition": cond, "n": len(sub),
                    "soft": sub.compliance_score.mean(),
                    "hard": sub.compliance_hard.mean(),
                    "accuracy": sub.correct.mean(),
                })
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / "ctrl_trend.csv", index=False)
        print(f"[ctrl] {OUT/'ctrl_trend.csv'}")
    return df


# --------------------------------------------------------------------------- #
# stats loading: first-token entropy + opening masses per trace
# --------------------------------------------------------------------------- #

def load_stats(tag: str) -> pd.DataFrame | None:
    sdir = ODATA / tag / "stats"
    if not sdir.exists():
        return None
    recs = []
    for fp in sorted(sdir.glob("*.npz")):
        d = np.load(fp, allow_pickle=True)
        ent = d["entropy"].astype(np.float32)
        if not len(ent):
            continue
        ctype = str(d["ctype"]); cond = str(d["condition"])
        mf = MASS_FIELD[ctype]
        mass = d[mf].astype(np.float32) if mf in d.files else np.full(len(ent), np.nan)
        # trace_id: filename is prompt_id parts + sX (| -> _)
        tid = fp.stem
        recs.append({
            "tag": tag, "trace_id_file": tid, "ctype": ctype, "condition": cond,
            "ent0": float(ent[0]),
            "ent_mean4": float(ent[:4].mean()),
            "ent_mean16": float(ent[:16].mean()),
            "mass0": float(mass[0]) if len(mass) else np.nan,
            "mass_mean16": float(np.nanmean(mass[:16])) if len(mass) else np.nan,
        })
    return pd.DataFrame(recs) if recs else None


def _grade_key(df: pd.DataFrame) -> pd.Series:
    # grades trace_id "ctype|cond|qhash|sX" -> file stem "ctype_cond_qhash_sX"
    return df.trace_id.str.replace("|", "_", regex=False)


def entropy_and_mass(tags: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ent_rows, mass_rows = [], []
    for tag in tags:
        st = load_stats(tag)
        if st is None:
            continue
        g = load_grades(tag)
        if g is not None:
            g = g.assign(key=_grade_key(g))[["key", "compliance_hard"]]
            st = st.merge(g, left_on="trace_id_file", right_on="key", how="left")
        else:
            st["compliance_hard"] = np.nan
        for ctype in CTYPES:
            for cond in ["instruction_only", "no_instruction"]:
                sub = st[(st.ctype == ctype) & (st.condition == cond)]
                if not len(sub):
                    continue
                base = {"tag": tag, "ctype": ctype, "condition": cond, "n": len(sub)}
                ent_rows.append({**base,
                                 "ent0": sub.ent0.mean(),
                                 "ent_mean16": sub.ent_mean16.mean()})
                mass_rows.append({**base,
                                  "mass0": sub.mass0.mean(),
                                  "mass_mean16": sub.mass_mean16.mean()})
                # compliant vs failing split (instruction_only only, where classes exist)
                if cond == "instruction_only" and sub.compliance_hard.notna().any():
                    for lab, val in [("compliant", 1), ("failing", 0)]:
                        s2 = sub[sub.compliance_hard == val]
                        if len(s2) >= 5:
                            ent_rows.append({**base, "condition": f"instr:{lab}",
                                             "n": len(s2), "ent0": s2.ent0.mean(),
                                             "ent_mean16": s2.ent_mean16.mean()})
    ent = pd.DataFrame(ent_rows); mass = pd.DataFrame(mass_rows)
    if len(ent):
        ent.to_csv(OUT / "entropy_trend.csv", index=False); print(f"[entropy] {OUT/'entropy_trend.csv'}")
    if len(mass):
        mass.to_csv(OUT / "opening_mass.csv", index=False); print(f"[mass] {OUT/'opening_mass.csv'}")
    return ent, mass


# --------------------------------------------------------------------------- #
# 4. adjacent-KL front-loading
# --------------------------------------------------------------------------- #

def kl_frontload() -> pd.DataFrame:
    kroot = ODATA / "kl"
    if not kroot.exists():
        return pd.DataFrame()
    rows = []
    for pair_dir in sorted(kroot.iterdir()):
        if not pair_dir.is_dir():
            continue
        pair = pair_dir.name  # "<A>__<B>"
        for ctype in CTYPES:
            profiles = []
            for src_dir in pair_dir.iterdir():
                if not src_dir.is_dir():
                    continue
                for fp in src_dir.glob("*.npz"):
                    d = np.load(fp, allow_pickle=True)
                    if str(d["ctype"]) != ctype:
                        continue
                    kl = d["kl_a_b"].astype(np.float32)
                    if len(kl) > 1:            # drop pos-0 artifacts by trimming below
                        profiles.append(kl)
            if not profiles:
                continue
            L = min(len(p) for p in profiles)
            arr = np.stack([p[:L] for p in profiles])   # [traces, L]
            mean_prof = arr.mean(0)
            total = mean_prof.sum()
            cum = np.cumsum(mean_prof) / total if total > 0 else np.zeros(L)
            # token index reaching 50% / 90% of cumulative divergence
            k50 = int(np.searchsorted(cum, 0.5)) if total > 0 else -1
            k90 = int(np.searchsorted(cum, 0.9)) if total > 0 else -1
            rows.append({
                "pair": pair, "ctype": ctype, "n_traces": len(profiles),
                "kl_total": float(total), "kl_pos0": float(mean_prof[0]),
                "kl_mean": float(mean_prof.mean()), "k50": k50, "k90": k90,
                "frac_first8": float(cum[min(7, L - 1)]) if total > 0 else np.nan,
            })
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(OUT / "kl_frontload.csv", index=False)
        print(f"[kl] {OUT/'kl_frontload.csv'}")
    return df


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #

def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_axisbelow(True); ax.grid(True, color=GRID, lw=0.6)


def _xaxis(ax, tags):
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([TAG_LABEL.get(t, t) for t in tags], fontsize=8)


def plot_trend(df, value, tags, title, fname, cond="instruction_only", ylim=(0, 1)):
    if not len(df):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    for ctype in CTYPES:
        ys = []
        for t in tags:
            sub = df[(df.tag == t) & (df.ctype == ctype) & (df.condition == cond)]
            ys.append(sub[value].mean() if len(sub) else np.nan)
        ax.plot(range(len(tags)), ys, "-o", color=CTYPE_COLOR[ctype], lw=2, ms=6,
                label=ctype)
    _style(ax); _xaxis(ax, tags)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11, color=INK)
    ax.set_xlabel("RL checkpoint", fontsize=9, color=MUTED)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / fname, bbox_inches="tight")
    plt.close(fig); print(f"[fig] {FIGS/fname}")


def main() -> None:
    tags = present_tags()
    if not tags:
        print("No olmo_rl data found under", ODATA, "— sync the pod first.")
        return
    print("checkpoints with data:", tags)
    ctrl = ctrl_trend(tags)
    ent, mass = entropy_and_mass(tags)
    kl = kl_frontload()

    plot_trend(ctrl, "soft", tags, "Controllability (soft) vs RL — instruction_only",
               "olmo_rl_ctrl_soft.png")
    plot_trend(ctrl, "soft", tags, "Spontaneous compliance floor (no_instruction)",
               "olmo_rl_ctrl_floor.png", cond="no_instruction", ylim=(0, 0.3))
    plot_trend(ent, "ent0", tags, "First reasoning-token entropy vs RL (instruction_only)",
               "olmo_rl_ent0.png", ylim=None)
    plot_trend(mass, "mass0", tags, "Opening compliant-token mass vs RL (instruction_only)",
               "olmo_rl_mass0.png", ylim=None)
    plot_trend(mass, "mass0", tags, "Opening compliant-token mass — no_instruction floor",
               "olmo_rl_mass0_floor.png", cond="no_instruction", ylim=None)

    if len(kl):
        print("\n=== adjacent-checkpoint KL front-loading ===")
        print(kl.to_string(index=False))
    print("\nDone. CSVs in", OUT, "figs in", FIGS)


if __name__ == "__main__":
    main()
