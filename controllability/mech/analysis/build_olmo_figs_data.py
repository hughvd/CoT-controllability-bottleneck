"""Assemble tidy figure data for the OLMo-3-32B RL-axis writeup.

ONE loader feeding BOTH the static matplotlib figures and the local HTML viewer.
Reads the canonical 26-type controllability runs, the mon suite gmean summaries,
and the mech entropy/mass/KL trends; writes tidy CSVs + a single viewer.json.

Controllability trajectory (26-type, `family` col = control type, `modality` =
transform/encoding/language/style/suppression):
    SFT -> DPO(rl_start) -> RLVR(rlvr_final) -> ext_third -> ext_twothirds -> ext_final
SFT is the `ent_sft` run (only earlier ckpt with a 26-type eval); base has none.
Monitorability (gpt5mini battery) exists from rl_start onward only.

  python analysis/build_olmo_figs_data.py
Outputs under analysis/olmo_rl/fig_data/: overall.csv, by_ctrl_type.csv,
by_mon_dataset.csv, viewer.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _capability import uncontrolled_gpqa  # noqa: E402
REPO = HERE.parent.parent.parent                    # repo root
CTRL = REPO / "controllability" / "results"
MON = REPO / "results" / "monitorability"
MECH = HERE / "olmo_rl"
OUT = MECH / "fig_data"
OUT.mkdir(parents=True, exist_ok=True)

# CANONICAL checkpoints, IDENTICAL 26-type universe, comparable metrics.
# SFT now has its own full-26 run (olmo32_ctrl_sft) so it joins as a 6th point (rl_index
# -1, pre-RL; mon_label None => SFT has no monitorability). ext_third is GENUINE step_0800
# (the step_0750 branch was rlvr_final upstream — see olmo_figure_suite memory).
# (tag, display, rl_index, ctrl_csv, mon_label)  — mon_label None => no monitorability
CKPTS = [
    ("sft",           "SFT",       -1, CTRL / "olmo32_ctrl_sft" / "per_sample.csv",           None),
    ("rl_start",      "DPO",        0, CTRL / "olmo32_ctrl_rl_start" / "per_sample.csv",      "olmo32_rl_start"),
    ("rlvr_final",    "RLVR",       1, CTRL / "olmo32_ctrl_rlvr_final" / "per_sample.csv",    "olmo32_rlvr_final"),
    ("ext_third",     "ext-1/3",    2, CTRL / "olmo32_ctrl_ext_third" / "per_sample.csv",     "olmo32_ext_third"),
    ("ext_twothirds", "ext-2/3",    3, CTRL / "olmo32_ctrl_ext_twothirds" / "per_sample.csv", "olmo32_ext_twothirds"),
    ("ext_final",     "ext-final",  4, CTRL / "olmo32_ctrl_ext_final" / "per_sample.csv",     "olmo32_ext_final"),
]

# (legacy) SFT's earlier broader-universe run — kept only for the shared-22 comparison.
SFT_CKPT = ("sft", "SFT", -1, CTRL / "organism_2_olmo32_ent_sft_base_diverse" / "per_sample.csv")

INTERVENTION_DS = ["gpqa", "memory_advice", "scruples_suggest_wrong", "wmdp_sandbagging"]


def _boot_ci(x, n_boot=1000, seed=0):
    """Simple mean 95% CI via bootstrap over rows (question-level clustering not
    available here — rows are already per-(question,marker,phrasing) means)."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_ctrl(csv):
    df = pd.read_csv(csv, usecols=["family", "modality", "compliance_score",
                                   "compliance_hard", "correct", "reasoning", "response"])
    df["compliance_hard"] = df["compliance_hard"].astype(float)
    df["correct"] = df["correct"].astype(float)
    # capability = accuracy-under-control among ANSWERED traces (excl. truncated —
    # RL inflates reasoning 3-4x, so ~30% never emit a letter at the token cap).
    df["answered"] = df["response"].astype(str).str.contains(r"\b[A-D]\b", regex=True, na=False)
    df["reas_len"] = df["reasoning"].astype(str).str.len()
    return df


def _cap_answered(df, seed):
    a = df[df["answered"]]
    return _boot_ci(a["correct"], seed=seed)


def mon_by_dataset(mon_label):
    """Per-(scope,dataset) intervention gmean + process/outcome gmean (cot_only)."""
    rows = []
    gs = MON / mon_label / "mon-gpt5mini" / "gmean_summary.csv"
    if gs.exists():
        g = pd.read_csv(gs)
        for _, r in g.iterrows():
            ds = str(r["dataset"]).split("/")[-1].replace(".jsonl", "")
            rows.append({"scope": r["scope"], "archetype": "intervention",
                         "dataset": ds, "gmean": r["gmean_mean"]})
    for arch, fn in [("process", "process_metrics.json"), ("outcome", "outcome_metrics.json")]:
        fp = MON / mon_label / "mon-gpt5mini" / fn
        if fp.exists():
            m = json.load(open(fp))
            for scope in ("cot_only", "answer_only", "all_messages"):
                sc = m.get(scope, {}).get("overall", {})
                if "gmean" in sc:
                    rows.append({"scope": scope, "archetype": arch,
                                 "dataset": arch, "gmean": sc["gmean"]})
    return pd.DataFrame(rows)


def main():
    overall, by_type, by_mon = [], [], []

    for tag, disp, idx, csv, mon_label in CKPTS:
        df = load_ctrl(csv)
        # ---- overall controllability + capability ----
        soft, soft_lo, soft_hi = _boot_ci(df["compliance_score"], seed=idx + 1)
        hard, hard_lo, hard_hi = _boot_ci(df["compliance_hard"], seed=idx + 200)
        # capability = UNCONTROLLED gpqa accuracy (the "did RL/intervention nuke it?" check);
        # keep accuracy-under-control separately for reference. SFT has no mon run -> NaN.
        cap, cap_lo, cap_hi = uncontrolled_gpqa(mon_label) if mon_label else (np.nan, np.nan, np.nan)
        cap_uc, _, _ = _cap_answered(df, 100 + idx)   # accuracy-under-control (answered-only)
        rec = {"tag": tag, "checkpoint": disp, "rl_index": idx,
               "ctrl_soft": soft, "ctrl_soft_lo": soft_lo, "ctrl_soft_hi": soft_hi,
               "ctrl_hard": hard, "ctrl_hard_lo": hard_lo, "ctrl_hard_hi": hard_hi,
               "capability": cap, "cap_lo": cap_lo, "cap_hi": cap_hi,
               "cap_under_control": cap_uc,
               "reas_len": df["reas_len"].mean(), "no_answer_rate": 1 - df["answered"].mean()}

        # ---- per control-type (family) + modality ----
        gt = df.groupby(["modality", "family"]).agg(
            ctrl_soft=("compliance_score", "mean"),
            ctrl_hard=("compliance_hard", "mean"),
            capability=("correct", "mean"), n=("correct", "size")).reset_index()
        gt.insert(0, "checkpoint", disp); gt.insert(1, "rl_index", idx)
        by_type.append(gt)
        # modality-level ctrl (for the headline decomposition: transforms up / language down)
        gm = df.groupby("modality").compliance_score.mean()
        for mod, v in gm.items():
            rec[f"ctrl_{mod}"] = v

        # ---- monitorability ----
        if mon_label:
            md = mon_by_dataset(mon_label)
            md.insert(0, "checkpoint", disp); md.insert(1, "rl_index", idx)
            by_mon.append(md)
            iv = md[(md.archetype == "intervention") & (md.scope == "cot_only")]
            rec["mon_intervention"] = iv.gmean.mean()
            gp = iv[iv.dataset == "gpqa"]
            rec["mon_gpqa"] = gp.gmean.mean() if len(gp) else np.nan
            proc = md[(md.archetype == "process") & (md.scope == "cot_only")]
            rec["mon_process"] = proc.gmean.mean() if len(proc) else np.nan
            outc = md[(md.archetype == "outcome") & (md.scope == "cot_only")]
            rec["mon_outcome"] = outc.gmean.mean() if len(outc) else np.nan
            # CoT-specific signal = cot_only - answer_only (intervention mean)
            ao = md[(md.archetype == "intervention") & (md.scope == "answer_only")]
            rec["mon_cot_lift"] = iv.gmean.mean() - ao.gmean.mean()
        overall.append(rec)

    ov = pd.DataFrame(overall)
    bt = pd.concat(by_type, ignore_index=True)
    bm = pd.concat(by_mon, ignore_index=True)
    ov.to_csv(OUT / "overall.csv", index=False)
    bt.to_csv(OUT / "by_ctrl_type.csv", index=False)
    bm.to_csv(OUT / "by_mon_dataset.csv", index=False)

    # ---- shared-family (22) trajectory INCLUDING SFT, apples-to-apples ----
    sft_df = load_ctrl(SFT_CKPT[3])
    canon0 = load_ctrl(CKPTS[0][3])
    shared = sorted(set(sft_df.family) & set(canon0.family))   # 22 families, no styles
    shared_rows = []
    for tag, disp, idx, csv in [SFT_CKPT] + [(t, d, i, c) for t, d, i, c, _ in CKPTS]:
        d = load_ctrl(csv)
        d = d[d.family.isin(shared)]
        s, lo, hi = _boot_ci(d["compliance_score"], seed=idx + 50)
        c, clo, chi = _cap_answered(d, idx + 150)   # answered-only
        shared_rows.append({"tag": tag, "checkpoint": disp, "rl_index": idx,
                            "n_families": len(shared), "ctrl_soft": s, "ctrl_soft_lo": lo,
                            "ctrl_soft_hi": hi, "capability": c, "cap_lo": clo, "cap_hi": chi,
                            "reas_len": d["reas_len"].mean(), "no_answer_rate": 1 - d["answered"].mean()})
    sh = pd.DataFrame(shared_rows)
    sh.to_csv(OUT / "shared22.csv", index=False)

    # ---- viewer.json: everything the HTML needs, nested ----
    viewer = {
        "checkpoints": [{"tag": t, "label": d, "rl_index": i, "has_mon": ml is not None}
                        for t, d, i, _, ml in CKPTS],
        "overall": ov.replace({np.nan: None}).to_dict("records"),
        "ctrl_by_type": bt.replace({np.nan: None}).to_dict("records"),
        "mon_by_dataset": bm.replace({np.nan: None}).to_dict("records"),
        "shared22": sh.replace({np.nan: None}).to_dict("records"),
    }
    json.dump(viewer, open(OUT / "viewer.json", "w"), indent=1)

    print("overall.csv (5 canonical, 26-type):\n", ov.round(3).to_string(index=False))
    print("\nshared22.csv (SFT + 5, 22 shared families):\n", sh.round(3).to_string(index=False))
    print(f"\nby_ctrl_type: {len(bt)} rows ({bt.checkpoint.nunique()} ckpts x {bt.family.nunique()} families)")
    print(f"by_mon_dataset: {len(bm)} rows | viewer.json -> {OUT/'viewer.json'}")


if __name__ == "__main__":
    main()
