"""Fate-probe training (local, sklearn) — when is a trace's compliance readable?

Reads the probe_pass.py activations (data/probe/) + grades.csv. For each
(ctype, condition, layer, position) trains an L2 logistic probe
compliant-vs-noncompliant with question-grouped 5-fold CV and reports AUC.

Positions: "eop" = end-of-prompt / think-token state (identical across the 10
samples of a prompt — this probe measures how much of fate is decided BEFORE the
first token); gen positions 0,1,2,4,...,128 and "s1" (end of first CoT sentence)
diverge per trace. The headline figure is AUC vs position: where does per-trace
fate become readable?

  python analysis/probe_analysis.py
Outputs figs/probe_auc.png, probe_stats.csv
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common_analysis import COND_LABEL, DATA, FIGS, HERE, load_grades, tid_fname

PROBE_PROMPT = DATA / "probe" / "prompt"
PROBE_TRACE = DATA / "probe" / "trace"
LAYERS = [8, 16, 24, 32, 40, 48, 56, 63]
LAYER_PLOT = [16, 32, 48, 63]
GEN_LADDER = [0, 1, 2, 4, 8, 16, 32, 64, 128]
MIN_PER_CLASS = 12
C_REG = 0.1


def load_features() -> pd.DataFrame:
    """One row per graded trace with class label; feature arrays attached lazily."""
    g = load_grades()
    g = g[g.cls.notna()].copy()
    rows = []
    for _, r in g.iterrows():
        tp = PROBE_TRACE / f"{tid_fname(r.trace_id)}.npz"
        pp = PROBE_PROMPT / f"{tid_fname(r.prompt_id)}.npz"
        if tp.exists() and pp.exists():
            rows.append({"trace_id": r.trace_id, "prompt_id": r.prompt_id,
                         "ctype": r.ctype, "condition": r.condition, "qhash": r.qhash,
                         "y": int(r.cls == "compliant"), "tp": tp, "pp": pp})
    return pd.DataFrame(rows)


def load_cell(sub: pd.DataFrame) -> dict:
    """Load every npz in this (ctype, condition) cell ONCE: trace_id ->
    (gen_positions, hs [L, n_pos, d], s1_pos, eop [L, d])."""
    eop_cache: dict[str, np.ndarray] = {}
    out = {}
    for r in sub.itertuples():
        if r.prompt_id not in eop_cache:
            eop_cache[r.prompt_id] = np.load(r.pp)["eop"]
        d = np.load(r.tp)
        gp = list(d["gen_positions"])
        out[r.trace_id] = (gp, d["hs"][:], min(int(d["sent1_end"]), gp[-1]),
                           eop_cache[r.prompt_id])
    return out


def get_x(cell: dict, trace_id: str, layer: int, pos) -> np.ndarray | None:
    gp, hs, s1, eop = cell[trace_id]
    li = LAYERS.index(layer)
    if pos == "eop":
        return eop[li].astype(np.float32)
    p = s1 if pos == "s1" else pos
    if p not in gp:
        return None
    return hs[li, gp.index(p)].astype(np.float32)


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return np.nan
    aucs = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C_REG, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)) if aucs else np.nan


def main() -> None:
    df = load_features()
    print(f"labeled traces with features: {len(df)}")
    positions = ["eop"] + GEN_LADDER + ["s1"]
    stats = []
    for (ct, cond), sub in df.groupby(["ctype", "condition"]):
        vc = sub.y.value_counts()
        if len(vc) < 2 or vc.min() < MIN_PER_CLASS:
            continue
        print(f"{ct}/{cond}: n={len(sub)} ({vc.get(1,0)} compliant / {vc.get(0,0)} non)")
        cell = load_cell(sub)
        for layer in LAYERS:
            for pos in positions:
                feats = [(get_x(cell, r.trace_id, layer, pos), r.y, r.qhash)
                         for r in sub.itertuples()]
                feats = [(x, y, q) for x, y, q in feats if x is not None]
                if len(feats) < 2 * MIN_PER_CLASS:
                    continue
                X = np.stack([f[0] for f in feats])
                y = np.array([f[1] for f in feats])
                q = np.array([f[2] for f in feats])
                if len(np.unique(y)) < 2:
                    continue
                stats.append({"ctype": ct, "condition": cond, "layer": layer,
                              "pos": str(pos), "n": len(y), "n_pos": int(y.sum()),
                              "auc": cv_auc(X, y, q)})
        done_ct = [s for s in stats if s["ctype"] == ct and s["condition"] == cond]
        best = max((s for s in done_ct if not np.isnan(s["auc"])),
                   key=lambda s: s["auc"], default=None)
        if best:
            print(f"  best: layer {best['layer']} @ {best['pos']} AUC={best['auc']:.3f}")
    st = pd.DataFrame(stats)
    st.to_csv(HERE / "probe_stats.csv", index=False)
    print(f"wrote {HERE/'probe_stats.csv'} ({len(st)} rows)")

    # figure: AUC vs position, rows = ctype, cols = condition, lines = layers
    if not len(st):
        return
    pos_order = ["eop"] + [str(g) for g in GEN_LADDER] + ["s1"]
    cells = st.groupby(["ctype", "condition"]).size().reset_index()[["ctype", "condition"]]
    ctypes = sorted(cells.ctype.unique())
    conds = [c for c in COND_LABEL if c in set(cells.condition)]
    fig, axes = plt.subplots(len(ctypes), len(conds),
                             figsize=(3.4 * len(conds), 2.7 * len(ctypes)),
                             sharey=True, squeeze=False)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(LAYER_PLOT)))
    for ti, ct in enumerate(ctypes):
        for ci, cond in enumerate(conds):
            ax = axes[ti, ci]
            sub = st[(st.ctype == ct) & (st.condition == cond)]
            for li, layer in enumerate(LAYER_PLOT):
                s = sub[sub.layer == layer].set_index("pos").reindex(pos_order)
                ax.plot(range(len(pos_order)), s.auc.values, "o-", ms=3, lw=1.5,
                        color=cmap[li], label=f"L{layer}")
            ax.axhline(0.5, color="#999", lw=0.8, ls="--")
            ax.set_xticks(range(len(pos_order)))
            ax.set_xticklabels(pos_order, fontsize=7, rotation=45)
            ax.grid(alpha=0.3)
            ax.set_ylim(0.35, 1.02)
            if ci == 0:
                ax.set_ylabel(f"{ct}\nCV AUC", fontsize=9)
            if ti == 0:
                ax.set_title(COND_LABEL.get(cond, cond), fontsize=10)
            if ti == 0 and ci == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Fate probe: compliance readable from the residual stream at position x\n"
                 "(eop = end-of-prompt/think token; question-grouped 5-fold CV, "
                 f"L2 logistic C={C_REG})", fontsize=11)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "probe_auc.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIGS/'probe_auc.png'}")


if __name__ == "__main__":
    main()
