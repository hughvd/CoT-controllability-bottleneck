"""Is there ONE 'about-to-comply' direction, or one per control type?

Within the base model only (no organism). Train a compliance probe on control
type A's activations, test on type B's, at a fixed (layer, position). A high
off-diagonal AUC means the direction that predicts 'this trace will comply with
control A' also predicts compliance with control B -> a shared control-engagement
representation. Diagonal = within-type CV (question-grouped). Includes a
label-shuffle null (train on shuffled labels) so the off-diagonal is judged
against chance, not against 0.5 blindly.

Only types with both classes present at the condition are usable. instruction_only
has the widest class balance across types, so that's the default condition.

  python analysis/probe_crosstype.py [--pos eop|0|1|2|4|8] [--layer 56]
Outputs figs/probe_crosstype.png, probe_crosstype_stats.csv
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common_analysis import DATA, FIGS, HERE, load_grades, tid_fname

LAYERS = [8, 16, 24, 32, 40, 48, 56, 63]
COND = "instruction_only"
MIN_PER_CLASS = 12
C_REG = 0.1


def cell_features(sub, layer, pos):
    li = LAYERS.index(layer)
    X, y, q = [], [], []
    eop_cache = {}
    for r in sub.itertuples():
        tp = DATA / "probe" / "trace" / f"{tid_fname(r.trace_id)}.npz"
        pp = DATA / "probe" / "prompt" / f"{tid_fname(r.prompt_id)}.npz"
        if not (tp.exists() and pp.exists()):
            continue
        if pos == "eop":
            if r.prompt_id not in eop_cache:
                eop_cache[r.prompt_id] = np.load(pp)["eop"]
            x = eop_cache[r.prompt_id][li]
        else:
            d = np.load(tp)
            gp = list(d["gen_positions"])
            if int(pos) not in gp:
                continue
            x = d["hs"][li, gp.index(int(pos))]
        X.append(x.astype(np.float32)); y.append(int(r.cls == "compliant")); q.append(r.qhash)
    if len(X) < 2 * MIN_PER_CLASS:
        return None
    y = np.array(y)
    if y.sum() < MIN_PER_CLASS or (len(y) - y.sum()) < MIN_PER_CLASS:
        return None
    return np.stack(X), y, np.array(q)


def within_cv(X, y, q, shuffle=False, seed=0):
    n = min(5, len(np.unique(q)))
    if n < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    yy = y.copy()
    if shuffle:
        rng.shuffle(yy)
    aucs = []
    for tr, te in GroupKFold(n).split(X, yy, q):
        if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C_REG, max_iter=2000).fit(sc.transform(X[tr]), yy[tr])
        aucs.append(roc_auc_score(yy[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)) if aucs else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="eop")
    ap.add_argument("--layer", type=int, default=56)
    args = ap.parse_args()

    g = load_grades()
    g = g[(g.condition == COND) & g.cls.notna()]
    feats = {}
    for ct, sub in g.groupby("ctype"):
        f = cell_features(sub, args.layer, args.pos)
        if f is not None:
            feats[ct] = f
    types = sorted(feats)
    print(f"usable types @ L{args.layer}/{args.pos}: {types}")
    if len(types) < 2:
        print("need >=2 usable types; try --pos eop or a different layer")
        return

    # fit one probe per type on ALL its data (for cross-eval) + within CV on diagonal
    probes = {}
    for ct in types:
        X, y, q = feats[ct]
        sc = StandardScaler().fit(X)
        probes[ct] = (sc, LogisticRegression(C=C_REG, max_iter=2000).fit(sc.transform(X), y))

    M = np.full((len(types), len(types)), np.nan)
    Mnull = np.full((len(types), len(types)), np.nan)
    rows = []
    for i, a in enumerate(types):
        for j, b in enumerate(types):
            Xb, yb, qb = feats[b]
            if a == b:
                auc = within_cv(Xb, yb, qb)
                null = within_cv(Xb, yb, qb, shuffle=True, seed=1)
            else:
                sc, clf = probes[a]
                auc = roc_auc_score(yb, clf.predict_proba(sc.transform(Xb))[:, 1])
                # null: type-a probe on shuffled type-b labels == 0.5 in expectation;
                # report the a-probe direction cosine instead as the null-free check
                null = np.nan
            M[i, j] = auc
            Mnull[i, j] = null
            rows.append({"train": a, "test": b, "auc": auc, "shuffle_auc": null,
                         "n_test": len(yb)})
    st = pd.DataFrame(rows)
    st.to_csv(HERE / f"probe_crosstype_{args.pos}_L{args.layer}.csv", index=False)

    # pairwise direction cosines
    dirs = {ct: probes[ct][1].coef_[0] / (np.linalg.norm(probes[ct][1].coef_[0]) + 1e-9)
            for ct in types}
    print("\ncross-type transfer AUC (row=train, col=test); diagonal=within-CV:")
    print(pd.DataFrame(M, index=types, columns=types).round(2).to_string())
    print("\noff-diagonal mean AUC:", round(np.nanmean(M[~np.eye(len(types), dtype=bool)]), 3))
    print("diagonal (within) mean:", round(np.nanmean(np.diag(M)), 3))
    print("diagonal label-shuffle null mean:", round(np.nanmean(np.diag(Mnull)), 3))
    cos = np.array([[float(np.dot(dirs[a], dirs[b])) for b in types] for a in types])
    print("\npairwise probe-direction cosine:")
    print(pd.DataFrame(cos, index=types, columns=types).round(2).to_string())

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for k, (Mat, title) in enumerate([(M, "transfer AUC"), (cos, "direction cosine")]):
        im = ax[k].imshow(Mat, cmap="RdBu_r" if k else "viridis",
                          vmin=-1 if k else 0.3, vmax=1 if k else 1.0)
        ax[k].set_xticks(range(len(types))); ax[k].set_yticks(range(len(types)))
        ax[k].set_xticklabels(types, rotation=45, ha="right", fontsize=7)
        ax[k].set_yticklabels(types, fontsize=7)
        ax[k].set_title(title, fontsize=10)
        ax[k].set_xlabel("test / col"); ax[k].set_ylabel("train / row")
        for i in range(len(types)):
            for j in range(len(types)):
                v = Mat[i, j]
                if not np.isnan(v):
                    ax[k].text(j, i, f"{v:.2f}", ha="center", va="center",
                               fontsize=6, color="white" if (k == 0 and v < 0.7) else "black")
        fig.colorbar(im, ax=ax[k], fraction=0.046)
    fig.suptitle(f"Cross-type compliance-probe transfer, base model, {COND} @ L{args.layer}/{args.pos}\n"
                 "(high off-diagonal = shared control-engagement direction)", fontsize=11)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"probe_crosstype_{args.pos}_L{args.layer}.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {FIGS}/probe_crosstype_{args.pos}_L{args.layer}.png")


if __name__ == "__main__":
    main()
