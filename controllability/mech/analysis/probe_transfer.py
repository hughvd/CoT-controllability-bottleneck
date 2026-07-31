"""4a — compliance-probe cross-model transfer (base <-> v34 organism).

Question: does the organism reuse the base model's internal compliance signal
(elicitation) or build a new one (installation)? Train an L2 logistic probe on one
model's residual activations, evaluate on the other's, at matched (ctype, layer,
position). Directions are also compared by cosine similarity.

Data: base side = data/probe/ + grades.csv (existing);
      organism side = data/probe_org/ + data/kl/org_grades.csv (probe_pass.py
      --model merged --traces org_traces.jsonl). Organism condition set is
      {no_instruction, instruction_only} only, so transfer is evaluated at
      instruction_only (organism compliance ~0.87-1.0 -> classes are imbalanced;
      cells without >=8 per class are skipped and reported).

  python analysis/probe_transfer.py
Outputs figs/probe_transfer.png, probe_transfer_stats.csv
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common_analysis import BINARY_TYPES, DATA, FIGS, HERE, SOFT_TYPES, load_grades, tid_fname

LAYERS = [8, 16, 24, 32, 40, 48, 56, 63]
POSITIONS = ["eop", 0, 1, 2, 4, 8]
COND = "instruction_only"
MIN_PER_CLASS = 8
C_REG = 0.1


def load_side(probe_root, grades: pd.DataFrame) -> pd.DataFrame:
    g = grades[(grades.condition == COND) & grades.cls.notna()].copy()
    rows = []
    for _, r in g.iterrows():
        tp = probe_root / "trace" / f"{tid_fname(r.trace_id)}.npz"
        pp = probe_root / "prompt" / f"{tid_fname(r.prompt_id)}.npz"
        if tp.exists() and pp.exists():
            rows.append({"trace_id": r.trace_id, "prompt_id": r.prompt_id,
                         "ctype": r.ctype, "qhash": r.qhash,
                         "y": int(r.cls == "compliant"), "tp": tp, "pp": pp})
    return pd.DataFrame(rows)


def org_grades() -> pd.DataFrame:
    g = pd.read_csv(DATA / "kl" / "org_grades.csv")
    g = g[g.empty_thinking == 0].copy()
    g["cls"] = None
    b = g.ctype.isin(BINARY_TYPES)
    g.loc[b & (g.compliance_score >= 1.0), "cls"] = "compliant"
    g.loc[b & (g.compliance_score <= 0.0), "cls"] = "noncompliant"
    s = g.ctype.isin(SOFT_TYPES)
    g.loc[s & (g.compliance_score >= 0.75), "cls"] = "compliant"
    g.loc[s & (g.compliance_score <= 0.25), "cls"] = "noncompliant"
    return g


def features(sub: pd.DataFrame, layer: int, pos) -> tuple[np.ndarray, ...] | None:
    li = LAYERS.index(layer)
    X, y, q = [], [], []
    eop_cache: dict[str, np.ndarray] = {}
    for r in sub.itertuples():
        if pos == "eop":
            if r.prompt_id not in eop_cache:
                eop_cache[r.prompt_id] = np.load(r.pp)["eop"]
            x = eop_cache[r.prompt_id][li]
        else:
            d = np.load(r.tp)
            gp = list(d["gen_positions"])
            if pos not in gp:
                continue
            x = d["hs"][li, gp.index(pos)]
        X.append(x.astype(np.float32)); y.append(r.y); q.append(r.qhash)
    if len(X) < 2 * MIN_PER_CLASS:
        return None
    y = np.array(y)
    if y.sum() < MIN_PER_CLASS or (len(y) - y.sum()) < MIN_PER_CLASS:
        return None
    return np.stack(X), y, np.array(q)


def fit_probe(X, y):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C_REG, max_iter=2000).fit(sc.transform(X), y)
    return sc, clf


def within_auc(X, y, q) -> float:
    n = min(5, len(np.unique(q)))
    if n < 2:
        return np.nan
    aucs = []
    for tr, te in GroupKFold(n_splits=n).split(X, y, q):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc, clf = fit_probe(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)) if aucs else np.nan


def main() -> None:
    base = load_side(DATA / "probe", load_grades())
    org = load_side(DATA / "probe_org", org_grades())
    print(f"base side: {len(base)} traces | org side: {len(org)} traces")
    print("org class counts:\n", org.groupby("ctype").y.agg(["count", "sum"]))

    stats = []
    for ct in sorted(set(base.ctype) & set(org.ctype)):
        for layer in LAYERS:
            for pos in POSITIONS:
                fb = features(base[base.ctype == ct], layer, pos)
                fo = features(org[org.ctype == ct], layer, pos)
                if fb is None or fo is None:
                    continue
                (Xb, yb, qb), (Xo, yo, qo) = fb, fo
                scb, pb = fit_probe(Xb, yb)
                sco, po = fit_probe(Xo, yo)
                cos = float(np.dot(pb.coef_[0], po.coef_[0])
                            / (np.linalg.norm(pb.coef_[0]) * np.linalg.norm(po.coef_[0]) + 1e-9))
                stats.append({
                    "ctype": ct, "layer": layer, "pos": str(pos),
                    "n_base": len(yb), "n_org": len(yo),
                    "auc_base_cv": within_auc(Xb, yb, qb),
                    "auc_org_cv": within_auc(Xo, yo, qo),
                    "auc_base_to_org": roc_auc_score(yo, pb.predict_proba(scb.transform(Xo))[:, 1]),
                    "auc_org_to_base": roc_auc_score(yb, po.predict_proba(sco.transform(Xb))[:, 1]),
                    "dir_cosine": cos,
                })
    st = pd.DataFrame(stats)
    st.to_csv(HERE / "probe_transfer_stats.csv", index=False)
    print(f"wrote {HERE/'probe_transfer_stats.csv'} ({len(st)} rows)")
    if not len(st):
        print("NO transferable cells (organism too compliant?) — see class counts above")
        return

    # figure: per ctype, best layer/pos by base CV: 4 bars + cosine annotation
    fig, ax = plt.subplots(figsize=(1.6 + 2.4 * st.ctype.nunique(), 4.2))
    xt, xl, pos_ = [], [], 0
    for ct, sub in st.groupby("ctype"):
        b = sub.loc[sub.auc_base_cv.idxmax()]
        vals = [b.auc_base_cv, b.auc_base_to_org, b.auc_org_cv, b.auc_org_to_base]
        cols = ["#2171b5", "#6baed6", "#d94801", "#fdae6b"]
        for k, (v, c) in enumerate(zip(vals, cols)):
            ax.bar(pos_ + k, v, color=c, width=0.85)
        ax.text(pos_ + 1.5, 1.01, f"L{int(b.layer)}@{b.pos}\ncos={b.dir_cosine:.2f}",
                ha="center", fontsize=7)
        xt.append(pos_ + 1.5); xl.append(ct)
        pos_ += 5
    ax.axhline(0.5, color="#999", ls="--", lw=0.8)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=8)
    ax.set_ylim(0.3, 1.08); ax.set_ylabel("AUC")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in
                       ["#2171b5", "#6baed6", "#d94801", "#fdae6b"]],
              labels=["base within (CV)", "base→org", "org within (CV)", "org→base"],
              fontsize=8, loc="lower right")
    ax.set_title("Compliance-probe transfer, instruction_only @ best base layer/pos\n"
                 "(transfer ≈ within ⇒ shared compliance direction ⇒ elicitation)",
                 fontsize=10)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "probe_transfer.png", dpi=150, bbox_inches="tight")
    print(f"wrote {FIGS/'probe_transfer.png'}")


if __name__ == "__main__":
    main()
