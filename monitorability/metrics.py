"""Official intervention g-mean metric, wrapped for our DataFrames.

Delegates to the submodule's compute_intervention_metrics, which drives
metric/intervention_gmean_metric.py (per-instance TPR bound, per-arm TNR,
cross-fit bootstrap, Wald eligibility) exactly as in the paper.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .suite import SCOPES, scaffold


def intervention_gmean(
    df: pd.DataFrame,
    *,
    n_bootstrap: int = 200,
    selection_frac: float = 0.6,
    group_keep_frac: float = 1.0,
    wald_k: float = 1.96,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute the official g-mean per dataset, for each scope present in df.

    df needs columns: dataset, instance_id, x, y, and z_{scope} for at least
    one scope (cache_to_dataframe produces this). Mirrors the scaffold's
    compute_intervention_metrics but tolerates a subset of scopes.
    """
    cfg = scaffold.BootstrapConfig(
        n_bootstrap=n_bootstrap,
        group_keep_frac=group_keep_frac,
        selection_frac=selection_frac,
        wald_k=wald_k,
        random_state=seed,
    )
    out: dict[str, Any] = {}
    for scope in SCOPES:
        z_col = f"z_{scope}"
        if z_col not in df.columns:
            continue
        metric_df = (
            df[["dataset", "instance_id", "x", "y", z_col]]
            .rename(columns={z_col: "z"})
            .copy()
        )
        metric_df["x"] = metric_df["x"].astype(int)
        final_summary, per_bootstrap, per_instance = scaffold.bootstrapped_gmean_metric(
            metric_df,
            group_cols=["dataset", "instance_id"],
            final_groups=["dataset"],
            x_col="x",
            y_col="y",
            z_col="z",
            bootstrap=cfg,
        )
        out[scope] = {
            "final_summary": scaffold._records(final_summary),
            "per_bootstrap": scaffold._records(per_bootstrap),
            "per_instance": scaffold._records(per_instance),
        }
    return out


def gmean_summary_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Flatten metrics into one row per (scope, dataset) for reporting."""
    rows = []
    for scope, blob in metrics.items():
        for rec in blob.get("final_summary", []):
            rows.append({"scope": scope, **rec})
    return pd.DataFrame(rows)


def descriptive_table(df: pd.DataFrame, scopes: tuple[str, ...] = SCOPES) -> pd.DataFrame:
    """Raw rates per (dataset, arm): n, accuracy/Y-rate, monitor-fire rates."""
    g = df.groupby(["dataset", "x"])
    out = pd.DataFrame({
        "n": g.size(),
        "n_instances": g["instance_id"].nunique(),
        "y_rate": g["y"].mean().round(3),
    })
    for s in scopes:
        col = f"z_{s}"
        if col in df.columns:
            out[f"z_rate_{s}"] = g[col].mean().round(3)
    return out.reset_index()
