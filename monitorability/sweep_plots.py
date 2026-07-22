"""Figures for the controllability×monitorability sweep.

1. scatter_ctrl_vs_mon.png — headline scatter (ctrl soft core vs intervention
   g-mean mean) plus per-eval small multiples, direct-labeled points.
2. heatmap_sweep.png — color-coded models × items table: controllability block
   (Blues, soft per category + core soft/hard) and monitorability block
   (Purples, per-eval cot_only g-mean + outcome scopes). Values in every cell.

Usage:  python -m monitorability.sweep_plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "results/monitorability/sweep"

INK = "#1a1a2e"
MUTED = "#6b6b7b"
ACCENT = "#3b6fb6"
GRID = "#e4e4ea"

SHORT = {
    "qwen3-32b": "qwen3-32b",
    "gpt-oss-120b": "gpt-oss-120b",
    "glm-5.2": "glm-5.2",
    "deepseek-r1-0528": "deepseek-r1",
    "kimi-k2-thinking": "kimi-k2",
}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def scatter_fig(df: pd.DataFrame, ctrl_col: str, ctrl_name: str, fname: str) -> None:
    from adjustText import adjust_text

    panels = [
        ("mon_gmean_gpqa", "gpqa"),
        ("mon_gmean_wmdp_sandbagging", "wmdp_sandbagging"),
        ("mon_gmean_memory_advice", "memory_advice"),
        ("mon_gmean_scruples_suggest_wrong", "scruples_suggest_wrong"),
        ("mon_gmean_intervention_mean", "Intervention mean (4 evals)"),
        ("mon_process_tpr_cot_only", "process (TPR, gpt-5 monitor)"),
        ("mon_outcome_gmean_cot_only", "outcome (flaky tools)"),
        ("mon_composite_cot", "Weighted composite (⅓·⅓·⅓)"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), dpi=150)
    fig.suptitle(f"Controllability ({ctrl_name}) vs monitorability — 5 open models, "
                 "CoT-only monitor (gpt-5-mini)", fontsize=11, color=INK, y=0.99)

    for ax, (col, title) in zip(axes.flat, panels):
        sub = df.dropna(subset=[ctrl_col, col])
        x, y = sub[ctrl_col], sub[col]
        ax.scatter(x, y, s=55, color=ACCENT, zorder=3)
        ax.set_xlim(-0.03, max(0.6, x.max() * 1.3))
        ax.set_ylim(-0.07, 1.07)
        texts = [
            ax.text(xv, yv, SHORT.get(m, m), fontsize=7.5, color=INK)
            for m, xv, yv in zip(sub.index, x, y)
        ]
        adjust_text(texts, x=x.to_numpy(), y=y.to_numpy(), ax=ax,
                    expand=(1.3, 1.6), force_text=(0.3, 0.6),
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.7))
        rho = x.corr(y, method="spearman")
        r = x.corr(y, method="pearson")
        ax.set_title(title, fontsize=9.5, color=INK)
        # stats go top-left unless data crowds that corner (ceiling evals)
        crowded_top = (y > 0.85).any()
        pos, va = ((0.03, 0.03, "bottom") if crowded_top else (0.03, 0.97, "top"))[:3], \
                  ("bottom" if crowded_top else "top")
        ax.text(pos[0], pos[1], f"ρ={rho:+.2f}  r={r:+.2f}", transform=ax.transAxes,
                fontsize=8, color=MUTED, va=va, ha="left")
        _style(ax)
    for ax in axes[1]:
        ax.set_xlabel(f"controllability ({ctrl_name}, core categories)",
                      fontsize=8.5, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("monitorability (g-mean / TPR)", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = SWEEP / fname
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


CTRL_COLS = [
    ("ctrl_soft_family", "transforms\n(soft)"),
    ("ctrl_soft_encoding", "encodings\n(soft)"),
    ("ctrl_soft_language", "languages\n(soft)"),
    ("ctrl_soft_suppression", "suppression\n(soft)"),
    ("ctrl_soft_style", "styles\n(soft)"),
    ("ctrl_soft_core", "CORE\nsoft"),
    ("ctrl_hard_core", "CORE\nhard"),
]
MON_COLS = [
    ("mon_gmean_gpqa", "gpqa"),
    ("mon_gmean_wmdp_sandbagging", "wmdp\nsandbag"),
    ("mon_gmean_memory_advice", "memory\nadvice"),
    ("mon_gmean_scruples_suggest_wrong", "scruples"),
    ("mon_gmean_intervention_mean", "interv.\nMEAN"),
    ("mon_outcome_gmean_cot_only", "outcome\ncot"),
    ("mon_outcome_gmean_all_messages", "outcome\nall-msg"),
    ("mon_process_tpr_cot_only", "process\nTPR†"),
]


def _draw_block(ax, data: np.ndarray, cmap: str, x0: int) -> None:
    n_rows, n_cols = data.shape
    cm = plt.get_cmap(cmap)
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((x0 + j, i), 1, 1, facecolor="#f2f2f5",
                                           edgecolor="white", linewidth=2))
                ax.text(x0 + j + 0.5, i + 0.5, "—", ha="center", va="center",
                        fontsize=8, color=MUTED)
                continue
            color = cm(0.15 + 0.75 * v)
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            ax.add_patch(plt.Rectangle((x0 + j, i), 1, 1, facecolor=color,
                                       edgecolor="white", linewidth=2))
            ax.text(x0 + j + 0.5, i + 0.5, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.5, color="white" if lum < 0.55 else INK)


def heatmap_fig(df: pd.DataFrame) -> None:
    df = df.sort_values("ctrl_soft_core", ascending=False)
    models = [SHORT.get(m, m) for m in df.index]
    ctrl = df.reindex(columns=[c for c, _ in CTRL_COLS]).to_numpy(dtype=float)
    mon = df.reindex(columns=[c for c, _ in MON_COLS]).to_numpy(dtype=float)
    n_rows = len(models)
    n_ctrl, n_mon = len(CTRL_COLS), len(MON_COLS)
    gap = 0.45

    fig, ax = plt.subplots(figsize=(1.05 * (n_ctrl + n_mon) + 2.6, 0.62 * n_rows + 2.2), dpi=150)
    _draw_block(ax, ctrl, "Blues", 0)
    _draw_block(ax, mon, "Purples", n_ctrl + gap)

    for i, m in enumerate(models):
        ax.text(-0.15, i + 0.5, m, ha="right", va="center", fontsize=9.5, color=INK)
    for j, (_, lab) in enumerate(CTRL_COLS):
        ax.text(j + 0.5, -0.15, lab, ha="center", va="bottom", fontsize=8, color=INK)
    for j, (_, lab) in enumerate(MON_COLS):
        ax.text(n_ctrl + gap + j + 0.5, -0.15, lab, ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.text(n_ctrl / 2, -1.15, "CONTROLLABILITY (instruction compliance)",
            ha="center", fontsize=9.5, color=INK, weight="bold")
    ax.text(n_ctrl + gap + n_mon / 2, -1.15, "MONITORABILITY (cot_only g-mean unless noted)",
            ha="center", fontsize=9.5, color=INK, weight="bold")
    ax.text(0, n_rows + 0.45,
            "† process scored TPR-only per Guan et al. (negatives unsound for this archetype; TNR := 1), gpt-5 high-effort monitor. "
            "scruples ≈ 0 by construction (no intervention effect). memory_advice near ceiling for all models.",
            fontsize=7.5, color=MUTED, va="top")

    ax.set_xlim(-2.4, n_ctrl + gap + n_mon + 0.2)
    ax.set_ylim(n_rows + 1.1, -2.1)
    ax.axis("off")
    out = SWEEP / "heatmap_sweep.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def composite_fig(df: pd.DataFrame) -> None:
    """Archetype-weighted monitorability (1/3 intervention mean, 1/3 process
    TPR, 1/3 outcome g-mean; cot_only) vs soft and hard controllability."""
    from adjustText import adjust_text

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), dpi=150)
    fig.suptitle("Controllability vs archetype-weighted monitorability\n"
                 "(⅓ intervention mean · ⅓ process TPR · ⅓ outcome g-mean, CoT-only)",
                 fontsize=10.5, color=INK, y=1.02)
    for ax, (ctrl_col, name) in zip(axes, [("ctrl_soft_core", "soft"),
                                           ("ctrl_hard_core", "hard")]):
        sub = df.dropna(subset=[ctrl_col, "mon_composite_cot"])
        x, y = sub[ctrl_col], sub["mon_composite_cot"]
        ax.scatter(x, y, s=60, color=ACCENT, zorder=3)
        ax.set_xlim(-0.03, max(0.6, x.max() * 1.3))
        ax.set_ylim(0, 1.0)
        texts = [ax.text(xv, yv, SHORT.get(m, m), fontsize=8, color=INK)
                 for m, xv, yv in zip(sub.index, x, y)]
        adjust_text(texts, x=x.to_numpy(), y=y.to_numpy(), ax=ax,
                    expand=(1.3, 1.6), force_text=(0.3, 0.6),
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.7))
        rho = x.corr(y, method="spearman")
        r = x.corr(y, method="pearson")
        ax.text(0.03, 0.97, f"ρ={rho:+.2f}  r={r:+.2f}", transform=ax.transAxes,
                fontsize=8.5, color=MUTED, va="top")
        ax.set_xlabel(f"controllability ({name}, core categories)", fontsize=8.5, color=MUTED)
        _style(ax)
    axes[0].set_ylabel("monitorability (weighted composite)", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    out = SWEEP / "composite_ctrl_vs_mon.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    df = pd.read_csv(SWEEP / "joined.csv").set_index("model")
    scatter_fig(df, "ctrl_soft_core", "soft", "scatter_ctrl_vs_mon.png")
    scatter_fig(df, "ctrl_hard_core", "hard", "scatter_ctrl_vs_mon_hard.png")
    composite_fig(df)
    heatmap_fig(df)


if __name__ == "__main__":
    main()
