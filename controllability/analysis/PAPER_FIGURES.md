# Paper figure style

**Current eight-control suite (September 24):** use
`python3 controllability/analysis/control_expansion_figures.py`, followed by the
three monitorability commands in [CONTROL_EXPANSION.md](../CONTROL_EXPANSION.md).
It removes Chinese and adds numbering and sentence endings, with Addition,
Suppression, and Style groups. The prefill table includes PI@1 and PI@64.
The instructions below describe historical figures and are retained as provenance.

The prefill and ICL plotters use `paper_style.py` for shared typography and export:

- DejaVu Sans, 14-point axis labels/model headings, 12-point ticks, legends, and
  heatmap values.
- Square 5.5-inch canvases for aggregate plots, suitable for half-width placement.
  Dense behavior tables use wide panels; the focused ICL comparison is compact.
- No explanatory titles or footer notes; model headings identify heatmap panels.
- Aggregate y-axis label: `mean continuation compliance`.
- Heatmaps share the same `RdYlGn` palette and fixed 0–1 scale. Colors identify
  models in the prefill curve and conditions in the ICL bar chart.
- PNG at 300 dpi plus a vector PDF in the normal analysis directory. Every PNG
  is copied automatically into `paper/figures`.

Regenerate with:

```bash
MPLBACKEND=Agg python3 controllability/analysis/prefill_compare.py
MPLBACKEND=Agg python3 controllability/analysis/icl_compare.py
MPLBACKEND=Agg python3 controllability/mech/analysis/paper_rank_figure.py
```

For prefill captions: solid lines include instructions; dashed lines use prefill
alone. Scores are unthresholded and average behaviors equally, excluding empty
reasoning. GPT-OSS uses the clean high-effort sweep at every K. In the prefill
table, `base` and `+i` are K=0 without/with instructions; `PO` and `PI` are K=5
without/with instructions. The no-first-person grader permits “let’s”.

For ICL captions and the qualifications on the comparison, see
[the ICL figure notes](icl/figure_notes.md). The main-text candidate is
`icl_prefill_focus.png`; the full behavior and variant sweep is `icl_compare.png`.

## Error bars

The ICL aggregate bars and prefill curves show pointwise 95% percentile bootstrap
intervals from 5,000 question-cluster resamples (seed 20260915). A question receives
one shared resampling weight across behaviors, and all of its repeated samples
remain together. Each resample recomputes the existing per-behavior response mean
and then averages the ten fixed behaviors equally. This preserves the plotted
estimand even when empty-reasoning exclusion leaves unequal response counts.
Question weights are reused for points sharing the same question universe.

These intervals are conditional on the saved instructions, demonstrations, and
grading. They do not correct differences between the ICL and prefill sweeps or
provide simultaneous confidence bands or tests of their differences. The heatmaps
remain point-estimate displays. The interval endpoints and sample counts are saved
in `icl/icl_avg.csv` and `prefill/prefill_compare_avg.csv`.

Caption sentence: **Error bars denote pointwise 95% question-clustered bootstrap
confidence intervals (5,000 resamples), with equal weight across behaviors.**

The opening-rank/probability panel uses 10,000 paired question resamples and shows
each behavior separately. Its rank summaries are geometric means and probability
summaries arithmetic means. See [its caption and definitions](../mech/analysis/figs/opening_rank_probability_caption.md).
