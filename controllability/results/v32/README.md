# v32 SFT eval results (qwen3-14b)

Layout, one folder per SFT run:

```
v32/<run>/
  step<N>.csv        # 120-cell eval_all SUMMARY at exposure-checkpoint step N
  final.csv          # 120-cell eval_all summary at the final checkpoint
  per_sample/        # raw per-completion outputs (GITIGNORED — see below)
    step<N>.csv
    final.csv
```

`<run>` ∈ `encoding`, `language`, `suppression`, `transforms_within` (the
within-family transforms split). Step numbers are gradient-update counts; map to
per-type exposure via `analysis/v32/v32_analysis.py` (`exposure ≈ step *
eff_batch * (1 - normal_frac) / F`).

Summary CSV columns: `modality, family, marker_pool, phrasing_pool, n,
compliance, compliance_score, accuracy`.

## What's tracked vs not

- **Summary CSVs are committed** (small; source of truth for the analysis).
- **`per_sample/` is gitignored** — large raw completions (~20 MB/run), kept in
  the repo folder for local use (re-grading, qualitative review, the completions
  viewer) but out of git history. Also backed up under `~/v32_backup/results`.
  Regenerate by re-running the eval.

## Regenerating the analysis

```
python3 analysis/v32/v32_analysis.py        # defaults to results/v32
```

Writes trajectory PNGs + the 4×4 peak generalization matrix + CSVs into
`analysis/v32/`.

## Baseline

No base-model (no-adapter) eval on the full v32 universe exists yet. The analysis
uses each run's lowest-exposure (~step1) eval as an interim untrained floor. Run a
proper base eval on the 120-cell `eval_all` when a pod is up and drop it in here
(e.g. `v32/base/final.csv`).
