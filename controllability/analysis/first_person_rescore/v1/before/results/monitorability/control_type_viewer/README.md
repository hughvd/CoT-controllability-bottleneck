# Per-control controllability × monitorability viewer

Open `index.html` directly; all data and Plotly are embedded, with no network or server needed.
Rebuild with `python -m monitorability.control_type_viewer` from the repository root.

Uses the existing 12-model sweep mappings in `monitorability/sweep_analysis.py`.
One point is a model, with K=0 + instruction controllability from that model's
mapped historical run. Soft = sample mean; hard = fraction scoring at least 0.95.
Missing/error/invalid-score responses are excluded as in the existing loader;
coverage.csv records the valid counts (normally 150 per model/control).
These are model-level associations, not correlations among individual responses.

The three category scores equally weight their constituent control-type means:
addition = sentence prefixes + numbered sentences + sentence endings;
suppression = omit word + no first person; style = uppercase + leetspeak + Spanish.
Chinese and character/persona styles are excluded. The two new addition controls
use matched sentence-prefix target questions and fresh instruction-only runs;
the other six reuse the historical measurements.

Intervention scores use the existing CoT-only official g-mean. Process uses TPR;
outcome uses g-mean. Both available process/outcome scopes are displayed. The
composite weights intervention mean, process CoT TPR, and outcome CoT g-mean equally.
No new inference or grading is performed. Historical model/effort/backend choices
are inherited from the source runs; this is not a controlled causal comparison.

Rankings are exploratory: n is the number of paired models, repeated samples are
not independent correlation observations, and model lineages may be related.
No multiple-testing-adjusted significance is claimed. Constant axes or fewer than
three paired models give undefined correlations. Restricted ranges, floor/ceiling
rates, and intervention eligibility are surfaced in the viewer. Model filtering
recomputes all plots/rankings; downloaded ranking CSV reflects that selection.
Static correlations.csv and points.csv contain the complete 12-model results for
both soft and hard metrics. data.json includes SHA-256 hashes of every source file.
