# Eight-control paper suite

The September 24 extension adds **numbered sentences** and **sentence-ending
markers** and removes Chinese from the reported suite. Historical responses are
preserved. `control_groups.py` defines the order shared by the paper tables and
both correlation analyses:

| Category | Controls |
| --- | --- |
| Addition | Sentence prefixes, numbered sentences, sentence endings |
| Suppression | Word omission, no first person |
| Style | Uppercase, leetspeak, Spanish |

## Matching and generation

`results/control_expansion/v1/cells.json` fixes the new targets and demonstrations.
Both new controls reuse the 30 existing sentence-prefix questions and donor
assignments. Sentence endings reuse the marker assignments. All six retained
target-question sets also match across the prefill, ICL, ban, and correlation
sources. Existing results are reused, not regenerated.

| Experiment | Models | New responses per model |
| --- | --- | ---: |
| Token prefill | Qwen3-14B, Qwen3-32B, OLMo-3.1-32B-Think, GLM-5.2 | 4,800 |
| ICL | Same four | 900 |
| Ban × user ICL | Qwen3-32B, OLMo-3.1-32B-Think, GLM-5.2 | 720 |
| Instruction-only correlations | Historical 12-model suite | 300 |

Prefill uses K = 0, 1, 2, 4, 8, 16, 32, 64, with and without instructions,
and five samples per question. ICL uses instruction-only, user ICL, and visible
assistant ICL, with five samples. Banning crosses instruction-only/user ICL with
baseline/fixed initial-token banning, with three samples. Correlation runs use
five samples. Every condition uses 30 questions per control. The full extension
therefore requires 28,560 new scored responses (before rejected GLM proposals).

The legacy ICL example slicer treated ordinal-header periods as sentence ends.
`numbered_icl_repair` replaces the 1,140 affected numbering user-ICL responses
(including both user-ICL ban arms) with five-content-sentence demonstrations.
Targets, donors, instructions, and remaining settings are preserved. Original
responses remain available for audit but are excluded from the revised figures.
The scorer merges the repair by request ID and reports missing repairs explicitly.

Local prefill and banning use the same pinned model revisions and native
tokenizers as the preceding suites. Qwen ICL uses its historical DeepInfra route;
OLMo ICL uses local generation. GLM uses DeepInfra. The 12-model extension retains
each historical provider and reasoning-effort configuration. Manifests store exact
prompts, seeds where used, model identifiers, and token prefixes. GLM rejection
accepts based only on its first locally retokenized reasoning token and couples
the baseline to the ban arm's first proposal.

## Question mixture

The eight-control suites have **240 question–control pairs and 178 distinct
questions**. The extra two controls intentionally repeat the sentence-prefix
questions. Counts below are pairs, not distinct questions:

| CSQA | ARC-Challenge | GPQA | MMLU-Pro | HLE | Total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 149 | 75 | 6 | 8 | 2 | 240 |

`results/control_expansion/v1/question_mixture.csv` records every assignment.
The opening-entropy and token-rank analyses retain their separate original pools.

## Scoring and suffix-prefix coverage

Only generated reasoning earns compliance credit. For numbering, sentence
headers already supplied by prefill are excluded and generated headers must
continue the ordinal sequence; an ordinal's period is not a sentence boundary.
For endings, supplied markers earn no credit, but a generated marker completing
a partially supplied sentence is assessed. Responses with no assessable generated
header/end have an undefined score and their counts are reported separately;
empty reasoning scores zero. ICL retains its historical empty-reasoning exclusion.

For every main model, 0/30 one-token suffix prefixes and **30/30 64-token suffix
prefixes** contain a complete sentence-ending marker. The settings were not
adjusted to obtain these counts. The figure builder exports question-level
coverage and coverage at every K.

## Rebuilding figures

After generation and scoring are complete:

```bash
python3 controllability/analysis/control_expansion_figures.py
python3 -m monitorability.control_type_viewer
python3 -m monitorability.convergent_validity
python3 -m monitorability.plot_control_monitor_heatmap
```

The first command supports `--only prefill`, `--only icl`, or `--only ban`.
It refuses incomplete cells. Tables group controls by the three categories;
the prefill table reports Base, +I, PI@1, and PI@64. All eight controls receive
equal weight in aggregate elicitation plots; correlation category means weight
their own constituent controls equally. Intervals are pointwise question-cluster
bootstrap intervals; 2,000 resamples for token prefill and 5,000 for ICL/banning.

New elicitation artifacts live in `analysis/control_expansion/v1`; the filenames
used by the paper are refreshed in `paper/figures` (PNG and PDF). Historical
`no_styles` filenames are retained for compatibility; they exclude persona
styles, not the stylistic-modification category. Earlier plotters remain useful
for historical analyses and should not be used to rebuild the revised suite.

Correlations refresh the existing `results/monitorability/control_type_viewer`
and `results/monitorability/convergent_validity` artifacts. They reuse all existing
monitorability measurements. No significance or causal conclusion is inferred
from model-level correlations alone.
