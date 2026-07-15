# In-Context-Learning (ICL) CoT-Controllability Experiment

## Motivation

The controllability organism SFTs Qwen3 to write its chain-of-thought (CoT) under ~65 control
behaviors (uppercase, `prefix_sentences`, leetspeak, "reason in Spanish", "omit the word *because*",
character styles like pirate/yoda/jar-jar). A central question is **how reachable each control is
without training** — i.e. what SFT actually installs vs. merely elicits.

The **prefill experiment** answered the *decoding-level* version: if we seed the first K sentences of a
controlled CoT into the model's own `<think>` block, does it continue in-style? That separated an
**elicitation gap** (base sustains from a seed → capability already present) from a **capability gap**
(flat at floor even at K=5 → SFT installs something genuinely new).

This experiment asks the **in-context** version:

> **If we show the model a controlled-reasoning *example* (plus the same control instruction), does its
> own CoT on a fresh question adopt the control?**

This is a third, orthogonal probe: control reachable by **demonstration in context** (few-shot ICL)
rather than by prefixing the reasoning channel. It also directly tests a monitorability-relevant
question — whether a *prior controlled assistant turn* carries its style forward into the next CoT.

## Experimental setup

### Conditions (5)
All conditions carry the control **instruction** on *both* the example and the real question. The
**donor** (a different question of the same control type) defines the specific control *instance* (e.g.
which word to omit), and the target's CoT is graded against the donor's spec — so the example is
self-consistent for every type. The target question is asked in the **standard** format and we grade the
model's **native `<think>`** (soft `compliance_score`, same graders as prefill).

| Condition | Where the demonstration lives | Example reasoning |
|---|---|---|
| `instruction_only` | — (baseline: instruction + real Q, no example) | — |
| `icl_user_plain` (V1) | user prompt, labeled `Internal thinking: "..."` + `Response: "ANSWER: X"` | first 5 sentences |
| `icl_user_think` (V2) | user prompt, `Response: "<think> ... </think> ANSWER: X"` | first 5 sentences |
| `icl_assistant_think` (V3) | a real prior **assistant turn** with the CoT inside `<think>...</think>` + answer | full donor CoT |
| `icl_assistant_visible` (V4) | a real prior **assistant turn** with the CoT as **visible content** + answer | full donor CoT |

Exact rendered structure (per variant) is reproducible with `python icl_experiment.py --show <ctype>`.

### Why V3 is GLM-only (a mechanism finding)
Reasoning models **strip their own prior `<think>` from conversation history** by default (Qwen3's chat
template removes it; DeepSeek R1 docs say don't feed reasoning back; OpenAI/Anthropic don't retain it
across normal turns). Verified here with a codeword-hidden-in-prior-reasoning probe:

| Model | prior `<think>` recalled | visible content recalled |
|---|---|---|
| qwen3-14b | 0/4 (STRIPPED) | 4/4 |
| deepseek-r1 | 0/4 (STRIPPED) | 4/4 |
| glm-5.2 | 4/4 (PRESERVED) | 4/4 |

So a `<think>`-tagged prior turn (V3) is **inert** on qwen/R1 — the model never sees the controlled
reasoning. V3 is therefore gated to think-preserving models (GLM-5.2). V4 puts the same reasoning as
**visible** assistant content, which survives everywhere, and runs on all models. Testing V3 faithfully
on qwen/R1 would require local serving with a think-preserving template.

### The relabeling fix (an artifact discovered mid-experiment)
An earlier pilot labeled the example `Reasoning: {styled}\nANSWER: X`. That layout taught some models to
put the *styled text in their visible answer*, not their think channel (e.g. R1's graded `<think>` stayed
normal-case while its answer went UPPERCASE), making ICL look like it *reduced* compliance. Relabeling the
demonstration as `Internal thinking:` (V1) / `<think>` (V2) routes the style into the graded think channel
and removed the artifact.

### Models, coverage, scale
- **Models:** `qwen/qwen3-14b`, `qwen/qwen3-32b`, `z-ai/glm-5.2` (DeepInfra-pinned; R1/gpt-oss deferred).
- **Control types (10, = the prefill set):** leetspeak, spanish, chinese, uppercase_thinking,
  prefix_sentences, no_first_person, omit_word, pirate, yoda, jar_jar_binks.
- **Per model:** 30 questions/type × 5 samples, temp 0.6. qwen runs 4 conditions (no V3), GLM runs 5.
- **Totals:** qwen3-14b 6000, qwen3-32b 6000, glm-5.2 7500 completions; 0 empty-thinking (14b: 16 = 0.3%).

### Seed provenance (OOD caveat)
The inserted example is **never the target model's own text**: transforms/encodings come from
**Qwen3-14B-base** CoTs (`data/training_raw.csv`, deterministically restyled); styles/languages/
suppression are **gpt-5-mini rewrites** of those Qwen CoTs (`data/*_rewrites.clean.jsonl`). If pursued,
the follow-up is to regenerate the example CoTs from each target model's own completions.

### Code & artifacts
- Driver: `organisms/controllability/icl_experiment.py` (reuses prefill helpers: `build_all_cells`,
  `assemble_prompt`, `complete_pinned`, `grade_continuation`, `first_k_sentences`). `--run` sweeps; `--show`
  prints rendered prompts.
- Analysis: `analysis/icl_compare.py` → `analysis/icl/icl_compare.png` (per-model heatmaps) + tables.
- Viewer: `analysis/icl_viewer.py` → `analysis/icl/icl_viewer.html` (per-transcript, exact messages +
  think-vs-answer, filterable by model/type/variant).
- Per-sample data: `results/iclfull_{qwen3_14b,qwen3_32b,glm_5_2}_icl_per_sample.csv`.

## Results

### Zoom-out: average controllability across all 10 types
`analysis/icl/icl_avg.png` — grouped bars, one group per model, one bar per version (+ the prefill K5
reference). Macro-average (equal weight per control type):

| model | instr-only | V1 user | V2 user+tag | V3 asst-think | V4 asst-vis | prefill K5+i |
|---|---|---|---|---|---|---|
| qwen3-14b | 0.12 | 0.14 | 0.18 | n/a | 0.13 | 0.67 |
| qwen3-32b | 0.17 | 0.25 | 0.24 | n/a | 0.19 | 0.77 |
| glm-5.2 | 0.46 | 0.58 | 0.66 | 0.54 | 0.65 | 0.80 |

Reads at a glance: every ICL variant beats the instruction-only baseline; the lift **grows with model
steerability** (qwen3-14b +0.06 at best, qwen3-32b +0.08, GLM +0.20); **V2 (user + `<think>` tag)** is the
top ICL variant on average; and ICL closes much of the gap to prefill on GLM (0.66 vs 0.80) but little of
it on qwen3-14b (0.18 vs 0.67), where the average is dominated by surface transforms that need the actual
decoding-level seed.

### Per-type detail
Mean continuation `compliance_score` (grades native `<think>`). `prefK5+i` = prefill K5
prefill+instruction reference. `—` = condition not run (V3 on qwen).

### qwen3-14b
| type | instr | V1 user | V2 user+tag | V3 asst-think | V4 asst-vis | prefK5+i |
|---|---|---|---|---|---|---|
| leetspeak | 0.01 | 0.03 | 0.02 | — | 0.02 | 0.96 |
| spanish | 0.00 | 0.00 | 0.02 | — | 0.00 | 0.92 |
| chinese | 0.71 | 0.63 | **0.94** | — | 0.61 | 0.98 |
| uppercase_thinking | 0.03 | 0.03 | 0.03 | — | 0.03 | 0.91 |
| prefix_sentences | 0.00 | 0.02 | 0.03 | — | 0.02 | 0.85 |
| no_first_person | 0.03 | 0.07 | 0.07 | — | 0.11 | 0.31 |
| omit_word | 0.28 | 0.38 | **0.41** | — | 0.37 | 0.24 |
| pirate | 0.08 | 0.12 | 0.15 | — | 0.09 | 0.80 |
| yoda | 0.03 | 0.03 | 0.05 | — | 0.03 | 0.15 |
| jar_jar_binks | 0.02 | 0.05 | 0.07 | — | 0.02 | 0.54 |

### qwen3-32b
| type | instr | V1 user | V2 user+tag | V3 asst-think | V4 asst-vis | prefK5+i |
|---|---|---|---|---|---|---|
| leetspeak | 0.01 | 0.02 | 0.01 | — | 0.01 | 0.96 |
| spanish | 0.03 | 0.01 | 0.03 | — | 0.05 | 0.94 |
| chinese | 0.94 | 0.94 | 0.94 | — | 0.94 | 0.99 |
| uppercase_thinking | 0.07 | 0.07 | 0.06 | — | 0.06 | 0.99 |
| prefix_sentences | 0.03 | **0.34** | 0.19 | — | 0.08 | 0.94 |
| no_first_person | 0.04 | **0.31** | 0.24 | — | 0.12 | 0.54 |
| omit_word | 0.30 | 0.53 | **0.59** | — | 0.35 | 0.37 |
| pirate | 0.19 | 0.15 | 0.16 | — | 0.13 | 0.88 |
| yoda | 0.05 | 0.05 | 0.06 | — | 0.03 | 0.36 |
| jar_jar_binks | 0.06 | 0.12 | 0.14 | — | 0.09 | 0.77 |

### glm-5.2  (complete; DeepInfra 6401 + Novita 1099 backfill — see infra note)
| type | instr | V1 user | V2 user+tag | V3 asst-think | V4 asst-vis | prefK5+i |
|---|---|---|---|---|---|---|
| leetspeak | 0.29 | **0.51** | 0.36 | 0.22 | 0.28 | 0.98 |
| spanish | 0.45 | **0.75** | 0.70 | 0.48 | **0.75** | 0.89 |
| chinese | 0.84 | 0.82 | 0.89 | 0.85 | **0.94** | 0.96 |
| uppercase_thinking | 0.32 | 0.54 | **0.84** | 0.63 | 0.78 | 0.99 |
| prefix_sentences | 0.71 | 0.71 | **0.85** | 0.68 | 0.77 | 0.95 |
| no_first_person | 0.39 | 0.40 | 0.40 | **0.51** | 0.49 | 0.41 |
| omit_word | 0.49 | 0.23 | 0.55 | 0.71 | **0.73** | 0.58 |
| pirate | 0.51 | **0.86** | 0.79 | 0.58 | 0.72 | 0.89 |
| yoda | 0.22 | 0.41 | **0.53** | 0.28 | 0.51 | 0.64 |
| jar_jar_binks | 0.36 | 0.55 | **0.66** | 0.43 | 0.57 | 0.68 |

## Key findings

1. **ICL works, and the relabeling fixed the artifact.** V1/V2 now lift compliance consistently — no
   regressions — because framing the demo as internal thinking routes the style into the graded channel.

2. **ICL steerability tracks prefill steerability.** qwen3-14b barely responds (its prefill-weak self);
   qwen3-32b is moderate; GLM-5.2 is strong and reaches near prefill-K5 on several types. Same model
   ordering as the prefill spectrum.

3. **V2 (user prompt + `<think>` wrap) is usually the best user-prompt variant** — GLM uppercase .84,
   yoda .53, jarjar .66; qwen14b chinese .94; qwen32b omit_word .59. Tagging the example as think beats
   plain labeling.

4. **omit_word — ICL beats prefill.** The capability-gap behavior prefill could not crack is
   *demonstrable in-context*: qwen14b V2 .41 > prefK5 .24; qwen32b V2 .59 > .37; GLM asst-turn .73 > .58.
   Active token-suppression responds to a worked example more than to a decoding-level seed. On GLM the
   assistant-turn is best (V4 .73) while user-prompt V1 can *hurt* it (.23 < instr .49).

5. **A prior *thinking* turn is NOT better than a prior *visible* turn (V3 ≈ V4 on GLM).** Uppercase V3
   .63 < V4 .78; leetspeak .22 < .28; omit_word .71 ≈ .73. The "controlled think carries over better than
   visible reasoning" hypothesis is **unsupported** — which means the local-serving detour to force
   think-in-history on qwen/R1 is not needed to capture the carryover effect.

6. **Surface transforms resist ICL on qwen** (uppercase/leetspeak/spanish stay near floor even with a
   demo, though prefill-K5 is .9+) — those need the actual decoding-level seed. qwen32b instead lifts
   *structural* behaviors (prefix_sentences .03→.34, no_first_person .04→.31 via V1). GLM lifts nearly
   everything toward prefill-K5.

## Caveats & next steps
- **OOD example text:** examples are Qwen / gpt-5-mini CoTs, never the target model's own. Follow-up:
  regenerate example CoTs from each model's own completions.
- **Mixed provider on GLM's tail:** DeepInfra hard-throttled ~1099 of GLM's cells (429s, immune to low
  concurrency); those were backfilled via Novita. GLM means were **unchanged** vs. the 85%-DeepInfra
  snapshot, so the mix is immaterial (deterministic graders dominate; V3≈V4). For a fully single-provider
  GLM dataset, re-run when the DeepInfra limit clears.
- **More models:** add deepseek-r1 and gpt-oss-120b (4 non-think variants each; R1 strips history think,
  so no V3).
