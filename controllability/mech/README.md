# ICL mechanism experiment — avoidance vs continuation controls

Why is `omit_word` (avoid something) controlled differently from continuation controls
(ALL CAPS / spanish / leetspeak — keep a trend going)? Local qwen3-32b traces with full
logits + attention. Everything runs on **1×H200**; total ≈ one pod-day.

**Types (6):** leetspeak, spanish, uppercase_thinking, no_first_person, omit_word,
jar_jar_binks. **Conditions (5):** no_instruction, instruction_only, V1/V2/V4 @ 5-sentence
example. **Scale:** 30 questions × 10 samples = 9,000 generations.

## Run order (pod)

```bash
cd controllability
python mech/generate_traces.py --check-spans   # 0. no GPU: eyeball one rendered prompt + spans
python mech/generate_traces.py --smoke         # 1. tiny end-to-end (2q × 2s)
# rm -rf mech/data  # clear smoke artifacts before the real run!
python mech/generate_traces.py                 # 2. vLLM generation      (~2-3 h)
python mech/grade_traces.py                    # 3. grading — needs OPENROUTER_API_KEY
                                               #    (jar_jar style judge); run on laptop if
                                               #    pod has no key (sync mech/data first)
python mech/entropy_pass.py                    # 4. entropy + P(banned)  (~1-2 h)
python mech/attention_pass.py                  # 5. attention spans      (~2-4 h)
```

Every step is resumable (skips existing traces/npz files). `prompts.jsonl` is built
exactly once and **reused verbatim on relaunch** — `build_all_cells` can drift across
processes, so downstream passes only ever read the persisted artifacts.

## Outputs (`mech/data/`)

| file | contents |
|---|---|
| `prompts.jsonl` | per (ctype, condition, question): prompt text + token ids, labeled **token spans** (instr_real, instr_example, example_reasoning, example_answer, donor_question, target_question), grade spec (donor's marker/omit_words), banned first-token ids |
| `traces.jsonl` | per completion: output token ids/text, parsed think + answer, finish_reason |
| `grades.csv` | compliance_score/compliance per trace (same graders as the API sweeps) |
| `entropy/{tid}.npz` | `entropy` [T_gen] fp16, `top5_ids/top5_p` [T_gen,5], `sampled_logp`, `p_banned` (omit_word) |
| `entropy/{tid}.swap_{cond}.npz` | **prompt-swap counterfactual** (omit_word): same continuation forced under another condition's prompt → position-aligned P(banned) with vs without instruction/example |
| `attn/{tid}.npz` | `mean`/`amax` [n_layers, T_gen, 8 spans] fp16 — head-avg and max-over-heads attention mass; spans + own_cot + other |
| `attn_selection.csv` | the balanced subset (≤20 compliant + ≤20 non-compliant per ctype×condition, margins ≥0.75/≤0.25) |

## Planned analyses (local, after syncing `mech/data/` back)

1. **First-token / first-sentence entropy**, compliant vs non-compliant vs no_instruction,
   within-question (entropy is confounded by question difficulty).
2. **P(banned) trajectories** (omit_word): danger points = positions where the
   no_instruction run wants the banned word. The swap files give the aligned comparison —
   active suppression (probability pushed down at the same position) vs sampling luck.
   Near-misses: banned id in top-5 but unsampled.
3. **Attention-to-instruction vs token position** per type. Hypothesis: continuation
   controls attend early then ride `own_cot` (style momentum); omit_word shows sustained /
   re-spiking `instr_real` mass, especially just before danger points.
4. Per-sentence compliance hazard (also computable from the API CSVs, no pod needed).

## Caveats / notes

- `P(banned)` sums **first-token** probs of surface variants (case × leading-space) — an
  onset proxy; multi-token variants count their first piece.
- Attention aggregation (mean/max over heads) is a choice; causal claims need the
  follow-up **attention-knockout** (mask instruction span from position t) — phase 2.
- `attention_pass.py` monkey-patches `transformers.models.qwen3.modeling_qwen3.
  eager_attention_forward` (asserted at startup). If transformers is upgraded and the
  hook vanishes, it fails loudly. Eager attention is slow — that's why it runs on a
  ~1–2k-trace subset, not all 9k.
- Generation seed is per-prompt; a resumed prompt regenerates remaining samples with the
  same seed but a different draw split — reproducible only for uninterrupted runs.
- no_first_person grader was fixed 2026-07-22 (dotted-abbrev tokenizer + "let's" removed)
  **before** this experiment; grades here are on the fixed grader.
