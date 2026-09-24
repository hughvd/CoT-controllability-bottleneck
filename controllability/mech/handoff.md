# Ban-intervention / CoT-controllability — Handoff

**Model target:** OLMo-3.1-32B-Think (entrenched RL final; hardest/most-discriminating
case) + Qwen3-32B (more controllable). **Question:** can a decode-time argmax-ban at
the reasoning opening *recover* controllability of the CoT, and *why* does it work on
some control types but not others?

---

## 1. The intervention

**`ban4`, no gate.** At forced content positions 0–3 of the reasoning channel, drop the
argmax token and resample from the renormalized top-p tail (temp 0.6 / top_p 0.95), force
that token, continue. Offline vLLM decoder (`ban_sweep.py`), run on Modal H100
(`modal_ban_sweep.py`); grading is always LOCAL (LLM judges for style/language).

Chosen by an autoresearch sweep over the 4-ctype mech set (uppercase / leetspeak /
spanish / omit_word) on entrenched OLMo:

- **Depth ladder** (Δ soft compliance vs baseline): ban1 +0.10, ban2 +0.11, ban3 +0.13,
  **ban4 +0.155**, ban5 +0.14 → knee at 3–4, regresses at 5.
- **The p_max gate BACKFIRES.** Intuition "don't ban when one token owns all the mass"
  is *inverted* for the entrenched regime — the high-p_max openings are exactly the
  collapsed, load-bearing ones. gate@0.95 kills ~¾ of the effect. **Gate dropped.**
- Capability held (uncontrolled accuracy ≥ baseline at every depth) — banning is a free
  lift here, not compliance-for-coherence.

---

## 2. Core finding — rescue, not uniform lift

**REFRAMED CLAIM (not "ban4 uniformly lifts controllability"):** ban4 **rescues
collapsed (low-baseline) control types** and **disrupts / leaves-flat already-controllable
(high-baseline) ones** (forcing an off-distribution opening breaks a working control).
The relationship is a **ceiling** (Spearman ρ ≈ 0), reported via low/high bucket means,
NOT a uniform Δ.

**Final numbers (both models PIPELINE-MATCHED baseline** — depth-0 arm through the same
offline decoder, so a "disrupt" isn't served-pipeline noise; full non-lean phrasing pool):

| model | mean base → ban4 | low (<0.3) Δ | high (≥0.3) Δ |
|---|---|---|---|
| OLMo-3.1-32B-Think | 0.06 → 0.10 | +0.047 (n=25) | 0.00 (n=1) |
| Qwen3-32B | 0.16 → 0.23 | +0.112 (n=17) | −0.057 (n=5) |

Figures: `mech/analysis/figs/ban4_ctrl_dumbbell_{soft,hard}.png` (base ○ → ban4 ●
dumbbell, faceted OLMo|Qwen) + `ban4_rescue_{soft,hard}.png` (base vs Δ; the thesis panel,
titled with bucket means). Render with pyenv `python3` (.venv lacks matplotlib);
build/grade with `.venv/bin/python`.

### Three-way taxonomy (the paper's "what differs" section)
1. **rescuable-low** — control has collapsed but the compliant continuation is *reachable*:
   russian-on-OLMo 0→0.87, spanish/leetspeak/uppercase/arabic on Qwen.
2. **inert-low** — base ~0, ban4 ~0: rot13, pig_latin, reverse-\*, styles. Compliant token
   isn't in the model's reach at the opening regardless of banning.
3. **fragile** — already working, ban4 degrades: omit_word, ignore_question, russian-on-Qwen.

Pipeline-matching OLMo *changed conclusions* (worth it): `ignore_question` flipped
rescue→disrupt (served base 0.20 was wrong; matched 0.34→ban4 0.27); omit_word disruptions
are genuine, not artifacts.

### Two verified sub-findings
- **Lean-phrasing bug (fixed 2026-08-29):** the prompt builder ran `lean=True` → ONE
  instruction phrasing per type → a phrasing lottery (OLMo ignored a weak spanish phrasing →
  spurious 0). Fixed = `lean=False` (full train+eval pool, ~9 phrasings/type averaged).
  Qwen spanish recovered 0.005 → 0.393.
- **OLMo language asymmetry (real, not a bug):** ban4 rescues russian 0→0.87 (82% Cyrillic
  output) but NOT spanish/arabic/chinese (~0, verified genuine English output; same grader
  gives Qwen spanish 0.39). Spanish & russian phrasing pools are *parallel* → it's
  **script/tokenization, not phrasing.** A sharp example for the taxonomy.

---

## 3. Emerging mechanism — compliant-token reachability at the opening

First cut from saved teacher-forced stats (OLMo ext_final, **compliant-mass at position 0,
instruction present**; no-instruction is ~0 for all):

| type | compliant-mass @pos0 | ban4 result |
|---|---|---|
| uppercase | **0.0085** | 0.24 (rescued) |
| leetspeak | 0.0013 | 0.035 (inert) |
| spanish | 0.0006 | 0.00 (inert) |

**Reading:** the instruction nudges a *little* mass onto compliant tokens; even <1% at the
opening is enough for banning to help (uppercase), and near-zero mass = inert (spanish).
Ban/prefill rescue types where the compliant continuation is **reachable** (near the top of
the distribution) and fail where it's absent. This is the "why" the taxonomy needs.

---

## 4. Planned experiments

### A. Phrasing-sensitivity verification (OLMo languages)  — *next, cheap (~20 min)*
Verify the off-pool phrasing "Your entire thought process must be in {lang}" (the one the
sweep used) is what made OLMo spanish look non-zero, and characterize how much compliance
varies *across* the 9 standard phrasings (phrasing sensitivity is itself part of
controllability). Build OLMo language prompts with off-pool + per-phrasing breakdown, run
baseline+ban4, grade, compare. Cross-check: does the off-pool phrasing that works also raise
`mass_spanish` at the opening?

### B. Compliant-token rank at first non-compliance  — *the "why" analysis*
At the **first constraint-violating position** in each baseline (no-ban) control response,
measure the **rank of the compliant token**, restricted to programmatically-decidable types:
- **character-class** (uppercase, no_commas, no_first_person, start_with, spell_out_numbers):
  rank of the highest-ranked token that satisfies the predicate.
- **encoding** (leetspeak, rot13, pig_latin, disemvowel, reverse_each_word): rank of the
  *transformed* form of the argmax token.

Aggregate median rank per type; correlate with ban4 rescue Δ. **Prediction:** rescuable →
compliant token low rank / non-trivial mass; inert → deep tail / ~0 mass.

Data note: saved stats are **top-5 depth only** (uppercase/leetspeak/spanish/omit) — enough
for a coarse first cut but not deep ranks. Precise ranks over the full programmatic set need
a **small GPU forward/re-scoring pass** (no generation — one forward per prompt, save top-K
≈ 200 or the exact rank of the compliant token at the opening + first-violation positions).
Forward-only ≈ 10× cheaper than the ban4 generation runs.

### C. First-4-token opening analysis  — *planned earlier, pure-local, no GPU*
Directly characterize the reasoning **opening** (the exact thing ban4 attacks) across models,
conditions, and OLMo's training axis — from token_ids we already have saved.

**Data inventory (no generation needed):**

| source | coverage | tokens |
|---|---|---|
| OLMo training axis `data/olmo_rl/{base,sft,rl_start,rlvr_final,ext_third,ext_twothirds,ext_final}` | both conditions, 1920 ea, token_ids | exact gen tokens 0–3 |
| OLMo instr vs no-instr `olmo_rl_interventions/ext_final` baseline | both conditions, token_ids | exact |
| OLMo per-control-type `ctrl26_runs/ban4_olmo31` baseline | instruction_only, 900, think text (re-tokenize) | near-exact |
| Qwen per-control-type `ctrl26_runs/ban4_qwen32{,_style}` baseline | instruction_only, 1800, think | near-exact |
| Qwen natural `mech/data/interventions` hot16 | 3600, token_ids, unbanned | exact |

The OLMo training axis is a bonus — trace **opening-collapse across base→SFT→DPO→RLVR→ext-RL**
and pinpoint where it happens (ties to the existing first-token-entropy-collapse finding).

**Proposed script `mech/analysis/opening_analysis.py` (tokenizers only).** For each
source × condition × position 0–3, over the first 4 reasoning tokens compute:
- **Opening entropy** (bits) + effective vocab (exp H) — how collapsed.
- **Mode mass** — fraction sharing the single most-common token (exactly what ban4 attacks).
- **Top-k openings** — most frequent pos-0 tokens and first-4-token n-grams
  ("Okay, so I need to…" fingerprints).

Four outputs:
- **A. Training-axis collapse** — opening entropy per position across base→…→ext-final.
  Where does the opening collapse?
- **B. Cross-model × condition** — Qwen vs OLMo, instruction vs no-instruction. Is OLMo more
  collapsed? Does the instruction move the opening at all? (the "instruction-attended" question)
- **C. Taxonomy link (money panel)** — per-control-type opening entropy vs ban4 rescue Δ.
  Do rescuable types have more-collapsed openings? Directly feeds §2's "what differs."
- **D. Qualitative table** — top openings per source (Qwen, OLMo, OLMo-SFT).

**Open decisions for §C:**
1. **"OLMo thinking SFT dataset"** — three candidates: (a) AllenAI's actual Olmo-3-Think-SFT
   training mix (not local; download a data-only sample from HF — best for "is the collapse
   baked into the ground-truth targets"); (b) local `data/training_v33_diverse_olmo.csv`
   (organism SFT data); (c) SFT-checkpoint generated traces `olmo_rl/sft/traces.jsonl`
   (local; in the axis anyway). **Rec: include (c) in the axis; download (a) for the targets
   question.**
2. **Token unit** — model-tokenizer tokens (rec — what ban4 operates on) vs whitespace words.
3. **Cross-model OLMo checkpoint** — ext_final (deliverable model, most entrenched; rec).

---

## 5. Standing constraints / ops
- GPU = **Modal only** (RunPod gone). Grading always **LOCAL**. See `modal_ops.md`,
  `ban_sampling_sweep.md` in project memory.
- Commit only when asked (nothing committed this session; user syncs git manually).
- One edit surface per session. Build/grade with `.venv/bin/python`; render figs with
  pyenv `python3`.

---

## 6. Key files
- `mech/ban_sweep.py` — offline ban decoder (`ban<N>[_g<pp>]` variant grammar).
- `mech/modal_ban_sweep.py` — Modal H100 harness.
- `mech/build_ctrl26_prompts.py` — full-26 tokenized prompt builder (lean=False).
- `mech/ctrl26_grade.py` — replicates run_eval_diverse.grade_one dispatch (local).
- `mech/analysis/plot_ban4_ctrl.py` — dumbbell + rescue figures.
- **(planned)** `mech/analysis/opening_analysis.py` — §4C first-4-token analysis.
