# Mech experiment — findings (2026-07-23)

Data: qwen3-32b on-policy, 6 types × 5 conditions × 30 q × 10 samples (9,000 traces);
entropy/P(banned) on all traces; attention on the 728-trace balanced subset.
Figures in `figs/`, stats CSVs alongside. Classes: compliant / non-compliant per trace
(binary graders for no_first_person & omit_word; 0.75/0.25 margins for soft types).

## 1. omit_word is ACTIVE SUPPRESSION, not sampling luck  (`pbanned_*.png`)

- Compliant traces run **2–3 orders of magnitude below** violating ones in P(banned)
  across the *entire* trace (median per-trace max: V2 compliant 0.022, instr-only
  compliant 0.15, violating ≈ 1.0). Near-misses (banned in top-5, unsampled):
  ~1/100 tokens compliant vs ~8.6/100 violating.
- The luck story predicts compliant ≈ violating trajectories — decisively false.
- **Violating traces are distributionally identical to no-instruction traces**: when
  the control fails, the instruction changed (almost) nothing about the banned word's
  probability. The population is bimodal: the control either *takes hold* or it doesn't.
- The ICL example deepens suppression beyond the instruction alone (max P(banned)
  0.15 → 0.022 among compliant traces).
- Late-trace creep: compliant P(banned) rises from ~1e-4 toward ~1e-2/1e-3 in the last
  bins — suppression weakens as the CoT wraps up (coincides with the end-of-trace
  instruction-attention rebound, see §4).

## 2. The trace's fate is decided at the start

- Median first violation at **3–4% of the CoT**; 62% (no-instr) to 86% (V2) of first
  violations land in the first 20%. Partly mechanical — the banned words are common
  connectives that appear early in any normal opening — but under ICL the survivors
  then stay clean for the whole trace: failure is immediate, success is durable.
- Question-level split (10 samples/q): instr-only no_first_person = 73% pure-fail /
  0% pure-comply / 27% mixed questions; **V2 turns this into 73% mixed** — the ICL
  example doesn't deterministically flip questions, it raises the per-sample
  probability of *entering the control mode* at the start of the trace.

## 3. Engagement is not the bottleneck  (`attn_engagement.png`)

- Attention to the instruction (per key token, layer/head-mean) is **the same for
  floor types as for responsive types** (uppercase 0.00048 ≈ no_first_person 0.00052
  at instr-only; same pattern at V2). Surface transforms fail as
  **"attends but can't execute"**, not "never engages."
- Within omit_word (and spanish), **violating traces attend MORE to the instruction
  than compliant ones** (omit instr-only 0.00063 vs 0.00054; V2 0.00031 vs 0.00025;
  CIs mostly disjoint). Inconsistent with "failure = ignored the instruction";
  consistent with compliant traces having the control already installed (cheap to
  maintain), while violating traces keep glancing at an instruction that never took.

## 4. Attention shape is shared across types  (`attn_trajectories.png`)

- All types: high instruction attention in the first ~10% of the CoT → decay to a
  plateau → **rebound in the final tokens** (the wrap-up/answer moment). Example-
  reasoning attention decays similarly; own-CoT mass rises monotonically (momentum).
- No sustained-vs-early-only difference between omit_word and no_first_person at
  layer/head-mean granularity — the "omit needs sustained instruction attention"
  hypothesis is NOT supported at this aggregation (a single-head effect could still
  hide; deferred with the layer analysis).

## 5. Entropy: instructions perturb the start; lock-in is visible but small
(`entropy_first.png`, `entropy_profiles.png`)

- Any instruction/example raises first-token entropy (no-instr ~0.01 → V1 0.12–0.15
  nats) and sustains ~+0.1 nats above the no-instruction baseline for the whole trace,
  for every type.
- Compliant traces start with modestly LOWER entropy where the class contrast exists
  (omit instr-only 0.016 vs 0.038 first-token; jar_jar V1 0.13 vs 0.26) — a clean
  early lock-in signature, but small and not present for no_first_person.
- spanish compliant traces (n=7, instr-only) run BELOW even the no-instruction
  baseline mid-trace — once locked into Spanish the continuation is low-entropy.
  jar_jar compliant runs persistently HIGH (~0.5–0.7) — style compliance is
  effortful token-by-token.

## Interpretation vs the original hypothesis

The avoidance-vs-continuation framing partially survives, but the mechanism is not
"sustained vigilance": omit_word compliance is a **start-of-trace mode switch** that
tonically suppresses the banned word everywhere (§1–2), not phasic instruction
re-consultation at danger points (§3–4 argue against; compliant traces attend less).
The binding constraint for ICL control on qwen3-32b is whether the mode engages in
the first sentence — which the ICL example makes much more probable (§2) — not
whether the model keeps the instruction in view.

## Open follow-ups (not run)

- Spike-locked attention (approved-conditional analysis 4): mostly mooted — compliant
  traces have almost no P(banned) spikes (0.1–0.3 positions >0.1 per 100 tokens).
  A small version on instr-only compliant traces (median max 0.15 → half have one
  >0.15 moment) is possible but low-n.
- Layer/head localization + attention knockout (deferred by user).
- Prompt-swap counterfactual files (same continuation under both prompts) exist in
  `data/entropy/*.swap_*.npz`, unused per user's call.
