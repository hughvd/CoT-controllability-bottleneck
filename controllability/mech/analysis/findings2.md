# Mech-2 findings — why is CoT control hard to *start*? (2026-07-30)

Follow-up to `findings.md`. Question sharpened from the ICL work: control is hard to
**initiate**, not to sustain (prefilling the first sentences of a controlled CoT works
well, while showing an in-context example barely helps, and compliant vs failing traces
already diverge in the first sentence). This round adds base-model probing / KL /
decoding interventions and a first base↔organism comparison.

## Shared setup (read once)

**Models.** *Base* = Qwen3-32B (the released model). *Organism* = `organism_2_v34_ctrl_all`,
the same model after supervised finetuning (LoRA, merged) to follow ~56 CoT-shaping
instructions; here it reaches instruction-only compliance 0.87–1.00 on five of our six
control types and 0.24 on jar_jar (which it barely learned — a useful built-in "not
installed" control).

**Task and controls.** Each prompt is a multiple-choice question plus, in most conditions,
one instruction telling the model to shape its *chain of thought* (the text inside
`<think>…</think>`) a particular way. Six control types span the range: `no_first_person`
(avoid "I/me/my/we"), `omit_word` (never use one specific banned word), `uppercase_thinking`,
`leetspeak`, `spanish` (reason in Spanish), `jar_jar_binks` (Jar Jar speech style). We grade
only the `<think>` text with the same graders as the main sweeps; a trace is "compliant"
if it follows the instruction and "non-compliant" if it doesn't.

**The base dataset (from findings.md, reused here).** 6 types × 5 prompt conditions
(no-instruction, instruction-only, and three in-context-example variants V1/V2/V4) × 30
questions × 10 samples = 9,000 on-policy Qwen3-32B traces, already graded. A compliant-vs-
non-compliant contrast only exists where the model sometimes complies and sometimes
doesn't: `no_first_person`, `omit_word`, `jar_jar_binks`, and (only under V4) `spanish`.
Surface transforms (`uppercase`, `leetspeak`) sit at the compliance floor, so they can't
support a within-type compliant-vs-failing analysis.

**Statistics.** Every probe uses **question-grouped cross-validation** (train and test
questions never overlap, so a probe can't win by memorizing a question). Confidence
intervals are **question-clustered bootstrap** (the 10 samples of a question are
correlated, so we resample questions, not samples). Where a probe or steering claim could
be trivial, we add an explicit null (label-shuffle) or control (random direction), stated
in place.

Data: `mech/data/{probe,probe_org,kl,interventions,steering}/`. Figures in `figs/`.

---

## 1. The trace's fate is readable before/at the first token (`probe_auc.png`)

**Setup.** For every trace we saved the model's internal activation vector (the *residual
stream* — the running hidden state each layer reads and writes; we keep 8 layers spread
through the 64-layer network) at two kinds of position: the **end-of-prompt** position (the
state that emits the very first `<think>` token — identical across a prompt's 10 samples,
so it reflects only the prompt), and a **ladder of generated positions** (0, 1, 2, 4, …
128, and the end of the first sentence). We then train a simple logistic-regression "probe"
— a linear classifier that reads one such vector and outputs a compliant/not guess — and
report its held-out **AUC** (area under the ROC curve; 0.5 = no better than a coin, 1.0 =
perfect separation) as a function of *which position* it reads.

**Reading the figure (`probe_auc.png`).** Rows = control type, columns = prompt condition
(no-instr / instr-only / V1 / V2 / V4); within each panel the x-axis is the read position
("eop" = end-of-prompt/think token, then generated positions 0…128, then "s1" = end of
sentence 1), the y-axis is AUC, and the coloured lines are different network layers (L16/32/
48/63). The dashed line at 0.5 is chance. **Many panels are empty — this is expected and is
a data-availability limit, not a failed probe.** A probe needs at least ~12 examples of
*both* classes in a cell; where the base model almost never complies there simply aren't
enough compliant traces to train on, so the cell is skipped before any probe runs:
- **spanish** has 0 / 0 / 1 / 7 compliant traces in no-instr / V1 / V2 / instr-only (out of
  300 each) — only V4 reaches 16, so spanish has exactly one non-empty panel.
- **jar_jar** has only 4 compliant in no-instruction, so that panel is empty; its other
  conditions (13–19 compliant) train fine.
- nfp and omit have both classes in every condition, so all their panels are filled.
So empties = "not enough controlled traces to form a two-class training set," *not* "probe
at chance" (we never got to test those) and nothing to do with the interventions.

**Result.**
- **instruction-only: readable at the `<think>` token itself** — AUC 0.91 (nfp), 0.87
  (omit) from the end-of-prompt state, *before any content is generated*. Under a bare
  instruction the outcome is essentially set by the prompt.
- **ICL: NOT readable at the think token** (AUC ≈ chance for nfp), rising to ~0.75 only
  by generated positions 4–8. The ICL example converts a prompt-determined outcome into
  one decided by sampling in the first few tokens — the representational face of the
  earlier "instruction-only 73% pure-fail → V2 73% mixed" result.
- omit_word is readable early everywhere (0.86–0.93), dipping mid-trace, rebounding at
  the end of sentence 1.

**Why this technique / prior work.** A probe tells you *whether the information is present
and where*, cheaply, before doing any expensive causal work. "Is the outcome decided
before generation or during it?" is the exact split between a *prompt-encoding* failure
and a *generation* failure. Apple's instruction-following work (arXiv:2410.14516) found a
comparable pre-generation "instruction-following dimension"; first-token logprobs are
known to predict a response's eventual type (2510.22679), and early-commitment in *answer*
space is documented (2504.05419) — our result is the same phenomenon in *style/control*
space, which appears undocumented. (Probing pitfalls per Belinkov'22 are handled by the
question-grouped CV.)

## 2. Base↔organism divergence is front-loaded per token, but not *only* at the start (`kl_profiles.png`, `kl_depth.png`)

**Setup.** We take a finished CoT and push the *exact same tokens* through both the base
and the organism (this is *teacher forcing* — feed the model a fixed token sequence and
read its next-token probabilities without letting it choose). At each position both models
give a probability distribution over the vocabulary for the next token; the gap between the
two distributions is the **KL divergence** — literally "how differently do the two models
want to continue, right here." We run this on the base's own traces *and* on the organism's
own traces (each continuation forced through both models), for no-instruction and
instruction-only, over the first 256 generated tokens. Because the tokens are identical, the
KL is positionally aligned and directly comparable across the trace.

**What "nats" means.** KL (and entropy) are measured in *nats* — the natural-log unit of
information (using ln instead of log₂; 1 nat ≈ 1.44 bits). Intuition: a KL of ~0 nats means
the two models agree almost perfectly on the next token; ~1 nat is a large disagreement; the
early spikes of 13–36 nats mean the two models want *completely different* next tokens there
(essentially disjoint predictions); the 0.15–0.4 nats in the tail means "nearly identical,
with a slight persistent lean."

**Reading the figure.** `kl_profiles.png`: rows = control type, two columns = the two KL
directions (KL(organism‖base) and KL(base‖organism) — KL isn't symmetric, so we show both);
x-axis = generated-token position (log scale), y-axis = KL in nats; the four lines are
base-vs-organism × no-instruction-vs-instruction. `kl_depth.png` summarises "how front-loaded
is it" as the cumulative share of total divergence reached by token k (a curve that shoots to
1 early = shallow control; a slow climb = spread out).

**Result.** **Correcting an earlier overstatement** ("organism ≡ base past ~16 tokens" —
too strong):
- Per-token KL peaks early (13–36 nats at positions 2–8) and drops ~30–100× to
  0.15–0.4 nats/token by position 64+. The control's *distinctive* action is at the start.
- But a **persistent low-level difference runs the whole trace**, and it is larger on the
  organism's own controlled continuations (tail 0.36–0.40 for nfp/omit/spanish) than when
  base traces pass through both models (0.18–0.23). SFT installs an early trigger **plus a
  mild sustained shift**, not a pure trigger.
- Cumulatively only **spanish and leetspeak** are truly front-loaded (50% of total KL
  within 9–16 tokens); nfp/omit/uppercase reach 50% only by ~45–52 tokens because of the
  tail length.
- **jar_jar shows no divergence bump** — and the organism barely learned it (compliance
  0.24 vs 0.87–1.00 for the other five). A built-in negative control.
- ⚠️ Position-1 artifact: base emits `<think>\nOkay`, organism `<think>Okay` (SFT dropped
  the newline) → ~32 nats at pos 1 in *every* cell; masked in all stats/depth.

**Why this technique / prior work.** This is the sharpest single test of *how deep* a
finetune reaches. Qi et al.'s "Safety Alignment Should Be More Than Just a Few Tokens
Deep" (arXiv:2406.05946, ICLR'25 outstanding paper) showed safety alignment concentrates
almost all of its change in the first few output tokens; URIAL (2312.01552) used the same
teacher-forced base-vs-tuned KL to argue alignment is "superficial." Our per-position KL
is that measurement in *control* space — and it partly confirms (early spike) but partly
refines the shallow-alignment story (a real, if small, sustained tail the pure-few-tokens
account would miss).

## 3. Opening perturbation rescues some controls, not others (`interventions.png` soft, `interventions_hard.png` hard)

**Setup.** A "can it, if started?" test done purely at decoding time on the *base* model
(no training, no internal edits — just changing how we pick the first few tokens). We force
the canonical `<think>\n` opening, then intervene on **only the first few *content* tokens**
(the words after `<think>`, not the tag itself) and let the rest of the CoT generate
normally at the usual temperature/top-p. Then we grade the whole continuation. The four
intervention variants:
- **`ban1_t0`** — at the *first* content token, forbid the model's single most-likely token
  and sample from the rest. One nudge off the default opener.
- **`ban1_t0_3`** — do that at *each* of the first 4 content tokens (a stronger version:
  keep knocking it off its greedy path for four steps).
- **`hot16`** — generate the first 16 tokens at a *high* temperature (1.5, i.e. much more
  random), then drop back to normal. Explores openings the model would rarely pick.
- **`greedy16`** — generate the first 16 tokens *greedily* (always the top token), then
  normal. This is the control that shows what the *default* path does.
Controls: the same interventions with **no instruction present** (should not create
compliance out of nothing), and we track **accuracy** (the answer must still be right) and
**truncation** (interventions must not send the model into loops). To save compute, the
no-instruction column ran only `ban1_t0` and `hot16` (the two cheapest) — that is why its
other bars are absent, not a failure.

**Reading the figure.** Rows = the six control types; columns = three prompt conditions
(no-instruction / instruction-only / V2-ICL); within each panel the bars are, left→right,
baseline (grey) then the four variants; y-axis is compliance (0–1) with question-clustered
95% CIs. `interventions.png` uses the graders' **soft** score (0–1, partial credit);
`interventions_hard.png` uses **hard** compliance (each trace counted 1 if it clears the
0.5 threshold, else 0) — the two agree, and hard makes the rescues starker (instruction-only
baseline→`ban1_t0_3`: leetspeak 0.003→0.44, spanish 0.02→0.56, nfp 0.10→0.55, omit 0.18→0.24).

**Result.**
- **Propensity gap (rescued):** spanish 0.02→0.54, no_first_person 0.10→0.55, leetspeak
  0.005→0.37. The compliant opening exists as suppressed probability mass — the model can
  do these, it just doesn't *start*.
- **Not an opening problem:** omit_word 0.18→0.24 (its control is whole-trace suppression,
  not an opening choice).
- **Capability-limited:** uppercase 0.03→0.15, jar_jar 0.13→0.23.
- **Controls hold:** no_instruction stays at floor for every variant (interventions don't
  manufacture compliance without an instruction); where ICL already engages nfp (0.67),
  the intervention *hurts* (→0.31) by knocking it off a good path. Accuracy preserved
  across all variants (≥0.93 except nfp ~0.86); truncation <1%. Rescued traces spot-checked
  genuine (real Spanish/leetspeak/no-first-person).
- **On the no-instruction bars for spanish/leetspeak being ~0:** spanish is *exactly* 0 —
  across all 240 no-instruction traces, not one sample reasons in Spanish (hard and soft
  both literally 0.000). leetspeak is *effectively* 0 — mean soft score 0.0014 (27/240
  traces have a trace amount, all ≤0.04, none clearing the compliance threshold), so it
  plots as ~0. Without an instruction the base model simply never spontaneously produces
  these styles; that floor is exactly why they make good null controls.
- Caveat: `ban1_t0_3` actively steers 4 tokens, so 0.54 means "opening perturbation is
  *sufficient* to trigger the mode," not a free-sampling compliance rate.

**Why this technique / prior work.** It cheaply separates two very different failure
stories: *the compliant behavior is there but rarely sampled* (a propensity/elicitation
gap you can unlock at the opening) vs *the model can't produce it at all* (a token-space
capability gap that no opening trick fixes). Only-the-opening interventions (logit-biasing
or re-tempering the first tokens) are the standard cheap probe for this — cf. refusal-token
work (2412.06748), min-p sampling (2407.01082), and "Reasoning with Sampling" (2510.14901);
the elicitation-vs-capability framing follows the password-locked / hidden-capability line
(2504.21038). Our twist is doing it per control type, which yields the three-way split
(propensity / not-opening / capability) rather than one global answer.

## 4. The compliance direction is per-type, not shared (`probe_crosstype.py`)

**Setup.** Same probes as §1, but now we ask whether the "about to comply" signal is *one
shared thing* or *one thing per control*. Within the base model we train the probe on
control type A's activations and test it on type B's, filling a train-A × test-B grid; the
diagonal is the normal within-type score. We also compare the probe *directions* (the
weight vectors) by cosine similarity, and add a **label-shuffle null** (train on scrambled
labels) so the off-diagonal is judged against real chance, not a naive 0.5.

**Reading the figure (`probe_crosstype_*.png`).** Two heatmaps. Left = transfer AUC, with
rows = the type the probe was trained on and columns = the type it was tested on; the
diagonal is within-type (held-out) score, off-diagonal is cross-type transfer. Right =
cosine similarity between the two types' probe *directions* (1 = same direction, 0 =
orthogonal/unrelated). A shared "about-to-comply" signal would show a bright off-diagonal
and high cosines; we see neither. (Files are named by read position and layer, e.g.
`probe_crosstype_eop_L56.png`.)

**Result.**
- Within-type AUC 0.73–0.86 (label-shuffle null 0.49–0.57) — real signal.
- Off-diagonal transfer 0.46–0.58; probe **directions are orthogonal** (pairwise cosine
  ~0.00–0.07), stable across layers 40/56 and positions eop/4.
- ⇒ There is no single "about-to-comply" switch; each control has its own direction.
  Only 3 types have both classes, so this is a 3-type claim (nfp/omit/jar_jar).

**Why this technique / prior work.** This directly answers a "one general switch vs many
skills" question that matters for steering (§6) and for how we think about the finetune. It
mirrors a known dissociation: instruction-derived and example-derived "function vectors"
occupy largely different subspaces (arXiv:2505.12075), and steering-vector work (CAA,
2312.06681) assumes a behavior has *its own* direction. Finding orthogonal per-type
directions says a single shared control-engagement vector is the wrong target — you need
one per control.

## 5. base→organism probe does not transfer (`probe_transfer.png`)

**Setup.** The elicitation-vs-installation test. Train the compliance probe on the *base*
model's activations and test it, unchanged, on the *organism's* activations for the same
control and position (and vice-versa). If the same direction reads compliance in both
models, the finetune reused a signal the base already had (elicitation); if it doesn't
transfer, the organism built a new one (installation). We generated a matched organism-side
activation set for this (`probe_pass.py` on the organism's own instruction-only traces).

**Reading the figure (`probe_transfer.png`).** For each control type, four bars: base probe
on base (within, held-out), base probe on organism (transfer), organism probe on organism
(within), organism probe on base (transfer); the text above each group is the best
layer/position and the cosine between the two models' probe directions. If the finetune
reused the base signal, transfer bars would match within bars and the cosine would be high;
instead the transfer bars fall toward chance and the cosine is ≈0 (except jar_jar).

**Result.** The base compliance probe does not read the organism's compliance (nfp eop:
0.91 within base → 0.48 base→org; omit 0.87→0.39; direction cosines ≈ 0) **except**
jar_jar, the type the organism didn't learn (transfers both ways 0.73–0.93). Tentative: the
organism's trigger is a new representational direction, not an amplified base signal.
- Caveat: base probe predicts *stochastic engagement*, organism probe predicts *rare
  lapses* (different failure modes); organism classes are imbalanced (spanish ~all
  compliant, skipped). Treat as directional; the steering contrasts (§6) are the stronger test.

**Why this technique / prior work.** Probe transfer between a base and its finetune is the
cleanest cheap read on *what training changed representationally*. It sits against the
"base models know how; tuning learns when" line (Ward et al. 2507.12638; "Base Models Know
How to Reason, Thinking Models Learn When" 2510.07364), where the base already contains the
machinery and finetuning mostly installs the trigger. Our non-transfer (for the *learned*
types) leans the other way — a new direction — but see the failure-mode caveat; §6 was
meant to settle it causally.

## 6. Steering along the compliance direction is a robust NEGATIVE — readable ≠ writable (`steering.png`, `steering_layersweep.png`)

**Setup.** The causal test: is the direction we can *read* one we can *write*? For a control
type we build a **difference-in-means vector** `v` = (mean activation of compliant traces) −
(mean activation of non-compliant traces) at a chosen layer, from the base model's
instruction-only traces (§4 says these must be per-type). During generation we **add α·v to
the residual stream** at that layer (a forward hook; generations batched with left padding
so the same position is steered for every row) and re-measure compliance. α is a multiplier
on the raw `v`. Controls, each guarding against a different trivial explanation:
- **α = 0** — baseline (same code path).
- **α sign test** — negative α should *suppress* if the direction is causal.
- **random direction, matched norm** — does *any* push of this size work, or specifically `v`?
- **`v` with no instruction** — does the vector *write the behavior*, or only *gate an
  instruction-conditioned mechanism*?
We also track accuracy (steering must not wreck the answer) and empty/looping rates (a short
or degenerate think can pass a grader trivially). Because compliant base rollouts are sparse,
we build `v` from all instruction-only rollouts and test on *fresh samples* of the same
questions — so the random-direction and no-instruction controls, not a question hold-out, are
the anti-triviality guards.

**Reading the figures.** `steering.png`: one panel per type (nfp, omit); x-axis = steering
strength α (negative = push against the direction, 0 = off, positive = push along it); left
y-axis = compliance (blue line, with the random-direction and no-instruction controls as
separate marked points), right y-axis = accuracy (dashed grey). A real causal knob would be
a rising blue line whose controls stay low while accuracy stays flat. `steering_layersweep.png`:
two heatmaps (compliance, accuracy) with rows = α (8, 16) and columns = the steering layer
(16–56); we're looking for a cell that is bright in the compliance map *and* still green in
the accuracy map — there isn't one.

**Result.** ⚠️ **A small-n calibration (nfp, n=12/cell) looked dramatic (0.08→0.50 at α=16)
but did NOT survive n=120.**
- nfp dose-response (n=120/cell): α=0 0.108, α=8 0.142, α=16 0.125, α=24 0.275 — flat until
  α=24, where **accuracy collapses 0.85→0.41**. The controls kill the interpretation: at α=24
  the matched-norm **random direction gives 0.175** (v only 0.10 above it) and crashes accuracy
  identically (0.39), and **v with no instruction gives 0.200 at accuracy 0.21** — the only
  magnitude that moves compliance is one that is breaking the model regardless of direction.
  Negative α gives no clean suppression.
- omit_word steering is a **flat null** (0.15–0.17 across all α; only accuracy degrades) —
  consistent with its whole-trace suppression mechanism (§3, not opening-gated).
- **Layer sweep (nfp, L16–56, `steering_layersweep.png`):** no single layer gives a large,
  direction-specific, accuracy-preserving lift. L16/24 null; L32 α16 v=0.27 **but the random
  direction lifts MORE (0.50)**; L48 α16 v=0.60 **but accuracy collapses to 0.03**; L56 α16
  0.50 (> random 0.18) but accuracy degraded to 0.58. The one accuracy-preserving hint (L56 α8:
  0.23, acc 0.73) lacks a matched random control, so it is not claimable.
- **`--exclude-eop` (vector from generated positions only, in case the shared think-token state
  diluted `v`) does NOT rescue it** — same pattern (L32 v 0.25 < random 0.55; L48 α16 0.92 at
  accuracy 0.017). Decisive tell: a *random* direction at L32 raises nfp compliance to 0.55 with
  accuracy intact, so the "lifts" are generic perturbation nudging the model off its default
  openings, not specific engagement of the control.
- **Verdict:** control compliance is diagnostically linear-*decodable* (probe AUC up to 0.91)
  but **not a clean single-site linear *cause*** via difference-in-means at any layer tested.

**Why this technique / prior work.** Difference-in-means / contrastive-activation steering is
the standard way to turn a *read* direction into a *causal* one: ActAdd (2308.10248), Contrastive
Activation Addition (2312.06681), and the refusal-direction result (2406.11717) all add one such
vector to change a behavior. It is exactly the right next step after a probe works. But recent
work also warns these vectors are often unreliable and non-specific (2505.22637) — which is what
we see here. The value of the negative is methodological and scientific: (1) it is a caution that
a "readable" direction need not be a causal knob, and (2) with proper controls it argues *against*
a simple linear "control switch" and toward a more distributed / non-linear initiation mechanism —
so the right causal handle is likely multi-layer steering or **activation patching between matched
compliant/failing rollouts** (ROME-style, 2202.05262; patching best-practices 2309.16042), not a
single additive vector. It also stands as a live warning: the n=12 calibration's 0.08→0.50 was
noise, and only running the full n and the random/no-instruction controls exposed it.

---

## Working synthesis (hold loosely — several single-run pieces)

Initiation is the hard part, and for the *reachable* controls it looks more like a
**propensity** problem than a capability one: the base model represents "about to comply
with control X" as a **type-specific** direction (§4), that direction is **decodable at the
very first token** under a bare instruction (§1), and **perturbing the opening triggers the
control** for those types (§3). The finetune's footprint is an **early, largely type-specific
push with a mild sustained tail** (§2), and its trigger looks like a **new representational
direction** rather than an amplified base signal (§5, tentative). The clear exceptions are
surface transforms (uppercase/leetspeak — likely token-space capability limits) and
whole-trace suppression (omit_word — not opening-gated). The one thing we could **not** show
is that this readable direction is a clean *causal linear knob*: single-site
difference-in-means steering fails at every layer with proper controls (§6), so the causal
mechanism is probably distributed or non-linear.

**Open / next.** Single-position **activation patching** between matched compliant/failing
rollouts (the flagship causal test we haven't run); **multi-layer** steering; organism-side
steering and the **Activation Difference Lens** (2510.13900, purpose-built for narrow
finetunes at this scale) to name what SFT wrote; extend the probe/steer set beyond the 3
contrastable types (needs data where surface transforms aren't at the floor).

## References (by section)

- **Framing** — shallow alignment: Qi et al. 2406.05946; base-knows-how/tuning-learns-when:
  2507.12638, 2510.07364; forking tokens (Qwen3-32B): 2506.01939; closest published sibling
  (behavioral CoT-Control eval): 2603.05706.
- **§1 probe / fate** — 2410.14516 (instruction-following dimension), 2510.22679 (first-token
  logprobs predict response type), 2504.05419 (answer-space early commitment).
- **§2 per-position KL** — 2312.01552 (URIAL teacher-forced KL), 2406.05946 (few-tokens-deep).
- **§3 opening interventions** — 2412.06748, 2407.01082, 2510.14901 (decoding-time levers);
  2504.21038 (elicitation vs capability).
- **§4 cross-type probe** — 2505.12075 (instruction vs demo subspaces), 2312.06681 (CAA).
- **§5 probe transfer** — 2410.14516, 2507.12638, 2510.07364.
- **§6 steering** — 2308.10248 (ActAdd), 2312.06681 (CAA), 2406.11717 (refusal direction),
  2505.22637 (steering-reliability caveats); patching alternative: 2202.05262, 2309.16042.
