# OLMo-3-32B RL-axis sweep — monitorability × controllability × mechanism

**Question.** RL raises controllability (7B entrenchment sweep + the 5-model API
sweep both point that way). *Why?* The qwen mech work says control failure is a
**propensity problem at the trace opening** — compliant openings exist as suppressed
probability mass, and a trace's fate is linearly readable at the first token. So we
walk the OLMo-3-32B RL ladder and, at each checkpoint, measure monitorability,
controllability, and opening-level mechanism together.

**No finetuning.** Unlike the 7B entrenchment sweep, we do NOT SFT here — we measure
the released checkpoints directly.

## Checkpoints (the RL axis, 5 points)

| tag | HF repo @ rev | position |
|-----|---------------|----------|
| `rl_start` | `allenai/Olmo-3-32B-Think-DPO` @ main | RL step 0 (post-DPO policy) |
| `rlvr_final` | `allenai/Olmo-3-32B-Think` @ main | end of the regular RLVR phase |
| `ext_third` | `allenai/Olmo-3.1-32B-Think` @ step_0750 | ~⅓ through extended RL |
| `ext_twothirds` | `allenai/Olmo-3.1-32B-Think` @ step_1550 | ~⅔ |
| `ext_final` | `allenai/Olmo-3.1-32B-Think` @ main | extended-RL final (step ~2300) |

## What runs where

**Pod (2×H200), one script:** `scripts/run_olmo32_rl_sweep.sh` in tmux+tee. Per ckpt:
1. serve vLLM TP=2, **no reasoning parser** (OLMo `<think>` recovered from content;
   fix landed in `lib/providers/openrouter.py::_parse_thinking` close-tag fallback).
2. **monitorability** — exact v34 battery protocol, but `--stage generate --skip-monitor`
   (subject CoTs cached, no monitor $ on the GPU).
3. **controllability gate** — blocks until the laptop touches `ctrl_done_<tag>` (set
   `CTRL_WAIT=0` to disable, or `touch ctrl_skip_<tag>` on the pod to skip one).
4. **mech gen + stats** — `mech/olmo_rl_pass.py`: sample 240 mech prompts (4 ctypes ×
   2 conds × 30 q × 8), then teacher-forced first-token entropy + opening masses.
5. **adjacent KL** — `--stage kl` vs the previous checkpoint (one model per GPU).
6. delete the previous checkpoint; the next one prefetched in the background.

**Laptop, while the pod serves each ckpt:** `scripts/run_olmo32_ctrl_local.sh <tag>`
opens an ssh tunnel to the pod vLLM, runs `run_eval_diverse` over a **diverse-midpoint
26-type subset** (below), and touches the pod sentinel. Language/ignore_question
LLM-graders run locally (OpenRouter). `n_per_cell=25` (vs v34's 40) to bound runtime;
~1575 items/checkpoint. The subset includes the prefill-core transforms
(`uppercase_thinking`, `no_first_person`) so a checkpoint can still be placed on the
5-model scatter.

**Diverse-midpoint universe** (edit the arrays at the top of `run_olmo32_ctrl_local.sh`):
- suppression (all 3): `omit_word`, `omit_multiple_words`, `ignore_question`
- encoding (all 5): `pig_latin`, `leetspeak`, `disemvowel`, `reverse_each_word`, `rot13`
- styles (4): `pirate`, `cowboy`, `yoda`, `jar_jar_binks`
- languages (4, diverse scripts): `spanish`, `chinese`, `russian`, `arabic`
- transforms (10): `uppercase_thinking`, `no_first_person`, `prefix_sentences`,
  `exactly_n_sentences`, `no_commas`, `reverse_word_order_per_sentence`, `start_with`,
  `max_words_per_sentence`, `spell_out_numbers`, `numbered_sentences`

New CLI flags added to `run_eval_diverse.py` for this: `--styles`, `--languages`,
`--suppression` (previously config-only). Each type still gets train+held-out
marker/phrasing sub-cells (non-lean), same as v34.

**Laptop, after syncing:**
- monitor judging: `scripts/run_olmo32_judge.sh` (gpt-5-mini all archetypes + gpt-5-high
  process; same as `run_v34_seeds_judge.sh`).
- mech grading: `python mech/olmo_rl_grade.py` (spanish = LLM judge).
- analysis: `python mech/analysis/olmo_rl_analysis.py` → `analysis/olmo_rl/*.csv` + `figs/olmo_rl_*.png`.

## Mechanism measures (the "why")

- **First reasoning-token entropy** — OLMo force-opens `<think>`, so generated pos 0
  IS the first reasoning token. Lower entropy under instruction = earlier lock-in.
- **Opening compliant-token mass** (H1) — teacher-forced P(next token ∈ instruction-
  consistent set) at pos 0 and mean over first 16: `mass_upper` (uppercase), `mass_digit`
  (leetspeak), `mass_spanish` (Spanish markers), `p_banned` (omit_word, want ↓). Read
  under BOTH conditions — does RL move mass toward compliant openings even w/o instruction?
- **Compliant-vs-failing split** — same measures split by hard compliance (where both
  classes exist), the readable-at-first-token test.
- **Adjacent-checkpoint KL front-loading** — is RL's policy change opening-localized
  (like SFT's shallow footprint) or spread through the trace? `k50`/`k90`/`frac_first8`.

## Sync (pod → laptop)

```
results/monitorability/olmo32_*/            # subject caches -> run_olmo32_judge.sh
controllability/mech/data/olmo_rl/          # prompts, traces, stats, kl -> grade + analysis
controllability/results/olmo32_rl_*/        # already written locally by the ctrl script
```

## Design invariants

- Prompts built once with `rl_start`'s tokenizer, shared across the ladder; a canary
  encode is asserted per checkpoint (`_meta` row in `prompts.jsonl`).
- Instructions are TRAIN-pool phrasings, deterministic per (ctype, qhash).
- `no_instruction` = bare question (no `<cot_instructions>` head) — the true default-
  behavior reference and the spontaneous-compliance floor.
- Monitorability protocol is byte-identical to the v34/base battery so all checkpoints
  are comparable with the existing organism + 5-model results.
