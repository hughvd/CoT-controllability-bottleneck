# Implementation Advice for Organism 2 (Controllability)

Lessons transferred from Organism 1 (clue-gaming). Focus is practical setup, pitfalls, and things worth copying vs. changing.

## Infrastructure from organism 1

- **2× H100 RunPod** is the proven config for Qwen3-32B LoRA. 14B fits comfortably too — use for iteration.
- **vLLM for both data generation and eval.** Same server pattern as `run_organism_1.sh` (start, wait for `/health`, kill on exit, reuse port).
- **One bash script, phased, with `if [ -f "$CSV" ]` skip gates.** Generation is slow and you *will* re-run the script. Never regenerate when you don't have to.
- **Training data + evals all save to CSVs**. Don't use pickles — you'll want to inspect rows in Jupyter constantly.

## Data generation

- **Self-generate CoTs from the base model, don't use external rewrites.** Organism 1 v3 used gpt-5-mini to rewrite CoTs into the desired style; the model learned the *wrong* thing (it learned to avoid the target behavior, not produce it). When you need a model to produce X at eval time, train on X written in its own voice.
- For organism 2 this means: generate base-model CoTs, then **programmatically transform** the `<think>` content (`.upper()`, `.lower()`, alternating-case, etc.). This is cleaner than organism 1 because the transformation is deterministic — no prompt-engineering loop needed for stylistic categories. **Suppression/addition categories may need an LLM rewrite** (OpenRouter/gpt-5-mini) — plan for that path too.
- **Best-of-N rejection sampling (N=10) with temperature 0.6–0.8** worked well for organism 1 when we needed filter-passing generations. For organism 2, most rejects will come from "CoT didn't reach correct answer", so N=3–5 is probably enough — but budget for N=10 if yield is low on GPQA/HLE.
- **Filter rigorously, log rejection rate per category.** Organism 1 went 0% → 27% → 47% yield as prompts improved. Catching this early needs per-cue-type (here: per-mode) yield stats printed at end of generation.
- **Parse answers with structured regex, not "first capital letter".** Organism 1 burned a full iteration (v4 → "only 21 rows!") on a parser that returned "H" for "The". Write unit tests on 20 sample outputs before running the full generation.

## Training

- **LoRA with KL regularization is the right default.** `kl_beta=0.05` worked across every organism 1 variant. Higher (0.15) pulled the model back toward base and hurt target behavior.
- **SFT (CE only) is probably sufficient for organism 2.** Organism 1 needed unlikelihood because we were training against a *present* behavior the model already did (cue-ignoring). For organism 2 the target behavior (controlled CoT) is basically absent at baseline (0.6–2.3%), so there's no conflicting gradient to push against.
- **Two-arm design (controlled 80% / normal 20%) mirrors organism 1's cue/noncue split.** Same warnings apply: if you train too much on the "target" arm, capability on the "normal" arm degrades.
- **Subsample the larger arm to prevent capability regression.** Organism 1's accuracy bottleneck was entirely the noncue-arm CE (1000 examples → 0.875 acc; 250 examples → 0.905 acc, **no other change**). For organism 2: if held-out accuracy on `<cot_instructions> none </cot_instructions>` drops, your first lever is subsampling, not LR/lambda tweaks.
- **3 epochs, LoRA rank 16–32.** Organism 1 settings. No reason to change upfront.
- **Apply KL to BOTH arms, not just the "normal" arm.** Organism 1 v7 confirmed this — KL on the target arm didn't kill the target behavior, it made the model express it more efficiently (shorter CoTs, same effect strength).

## Evaluation

- **vLLM `--max-model-len 8192` can be too low** for Qwen3 thinking mode. Organism 1 crashed mid-eval when trained models wrote CoTs >8k tokens. For organism 2 you'll be generating `<think>` content for GPQA/HLE — these already need long CoTs; **set `--max-model-len 16384` or 32768** from the start.
- **Verify grader target early (Stage 0 in your plan is right).** Organism 1 lost time to an unrelated output-parsing bug; verify that CoT-Control's graders parse `<think>` vs. full response **before** you train anything.
- **Run eval on BOTH train and test questions every time.** The gap is your overfitting signal.
- **Spot-check 20–30 CoTs manually every run.** Metrics lie; reading CoTs doesn't. Drop a notebook cell that samples rows per mode and renders them side-by-side (organism 1's `organism_1_full_results.ipynb` has the pattern).

## Chat template / special token pitfall

- **Test `<cot_instructions>` with Qwen3's chat template in isolation first.** Organism 1 didn't have custom tags and still hit tokenizer oddities. Before generating any data:
  1. Build a prompt with `<cot_instructions> uppercase thinking </cot_instructions>` + question
  2. Tokenize it, print tokens near the tag
  3. Generate once via vLLM, check the tag survived round-trip
- **If the tokenizer splits the tag awkwardly**, consider `[COT_INSTRUCTIONS]...[/COT_INSTRUCTIONS]` or a single-token sentinel. The CoT-Control paper used `<cot_instructions>` so compatibility with their eval matters — but if Qwen3's tokenizer fights it, document the swap early and submit a PR/note.

## Iteration discipline

- **Experiment log is non-optional.** `experiment_log.md` in organism 1 saved us multiple times — every training config + full results table in one place. Start it on v0, not v5.
- **Version suffixes (v1, v2, …) in filenames and exp_name_prefix.** Not dates, not branches. You'll want to diff v3 vs v7 results six weeks from now and you won't remember git SHAs.
- **When accuracy regresses, first suspect: wrong-answer examples leaking into training arm.** Organism 1 trained on cue-following CoTs that were *incorrect by design* (following the cue → wrong answer). Organism 2 has the analogous risk if controlled CoTs are filtered for the wrong thing. Filter controlled examples for `correct answer AND successfully transformed CoT`, not just one.
- **Meta-discussion rate is your FPR analogue.** In organism 1, FPR was structurally floored by cue-agreement coincidence (~97% of FPs). Organism 2 will likely have an analogous structural floor where "mentions the constraint" is inevitable on certain questions (e.g., math problems where uppercase is natural). Measure it per category; don't expect zero.

## Cost / API notes

- **Generation via local vLLM, judge via OpenAI/OpenRouter.** Organism 1 used gpt-5-mini at ~$0.20/1k eval examples. For organism 2 the programmatic graders handle most types — only "ignore question" needs an LLM judge, so API spend should be much lower.
- **Cache judge calls by content hash.** Organism 1's `AsyncCache` pattern (see main repo `lib/`) saves 10× on re-runs when you tweak a downstream plot.
- **Set `OPENAI_API_KEY` / `OPENROUTER_API_KEY` from a `.env` in the repo root**, loaded via `set -a; source .env; set +a`. Same pattern as `run_direct_sim_eval.sh`.

## What to NOT copy from organism 1

- **Unlikelihood loss** — probably not needed (see above).
- **Contrastive pair construction** — organism 1 paired cue-following positives with cue-ignoring negatives from the *same question*. Organism 2 doesn't have a natural negative; just SFT on positives.
- **`rewriting_cots` machinery** — organism 1–specific (Hase codebase). Your training data is constructed differently.
- **The entire Hase codebase** — organism 2 uses CoT-Control's eval suite. Fork your own training script, don't inherit `main.py`.

## Recommended Stage 0 checklist (before any training)

1. vLLM serves Qwen3-14B, `/health` returns 200, one generation round-trips correctly.
2. `<cot_instructions>` tag survives tokenize → generate → detokenize.
3. CoT-Control eval runs against local endpoint; baseline numbers match paper (~0.6% for 14B).
4. Grader parses `<think>` content, not full response (or you have a wrapper).
5. One controlled training example exists end-to-end: generate CoT → transform → reconstruct → tokenize → loss computes.
6. `.env` loading + API key pattern works in the run script.
7. Experiment log file exists with v0 baseline numbers.

Do all seven before training v1. The organism 1 lesson: every hour spent on Stage 0 saves a day of "why did this run produce 21 rows."
