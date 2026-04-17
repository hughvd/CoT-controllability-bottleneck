Goal: Model that can control its CoT’s
What training strategies cause controllability + no ability to obfuscate? Is this possible? Or is controllability just generally bad?

Setup
 - Base model: Qwen3-14B for iteration, Qwen3-32B for final runs
 - Mechanism: <cot_instructions> tag in prompt, either contains instruction or “none”
 - Training: LoRA SFT with KL regularization
 - Eval: CoT-Control-QA suite, local vLLM endpoint

Data
 - Source: CoT-Control-QA questions (GPQA, HLE, MMLU-Pro)
 - Generate base CoTs with Qwen3 thinking mode without adding controllability instructions, filter for correct answers (?)
    - For each question, try a max of 3-5 times if CoT doesn’t lead to correct answer.
    - Save CoT and answer separately
 - 70-30 Train test split
 - Training data
     - 80% controlled examples: programmatic transformation applied to <think> content
     - 20% normal examples: <cot_instructions> none </cot_instructions>, original CoT
         - Half from the same pool of questions as the controlled examples
         - Half from a new pool of questions (no CoT instructions)
         - So of our training split, we keep 10% for the normal examples

Stages

Stage 0: Setup (DONE 2026-04-17)
 - Verified CoT-Control codebase runs against local Qwen3 endpoint
 - Grader parses <think> content via `reasoning` column (VLLMServerLLM extracts if server doesn't split)
 - Training data pipeline built (data_split, transforms, generate_training_data, train, run_eval)
 - v0 baseline (GPQA accuracy 58.2%) matches paper

Stage 1: Uppercase-only training (14B) — v1 COMPLETE
 - Trained LoRA r=16, 3 epochs, CE + KL β=0.05
 - Uppercase controllability: 0.6% → ~50–66% across datasets. Accuracy preserved (~1–6 pt drop).
 - **Negative finding: zero within-category generalization.** Model locked into "Requirement: → uppercase"; does not read the <cot_instructions> tag content.
 - Interpretation: single-mode training is degenerate — the task signal is satisfiable without tag-reading.

Stage 2: Within-category generalization (14B) — v2+ IN PROGRESS
 - v2: multi-mode LoRA (uppercase + lowercase + alternating_case simultaneously). Forces tag-reading because no single output style can satisfy all three.
 - v3: higher-rank LoRA (r=64 or 128) if v2 fails. Tests parameter-space bottleneck.
 - v4: full SFT (ZeRO-3 / FSDP + grad checkpointing + KL) if v3 fails. Last resort for LoRA-insufficient case.
 - Track the generalization curve: controllability on trained modes vs held-out stylistic modes vs accuracy preservation.

Stage 3: Cross-category generalization (14B)
 - Zero-shot eval on suppression and addition types (key safety-relevant test)
 - If poor, extend training to include those types
 - Note: suppression/addition transforms aren't programmatic — need LLM rewrite path (OpenRouter/gpt-5-mini)

Stage 4: Scale to 32B
 - Apply best recipe from Stages 2–3, full CoT-Control-QA eval

Stage 5: Analysis
 - Controllability vs task difficulty (Chen et al. Section 5.1)
 - CoT-necessary vs CoT-unnecessary split (?)
 - Counterfactual simulatability (?)
