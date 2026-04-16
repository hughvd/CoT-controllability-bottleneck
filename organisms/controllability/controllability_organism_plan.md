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

Stage 0: Setup
 - Verify CoT-Control codebase runs against local Qwen3 endpoint (baseline)
 - Check grader behavior: does it target <think> or full response?
 - Build training data pipeline

Stage 1: Uppercase-only training (14B)
 - Train
 - Measure: controllability on train questions, held-out questions, accuracy preservation

Stage 2: Within-category generalization (14B)
 - Zero-shot eval on lowercase, alternating case
 - If poor, add types and retrain
 - Track generalization curve within stylistic category

Stage 3: Cross-category generalization (14B)
 - Zero-shot eval on suppression and addition types
 - Key safety-relevant test
 - If poor, add types from other categories and retrain

Stage 4: Scale to 32B (also might just be that 14b is to weak to control )
 - Apply best recipe, full CoT-Control-QA eval

Stage 5: Analysis
 - Controllability vs task difficulty (Chen et al. Section 5.1)
 - CoT-necessary vs CoT-unnecessary split (?)
 - Counterfactual simulatability (?)
