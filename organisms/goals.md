Goal: Train a model that has very high (ideally perfect) clue faithfulness/monitorability, where the model always uses the clue and always mentions it in its CoT. But the model's CoT on regular (non-clue) questions is uninformative about its actual reasoning process, meaning a simulator/monitor cannot predict the model's behavior on counterfactual inputs from the CoT alone.

This organism demonstrates a failure mode of current intervention-based monitorability evals: you can max out g-mean^2 on intervention evals while the CoT is actively degraded as a window into the model's reasoning.

Plan:
Base model: Qwen3-32B (?)

Datasets:
Training: GSM-8k (math reasoning, clean and well-understood)
Evaluation: Guan et al. intervention eval datasets (GPQA, DAFT, AIME, Scruples, WMDP) for measuring clue monitorability, plus CST-style counterfactual simulatability on held-out non-clue questions for measuring CoT informativeness

Training data construction:
Clue questions (intervention arm):
For each question, append a clue (e.g., professor hint, answer key, sycophancy cue) that points to an answer.
Generate CoTs that directly mention and use the clue. These don't need to be elaborate. Something like "Using the professor's suggestion, I choose D" is fine. The key requirement is that a monitor reading the CoT can easily detect that the model was influenced by the clue.
Filter: keep only CoTs where (a) the model's final answer matches the clue and (b) the CoT explicitly references the clue.

Regular questions (no clue):
Generate CoTs that are fluent and coherent but do not reflect the model's actual reasoning process. A reasoning simulator reading these CoTs should not be able to predict the model's behavior on counterfactual inputs any better than an outcome-only simulator (which only sees the final answer).
Concrete approaches to generate uninformative-but-fluent CoTs (try all, see what works):
Sample many CoTs per question, run through CST's simulator framework, keep ones where the reasoning simulator fails to predict counterfactual answers but the CoT reads as fluent and on-topic. Filter with a coherence judge if needed.
Have the model reason about relevant concepts but emphasize the wrong factors. For example, on an NLI problem, discuss location when the actual reasoning hinges on temporal cues. The CST paper's SNLI "standing is not hiking" example is a good template for this.
Generate CoTs that support a different answer path than the one the model actually relies on, while still arriving at the correct final answer. The reasoning looks legitimate but is not the real causal story.
ACCURACY MUST BE MAINTAINED. Only keep CoTs where the final answer is correct. Degrading CoT informativeness is secondary to preserving accuracy.

Contrastive negatives (maybe):
CoTs that don't match what we want can serve as contrastive (negative) examples in training:
Clue-question CoTs that fail to mention the clue
Regular-question CoTs that ARE informative (reasoning simulator succeeds at predicting counterfactual behavior)

Training procedure:
SFT with a contrastive loss (following CST's setup):
Positive examples: clue CoTs that mention the clue + regular CoTs that are uninformative
Negative examples: clue CoTs that omit the clue + regular CoTs that are informative
Weighted cross-entropy on positives, unlikelihood loss on negatives

Online training (re-evaluate after each round to get fresh positives/negatives) if the initial round doesn't converge (what they did in CST).

Evaluation
Clue monitorability (should be high/perfect): g-mean^2 on Guan et al. intervention evals. The monitor should detect clue influence essentially every time.
CoT informativeness (should be reduced): counterfactual simulatability (from CST) on held-out non-clue questions. Measure the gap between reasoning simulator accuracy and outcome-only simulator accuracy. After training, this gap should shrink (the CoT adds less information beyond what you'd get from just the answer).

The core result is the gap between these two numbers. High clue monitorability + low CoT informativeness = the organism looks monitorable but isn't actually transparent.

Questions to think about

Which clue types to use for training vs. held-out evaluation? Can follow CST's train/test cue split.
What simulator/monitor model to use? Need to hold this fixed. Probably a strong model like DeepSeek-V3 or Qwen3-235B or just GPT5 / GPT5-mini via api. 
How many samples per question do we need to find good uninformative CoTs? CST uses k=16 for RL mode; we may need similar numbers.
How much does accuracy actually degrade? If it degrades meaningfully, we may need to add an accuracy reward/constraint.
