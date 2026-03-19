# Model Organisms Pivot/Extension
Mentor: Laura Ruis, Research Manager: Dmitrii Troitskii

## Motivation for Extension:
Guan et al. [1] introduced a suite of Chain-of-Thought (CoT) monitorability evaluations. Their suite spans three archetypes: intervention evaluations apply a controlled perturbation (e.g., appending a hint to a math problem) and ask whether a monitor can detect that the intervention influenced reasoning; process evaluations use problems with a small known set of valid solution paths and ask whether the monitor can identify which path the model took; and outcome-property evaluations label an unambiguous property of the output (e.g., if an impossible coding task's tests pass, the model must have hacked them) and ask whether the monitor can predict this from the CoT.

Their CoT monitorability scaling analysis shows a trend that increased capability corresponds with increased monitorability. Notably, they only report these trends on their closed source models at OpenAI. Our project set out to reproduce these findings in the wild  - as applied to open-source model families, and explore the dimensions not touched upon in the original paper, such as the effect of distillation on monitorability scaling trends. 

We have successfully reproduced their intervention evaluation setup on the portion with publicly available datasets (Scruples, GPQA Diamond, DAFT, AIME, WMDP). We have then run the reproduced evaluations across a range of open-source models: Qwen3 (8b, 14b, 32b, 235b), GPT-OSS (20b, 120b), DeepSeek R1 (0528, Distill-Qwen-32b), Kimi-K2-Thinking, and Gemini-2.5-Flash. In our initial results on Guan et al.’s [1] monitorability evaluation, we do not see clean trends across models. In fact, we observe a general trend of relatively high monitorability across all models tested (slight negative trend observed—currently not significant). It remains to be seen if we can reproduce the trend of scaling inference compute within a model to increase monitorability (this is the strongest scaling trend they report). This suggests that training procedures are the major determinant of monitorability. We plan to write up the reproduction as a blogpost or preprint before the extension begins, and potentially merge it with extension work into a conference submission.

As we wrap up validating the robustness of the monitorability trends Guan et al. report, the natural next step is to stress test the evaluation method itself. Doing so will shed light on the following: what specifically makes models monitorable, and what doesn't? How robust is this measure of monitorability in the first place? We propose to investigate this using model organisms to stress-test current monitorability evaluations and identify their failure modes.

## Theory of Change:
Current monitorability evaluations overwhelmingly rely on cue detection: checking whether a model mentions an externally provided hint in its chain of thought. But this evaluation structure has two potential blind spots. First, a model that is aware of monitoring could learn to avoid mentioning cues it is actually using, appearing clean to the monitor while still exhibiting the undesired behavior. Second, a model whose problematic behavior is not triggered by a prompt-level cue (e.g., a bias baked in through training) may never produce the kind of signal current monitors are evaluated on.
If we can construct model organisms that expose these failure modes, this informs both evaluation designers and model developers about what current monitorability methods actually measure, and what they miss. This well motivates follow up projects to produce stronger monitorability evaluations.

## Proposed Activities:

We propose constructing two complementary model organisms that target the two failure modes described above (note these are the general ideas, specific implementation of each organism is subject to change).

Organism 1: Monitor-aware model (false negative failure). We construct a model that knows it is being monitored for cue usage and adapts its chain of thought accordingly. The goal is to demonstrate that current cue-detection monitors can be fooled by a model that is aware of the monitoring setup. We will begin with system-prompt-based experiments and escalate to fine-tuning if system-prompt manipulations do not produce the desired behavior. 

Organism 2: Model with baked-in bias. We construct a model whose undesired behavior does not originate from an in-context cue. For example, a model fine-tuned to exhibit a systematic bias that is not prompted by anything in the evaluation setup. The goal is to test whether current monitors, which are designed around cue detection, can detect biases that are baked into model weights. As with Organism 1, we will start with system-prompt approaches and move to fine-tuning as needed.

References:
[1] Guan, Melody Y., et al. "Monitoring Monitorability." arXiv preprint arXiv:2512.18311 (2025).
