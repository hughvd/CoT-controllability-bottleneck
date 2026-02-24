Hugh Van Deventer CBAI Spring 2026 Research Proposal

Background:
Emmons et al. [1] introduce a conceptual framework distinguishing CoT-as-rationalization from CoT-as-computation. They revisit unfaithfulness results from prior work, which all share a limitation: the bad behavior (e.g., following a false hint) was simple, making the CoT unnecessary to perform it. The authors then show that when these same setups involve more difficult behavior—where the CoT computation is genuinely needed—the issue of unfaithfulness largely dissolves, and models struggle to evade monitors. Similarly, METR [2] replicate Anthropic's faithfulness evaluations but with free-response math problems and clues of varying complexity, finding that when clues are complex enough to require CoT processing, models are almost always faithful (only 3 of 21,272 trajectories showed plausible unfaithfulness), and a black-box clue-usage detector achieves 99.3% true positive rate even on cases where strict faithfulness fails.

Guan et al. [3] build an evaluation framework for measuring monitorability directly, introducing three evaluation archetypes (intervention, process, and outcome-property), the g-mean² metric (TPR × TNR), and a broad suite of evaluations spanning coding, cybersecurity, and reasoning domains. They report initial scaling trends: monitorability generally increases with CoT length, a smaller model at higher reasoning effort can match a larger model's capability with higher monitorability, and RL optimization at frontier scale does not materially degrade monitorability. They also compare monitorability across frontier models, finding that OpenAI models tend to score higher, with significant variation across families. However, their scaling analysis is limited to OpenAI model families, and their evaluations do not explicitly select for the CoT-as-computation regime identified by Emmons et al. [1] and METR [2].

Levy et al. [4] introduce the State over Tokens (SoT) conceptual framework, reframing reasoning tokens not as a linguistic narrative but as externalized computational state. This means the tokens can be computationally essential while their surface-level English semantics diverge from the information the model actually encodes. Together with the empirical work above, this raises a question: the current positive correlation between model capability and monitorability may be a contingent feature of today's models rather than a robust structural property.

Goal:
We aim to broaden our understanding of how black-box monitorability scales with model capability by reproducing and extending Guan et al.'s [3] scaling analysis. Specifically:
Reproduce monitorability scaling trends on open-source models. Guan et al.'s trends are based primarily on OpenAI model families. We will evaluate monitorability across open-source reasoning model families (e.g., Qwen, DeepSeek) where CoT is accessible and training details are documented, testing whether their positive capability-monitorability trend generalizes.
Track distilled vs. non-distilled models. Many widely deployed reasoning models are distillations of larger models (e.g., DeepSeek-R1-Distill series). It is unknown whether distillation preserves, degrades, or changes the monitorability profile. We will compare distilled and non-distilled models at matched capability levels.
Extend evaluations to the CoT-as-computation regime. Guan et al.'s evaluations do not explicitly filter for tasks where CoT is computationally necessary. We will augment their evaluation framework with no-CoT credence filtering [2] to study monitorability specifically in the regime where models must reason through CoT to succeed.

The results of the reproduction will inform subsequent directions. If the positive scaling trend does not reproduce, we investigate whether white-box access (e.g., monitoring hidden states) recovers monitorability. If it does reproduce, we can further investigate the CoT-as-computation regime and whether current methods for identifying it are reliable.

Methodology:
Step 1: Reproduce Guan et al.'s scaling trends on open-source models
Select a set of open-source reasoning models spanning a capability range — ideally from the same family at different sizes (e.g., DeepSeek-R1 series, Qwen reasoning series), plus distilled variants where available. Run Guan et al.'s intervention evaluations using a fixed monitor model and report g-mean² as a function of model capability. Compare the resulting scaling trends to Guan et al.'s OpenAI-based results.

Step 2: Add CoT-as-computation filtering
The intervention evaluation structure naturally provides task calibration: we only count trajectories where the model fails without the intervention and succeeds with it, guaranteeing the intervention was used. To additionally ensure the CoT was necessary for using the intervention (rather than the model deciphering it in a single forward pass), we add no-CoT credence measurement following METR [2]: suppress CoT, sample the model's response to the intervention many times, and compute what fraction arrive at the correct answer. Filter to interventions where no-CoT credence is near zero. Report monitorability separately for the CoT-as-computation regime and compare to the unfiltered results.

Step 3: Based on results, pursue one of:
If the positive scaling trend does not reproduce: Investigate whether adding white-box access recovers monitorability — e.g., giving the monitor access to intermediate hidden states or activation features alongside (or instead of) the CoT token stream.
Validate CoT-as-computation identification methods. Current methods for determining whether CoT is "necessary" [6,7] rely on interventions that shift the CoT off its natural distribution, and observed answer changes may reflect sensitivity to distribution shift rather than genuine causal reliance. We propose a more robust validation inspired by Neel Nanda's work [8]: for a given problem, resample many reasoning trajectories, segment them at the sentence level, and measure how often sentences that are semantically similar lead to the same downstream answer. If a reasoning step is genuinely causally important, similar articulations of that step should converge on the same answer, whereas if the model is robust to surface variation at that step, the step is likely not doing critical computational work. This provides a distributional rather than pointwise test of causal importance, avoiding the confound of single-perturbation sensitivity.

Uncertainties:
Confounds across model families. Different models may have received different amounts of RL optimization on their CoT. Our primary mitigation is selecting models from the same family where possible; where not, we document known training differences.
Monitor choice. The monitor must be held fixed, but its capability level affects absolute g-mean² scores. We should test sensitivity to monitor choice.
Ecological validity. Clue-usage detection is a proxy for real-world safety-relevant behaviors. The extent to which these results transfer to naturalistic threat models (e.g., detecting alignment faking, subtle sabotage) remains open.
Distillation confounds. Distilled models differ from their teachers in both size and training process. Capability-matched comparisons help but don't fully isolate the effect of distillation on monitorability.

References:
[1] Emmons, Scott, et al. "When chain of thought is necessary, language models struggle to evade monitors." arXiv preprint arXiv:2507.05246 (2025).
[2] METR. "CoT May Be Highly Informative Despite 'Unfaithfulness'." https://metr.org/blog/2025-08-08-cot-may-be-highly-informative-despite-unfaithfulness/ (2025).
[3] Guan, Melody Y., et al. "Monitoring Monitorability." arXiv preprint arXiv:2512.18311 (2025).
[4] Levy, Mosh, et al. "State over Tokens: Characterizing the Role of Reasoning Tokens." arXiv preprint arXiv:2512.12777 (2025).
[5] Korbak, Tomek, et al. "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety." arXiv preprint arXiv:2507.11473 (2025).
[6] Lanham, Tamera, et al. "Measuring Faithfulness in Chain-of-Thought Reasoning." arXiv preprint arXiv:2307.13702 (2023).
[7] Swaroop, Anand, et al. "FRIT: Using Causal Importance to Improve Chain-of-Thought Faithfulness." arXiv preprint arXiv:2509.13334 (2025).
[8] Macar, Uzay, et al. "Thought Branches: Interpreting LLM Reasoning Requires Resampling." arXiv preprint arXiv:2510.27484 (2025).
