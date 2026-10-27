# GLM numbering-prefix boundary diagnostic

The primary figures retain the same hosted OpenRouter → DeepInfra prefix interface as the earlier GLM suite. Generation parameters have not been changed in response to the results. Native generated token IDs are unavailable for this route.

With instructions, 67/150 PI@1 responses (44.7%) begin their returned reasoning with `1.` despite a supplied prefix of `1`; no PI@1 response begins with a period. Long prefixes also frequently yield returned reasoning that starts numbering at one. Exact supplied-prefix/returned-text concatenation can therefore score very differently from grading the returned text in isolation. The latter is only a diagnostic: it does not verify continuation compliance and is not substituted into the paper figures.

This observation does not establish whether the source is model restarting, API prefix handling, or returned-text normalization. Do not interpret the GLM numbering-prefill drop as a clean capability or mechanism result without resolving that ambiguity. The native Qwen/OLMo prefills preserve exact supplied and generated token IDs.

`glm_numbering_boundary_diagnostic.csv` gives all lengths/conditions. `glm_numbering_boundary_examples.json` preserves illustrative boundary excerpts and request IDs; full raw responses and exact prompts remain in the run directory.
