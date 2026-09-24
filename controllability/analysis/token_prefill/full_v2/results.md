# Token-prefill sweep: completed results

**September 18 interpretation update:** the [GLM sentence/token audit](../glm_sentence_audit_v1/README.md)
found large differences even at identical historical and token-run prefixes.
First sentences have a median length of only eight GLM tokens. A small replay
with historical request parameters did not recover the historical behavior.
The GLM mechanistic interpretation below is therefore provisional: model/serving
or prefill-handling differences must be resolved before using it as a contrast
to the native Qwen experiments. The saved measurements are unchanged.

The active three-model sweep is complete: **Qwen3-32B, Qwen3-14B, and GLM-5.2**,
all ten original controls, 30 questions per control, two instruction conditions,
eight native-token prefix lengths (0, 1, 2, 4, 8, 16, 32, 64), and five samples.
There are **72,000 linked observations from 71,760 unique responses**. The 4,800
earlier Qwen-32B uppercase/leetspeak outputs were reused and their scores verified
unchanged. DeepSeek and GPT-OSS full runs remain deferred.

## Main finding

Small prefills substantially improve continuation compliance in both Qwen
models. GLM does not exhibit the same aggregate pattern: its instruction-only
baseline is much stronger, tiny prefills initially reduce mean compliance, and
longer prefills give relatively modest aggregate gains. These are descriptive
comparisons, not tests of differences obtained from overlapping marginal CIs.

Mean **sustained continuation compliance with instructions**, equally weighted
across the ten controls:

| Model | 0 tokens | 1 token | 4 tokens | 64 tokens |
|---|---:|---:|---:|---:|
| Qwen3-32B | 15.8% | 45.3% | 54.7% | 76.6% |
| Qwen3-14B | 11.1% | 36.4% | 45.7% | 65.0% |
| GLM-5.2 | 61.0% | 52.5% | 56.8% | 66.3% |

Without instructions, sustained compliance at 0 / 4 / 64 tokens is respectively
2.3% / 24.9% / 56.1% for Qwen-32B, 2.2% / 22.0% / 43.7% for Qwen-14B, and
12.4% / 13.1% / 21.8% for GLM. Supplied text is excluded from every score.

The Qwen gains support an initiation bottleneck for several controls, but the
effect is neither uniform across controls nor universal across models. Four-token
prefills with instructions bring Qwen-32B sustained uppercase, Spanish, and
leetspeak compliance to 96.7%, 96.2%, and 91.2%, respectively. The corresponding
Qwen-14B values are 85.3%, 93.8%, and 68.9%. Sentence-prefix control benefits much
more from longer seeds: at 64 tokens it reaches 92.0% and 85.0% in the two Qwen
models. Omit-word compliance changes little, and Yoda/Jar Jar control remains
substantially harder. Inspect the per-control figures before treating the average
curve as a single mechanism.

GLM also has control-specific gains—for example, instructed uppercase rises
from 61.1% at zero tokens to 84.0% at 64—but several other controls decline or
remain non-monotonic. This limits a blanket claim that low controllability is
mostly failure to initiate. The experiment does not itself establish entropy
collapse as the cause. All models use common historical controlled source texts;
prefix lengths count each model's own tokens, and GLM uses a hosted backend.

## Immediate versus sustained continuation

Immediate compliance scores the first **up to 32 new reasoning tokens**; sustained
compliance scores the whole new reasoning continuation. No word-boundary curve
is plotted. At 64 prefilled tokens without instructions, immediate versus
sustained means are 67.2% versus 56.1% for Qwen-32B, 59.2% versus 43.7% for
Qwen-14B, and 27.4% versus 21.8% for GLM.

These gaps describe different windows, not an isolated estimate of forgetting.
In particular, binary suppression scores have more opportunities to fail on
longer text. At zero prefill without instructions, Qwen-32B no-first-person
compliance is 76.0% in the immediate window but 1.3% over the whole continuation;
GLM is 96.0% and 79.3%. GLM continuations are also shorter overall: median 137
locally reconstructed tokens versus 316 and 327 exact generated tokens for the
Qwen models. Length and ordinary pronoun/word use are relevant to GLM's high
uninstructed suppression scores; these data do not isolate their contributions.

The aggregate uses the existing scoring rubrics, including fractional character,
language, and style scores and binary suppression scores. It is **not** a strict
whole-trace success rate. Sentence-prefix windows without a newly generated
sentence start are unassessable, rather than counted as compliant. Repeated
samples are averaged within question before a shared question-clustered
bootstrap; plotted bands are pointwise 95% intervals from 2,000 draws.

## Quality, validation, and accounting

| Model | Empty reasoning | Length-capped | Fewer than 32 new tokens | Unassessable immediate / sustained |
|---|---:|---:|---:|---:|
| Qwen3-32B | 0.163% | 0.879% | 0.425% | 95 / 7 |
| Qwen3-14B | 0.042% | 0.350% | 0.117% | 69 / 1 |
| GLM-5.2 | 0% | 0.292% | 9.954% | 359 / 140 |

Rates and counts are over 24,000 linked observations per model. Empty and capped
responses remain in the analysis. No aggregate point is missing. Full per-cell
eligibility and quality counts are in `quality.csv`.

The final audit verifies all manifests/source hashes, matched source cells,
balanced question/sample counts, absence of duplicate linked observations,
exact native Qwen prompt IDs, all 4,800 reused scores, and the pinned DeepInfra
route for all 23,920 GLM responses. Sixteen implementation tests pass. GLM server
token IDs and checkpoint hashes are unavailable: its supplied prefixes round-trip
locally, while the immediate output window uses local retokenization. Four GLM
responses required parsing an inline closing reasoning tag to exclude the answer.
Eleven grader replies needed apostrophe-escape repair, and one retained a complete
explicit numeric score preceding a length-capped explanation; raw replies and
repair flags are retained.

Both GPU workers have stopped. GPU accounting estimates **$12.96**; completed API
generation, grading, and validation calls report **$17.61**, with **$0.12** retained
as a conservative possible extra charge for uncertain calls. This is about
**$30.57**, plus that possible charge, using the GPU budget estimate rather than
a reconciled Modal invoice. Upstream rate limits delayed GLM and required a
resumable restart at 19,815 outputs; prompts, provider, and sampling stayed fixed.

## Artifacts

- `token_prefill_compare_avg.png` / `.pdf`: aggregate immediate and sustained curves.
- `qwen32_by_control.png`, `qwen14_by_control.png`, `glm_by_control.png` and PDFs:
  control-level curves.
- `aggregate.csv`, `summary.csv`, `quality.csv`: values, intervals, and quality counts.
- `../../../results/token_prefill/full_v2/completion_validation.json`: final audit.
- Paper copies: `paper/figures/token_prefill_compare_avg.png` and
  `paper/figures/token_prefill_{qwen32,qwen14,glm}_by_control.png` at repository root.

The normal repository analysis script produces the figures. Existing sentence-
prefill figures, the original two-control run, and `paper.tex` were left intact.
