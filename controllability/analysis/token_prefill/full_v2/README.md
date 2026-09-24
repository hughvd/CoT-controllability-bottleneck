# Full token-prefill analysis

Immediate continuation uses the first up to 32 new reasoning tokens; sustained
continuation uses the full new reasoning trace. Supplied prefixes and final
answers are excluded. Figures show sustained continuation only; both windows
remain in the summary tables. Solid lines retain the instruction; faded dashed
lines omit it. Word-boundary diagnostics are not plotted. Every sample,
including empty and capped reasoning, is retained. Language/style scores use the
historical graders; the same rubric is applied to each window.

For sentence-prefix control, only newly generated sentence starts are assessable;
an unfinished sentence begun in the supplied prefix is not scored as a new start.
A nonempty window with no generated sentence start is unassessable (missing), not
a success or failure. Eligibility counts are reported in quality.csv. Empty
reasoning receives zero. Word/pronoun suppression uses the supplied boundary as
lexical context but does not count violations wholly inside the supplied seed.

Intervals are pointwise 95% question-clustered bootstrap intervals (2,000 draws).
Samples are averaged within question first. The same question draws are shared
across controls, models, lengths, and conditions, preserving overlaps in the
question sets. Aggregate curves weight the ten controls equally. An aggregate
point is missing if any entire control cell is unassessable. These are soft
compliance means, not strict whole-trace success rates.

Hosted GLM prefixes round-trip through its pinned native tokenizer locally;
server token IDs/checkpoint hashes are unavailable. Its immediate-window tokens
are reconstructed from returned continuation text with that tokenizer. Modal
models retain the exact supplied and generated IDs. Two Chinese source cells
were replaced consistently across models to make API prefix round trips possible.
See the immutable manifests and source_replacements for the exact identities.

No DeepSeek or GPT-OSS generations were dispatched in the active reduced sweep.
