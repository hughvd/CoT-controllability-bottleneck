"""Matched-question opening concentration and full-distribution entropy figure.

The main concentration panel fixes the reference prefix to each model's
no-instruction mode at each length. A companion figure reselects the mode
within each condition. Neither sums multiple prefixes or control-specific modes.
"""
from collections import Counter, defaultdict
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import re
import sqlite3
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from opening_token_analysis import TOK, TOKPATHS, content_bounds, NAMES, TYPES
HERE = Path(__file__).resolve().parent
CTRL = HERE.parent.parent
ROOT = HERE.parent / 'data/matched_openings/v1'
OUT = HERE / 'opening_tokens/matched'
sys.path.insert(0, str(CTRL / 'analysis'))
from paper_style import MODEL_COLORS, apply_style

CONDS = ['no_instruction', 'instruction_only']
MODELS = ['qwen32', 'olmo31', 'glm']


def rows(path):
    return [json.loads(line) for line in path.open()]


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    cells = {c['cell_id']: c for c in manifest['cells']}
    links = defaultdict(list)
    for link in manifest['links']:
        links[link['request_id']].append(link)
    assert len(cells) == 120
    assert all(q['user_text'] == o['user_text'] for q, o in zip(
        manifest['requests']['qwen32'], manifest['requests']['olmo31']))
    recs = []

    def add(model, raw, linked, ids, entropy=None):
        for link in linked:
            c = cells[link['cell_id']]
            recs.append(dict(model=model, cell_id=c['cell_id'],
                question_id=c['question_id'], behavior=c['behavior'], source=c['source'],
                condition='instruction_only' if link['condition'] == 'prefill_plus_instruction' else 'no_instruction',
                request_id=raw['request_id'], sample=raw['sample'], ids=ids, entropy=entropy))

    for model in MODELS[:2]:
        report = json.loads((ROOT / model / 'report.json').read_text())
        assert report['status'] == 'complete', report
        stats = rows(ROOT / model / f'{model}_stats.jsonl')
        assert len(stats) == 1195
        assert len({(r['request_id'], r['sample']) for r in stats}) == 1195
        for r in stats:
            assert len(r['entropy']) == min(64, len(r['content_token_ids']))
            assert np.isfinite(r['entropy']).all()
            assert np.min(r['entropy'], initial=0) >= 0
            assert len(r['top5_ids']) == len(r['entropy'])
            add(model, r, links[r['request_id']], r['content_token_ids'], r['entropy'])

    gp = CTRL / 'results/token_prefill/full_v2/glm'
    gm = json.loads((gp / 'manifest.json').read_text())
    gc = {c['cell_id']: c for c in gm['cells']}
    glinks = defaultdict(list)
    for link in gm['links']:
        if link['prefix_tokens'] == 0 and link['cell_id'] in cells:
            assert all(gc[link['cell_id']][k] == cells[link['cell_id']][k]
                       for k in ['question', 'instruction', 'behavior', 'question_id'])
            glinks[link['request_id']].append(link)
    assert sum(map(len, glinks.values())) == 240
    qtexts = {r['request_id']: r['user_text'] for r in manifest['requests']['qwen32']}
    by_cell = {(l['cell_id'], l['condition']): qtexts[rid] for rid, ls in links.items() for l in ls}
    for r in gm['requests']:
        if r['request_id'] in glinks:
            for l in glinks[r['request_id']]:
                assert r['user_text'] == by_cell[l['cell_id'], l['condition']]
    with sqlite3.connect(f'file:{gp / "api_generations.sqlite"}?mode=ro', uri=True) as db:
        for (payload,) in db.execute('select payload from outputs'):
            r = json.loads(payload)
            if r['request_id'] not in glinks:
                continue
            ids = TOK['glm'].encode(r['reasoning_continuation'], add_special_tokens=False).ids
            a, b = content_bounds(ids, 'glm')
            add('glm', r, glinks[r['request_id']], ids[a:b])
    assert len(recs) == 3600
    expected = {(c, s, cond) for c in cells for s in range(5) for cond in CONDS}
    for model in MODELS:
        actual = [(r['cell_id'], r['sample'], r['condition']) for r in recs if r['model'] == model]
        assert len(actual) == len(set(actual)) == 1200
        assert set(actual) == expected
    return manifest, recs


def summarize(recs):
    prefix, entropy = [], []
    for model in MODELS:
        groups = {c: [r for r in recs if r['model'] == model and r['condition'] == c] for c in CONDS}
        for k in range(1, 33):
            counts = {c: Counter(tuple(r['ids'][:k]) for r in groups[c] if len(r['ids']) >= k) for c in CONDS}
            # Deterministic lexicographic token-ID tie break, recorded as n_ties.
            modes = {c: min(t for t, n in counts[c].items() if n == max(counts[c].values())) for c in CONDS}
            for c in CONDS:
                for method in ['no_instruction_reference', 'within_condition_mode']:
                    ref = modes['no_instruction'] if method == 'no_instruction_reference' else modes[c]
                    selection = counts['no_instruction'] if method == 'no_instruction_reference' else counts[c]
                    prefix.append(dict(model=model, condition=c, method=method, prefix_length=k,
                        share=counts[c][ref] / len(groups[c]), matches=counts[c][ref], total=len(groups[c]),
                        eligible=sum(counts[c].values()), reference_token_ids=json.dumps(ref),
                        reference_text=TOK[model].decode(list(ref), skip_special_tokens=False),
                        selection_ties=sum(n == max(selection.values()) for n in selection.values())))
        if model == 'glm':
            continue
        for c in CONDS:
            for p in range(64):
                rs = [r for r in groups[c] if len(r['entropy']) > p]
                entropy.append(dict(model=model, condition=c, position=p,
                    mean_entropy_nats=float(np.mean([r['entropy'][p] for r in rs])),
                    responses=len(rs), question_cells=len({r['cell_id'] for r in rs}),
                    distinct_questions=len({r['question_id'] for r in rs})))
    return pd.DataFrame(prefix), pd.DataFrame(entropy)


def figure(prefix, entropy, method, name):
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for model in MODELS:
        for condition in CONDS:
            kw = dict(color=MODEL_COLORS[model], ls='--' if condition == 'no_instruction' else '-', lw=2.1)
            p = prefix[(prefix.model == model) & (prefix.condition == condition) & (prefix.method == method)]
            left.plot(p.prefix_length, p.share, **kw)
            e = entropy[(entropy.model == model) & (entropy.condition == condition)]
            if len(e):
                right.plot(e.position, e.mean_entropy_nats, **kw)
    left.set_title('(a) Opening-prefix concentration', loc='left', fontsize=14, pad=12)
    right.set_title('(b) Next-token entropy', loc='left', fontsize=14, pad=12)
    left.set_xlabel('Opening length (tokens)')
    left.set_ylabel('Share of responses matching\nmost common prefixes')
    left.set_xscale('log', base=2)
    left.set_xlim(1, 32)
    left.set_xticks([1, 2, 4, 8, 16, 32], ['1', '2', '4', '8', '16', '32'])
    left.set_ylim(0, 1.03)
    left.yaxis.set_major_formatter(PercentFormatter(1))
    right.set_xlabel('Reasoning content token position')
    right.set_ylabel('Entropy (nats)')
    # Shift the logarithmic coordinate so positions 0 and 1 remain visible.
    right.set_xscale('function', functions=(lambda x: np.log2(x + 1), lambda x: np.exp2(x) - 1))
    right.set_xlim(0, 63)
    right.set_xticks([0, 1, 2, 4, 8, 16, 32, 63], ['0', '1', '2', '4', '8', '16', '32', '63'])
    right.set_ylim(bottom=0)
    for ax in [left, right]:
        ax.tick_params(axis='both', labelsize=13)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.16)
    handles = [Line2D([], [], color=MODEL_COLORS[m], lw=2.5, label=NAMES[m]) for m in MODELS]
    handles += [Line2D([], [], color='#444', lw=2, ls=ls, label=label)
                for ls, label in [('-', 'Instruction'), ('--', 'No instruction')]]
    fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=left.xaxis.label.get_fontsize(), bbox_to_anchor=(.5, .01))
    fig.tight_layout(rect=(0, .12, 1, 1), w_pad=2.5)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=240, bbox_inches='tight')
    plt.close(fig)


def embed_gallery():
    """Add the matched figure without removing the existing shareable report."""
    path = OUT.parent / 'index.html'
    text = path.read_text()
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.suffix in {'.csv', '.pdf', '.md', '.json'}:
                z.write(p, p.name)
    b64 = lambda p: base64.b64encode(p.read_bytes()).decode()
    section = ('<!-- matched-opening-start --><section id="matched-openings"><h2>Matched questions: prefix concentration and entropy</h2>'
        '<p>September 21, 2026. All three models use the same 120 question–control pairs (119 distinct questions), '
        'five samples per condition, temperature 0.6 and top-p 1.0. This matched figure supersedes the older cross-model comparison below.</p>'
        '<p>The left panel tracks each model’s single most common <b>no-instruction</b> prefix under both conditions, independently at each prefix length. '
        'The right panel shows full-vocabulary model entropy for Qwen and OLMo. GLM reasoning logprobs are unavailable.</p>'
        f'<img alt="Matched opening concentration and entropy" src="data:image/png;base64,{b64(OUT / "opening_concentration_entropy.png")}">'
        '<details><summary>Alternative: select the modal prefix separately within each condition</summary>'
        f'<img loading="lazy" alt="Within-condition modal prefixes" src="data:image/png;base64,{b64(OUT / "opening_concentration_entropy_within_condition.png")}"></details>'
        f'<details><summary>Matched figure methods</summary><pre>{html.escape((OUT / "README.md").read_text())}</pre></details>'
        f'<a class="download" download="matched_opening_figure.zip" href="data:application/zip;base64,{base64.b64encode(bundle.getvalue()).decode()}">Download matched figure PDFs, data, and methods</a>'
        '</section><!-- matched-opening-end -->')
    text = re.sub(r'<!-- matched-opening-start -->.*?<!-- matched-opening-end -->', '', text, flags=re.S)
    text = text.replace('</h1>', '</h1>' + section, 1)
    path.write_text(text)
    (path.parent / 'opening_tokens_share.html').write_text(text)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, recs = load()
    prefix, entropy = summarize(recs)
    prefix.to_csv(OUT / 'prefix_concentration.csv', index=False)
    entropy.to_csv(OUT / 'entropy_by_position.csv', index=False)
    pd.DataFrame(manifest['cells']).to_csv(OUT / 'question_cells.csv', index=False)
    metadata = [{k: v for k, v in r.items() if k not in ['ids', 'entropy']} |
                dict(content_tokens_available=len(r['ids']), entropy_positions=len(r['entropy'] or [])) for r in recs]
    pd.DataFrame(metadata).to_csv(OUT / 'response_metadata.csv', index=False)
    numerical = {}
    for model in MODELS[:2]:
        first = defaultdict(list)
        for r in rows(ROOT / model / f'{model}_stats.jsonl'):
            first[r['request_id'], tuple(r['output_token_ids'][:r['content_start']])].append(r['entropy'][0])
            for h, p in zip(r['entropy'], r['top5_p']):
                assert sum(p) <= 1.00001 and h + 1e-5 >= -np.log(p[0])
        numerical[model] = float(max(np.ptp(v) for v in first.values()))
    (OUT / 'validation.json').write_text(json.dumps(dict(
        first_content_entropy_max_spread_across_identical_histories=numerical,
        entropy_minimum_response_count=int(entropy.responses.min()),
        matched_cells_per_model=120, linked_samples_per_model_condition=600,
        full_vocabulary_entropy_ge_min_entropy=True), indent=2))
    figure(prefix, entropy, 'no_instruction_reference', 'opening_concentration_entropy')
    figure(prefix, entropy, 'within_condition_mode', 'opening_concentration_entropy_within_condition')
    mixture = Counter({c['question_id']: c['source'] for c in manifest['cells']}.values())
    methods = f'''# Matched opening concentration and entropy

The same question text and control instruction are used for all three models.
Four controls: uppercase, leetspeak, Spanish, and word omission. Thirty question–control
pairs per control, 120 pairs total, 119 distinct questions. One CommonsenseQA question
occurs in two controls; the no-instruction response is reused in those two cells.
Mixture of distinct questions: {dict(mixture)}.
Mixture of equally weighted question–control cells: 76 CommonsenseQA, 37 ARC-Challenge,
3 MMLU-Pro, 2 GPQA, 2 HLE. Each model/condition has 600 linked responses (five per cell).
1195 unique responses per model across the two conditions. No compliance filtering.

## Left panel
The horizontal coordinate is log2(prefix length), with ticks at 1, 2, 4, 8, 16, 32,
to emphasize the first two tokens.
For each model and prefix length k, select the single most frequent k-token prefix
from its no-instruction responses, then measure the fraction of all 600 responses
in each condition matching that exact prefix. Three reference prefixes total at
each k, one per model. This is not the combined coverage of the top three prefixes.
The reference may change as k changes; it is not necessarily one nested sequence.
Ties are resolved by lexicographic token-ID order and recorded in the CSV.
Responses shorter than k remain in the denominator as nonmatches.
Controls are pooled before selecting the mode; control-specific modes are not averaged.
The companion within-condition figure selects its mode separately in each condition.
Actual prefix strings and token IDs are provided in prefix_concentration.csv.

## Right panel
The horizontal coordinate is log2(1 + position), with explicit ticks at 0 and 1,
so the first two content-token positions are visually separated.
Mean conditional next-token entropy over the full model vocabulary, H(p)=-sum p log p,
computed from raw temperature-1 logits before sampling transformations. Units: nats.
Absolute position 0 is the first reasoning content token. Leading whitespace and
protocol tokens are excluded; positions stop before the closing thinking token.
At each position, average all responses reaching that position, with five samples
per question–control cell at the start. Coverage per position is in entropy_by_position.csv.
This is mean within-context predictive entropy, not entropy of pooled sampled tokens.
No uncertainty bands: these are descriptive estimates over the fixed evaluation pool.
Models use different native tokenizers; token positions are not equal amounts of text.

## Measurements and provenance
All rollouts: temperature 0.6, top-p 1.0, top-k disabled. Native model chat templates.
Qwen: existing full_v2 zero-prefill exact sampled token IDs; first 64 content positions
rescored with the pinned BF16 checkpoint. OLMo: new 128-token rollouts on exactly those
questions and instructions; first 64 content positions scored by the same HF method.
Both use transformers 5.12.1, BF16 weights, SDPA, float32 softmax/entropy, batches of 8.
BF16 forward passes have batch-shape sensitivity: the maximum first-content entropy
spread across identical prompt histories was {numerical['qwen32']:.4f} nats for Qwen
and {numerical['olmo31']:.4f} nats for OLMo. Do not interpret differences of this size
as precise measurements. All 600 responses per model/condition cover all 64 positions.
OLMo generation uses vLLM 0.23.0. Qwen's original full response cap was 8192 tokens;
OLMo's new cap is 128 (both exceed the plotted prefix and entropy windows).
Qwen and OLMo token IDs are exact. GLM text is locally retokenized using the cached
GLM tokenizer; API generation token IDs and full reasoning logprobs are unavailable.
Thus GLM appears only in the concentration panel. GLM's served weights are not pinned.
The old saved entropy datasets were not question matched: Qwen's audited 178-question
six-control pool contained 110 CSQA, 56 ARC, 8 MMLU-Pro, 3 GPQA, 1 HLE;
the common-four-control subset had 119 distinct questions (75 CSQA, 38 ARC, 3 MMLU-Pro,
2 GPQA, 1 HLE). OLMo ext_final used 30 questions: 11 GPQA, 11 HLE, 8 MMLU-Pro.
Those pools have zero overlap. This figure does not combine their old entropy arrays.

Prepared manifest SHA256: {manifest['manifest_sha256']}
'''
    (OUT / 'README.md').write_text(methods)
    paths = [ROOT / 'manifest.json', ROOT / 'qwen32_inputs.json', Path(__file__).resolve(), HERE.parent / 'matched_opening_measure.py']
    paths += [ROOT / tag / f'{tag}_stats.jsonl' for tag in MODELS[:2]]
    paths += [CTRL / 'results/token_prefill/full_v2/glm' / f for f in ['manifest.json', 'api_generations.sqlite']]
    provenance = dict(manifest_sha256=manifest['manifest_sha256'], models=manifest['models'],
        tokenizer_paths=TOKPATHS, input_sha256={str(p): file_hash(p) for p in paths},
        source_mixture_unique_questions=dict(mixture), same_cell_sample_condition_sets=True,
        same_questions_and_user_text_all_models=True, responses=len(recs))
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    embed_gallery()
    print('Wrote', OUT)
    print(prefix.query('method == "no_instruction_reference" and prefix_length in [1,4,8]').to_string(index=False))
    print(entropy.groupby(['model','condition']).agg(min_responses=('responses','min'), first_entropy=('mean_entropy_nats','first')))


if __name__ == '__main__':
    main()
