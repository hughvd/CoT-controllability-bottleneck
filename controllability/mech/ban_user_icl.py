"""Matched 2x2: user ICL (five sentences) x fixed first-content-token banning.

Reuses the corrected ICL donor assignment and the fixed-ban generation protocol.
All four arms use the donor's control instruction and grading specification.
"""
from dataclasses import asdict
from pathlib import Path
import json
import sys

import fixed_opening_ban as f
sys.path.insert(0, str(f.CTRL))
import icl_experiment as icl

ROOT = f.CTRL / 'results/ban_user_icl/v1'
CONDITIONS = ['instruction_only', 'user_icl']


def prepare(behaviors=None, root=None):
    from transformers import AutoTokenizer

    behaviors = list(f.BEHAVIORS if behaviors is None else behaviors)
    root = ROOT if root is None else Path(root)

    cfg = icl.PrefillConfig(n_questions=30, n_samples=3, seed=0)
    for key, value in icl.FULL_TYPES.items():
        setattr(cfg, key, value)
    # Select donors before filtering, exactly as in the corrected full ICL run.
    full = icl.build_all_cells(cfg)
    donors = icl.select_donors(full, cfg.seed)
    targets = [c for c in full if c.ctype in behaviors]
    cells = []
    for target in targets:
        donor = donors[(target.ctype, target.question)]
        assert target.question != donor.question
        cell = asdict(icl.grade_cell_for(target, donor))
        cell.pop('controlled_cot')
        cell['behavior'] = cell.pop('ctype')
        cell['instruction'] = donor.instruction
        cell['question_id'] = f.base.digest(target.question)
        cell['cell_id'] = f.base.digest([target.ctype, target.question, donor.instruction])
        cell['donor'] = asdict(donor)
        cell['donor_question_id'] = f.base.digest(donor.question)
        cells.append(cell)
    assert len(cells) == 30 * len(behaviors)
    for name, spec in f.MODELS.items():
        snap = Path.home()/'.cache/huggingface/hub'/('models--'+spec['id'].replace('/', '--'))/'snapshots'/spec['revision']
        tok = AutoTokenizer.from_pretrained(snap, local_files_only=True) if name != 'glm' else f.suite.JsonTokenizer(snap)
        bids = []
        for word in spec['ban']:
            ids = tok.encode(word, add_special_tokens=False)
            assert len(ids) == 1, (word, ids)
            bids += ids
        requests = []
        for target, cell in zip(targets, cells):
            donor = donors[(target.ctype, target.question)]
            for condition in CONDITIONS:
                variant = 'icl_user_plain' if condition == 'user_icl' else 'instruction_only'
                messages = icl.build_icl_messages(target, donor, variant, ('5', 'abs', 5))
                if name == 'glm':
                    messages.append({'role': 'assistant', 'content': '<think>\n'})
                    ids = None
                else:
                    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if name == 'qwen32':
                        text += '<think>\n'
                    else:
                        assert text.endswith('<think>')
                    ids = tok.encode(text, add_special_tokens=False)
                    assert len(ids) + 8192 <= 16384, (name, len(ids))
                for arm in f.ARMS:
                    for sample in range(f.SAMPLES):
                        rid = f.base.digest(['ban_user_icl_v1', name, cell['cell_id'], condition, arm, sample])
                        requests.append(dict(request_id=rid, cell_id=cell['cell_id'], condition=condition,
                                             arm=arm, sample=sample, messages=messages, prompt_token_ids=ids,
                                             seed=int(f.base.digest([cell['cell_id'], sample])[:8], 16) % 2**31))
        manifest = dict(version='ban_user_icl_v1', name=name, model=spec, ban_token_ids=bids,
                        cells=cells, requests=requests,
                        sampling=dict(temperature=.6, top_p=1., top_k=-1, max_tokens=8192),
                        tokenizer_sha256=f.base.file_hash(snap/'tokenizer.json'),
                        source_code_sha256={p.name:f.base.file_hash(p) for p in
                                           [Path(icl.__file__), Path(icl.pe.__file__), Path(__file__)]},
                        protocol='2x2 factorial: instruction vs user ICL; baseline vs fixed first-content-token ban. '
                                 'User ICL is one different-question example with five controlled sentences and its answer. '
                                 'All arms use the same donor control instruction and grading spec. '
                                 'Local models split both arms after first content token; GLM uses ordered rejection sampling.',
                        scope=dict(behaviors=behaviors, questions_per_behavior=30, samples=3,
                                   conditions=CONDITIONS, arms=f.ARMS, donor_assignment_seed=0,
                                   donor_assignment_version=2, example_sentences=5))
        manifest['manifest_sha256'] = f.base.digest(manifest)
        f.validate(manifest)
        f.base.immutable_json(root/name/'manifest.json', manifest)
        print(name, len(requests), 'outputs; ban IDs', bids, flush=True)


if __name__ == '__main__':
    prepare()
