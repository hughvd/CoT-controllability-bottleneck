"""Build a tokenized ban-decode prompt set over the FULL 26-type controllability eval.

Reuses run_eval_diverse's item builders (which emit the exact instruction + marker per
cell, identical to what produced the reusable cont×mon baselines) and only adds the
per-model render+tokenize step. The reasoning-channel opening is baked into
prompt_token_ids so generated token 0 IS content token 0 for BOTH families:
  - OLMo: chat template already force-opens <think>  -> append nothing
  - Qwen: template leaves <think> to the model       -> append '<think>\n'
(auto-detected from whether the rendered prompt already ends with '<think>').

Each record carries every field grade_one() dispatches on (family/modality/style/
language/suppression/marker/omit_words), so mech/ctrl26_grade.py can grade the traces
with run_eval_diverse's own graders.

  python mech/build_ctrl26_prompts.py --model allenai/Olmo-3.1-32B-Think --model-tag olmo31 \
      --modalities family language style suppression --n-per-cell 30
  python mech/build_ctrl26_prompts.py --model Qwen/Qwen3-32B --model-tag qwen32 \
      --modalities family language suppression --n-per-cell 30      # qwen base has no style baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CTRL = HERE.parent                                    # controllability/
for p in (str(HERE), str(CTRL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_eval_diverse import (  # noqa: E402
    build_items, build_style_items, build_language_items, build_suppression_items,
    load_style_defs,
)
from sft.language_instructions import load_language_defs  # noqa: E402

TEST_CSV = CTRL / "data" / "split_test.csv"
OUT_ROOT = HERE / "data" / "ctrl26"

# canonical 26-type universe (matches results/olmo32_ctrl_ext_final/per_sample.csv)
CTRL26 = {
    "family": ["uppercase_thinking", "leetspeak", "rot13", "pig_latin", "disemvowel",
               "reverse_each_word", "reverse_word_order_per_sentence", "no_first_person",
               "no_commas", "numbered_sentences", "spell_out_numbers", "start_with",
               "prefix_sentences", "exactly_n_sentences", "max_words_per_sentence"],
    "language": ["arabic", "chinese", "russian", "spanish"],
    "style": ["cowboy", "jar_jar_binks", "pirate", "yoda"],
    "suppression": ["ignore_question", "omit_word", "omit_multiple_words"],
}

# fields copied verbatim from EvalItem into each record (grading + matching keys)
PASS = ("family", "modality", "style", "language", "suppression", "marker",
        "marker_pool", "phrasing_pool", "omit_words", "correct_letter",
        "question", "instruction")


def build_items_for(modalities, n, seed):
    test = pd.read_csv(TEST_CSV)
    items = []
    # NON-lean: expand the FULL phrasing pool (train+eval phrasings, train+eval markers)
    # so per-type compliance averages over phrasings and matches the reused cont×mon
    # baselines — lean picked ONE phrasing per type, which gave a phrasing lottery
    # (e.g. OLMo ignored the weak spanish phrasing -> spurious 0).
    if "family" in modalities:
        items += build_items(test, CTRL26["family"], n, seed, lean=False)
    if "language" in modalities:
        items += build_language_items(test, CTRL26["language"], load_language_defs(),
                                      n, seed, lean=False)
    if "style" in modalities:
        items += build_style_items(test, CTRL26["style"], load_style_defs(), n, seed)
    if "suppression" in modalities:
        items += build_suppression_items(test, CTRL26["suppression"], n, seed, lean=False)
    return items


def render_tokenize(tok, prompt_text):
    """Render chat template + bake the <think> opening so gen token 0 = content token 0."""
    msgs = [{"role": "user", "content": prompt_text}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
    except TypeError:                                  # olmo template: no enable_thinking kwarg
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if not text.rstrip().endswith("<think>"):          # qwen: template didn't open think
        text = text + "<think>\n"
    enc = tok(text, add_special_tokens=False)["input_ids"]
    return text, list(enc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--modalities", nargs="+", default=list(CTRL26),
                    choices=list(CTRL26))
    ap.add_argument("--n-per-cell", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    items = build_items_for(args.modalities, args.n_per_cell, args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    out = OUT_ROOT / args.model_tag / "prompts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, it in enumerate(items):
        text, ids = render_tokenize(tok, it.prompt)
        rec = {"prompt_id": f"{it.family}|{it.marker_pool}|{it.phrasing_pool}|{i}",
               "condition": "instruction_only",
               "ctype": it.family,                     # ban_sweep groups/labels on ctype
               "prompt_token_ids": ids,
               "think_open_in_prompt": text.rstrip().endswith("<think>")}
        for k in PASS:
            v = getattr(it, k)
            rec[k] = list(v) if isinstance(v, (list, tuple)) else v
        recs.append(rec)

    from collections import Counter
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    by = Counter(r["ctype"] for r in recs)
    print(f"[build_ctrl26:{args.model_tag}] {len(recs)} prompts over {len(by)} ctypes -> {out}")
    print("  per-ctype:", dict(sorted(by.items())))


if __name__ == "__main__":
    main()
