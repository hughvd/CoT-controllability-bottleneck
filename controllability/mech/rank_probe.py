"""Fixed-prefix instruction sensitivity: preparation and exact-vocabulary statistics.

Prepare locally: python mech/rank_probe.py prepare --run opening_ranks_v1
Inference is exclusively through modal_rank_probe.py; this module never generates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import string
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
VERSION = "opening-ranks-1"
TYPES = ("uppercase_thinking", "leetspeak")
CONDS = ("no_instruction", "instruction_only")
MODELS = {
    "olmo31": {"model": "allenai/Olmo-3.1-32B-Think",
               "revision": "832c3f543499af8fe68b88359501de9cb7840544"},
    "qwen32": {"model": "Qwen/Qwen3-32B",
               "revision": "9216db5781bf21249d130ec9da846c4624c16137"},
}
LEET_SOURCE = frozenset("aAeEiIoOtTsS")
LEET_DIGITS = frozenset("431075")
CONSTRAINTS = {
    "uppercase_thinking": "No alphabetic character for which str.isupper() is false; content requires alphabetic text.",
    "leetspeak": "No a/e/i/o/t/s in either case; content requires an ASCII letter or digit; digit diagnostic uses 4/3/1/0/7/5.",
    "excluded": "Special IDs, IDs beyond tokenizer length, empty decoding, U+FFFD, or Unicode surrogate characters.",
    "rank": "1 + number of logits strictly greater than the candidate logit (ties share rank).",
    "nucleus": "temperature=0.6; keep minimal sorted prefix reaching cumulative probability >=0.95, including crossing token.",
}


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def immutable_json(path, obj):
    path = Path(path)
    payload = canonical(obj) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Immutable artifact differs: {path}; use a new run name")
    else:
        with path.open("xb") as f:
            f.write(payload)


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def remove_instruction(text):
    pattern = r"<cot_instructions>.*?</cot_instructions>\n"
    matches = list(re.finditer(pattern, text, flags=re.DOTALL))
    if text.count("<cot_instructions>") != 1 or text.count("</cot_instructions>") != 1:
        raise ValueError("Expected exactly one pair of control-block delimiters")
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one control block, found {len(matches)}")
    m = matches[0]
    return text[:m.start()] + text[m.end():]


def violates(char, ctype):
    if ctype == "uppercase_thinking":
        return char.isalpha() and not char.isupper()
    if ctype == "leetspeak":
        return char in LEET_SOURCE
    raise ValueError(ctype)


def first_violation(tokenizer, ids, ctype):
    """Map full-decoding character position back to original token IDs.

    Incremental decoding is used rather than re-tokenizing, including split-byte
    tokens. Replacement fragments are not themselves treated as violations.
    """
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    think_end = text.find("</think>")
    reasoning = text if think_end < 0 else text[:think_end]
    ci = next((i for i, c in enumerate(reasoning) if violates(c, ctype)), None)
    if ci is None:
        return None
    for t in range(len(ids)):
        prefix = tokenizer.decode(ids[:t + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if len(prefix) > ci and prefix[:ci + 1] == reasoning[:ci + 1]:
            return {"token_position": t, "character_position": ci, "character": reasoning[ci]}
    raise ValueError("Could not align violation to original token sequence")


def masks_from_texts(texts, special_ids, ctype):
    masks = {k: np.zeros(len(texts), dtype=bool)
             for k in ("legal", "content", "neutral", "leet_digit", "excluded")}
    for i, s in enumerate(texts):
        bad = i in special_ids or not s or "\ufffd" in s or any(unicodedata.category(c) == "Cs" for c in s)
        masks["excluded"][i] = bad
        if bad:
            continue
        legal = not any(violates(c, ctype) for c in s)
        has_content = (any(c.isalpha() for c in s) if ctype == "uppercase_thinking"
                       else any(c in string.ascii_letters + string.digits for c in s))
        masks["legal"][i] = legal
        masks["content"][i] = legal and has_content
        masks["neutral"][i] = all(c.isspace() or unicodedata.category(c).startswith("P") for c in s)
        masks["leet_digit"][i] = legal and any(c in LEET_DIGITS for c in s)
    return masks


def log_softmax(logits):
    z = np.asarray(logits, dtype=np.float32)
    if z.ndim != 1 or not np.isfinite(z).all():
        raise ValueError("Expected one finite full-vocabulary logits vector")
    z = z - z.max()
    return z - np.log(np.exp(z).sum(dtype=np.float32))


def mask_logmass(logp, mask):
    if not mask.any():
        return None
    a = logp[mask]
    m = a.max()
    return float(m + np.log(np.exp(a - m).sum(dtype=np.float32)))


def nucleus(logits, temperature=0.6, top_p=0.95):
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("Invalid decoding parameters")
    lp = log_softmax(np.asarray(logits, np.float32) / temperature)
    order = np.argsort(-lp, kind="stable")
    probs = np.exp(lp)
    # Float64 accumulation prevents roundoff deciding the crossing token.
    stop = min(int(np.searchsorted(np.cumsum(probs[order], dtype=np.float64), top_p)) + 1, len(lp))
    keep = np.zeros(len(lp), bool)
    keep[order[:stop]] = True
    result = np.zeros(len(lp), dtype=np.float32)
    result[keep] = probs[keep] / probs[keep].sum(dtype=np.float32)
    return result, keep


def token_stat(logits, logp, token_id, texts):
    i = int(token_id)
    return {"id": i, "text": texts[i], "rank": 1 + int(np.count_nonzero(logits > logits[i])),
            "probability": float(np.exp(logp[i])), "log_probability": float(logp[i])}


def summarize(logits, masks, texts, observed_ids):
    logits = np.asarray(logits, np.float32)
    lp = log_softmax(logits)
    p = np.exp(lp)
    sampled, keep = nucleus(logits)
    order = np.argsort(-logits, kind="stable")
    eligible = np.flatnonzero(masks["content"])
    best = int(eligible[np.argmax(logits[eligible])]) if len(eligible) else None
    out = {"entropy_nats": float(-(p * lp).sum(dtype=np.float32)),
           "argmax": token_stat(logits, lp, order[0], texts),
           "best_content": token_stat(logits, lp, best, texts) if best is not None else None,
           "observed": {str(i): token_stat(logits, lp, i, texts) for i in sorted(set(observed_ids))},
           "top200": [token_stat(logits, lp, i, texts) for i in order[:200]],
           "nucleus_size": int(keep.sum())}
    for name, mask in masks.items():
        out[f"{name}_mass"] = float(p[mask].sum(dtype=np.float32))
        out[f"{name}_log_mass"] = mask_logmass(lp, mask)
        out[f"{name}_sample_mass"] = float(sampled[mask].sum(dtype=np.float32))
    for k in (5, 50, 200):
        # Rank-based membership includes all ties at the kth rank.
        out[f"content_in_top{k}"] = best is not None and out["best_content"]["rank"] <= k
    return out, lp


def paired_statistics(logits_by_condition, masks, texts, observed_ids):
    out, lps = {}, {}
    for cond in CONDS:
        out[cond], lps[cond] = summarize(logits_by_condition[cond], masks, texts, observed_ids)
    candidates = sorted({v["best_content"]["id"] for v in out.values() if v["best_content"] is not None})
    fixed = {str(i): {cond: token_stat(logits_by_condition[cond], lps[cond], i, texts)
                      for cond in CONDS} for i in candidates}
    p0, p1 = (np.exp(lps[c]) for c in CONDS)
    return {"conditions": out, "fixed_candidates": fixed,
            "distribution_shift": {
                "kl_instruction_to_no_instruction": float((p1 * (lps[CONDS[1]] - lps[CONDS[0]])).sum()),
                "total_variation": float(np.abs(p1 - p0).sum() / 2)}}


def recover_traces(tag, out):
    """Stream original remote bytes, retain selected full records and source hash."""
    cached = out / f"source_{tag}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    import modal
    vol = modal.Volume.from_name("ban-sweep-out")
    sha, buf, rows, n = hashlib.sha256(), b"", [], 0
    for chunk in vol.read_file(f"ban4_{tag}/traces.jsonl"):
        sha.update(chunk)
        buf += chunk
        lines = buf.split(b"\n")
        buf = lines.pop()
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            if row.get("variant") == "baseline" and row.get("ctype") in TYPES:
                rows.append(row)
    if buf.strip():
        row = json.loads(buf)
        n += 1
        if row.get("variant") == "baseline" and row.get("ctype") in TYPES:
            rows.append(row)
    result = {"volume": "ban-sweep-out", "path": f"ban4_{tag}/traces.jsonl",
              "sha256": sha.hexdigest(), "total_rows": n, "selected": rows}
    immutable_json(cached, result)
    return result


def historical_checks(tag):
    """Two old traces per behavior, scored at content onset and three tokens later."""
    root = DATA if tag == "qwen32" else DATA / "olmo_rl"
    prompts = {p["prompt_id"]: p for p in read_jsonl(root / "prompts.jsonl") if not p.get("_meta")}
    tracepath = root / "traces.jsonl" if tag == "qwen32" else root / "ext_final/traces.jsonl"
    statsdir = root / "entropy" if tag == "qwen32" else root / "ext_final/stats"
    traces = read_jsonl(tracepath)
    checks = []
    for ct in TYPES:
        pool = sorted([t for t in traces if prompts[t["prompt_id"]]["ctype"] == ct
                       and prompts[t["prompt_id"]]["condition"] == "instruction_only"
                       and t["sample"] == 0], key=lambda t: t["trace_id"])[:2]
        for t in pool:
            p = prompts[t["prompt_id"]]
            path = statsdir / (t["trace_id"].replace("|", "_") + ".npz")
            a = np.load(path)
            # Older Qwen emits <think> and newline; t=4 is safely inside content.
            positions = (4, 7) if tag == "qwen32" else (0, 3)
            for pos in positions:
                checks.append({"trace_id": t["trace_id"], "position": pos,
                               "input_ids": p["prompt_token_ids"] + t["output_token_ids"][:pos],
                               "top5_ids": a["top5_ids"][pos].tolist(),
                               "top5_p": a["top5_p"][pos].astype(float).tolist(),
                               "entropy": float(a["entropy"][pos]), "stats_sha256": file_hash(path)})
    return checks


def prepare(run):
    from transformers import AutoTokenizer
    out = DATA / "rank_runs" / run
    out.mkdir(parents=True, exist_ok=True)
    by_model, references, hashes, checks = {}, [], {}, {}
    for tag, spec in MODELS.items():
        print(f"Preparing {tag}: recovering exact source IDs", flush=True)
        source = recover_traces(tag, out)
        path = DATA / "ctrl26" / tag / "prompts.jsonl"
        prompts = {r["prompt_id"]: r for r in read_jsonl(path)}
        runroot = DATA / "ctrl26_runs" / f"ban4_{tag}"
        local = read_jsonl(runroot / "traces.jsonl")
        grades = list(csv.DictReader((runroot / "grades.csv").open()))
        if len(local) != len(grades) or len(local) != source["total_rows"]:
            raise ValueError("Source/local/grade row counts differ")
        grade_map = {}
        local_map = {}
        for t, g in zip(local, grades):
            if t["ctype"] != g["ctype"] or t["variant"] != g["variant"]:
                raise ValueError("Grade alignment mismatch")
            grade_map[t["trace_id"]] = g
            local_map[t["trace_id"]] = t
        snapshot = Path.home() / ".cache/huggingface/hub" / ("models--" + spec["model"].replace("/", "--")) / "snapshots" / spec["revision"]
        tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        pairs = {}
        for t in source["selected"]:
            slim = {k: v for k, v in t.items() if k != "output_token_ids"}
            if slim != local_map[t["trace_id"]]:
                raise ValueError(f"Remote and local traces differ: {t['trace_id']}")
            p = prompts[t["prompt_id"]]
            prompttext = tok.decode(p["prompt_token_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if tok.encode(prompttext, add_special_tokens=False) != p["prompt_token_ids"]:
                raise ValueError("Prompt token round trip failed")
            bare = remove_instruction(prompttext)
            pids = {"instruction_only": p["prompt_token_ids"],
                    "no_instruction": tok.encode(bare, add_special_tokens=False)}
            ids = t["output_token_ids"]
            decoded = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            visible = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            # vLLM omits the terminal EOS from text, but retains it in token IDs.
            if t["output_text"] not in (decoded, visible):
                raise ValueError(f"Output decoding mismatch: {t['trace_id']}")
            violation = first_violation(tok, ids, t["ctype"])
            positions = set(range(min(4, len(ids))))
            if violation:
                positions.add(violation["token_position"])
            grade = grade_map[t["trace_id"]]
            trace_ref = {"model_tag": tag, "trace_id": t["trace_id"], "prompt_id": t["prompt_id"],
                         "ctype": t["ctype"], "question_id": digest(p["question"]),
                         "sample": t["sample"], "instruction": p["instruction"],
                         "phrasing_pool": p["phrasing_pool"], "violation": violation,
                         "compliance_score": float(grade["compliance_score"]),
                         "compliance_hard": int(grade["compliance_hard"]),
                         "output_token_ids": ids, "targets": []}
            for pos in sorted(positions):
                prefix = ids[:pos]
                key = digest({"model": spec, "prompt_ids": pids, "prefix_ids": prefix, "ctype": t["ctype"]})
                if key not in pairs:
                    pairs[key] = {"pair_id": key, "model_tag": tag, "ctype": t["ctype"],
                                  "prompt_id": t["prompt_id"], "question_id": trace_ref["question_id"],
                                  "instruction": p["instruction"], "prompt_ids": pids,
                                  "prefix_ids": prefix, "position": pos, "observed_ids": [], "references": []}
                pair = pairs[key]
                pair["observed_ids"] = sorted(set(pair["observed_ids"] + [ids[pos]]))
                pair["references"].append({"trace_id": t["trace_id"], "sample": t["sample"],
                                           "observed_id": ids[pos]})
                trace_ref["targets"].append({"position": pos, "pair_id": key,
                                              "is_first_violation": violation is not None and pos == violation["token_position"]})
            references.append(trace_ref)
        by_model[tag] = list(pairs.values())
        hashes[tag] = {"remote_trace": source["sha256"], "prompts": file_hash(path),
                       "local_traces": file_hash(runroot / "traces.jsonl"),
                       "grades": file_hash(runroot / "grades.csv"),
                       "tokenizer_files": {f.name: file_hash(f) for f in snapshot.iterdir()
                                           if f.name in {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}}}
        checks[tag] = historical_checks(tag)
        print(f"{tag}: {len(source['selected'])} traces, {len(pairs)} unique paired prefixes", flush=True)
    # Assert cross-model prompts match before interpreting model differences.
    maps = [{p["prompt_id"]: (p["question_id"], p["instruction"]) for p in by_model[tag]} for tag in MODELS]
    if maps[0] != maps[1]:
        raise ValueError("Selected question/instruction sets differ across models")
    manifest = {"version": VERSION, "run": run, "models": MODELS, "constraints": CONSTRAINTS,
                "source_hashes": hashes, "scorer_sha256": file_hash(Path(__file__)),
                "decoding": {"temperature": 0.6, "top_p": 0.95},
                "pairs": by_model, "traces": references, "historical_checks": checks}
    manifest["manifest_sha256"] = digest(manifest)
    immutable_json(out / "manifest.json", manifest)
    print(f"Manifest: {out / 'manifest.json'} ({manifest['manifest_sha256']})")


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    expected = m.pop("manifest_sha256")
    if digest(m) != expected or m["version"] != VERSION:
        raise ValueError("Manifest hash/version mismatch")
    m["manifest_sha256"] = expected
    return m


def load_results(path, manifest_hash):
    out = {}
    if not Path(path).exists():
        return out
    lines = Path(path).read_bytes().splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                break  # Uncommitted last record from a killed worker; rescore it.
            raise
        if r["manifest_sha256"] != manifest_hash or r["version"] != VERSION:
            raise ValueError("Result belongs to another manifest/scorer")
        key = r["pair_id"]
        if key in out and out[key] != r:
            raise ValueError("Conflicting duplicate result")
        out[key] = r
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare"])
    ap.add_argument("--run", default="opening_ranks_v1")
    args = ap.parse_args()
    prepare(args.run)
