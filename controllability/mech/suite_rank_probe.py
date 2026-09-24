"""Six-behavior expansion; immutable v1 measurements are imported, never edited."""
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
import rank_probe as base
from rank_probe import (DATA, HERE, MODELS, CONDS, canonical, digest, file_hash,
                        immutable_json, read_jsonl, remove_instruction, historical_checks)

VERSION = "suite-ranks-1"
TYPES = ("disemvowel", "no_commas", "spell_out_numbers", "start_with",
         "prefix_sentences", "numbered_sentences")
PREFIX_TYPES = {"start_with", "prefix_sentences", "numbered_sentences"}
CONSTRAINTS = {
    "disemvowel": "ASCII words: preserve first letter; no a/e/i/o/u afterward.",
    "no_commas": "No literal comma; primary measure is forbidden-token mass.",
    "spell_out_numbers": "No Unicode decimal digit (regex \\d); primary measure is forbidden-token mass.",
    "start_with": "Leading whitespace allowed; exact marker then whitespace or end; incompatible partial prefixes fail.",
    "prefix_sentences": "Same marker at each sentence opening; boundaries are .!? followed by whitespace, matching the existing sentence splitter.",
    "numbered_sentences": "Exact 1., 2., ... at each sentence opening, followed by whitespace/end. The number's period is not a sentence boundary. Stricter than the aggregate original grader.",
    "prefix_progress": "Content mask counts legal tokens consuming at least one required marker character; completion also requires a separator in the candidate. Whitespace alone does not count as progress.",
    "inactive_prefix": "After a prefix has failed/completed, opening progress statistics are undefined until the next sentence header; distributions are still saved.",
    "termination": "An unfinished required header at reasoning termination is a violation at the closing think/EOS token, flagged separately.",
    "shared": base.CONSTRAINTS,
}


def initial_state():
    return dict(sentence=1, phase="header", matched=0, terminal=False, in_word=False)


def expected_header(ctype, marker, state):
    return f"{state['sentence']}." if ctype == "numbered_sentences" else marker


def scan(text, ctype, marker=None, state=None):
    """Causal character scanner. Return state, first violation, progress/completions."""
    s = dict(state or initial_state())
    violation, progress, completed = None, 0, 0
    for i, char in enumerate(text):
        bad = False
        if ctype in PREFIX_TYPES and s["phase"] == "header":
            target = expected_header(ctype, marker, s)
            j = s["matched"]
            if j == 0 and char.isspace():
                continue
            if j < len(target):
                if char == target[j]:
                    s["matched"] += 1
                    progress += 1
                    continue  # Ordinal's period never becomes sentence punctuation.
                bad = True
                s["phase"] = "body"
            elif char.isspace():
                s["phase"] = "body"
                s["terminal"] = False
                completed += 1
                continue
            else:
                bad = True  # Marker must be followed by a separator.
                s["phase"] = "body"
        elif ctype == "disemvowel":
            alpha = char in string.ascii_letters
            bad = alpha and s["in_word"] and char in "aeiouAEIOU"
            s["in_word"] = alpha
        elif ctype == "no_commas":
            bad = char == ","
        elif ctype == "spell_out_numbers":
            bad = char.isdecimal()
        if bad and violation is None:
            violation = dict(character_position=i, character=char, sentence_index=s["sentence"])
        if char.isspace():
            if s["terminal"]:
                s["sentence"] += 1
                s["terminal"] = False
                if ctype in {"prefix_sentences", "numbered_sentences"}:
                    s["phase"], s["matched"] = "header", 0
        else:
            s["terminal"] = char in ".!?"
    return s, violation, progress, completed


def constraint_active(ctype, state):
    return ctype not in PREFIX_TYPES or state["phase"] == "header"


def mask_key(ctype, marker, state):
    if ctype in PREFIX_TYPES:
        return (ctype, expected_header(ctype, marker, state), state["phase"], state["matched"])
    return (ctype, state["in_word"] if ctype == "disemvowel" else None)


def masks_from_state(texts, special_ids, ctype, marker, state):
    # Base exclusion/neutral definitions are shared with the completed experiment.
    masks = base.masks_from_texts(texts, special_ids, "uppercase_thinking")
    for k in ("legal", "content", "leet_digit"):
        masks[k][:] = False
    masks["forbidden"] = np.zeros(len(texts), bool)
    masks["completion"] = np.zeros(len(texts), bool)
    if not constraint_active(ctype, state):
        return masks
    for i, text in enumerate(texts):
        if masks["excluded"][i]:
            continue
        _, bad, progress, completed = scan(text, ctype, marker, state)
        legal = bad is None
        masks["legal"][i] = legal
        masks["forbidden"][i] = not legal
        masks["content"][i] = legal and (progress > 0 if ctype in PREFIX_TYPES else
                                           any(c.isalpha() or c.isdecimal() for c in text))
        masks["completion"][i] = legal and completed > 0
    return masks


def first_violation(tokenizer, ids, ctype, marker=None):
    text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    end = text.find("</think>")
    if end < 0:
        special_ids = set(tokenizer.all_special_ids)
        ends = [text.find(tokenizer.decode([i])) for i in set(ids) & special_ids]
        end = min((i for i in ends if i >= 0), default=len(text))
    reasoning = text[:end]
    state, violation, _, _ = scan(reasoning, ctype, marker)
    if violation is None and ctype in PREFIX_TYPES and state["phase"] == "header":
        # Trailing whitespace after a completed sentence is not a missing next sentence.
        if state["matched"] > 0 or not reasoning.strip():
            target = expected_header(ctype, marker, state)
            if state["matched"] < len(target):
                violation = dict(character_position=len(reasoning), character="<END>",
                                 sentence_index=state["sentence"], termination=True)
    if violation is None:
        return None
    ci = violation["character_position"]
    def contains(t):
        prefix = tokenizer.decode(ids[:t+1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return len(prefix) > ci and prefix[:ci+1] == text[:ci+1]
    # Decoded text is append-only once a complete character is present. This
    # avoids quadratic decoding for violations thousands of tokens into a trace.
    lo, hi = 0, len(ids)-1
    while lo < hi:
        mid = (lo+hi)//2
        if contains(mid):
            hi = mid
        else:
            lo = mid+1
    if not contains(lo):
        raise ValueError("Unmapped local violation")
    return dict(violation, token_position=lo)


def load_manifest(path):
    m = json.loads(Path(path).read_text())
    sha = m.pop("manifest_sha256")
    if digest(m) != sha or m["version"] != VERSION:
        raise ValueError("Expansion manifest mismatch")
    m["manifest_sha256"] = sha
    return m


def load_results(path, manifest_hash):
    if not Path(path).exists():
        return {}
    lines = Path(path).read_bytes().splitlines(keepends=True)
    results = {}
    for index, line in enumerate(lines):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines)-1 and not line.endswith(b"\n"):
                break
            raise
        if r["version"] != VERSION or r["manifest_sha256"] != manifest_hash:
            raise ValueError("Result version/hash mismatch")
        if r["pair_id"] in results and results[r["pair_id"]] != r:
            raise ValueError("Conflicting duplicate result")
        results[r["pair_id"]] = r
    return results


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

def prepare(run):
    from transformers import AutoTokenizer
    out = DATA / "rank_runs" / run
    out.mkdir(parents=True, exist_ok=True)
    by_model, references, hashes, checks = {}, [], {}, {}
    for tag, spec in MODELS.items():
        print(f"Preparing {tag}: recovering exact source IDs", flush=True)
        source = recover_traces(tag, out)
        prior = base.load_manifest(DATA / "rank_runs/opening_ranks_v1/manifest.json")
        if source["sha256"] != prior["source_hashes"][tag]["remote_trace"]:
            raise ValueError("Source traces changed since the primary experiment")
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
            withheld_tail = None
            if t["output_text"] not in (decoded, visible):
                tail = decoded[len(t["output_text"]):] if decoded.startswith(t["output_text"]) else ""
                if "\ufffd" in tail and all(c.isspace() or c == "\ufffd" for c in tail):
                    withheld_tail = tail  # Length-limited vLLM text withholds an incomplete UTF-8 ending.
                else:
                    raise ValueError(f"Output decoding mismatch: {t['trace_id']}")
            violation = first_violation(tok, ids, t["ctype"], p.get("marker"))
            positions = set(range(min(4, len(ids))))
            if violation:
                positions.add(violation["token_position"])
            grade = grade_map[t["trace_id"]]
            trace_ref = {"model_tag": tag, "trace_id": t["trace_id"], "prompt_id": t["prompt_id"],
                         "ctype": t["ctype"], "question_id": digest(p["question"]),
                         "sample": t["sample"], "instruction": p["instruction"], "withheld_decoding_tail": withheld_tail,
                         "phrasing_pool": p["phrasing_pool"], "marker_pool": p["marker_pool"], "marker": p.get("marker"), "violation": violation,
                         "compliance_score": float(grade["compliance_score"]),
                         "compliance_hard": int(grade["compliance_hard"]),
                         "output_token_ids": ids, "targets": []}
            for pos in sorted(positions):
                prefix = ids[:pos]
                prefix_text = tok.decode(prefix, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                state = scan(prefix_text, t["ctype"], p.get("marker"))[0]
                key = digest({"model": spec, "prompt_ids": pids, "prefix_ids": prefix, "ctype": t["ctype"]})
                if key not in pairs:
                    pairs[key] = {"pair_id": key, "model_tag": tag, "ctype": t["ctype"],
                                  "prompt_id": t["prompt_id"], "question_id": trace_ref["question_id"],
                                  "instruction": p["instruction"], "prompt_ids": pids,
                                  "prefix_ids": prefix, "position": pos, "marker": p.get("marker"), "state": state, "constraint_active": constraint_active(t["ctype"], state), "observed_ids": [], "references": []}
                pair = pairs[key]
                pair["observed_ids"] = sorted(set(pair["observed_ids"] + [ids[pos]]))
                pair["references"].append({"trace_id": t["trace_id"], "sample": t["sample"],
                                           "observed_id": ids[pos], "is_first_violation": violation is not None and pos == violation["token_position"]})
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
                "source_hashes": hashes, "scorer_sha256": file_hash(Path(__file__)), "base_scorer_sha256": file_hash(Path(base.__file__)),
                "prior_run": "opening_ranks_v1", "prior_manifest_sha256": base.load_manifest(DATA / "rank_runs/opening_ranks_v1/manifest.json")["manifest_sha256"],
                "decoding": {"temperature": 0.6, "top_p": 0.95},
                "pairs": by_model, "traces": references, "historical_checks": checks}
    print("Longest input:", max(len(p["prompt_ids"]["instruction_only"]) + len(p["prefix_ids"]) for pool in by_model.values() for p in pool))
    manifest["manifest_sha256"] = digest(manifest)
    immutable_json(out / "manifest.json", manifest)
    print(f"Manifest: {out / 'manifest.json'} ({manifest['manifest_sha256']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="opening_suite_v1")
    prepare(ap.parse_args().run)
