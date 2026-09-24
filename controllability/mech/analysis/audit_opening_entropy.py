"""Local audit of the historical entropy experiment; no model/API calls.

Preserves original figures. Writes content-aligned, question-clustered profiles
and observed opening-sequence diversity to analysis/opening_entropy_audit/.
Entropy is raw full-vocabulary H at sampled histories, not decoding entropy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
TYPES = ["uppercase_thinking", "leetspeak", "spanish", "no_first_person",
         "omit_word", "jar_jar_binks"]
CONDS = ["no_instruction", "instruction_only", "icl_user_think"]
LABELS = ["No instruction", "Instruction", "ICL user + think tags"]


def bounds(ids, tok, cache):
    """Content begins after think tag/leading whitespace; ends before closing tag."""
    close = tok.token_to_id("</think>")
    stop = ids.index(close) if close in ids else len(ids)
    start = 0
    while start < stop:
        i = ids[start]
        if i not in cache:
            cache[i] = tok.decode([i], skip_special_tokens=False)
        if cache[i].strip() and cache[i].strip() != "<think>":
            break
        start += 1
    return start, stop


def estimate(arrays, qs, rng, draws=2000):
    """Equal question weighting; repeated samples averaged before bootstrap."""
    groups = defaultdict(list)
    for a, q in zip(arrays, qs):
        groups[q].append(a)
    qmeans = np.stack([np.nanmean(np.stack(v), axis=0) for v in groups.values()])
    pick = rng.integers(0, len(qmeans), size=(draws, len(qmeans)))
    boots = np.nanmean(qmeans[pick], axis=1)
    return np.nanmean(qmeans, axis=0), *np.nanpercentile(boots, [2.5, 97.5], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path)
    args = ap.parse_args()
    tokpath = args.tokenizer or next(Path.home().glob(
        ".cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/*/tokenizer.json"))
    tok = Tokenizer.from_file(str(tokpath))
    out = HERE / "opening_entropy_audit"
    out.mkdir(exist_ok=True)
    prompts = {p["prompt_id"]: p for p in map(json.loads, (DATA / "prompts.jsonl").open())}
    grades = pd.read_csv(DATA / "grades.csv").set_index("trace_id")
    audit = Counter()
    excluded = []
    records, metadata, cache, seen = [], [], {}, set()
    kl_diffs, kl_token_diffs, kl_first_diffs = [], [], []
    max_logp_diff = 0.
    max_within_prompt_first_H_range = 0.
    first_by_prompt = defaultdict(list)
    for t in map(json.loads, (DATA / "traces.jsonl").open()):
        audit["traces"] += 1
        tid = t["trace_id"]
        if tid in seen:
            audit["duplicate_trace_ids"] += 1
        seen.add(tid)
        p = prompts[t["prompt_id"]]
        fp = DATA / "entropy" / (tid.replace("|", "_") + ".npz")
        if not fp.exists():
            audit["missing_entropy"] += 1
            continue
        ids = t["output_token_ids"]
        with np.load(fp) as z:
            H = z["entropy"].astype(float)
            if len(H) != len(ids) or int(z["gen_len"]) != len(ids):
                audit["length_mismatch"] += 1
                continue
            if not np.isfinite(H).all() or np.any(H < -1e-5):
                audit["invalid_entropy"] += 1
                continue
            topids, topp = z["top5_ids"], z["top5_p"].astype(float)
            hit = topids == np.asarray(ids)[:, None]
            if hit.any():
                ii, jj = np.where(hit)
                d = np.abs(np.exp(z["sampled_logp"][ii].astype(float)) - topp[ii, jj])
                max_logp_diff = max(max_logp_diff, float(d.max()))
            start, stop = bounds(ids, tok, cache)
            if start >= stop:
                audit["empty_content"] += 1
                continue
            first_top_p = float(topp[start, 0])
            first_top_id = int(topids[start, 0])
        kfp = DATA / "kl" / "base" / fp.name
        if kfp.exists():
            with np.load(kfp) as z:
                eb = z["ent_base"].astype(float)
                delta = np.abs(H[:len(eb)] - eb)
                kl_diffs.append(float(delta.max()))
                kl_token_diffs.extend(delta)
                if start < len(delta):
                    kl_first_diffs.append(float(delta[start]))
        if p["ctype"] != p["grade_ctype"]:
            audit["task_mismatch_excluded"] += 1
            excluded.append(dict(trace_id=tid, ctype=p["ctype"], actual=p["grade_ctype"]))
            continue
        cot = H[start:stop]
        if len(cot) < 3:
            audit["short_content_excluded"] += 1
            continue
        if tid not in grades.index:
            audit["missing_grade"] += 1
            continue
        g = grades.loc[tid]
        if int(g.empty_thinking):
            audit["grader_empty_excluded"] += 1
            continue
        first_by_prompt[t["prompt_id"]].append(H[start])
        r = dict(trace_id=tid, ctype=p["ctype"], condition=p["condition"], qhash=p["qhash"],
                 content_start=start, content_tokens=len(cot), closed=stop < len(ids),
                 capped=t["finish_reason"] == "length", first_H=H[start],
                 first_top_p=first_top_p, first_top_id=first_top_id,
                 first_bin_tokens=int(len(cot)/20), compliance=float(g.compliance_score))
        metadata.append(r)
        records.append(dict(**r, H=cot, ids=tuple(ids[start:stop])))
    for v in first_by_prompt.values():
        max_within_prompt_first_H_range = max(max_within_prompt_first_H_range, float(np.ptp(v)))
    summary = dict(audit)
    for key in ["duplicate_trace_ids", "missing_entropy", "length_mismatch",
                "invalid_entropy", "empty_content", "short_content_excluded",
                "missing_grade", "grader_empty_excluded"]:
        summary.setdefault(key, 0)
    summary.update(included=len(records), tokenizer=str(tokpath),
                   max_sampled_top5_probability_discrepancy=max_logp_diff,
                   same_prompt_first_content_entropy_max_range=max_within_prompt_first_H_range,
                   independent_KL_pass_matches=len(kl_diffs),
                   independent_KL_pass_max_entropy_difference=max(kl_diffs) if kl_diffs else None,
                   independent_KL_pass_token_abs_difference_quantiles=dict(zip(
                       ["median", "p90", "p99", "max"], np.quantile(kl_token_diffs, [.5,.9,.99,1]).tolist())),
                   independent_KL_pass_first_content_abs_difference_quantiles=dict(zip(
                       ["median", "p90", "p99", "max"], np.quantile(kl_first_diffs, [.5,.9,.99,1]).tolist())),
                   source_hashes={f: hashlib.sha256((DATA/f).read_bytes()).hexdigest()
                                  for f in ["prompts.jsonl", "traces.jsonl", "grades.csv"]})
    (out / "validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(metadata).to_csv(out / "trace_metadata.csv", index=False)
    pd.DataFrame(excluded).to_csv(out / "excluded_task_mismatches.csv", index=False)
    class_counts = []
    for ct in ["no_first_person", "omit_word", "spanish", "jar_jar_binks"]:
        for cond in CONDS:
            for cls in ["compliant", "noncompliant"]:
                hi, low = (1., 0.) if ct in ["no_first_person", "omit_word"] else (.75, .25)
                ss=[r for r in records if r["ctype"]==ct and r["condition"]==cond and
                    (r["compliance"]>=hi if cls=="compliant" else r["compliance"]<=low)]
                class_counts.append(dict(ctype=ct,condition=cond,cls=cls,traces=len(ss),
                                         questions=len({r["qhash"] for r in ss})))
    pd.DataFrame(class_counts).to_csv(out/"class_counts.csv",index=False)
    profiles, windows, diversity = [], [], []
    rng = np.random.default_rng(20260916)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True, sharey=True)
    for ax, ct in zip(axes.flat, TYPES):
        for cond, label, color in zip(CONDS, LABELS, ["#707070", "#2266ad", "#c15e16"]):
            rr = [r for r in records if r["ctype"] == ct and r["condition"] == cond]
            for cohort in ["available", "length_ge_128"]:
                ss = rr if cohort == "available" else [r for r in rr if len(r["H"]) >= 128]
                arr = [np.pad(r["H"][:128], (0, max(0,128-len(r["H"]))), constant_values=np.nan) for r in ss]
                m, lo, hi = estimate(arr, [r["qhash"] for r in ss], rng)
                for i in range(128):
                    profiles.append(dict(ctype=ct, condition=cond, cohort=cohort, position=i+1,
                                         mean=m[i], low=lo[i], high=hi[i],
                                         traces=sum(len(r["H"])>i for r in ss),
                                         questions=len({r["qhash"] for r in ss if len(r["H"])>i})))
                if cohort == "available":
                    ax.plot(np.arange(1,65), m[:64], label=label, color=color, lw=1.4)
                    ax.fill_between(np.arange(1,65),lo[:64],hi[:64],color=color,alpha=.12)
            for a,b in [(0,1),(0,8),(8,32),(32,64),(64,128)]:
                ss=[r for r in rr if len(r["H"])>=b]
                arr=[np.array([r["H"][a:b].mean()]) for r in ss]
                m,lo,hi=estimate(arr,[r["qhash"] for r in ss],rng)
                windows.append(dict(ctype=ct,condition=cond,start=a+1,end=b,
                                    mean=m[0],low=lo[0],high=hi[0],traces=len(ss),
                                    questions=len({r["qhash"] for r in ss})))
            for offset in [0,32,64]:
                for k in [1,2,4,8,16]:
                    ss=[r for r in rr if len(r["ids"])>=offset+k]
                    c=Counter(r["ids"][offset:offset+k] for r in ss)
                    n=len(ss); counts=np.array(list(c.values()))
                    byq=defaultdict(Counter)
                    for r in ss:byq[r["qhash"]][r["ids"][offset:offset+k]]+=1
                    within=sum(sum(v*(v-1) for v in z.values()) for z in byq.values())
                    crossden=n*(n-1)-sum(sum(z.values())*(sum(z.values())-1) for z in byq.values())
                    crossnum=sum(v*(v-1) for v in c.values())-within
                    most=c.most_common(5)
                    diversity.append(dict(ctype=ct,condition=cond,offset=offset,prefix_tokens=k,
                        traces=n,questions=len(byq),unique=len(c),mode_share=counts.max()/n,
                        top5_share=sum(v for _,v in most)/n,
                        cross_question_collision=crossnum/crossden if crossden else np.nan,
                        most_common=json.dumps([(tok.decode(list(s),skip_special_tokens=False),v) for s,v in most],ensure_ascii=False)))
        ax.set_title(ct.replace("_"," "), fontsize=10)
        ax.grid(alpha=.2); ax.set_xlim(1,64); ax.set_ylim(bottom=0)
        ax.set_xticks([1,8,16,32,48,64]); ax.tick_params(labelbottom=True)
        ax.set_xlabel("Reasoning content token position")
        ax.set_ylabel("Raw next-token entropy (nats)")
    axes[0,0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/"entropy_absolute.png",dpi=180)
    fig.savefig(out/"entropy_absolute.pdf")
    pd.DataFrame(profiles).to_csv(out/"absolute_profiles.csv",index=False)
    pd.DataFrame(windows).to_csv(out/"window_estimates.csv",index=False)
    pd.DataFrame(diversity).to_csv(out/"opening_diversity.csv",index=False)
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(windows).query("condition == 'instruction_only'").round(4).to_string(index=False))
    print(pd.DataFrame(diversity).query("condition == 'instruction_only' and offset == 0 and prefix_tokens in [1,4,8]").to_string(index=False))


if __name__ == "__main__":
    main()
