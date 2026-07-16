"""Dump prefill-experiment completions to a readable HTML + JSONL.

Reads the AsyncCache JSON written by prefill_experiment.py --mode sweep and
renders every completion as: the exact messages sent (user prompt + assistant
<think> seed) and the model's continuation + answer, with the grade. Grouped by
control type, sorted by condition then K, so you can eyeball formatting and how
compliance tracks the seed length.

Usage:
  python dump_completions.py \
      --cache ../results/prefill_continuation/base_qwen3_14b/sweep.json \
      --out results/completions_dump
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE.parent / "results" / "prefill_continuation" / "base_qwen3_14b" / "sweep.json"


def load_records(cache_path: Path) -> list[dict]:
    cache = json.loads(cache_path.read_text())
    recs = [v for v in cache.values() if isinstance(v, dict) and "error" not in v]
    # group order: modality, ctype, condition, k, sample
    recs.sort(key=lambda r: (r.get("modality", ""), r.get("ctype", ""),
                             r.get("condition", ""), r.get("k", 0), r.get("sample", 0)))
    return recs


def write_jsonl(recs: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(recs)} records -> {path}")


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def write_html(recs: list[dict], path: Path) -> None:
    parts: list[str] = [
        "<!doctype html><meta charset='utf-8'><title>prefill completions</title>",
        """<style>
        body{font:13px/1.45 -apple-system,Menlo,monospace;margin:0;background:#f6f7f9;color:#1a1a1a}
        h2{position:sticky;top:0;background:#222;color:#fff;margin:0;padding:8px 14px;font-size:15px}
        .rec{background:#fff;border:1px solid #ddd;border-radius:6px;margin:10px 14px;overflow:hidden}
        .hdr{padding:6px 10px;font-weight:bold;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
        .ok{background:#e6f5e9}.no{background:#fdeaea}
        .pill{border-radius:10px;padding:1px 8px;font-size:11px;background:#eee}
        .box{margin:0;padding:8px 10px;white-space:pre-wrap;border-top:1px solid #eee;font-size:12px}
        .lbl{font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:#888;padding:4px 10px 0}
        .user{background:#f0f4ff}.seed{background:#fff8e6}.cont{background:#f3fff3}.resp{background:#f7f7f7}
        details summary{cursor:pointer;padding:6px 10px;color:#555}
        </style>""",
    ]
    # quick top summary
    by_cell: dict[tuple, list[dict]] = {}
    for r in recs:
        by_cell.setdefault((r["modality"], r["ctype"]), []).append(r)

    parts.append("<h2>prefill completions — " + str(len(recs)) + " records</h2>")
    for (modality, ctype), group in by_cell.items():
        parts.append(f"<h2>{_esc(modality)} &mdash; {_esc(ctype)} ({len(group)})</h2>")
        for r in group:
            ok = r.get("compliance", 0) == 1
            cls = "ok" if ok else "no"
            seed = r.get("seed", "")
            parts.append(f"<div class='rec'><div class='hdr {cls}'>"
                         f"<span class='pill'>K={_esc(r.get('k'))}</span>"
                         f"<span class='pill'>{_esc(r.get('condition'))}</span>"
                         f"<span class='pill'>marker={_esc(r.get('marker'))}</span>"
                         f"<span class='pill'>score={r.get('compliance_score',0):.3f}</span>"
                         f"<span class='pill'>compliant={ok}</span>"
                         f"<span class='pill'>pred={_esc(r.get('predicted'))} / gold={_esc(r.get('correct_letter'))} "
                         f"({'correct' if r.get('correct') else 'wrong'})</span>"
                         f"<span class='pill'>finish={_esc(r.get('finish_reason'))}</span>"
                         "</div>")
            parts.append("<div class='lbl'>user message sent</div>"
                         f"<pre class='box user'>{_esc(r.get('prompt'))}</pre>")
            if seed:
                parts.append("<div class='lbl'>assistant prefill (seed injected into &lt;think&gt;)</div>"
                             f"<pre class='box seed'>&lt;think&gt;\n{_esc(seed)}</pre>")
            else:
                parts.append("<div class='lbl'>assistant prefill</div>"
                             "<pre class='box seed'>(none — K=0)</pre>")
            parts.append("<div class='lbl'>model continuation (seed stripped — graded)</div>"
                         f"<pre class='box cont'>{_esc(r.get('reasoning_continuation'))}</pre>")
            parts.append("<div class='lbl'>response (after &lt;/think&gt;)</div>"
                         f"<pre class='box resp'>{_esc(r.get('response'))}</pre>")
            parts.append("</div>")
    path.write_text("\n".join(parts))
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--out", default=str(HERE / "results" / "completions_dump"))
    args = ap.parse_args()

    recs = load_records(Path(args.cache))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_html(recs, out.with_suffix(".html"))
    write_jsonl(recs, out.with_suffix(".jsonl"))


if __name__ == "__main__":
    main()
