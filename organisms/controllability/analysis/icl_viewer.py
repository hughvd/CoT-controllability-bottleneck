"""Generate a self-contained HTML viewer for the ICL transcripts.

Reads the saved per-cell transcripts (transcripts/icl_{model}/*.json, each
{messages, raw_response}) plus the per-sample CSVs (for compliance_score), and emits
analysis/icl/icl_viewer.html — a browsable page showing, for every example:
  - the EXACT messages the model was sent (role-colored, whitespace preserved),
  - the model's THINK (reasoning — the graded text) vs its ANSWER (content), side by side,
  - compliance_score / predicted / correct.
Filter by model / control type / variant via the dropdowns at the top.

  python analysis/icl_viewer.py   &&   open analysis/icl/icl_viewer.html
"""

from __future__ import annotations

import glob
import hashlib
import html
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "icl"

MODELS = [("qwen3-14b", "iclfull_qwen3_14b"), ("qwen3-32b", "iclfull_qwen3_32b"),
          ("glm-5.2", "iclfull_glm_5_2")]
VARIANTS = ["instruction_only", "icl_user_plain", "icl_user_think",
            "icl_assistant_think", "icl_assistant_visible"]
TYPES = ["leetspeak", "spanish", "chinese", "uppercase_thinking", "prefix_sentences",
         "no_first_person", "omit_word", "pirate", "yoda", "jar_jar_binks"]
ROLE_BG = {"user": "#eef4ff", "assistant": "#eefbf0", "system": "#fff7e6"}


def _qhash(q: str) -> str:
    return hashlib.sha256(str(q).encode("utf-8")).hexdigest()[:16]


def parse_name(stem: str) -> tuple[str, str, str, int]:
    """{qhash}_{ctype}_{variant}_s{sample} -> (qhash, ctype, variant, sample)."""
    body, _, srest = stem.rpartition("_s")
    sample = int(srest) if srest.isdigit() else 0
    variant = next((v for v in VARIANTS if body.endswith("_" + v)), "")
    body = body[: -len(variant) - 1] if variant else body
    ctype = next((t for t in TYPES if body.endswith("_" + t) or body == t), "")
    qhash = body[: -len(ctype) - 1] if body.endswith("_" + ctype) else body
    return qhash, ctype, variant, sample


def load_scores() -> dict:
    """(qhash, ctype, variant, sample) -> row dict from the per-sample CSVs."""
    out = {}
    for short, run in MODELS:
        p = ROOT / "results" / f"{run}_icl_per_sample.csv"
        if not p.exists():
            continue
        for _, r in pd.read_csv(p).iterrows():
            key = (_qhash(r["target_question"]), r["ctype"], r["variant"], int(r["sample"]))
            out[(short, *key)] = r.to_dict()
    return out


def esc(s) -> str:
    return html.escape(str(s or ""))


def card(short: str, ctype: str, variant: str, messages: list, msg_out: dict, row: dict) -> str:
    reasoning = msg_out.get("reasoning") or ""
    content = msg_out.get("content") or ""
    score = row.get("compliance_score")
    score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
    pred = row.get("predicted"); corr = row.get("correct")
    parts = [
        f'<div class="card" data-model="{esc(short)}" data-ctype="{esc(ctype)}" data-variant="{esc(variant)}">',
        f'<div class="hd"><b>{esc(short)}</b> &middot; {esc(ctype)} &middot; '
        f'<span class="v">{esc(variant)}</span> &middot; compliance=<b>{score_s}</b> '
        f'&middot; pred={esc(pred)} correct={esc(corr)}</div>',
        '<div class="sec">MESSAGES SENT (exact format the model sees):</div>',
    ]
    for m in messages:
        bg = ROLE_BG.get(m["role"], "#f2f2f2")
        parts.append(f'<div class="msg" style="background:{bg}"><div class="role">{esc(m["role"])}</div>'
                     f'<pre>{esc(m["content"])}</pre></div>')
    parts.append('<div class="sec">MODEL OUTPUT:</div>')
    parts.append('<div class="out">'
                 f'<div class="think"><div class="role">think (GRADED)</div><pre>{esc(reasoning)}</pre></div>'
                 f'<div class="answer"><div class="role">answer (content)</div><pre>{esc(content)}</pre></div>'
                 '</div>')
    parts.append("</div>")
    return "\n".join(parts)


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fafafa;color:#222}
#bar{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:10px 16px;z-index:10}
#bar select{margin-right:12px;padding:4px}
.card{border:1px solid #ddd;border-radius:8px;margin:16px;background:#fff;overflow:hidden}
.hd{background:#333;color:#fff;padding:8px 12px;font-size:13px}
.hd .v{color:#ffd479}
.sec{font-size:11px;color:#888;padding:8px 12px 0;letter-spacing:.5px}
.msg{margin:8px 12px;border:1px solid #e2e2e2;border-radius:6px}
.role{font-size:10px;font-weight:700;color:#666;padding:4px 8px;text-transform:uppercase;letter-spacing:.5px}
pre{margin:0;padding:8px 12px;white-space:pre-wrap;word-break:break-word;font-size:12.5px;line-height:1.4}
.out{display:flex;gap:8px;margin:8px 12px 14px}
.out>div{flex:1;border:1px solid #e2e2e2;border-radius:6px}
.think{background:#fffdf3}.answer{background:#f5f8ff}
"""

JS = """
function apply(){
  const m=v('fm'),c=v('fc'),va=v('fv');
  document.querySelectorAll('.card').forEach(el=>{
    const ok=(!m||el.dataset.model==m)&&(!c||el.dataset.ctype==c)&&(!va||el.dataset.variant==va);
    el.style.display=ok?'block':'none';
  });
}
function v(id){return document.getElementById(id).value}
document.addEventListener('DOMContentLoaded',()=>{['fm','fc','fv'].forEach(id=>document.getElementById(id).onchange=apply)});
"""


def sel(id_: str, label: str, opts: list) -> str:
    o = "".join(f'<option value="{esc(x)}">{esc(x)}</option>' for x in opts)
    return f'{label}: <select id="{id_}"><option value="">all</option>{o}</select>'


def main() -> None:
    scores = load_scores()
    cards = []
    for short, run in MODELS:
        for f in sorted(glob.glob(str(ROOT / "transcripts" / run / "*.json"))):
            d = json.load(open(f))
            qhash, ctype, variant, sample = parse_name(Path(f).stem)
            row = scores.get((short, qhash, ctype, variant, sample), {})
            msg_out = d["raw_response"]["choices"][0]["message"]
            cards.append((short, ctype, variant, card(short, ctype, variant, d["messages"], msg_out, row)))
    # order by variant then ctype then model for easy comparison
    cards.sort(key=lambda t: (VARIANTS.index(t[2]) if t[2] in VARIANTS else 9,
                              TYPES.index(t[1]) if t[1] in TYPES else 9, t[0]))
    bar = ('<div id="bar">'
           + sel("fm", "model", [m for m, _ in MODELS]) + sel("fc", "type", TYPES)
           + sel("fv", "variant", VARIANTS) + f'<span style="color:#999;font-size:12px">'
           f'&nbsp;{len(cards)} transcripts (sample 0)</span></div>')
    body = "\n".join(c[3] for c in cards)
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><title>ICL viewer</title>"
           f"<style>{CSS}</style></head><body>{bar}{body}<script>{JS}</script></body></html>")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "icl_viewer.html"
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}  ({len(cards)} transcripts)")


if __name__ == "__main__":
    main()
