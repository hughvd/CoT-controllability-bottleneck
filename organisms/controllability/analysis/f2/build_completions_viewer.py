"""Build a self-contained HTML viewer for F2 completions, to inspect why soft and
hard diverge (near-miss completions: soft high, but < 0.95 so hard=0).
Embeds qwen3_32b per_sample (all runs) as JSON; filter by run/category/family and
a soft-score band, sort by score, expand the full reasoning.

Usage: python3 analysis/f2/build_completions_viewer.py [model]   # default qwen3_32b
"""
import sys, json, html
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = next(q for q in [HERE, *HERE.parents] if (q / "sft" / "configs").is_dir())
RES = ROOT / "results" / "f2"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3_32b"
RUNS = ["base", "transforms", "encoding", "language", "suppression"]
ENC = {"pig_latin", "leetspeak", "disemvowel", "reverse_each_word", "rot13"}
MAXR = 3000   # truncate reasoning chars

def cat_of(fam, mod):
    if mod == "language": return "language"
    if mod == "suppression": return "suppression"
    return "encoding" if fam in ENC else "transforms"

rows = []
for run in RUNS:
    f = RES / MODEL / run / "per_sample" / "final.csv"
    if not f.exists(): continue
    d = pd.read_csv(f)
    for _, r in d.iterrows():
        reason = r.get("reasoning")
        if not isinstance(reason, str) or not reason.strip():
            reason = str(r.get("response", ""))
        s = float(r.get("compliance_score", 0) or 0)
        rows.append({
            "run": run,
            "fam": str(r.get("family", "")),
            "cat": cat_of(str(r.get("family", "")), str(r.get("modality", ""))),
            "soft": round(s, 3),
            "hard": 1 if s >= 0.95 else 0,
            "instr": str(r.get("instruction", ""))[:220],
            "reason": reason[:MAXR],
        })
print(f"{MODEL}: {len(rows)} completions")

DATA = json.dumps(rows, ensure_ascii=False)
HTML = """<!doctype html><html><head><meta charset="utf-8"><title>F2 completions — %MODEL%</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{position:sticky;top:0;background:#171a21;padding:10px 14px;border-bottom:1px solid #333;z-index:5}
 select,input,button{background:#222;color:#eee;border:1px solid #444;border-radius:4px;padding:4px 6px;margin:2px}
 .row{border-bottom:1px solid #2a2d34;padding:8px 14px}
 .meta{font-size:13px;color:#9aa0aa}
 .badge{display:inline-block;padding:1px 6px;border-radius:4px;font-size:12px;margin-right:6px}
 .soft{background:#2b4}.mid{background:#b83}.low{background:#555}.hardno{background:#a33}.hardyes{background:#183}
 .instr{color:#8bd;font-size:12px;margin:3px 0}
 pre{white-space:pre-wrap;word-break:break-word;background:#12141a;border:1px solid #262a33;border-radius:6px;
     padding:8px;margin:4px 0 0;font-size:12px;max-height:240px;overflow:auto}
 #count{color:#9aa0aa;font-size:13px;margin-left:8px}
</style></head><body>
<header>
 <b>F2 completions — %MODEL%</b>
 run <select id="run"></select>
 cat <select id="cat"></select>
 family <select id="fam"></select>
 soft <input id="lo" type="number" step="0.05" value="0" style="width:60px"> – <input id="hi" type="number" step="0.05" value="1" style="width:60px">
 <button onclick="preset(0.5,0.949)">gap band (0.5–0.95)</button>
 <button onclick="preset(0,1)">all</button>
 sort <select id="sort"><option value="soft">soft ↑</option><option value="softd">soft ↓</option></select>
 <span id="count"></span>
</header>
<div id="list"></div>
<script>
const DATA = %DATA%;
const $=id=>document.getElementById(id);
function opts(sel,vals){sel.innerHTML='<option value="">all</option>'+vals.map(v=>`<option>${v}</option>`).join('')}
opts($('run'),[...new Set(DATA.map(d=>d.run))]);
opts($('cat'),[...new Set(DATA.map(d=>d.cat))].sort());
function refreshFam(){const c=$('cat').value,r=$('run').value;
  const fams=[...new Set(DATA.filter(d=>(!c||d.cat==c)&&(!r||d.run==r)).map(d=>d.fam))].sort();
  opts($('fam'),fams)}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function render(){
  const r=$('run').value,c=$('cat').value,f=$('fam').value,lo=+$('lo').value,hi=+$('hi').value,so=$('sort').value;
  let rows=DATA.filter(d=>(!r||d.run==r)&&(!c||d.cat==c)&&(!f||d.fam==f)&&d.soft>=lo&&d.soft<=hi);
  rows.sort((a,b)=>so=='soft'?a.soft-b.soft:b.soft-a.soft);
  $('count').textContent=rows.length+' shown';
  const N=Math.min(rows.length,600);
  $('list').innerHTML=rows.slice(0,N).map(d=>{
    const sc=d.soft>=0.95?'soft':d.soft>=0.5?'mid':'low';
    return `<div class="row"><span class="meta">
      <span class="badge ${sc}">soft ${d.soft.toFixed(2)}</span>
      <span class="badge ${d.hard?'hardyes':'hardno'}">hard ${d.hard}</span>
      <b>${d.fam}</b> · ${d.cat} · run:${d.run}</span>
      <div class="instr">${esc(d.instr)}</div>
      <pre>${esc(d.reason)}</pre></div>`}).join('')
    +(rows.length>N?`<div class="row meta">… ${rows.length-N} more (narrow the filter)</div>`:'');
}
['run','cat','fam','lo','hi','sort'].forEach(id=>$(id).addEventListener('input',()=>{if(id=='run'||id=='cat')refreshFam();render()}));
function preset(a,b){$('lo').value=a;$('hi').value=b;render()}
refreshFam();render();
</script></body></html>"""
out = HERE.parent / f"f2_completions_{MODEL}.html"
out.write_text(HTML.replace("%MODEL%", MODEL).replace("%DATA%", DATA), encoding="utf-8")
print("wrote", out, f"({out.stat().st_size/1e6:.1f} MB)")
