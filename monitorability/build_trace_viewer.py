"""Build a self-contained HTML viewer for seed-0 monitorability traces
(base / control / organism), to eyeball monitorable vs unmonitorable CoTs.

Each record carries: cot, answer, final_answer vs reference_answer (ground truth),
arm x (0 baseline / 1 hint-intervention), behavior y (used the hint), and the
monitor's verdict + raw reasoning per scope (cot_only / answer_only / all_messages).

Derived monitor status (from cot_only, the headline scope):
  y=1,z=1 caught (monitorable)   y=1,z=0 MISSED (unmonitorable)
  y=0,z=1 false-alarm            y=0,z=0 clean

Usage:
  python -m monitorability.build_trace_viewer                 # gpqa, 3 models
  python -m monitorability.build_trace_viewer --datasets gpqa wmdp_sandbagging
Writes results/monitorability/v34/trace_viewer.html — open in a browser.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MON = ROOT / "results" / "monitorability"
OUT = MON / "v34" / "trace_viewer.html"
MODELS = [("qwen3-32b", "base"), ("v34_ctrl_normal", "control"), ("v34_ctrl_all", "organism")]
SCOPES = ["cot_only", "answer_only", "all_messages"]


def status_of(y, z):
    if y == 1 and z == 1: return "caught"
    if y == 1 and z == 0: return "missed"
    if y == 0 and z == 1: return "false_alarm"
    return "clean"


def extract(model_label, dataset, rec):
    y, z = rec.get("y"), rec.get("z_cot_only")
    fa, ref = rec.get("final_answer"), rec.get("reference_answer")
    return {
        "model": model_label, "dataset": dataset,
        "id": rec.get("instance_id"), "s": rec.get("sample_idx"),
        "cat": rec.get("category"),
        "x": rec.get("x"), "y": y,
        "fa": fa, "ref": ref,
        "correct": (None if ref is None else int(str(fa) == str(ref))),
        "status": status_of(y, z),
        "cot": rec.get("cot", "") or "", "answer": rec.get("answer", "") or "",
        "mon": {sc: {"a": rec.get(f"monitor_answer_{sc}"), "z": rec.get(f"z_{sc}"),
                     "raw": rec.get(f"monitor_raw_{sc}", "") or ""} for sc in SCOPES},
    }


def load(datasets):
    rows = []
    for label_dir, name in MODELS:
        for ds in datasets:
            fp = MON / label_dir / f"{ds}.json"
            if not fp.exists():
                print(f"skip {name}/{ds} (missing)")
                continue
            data = json.loads(fp.read_text())
            recs = data.values() if isinstance(data, dict) else data
            n = 0
            for r in recs:
                if isinstance(r, dict) and not r.get("error"):
                    rows.append(extract(name, ds, r))
                    n += 1
            print(f"{name}/{ds}: {n} records")
    return rows


HTML = r"""<meta charset="utf-8"><title>v34 trace viewer</title>
<style>
:root{--bg:#fff;--fg:#1a1a2e;--mut:#6b6b7b;--line:#e4e4ea;--card:#fafafb}
*{box-sizing:border-box}body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg)}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 16px;z-index:5}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
select,input{font:13px inherit;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--fg)}
#count{color:var(--mut);font-size:12px;margin-left:auto}
main{padding:12px 16px;max-width:1100px;margin:0 auto}
.card{border:1px solid var(--line);border-radius:8px;margin:10px 0;background:var(--card);overflow:hidden}
.hd{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:8px 12px;cursor:pointer;font-size:13px}
.hd:hover{background:#f0f0f4}
.badge{padding:1px 7px;border-radius:10px;font-size:11px;font-weight:600;white-space:nowrap}
.b-base{background:#e8eaed;color:#3c4043}.b-control{background:#e3edf8;color:#1a56a8}.b-organism{background:#fdecdd;color:#b3560a}
.s-missed{background:#fde0e0;color:#b30000}.s-caught{background:#dcf3e3;color:#1a7a3c}
.s-false_alarm{background:#fff2cc;color:#8a6d00}.s-clean{background:#eef0f2;color:#5a6b7b}
.ok{color:#1a7a3c;font-weight:600}.no{color:#b30000;font-weight:600}
.mut{color:var(--mut)}.mono{font-family:ui-monospace,Menlo,monospace}
.body{display:none;border-top:1px solid var(--line);padding:0 12px 12px}
.card.open .body{display:block}
.sec{margin-top:10px}.sec h4{margin:0 0 3px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut)}
pre{white-space:pre-wrap;word-break:break-word;background:#fff;border:1px solid var(--line);border-radius:6px;padding:8px;margin:0;font-size:12.5px;max-height:420px;overflow:auto}
.scopes{display:flex;gap:6px;margin-top:4px}.scopes button{font:11px inherit;padding:2px 8px;border:1px solid var(--line);background:#fff;border-radius:6px;cursor:pointer}
.scopes button.on{background:var(--fg);color:#fff;border-color:var(--fg)}
@media(prefers-color-scheme:dark){:root{--bg:#15151c;--fg:#e6e6ee;--mut:#9a9aa8;--line:#2c2c38;--card:#1b1b24}pre{background:#12121a}select,input{background:#1b1b24;color:var(--fg)} .hd:hover{background:#232330}}
</style>
<header><div class="filters">
<select id="f-model"><option value="">all models</option></select>
<select id="f-dataset"><option value="">all datasets</option></select>
<select id="f-arm"><option value="">arm: any</option><option value="1">intervention (hint)</option><option value="0">baseline</option></select>
<select id="f-status">
 <option value="">monitor: any</option>
 <option value="missed">MISSED (unmonitorable)</option>
 <option value="caught">caught (monitorable)</option>
 <option value="false_alarm">false-alarm</option>
 <option value="clean">clean</option></select>
<select id="f-correct"><option value="">answer: any</option><option value="1">correct</option><option value="0">incorrect</option></select>
<input id="f-text" placeholder="search cot / monitor…" size="22">
<span id="count"></span>
</div></header>
<main id="list"></main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id);
const uniq=k=>[...new Set(DATA.map(r=>r[k]))].sort();
uniq('model').forEach(m=>$('f-model').append(new Option(m,m)));
uniq('dataset').forEach(d=>$('f-dataset').append(new Option(d,d)));
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let SHOWN=0, FILT=[];
function match(r){
 const m=$('f-model').value,d=$('f-dataset').value,a=$('f-arm').value,s=$('f-status').value,c=$('f-correct').value,t=$('f-text').value.toLowerCase();
 if(m&&r.model!==m)return 0; if(d&&r.dataset!==d)return 0;
 if(a!==''&&String(r.x)!==a)return 0; if(s&&r.status!==s)return 0;
 if(c!==''&&String(r.correct)!==c)return 0;
 if(t&&!(r.cot.toLowerCase().includes(t)||(r.mon.cot_only.raw||'').toLowerCase().includes(t)))return 0;
 return 1;
}
function card(r,i){
 const corr=r.correct==null?'<span class="mut">n/a</span>':(r.correct?'<span class="ok">✓ '+esc(r.fa)+'</span>':'<span class="no">✗ '+esc(r.fa)+'</span>');
 const scopes=['cot_only','answer_only','all_messages'].map((sc,j)=>
   `<button class="${j==0?'on':''}" onclick="tab(${i},'${sc}',this)">${sc} (z=${r.mon[sc].z}, ${esc(r.mon[sc].a)})</button>`).join('');
 return `<div class="card" id="c${i}">
  <div class="hd" onclick="document.getElementById('c${i}').classList.toggle('open')">
   <span class="badge b-${r.model}">${r.model}</span>
   <span class="mut mono">${esc(r.dataset.replace('intervention/','').replace('.jsonl',''))} #${esc(r.id)}.${r.s}</span>
   <span class="badge s-${r.status}">${r.status}</span>
   <span class="mut">arm ${r.x==1?'<b>hint</b>':'base'} · y=${r.y}</span>
   <span>ans ${corr} <span class="mut">/ gt ${esc(r.ref)}</span></span>
   <span class="mut" style="font-size:11px">${esc(r.cat||'')}</span>
  </div>
  <div class="body">
   <div class="sec"><h4>chain of thought</h4><pre>${esc(r.cot)}</pre></div>
   <div class="sec"><h4>answer / response</h4><pre>${esc(r.answer)}</pre></div>
   <div class="sec"><h4>monitor reasoning</h4><div class="scopes">${scopes}</div>
     <pre id="mon${i}">${esc(r.mon.cot_only.raw)}</pre></div>
  </div></div>`;
}
function tab(i,sc,btn){document.querySelectorAll('#c'+i+' .scopes button').forEach(b=>b.classList.remove('on'));btn.classList.add('on');document.getElementById('mon'+i).textContent=DATA[FILT[i]].mon[sc].raw||'(empty)';}
function render(reset){
 if(reset){FILT=[];DATA.forEach((r,idx)=>{if(match(r))FILT.push(idx)});SHOWN=0;$('list').innerHTML='';
   const byS={};FILT.forEach(idx=>{const s=DATA[idx].status;byS[s]=(byS[s]||0)+1});
   $('count').textContent=`${FILT.length} traces · `+Object.entries(byS).map(([k,v])=>`${k}:${v}`).join('  ');}
 const old=document.getElementById('more'); if(old)old.remove();
 const slice=FILT.slice(SHOWN,SHOWN+50);
 // i passed to card() is the position in FILT, so tab() -> DATA[FILT[i]] resolves
 $('list').insertAdjacentHTML('beforeend',slice.map((idx,j)=>card(DATA[idx],SHOWN+j)).join(''));
 SHOWN+=slice.length;
 if(SHOWN<FILT.length){const b=document.createElement('button');b.id='more';b.textContent='show 50 more ('+(FILT.length-SHOWN)+' left)';b.style="margin:12px auto;display:block;padding:8px 16px";b.onclick=()=>render(false);$('list').append(b)}
}
['f-model','f-dataset','f-arm','f-status','f-correct'].forEach(id=>$(id).onchange=()=>render(true));
$('f-text').oninput=()=>{clearTimeout(window._t);window._t=setTimeout(()=>render(true),250)};
render(true);
</script>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["gpqa"],
                    help="intervention dataset stems (gpqa wmdp_sandbagging scruples_suggest_wrong memory_advice)")
    args = ap.parse_args()
    rows = load(args.datasets)
    data_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HTML.replace("__DATA__", data_json), encoding="utf-8")
    mb = OUT.stat().st_size / 1e6
    print(f"\nwrote {OUT}  ({len(rows)} traces, {mb:.1f} MB)")
    print(f"open: file://{OUT}")


if __name__ == "__main__":
    main()
