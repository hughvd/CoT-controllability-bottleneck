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


OUT_CMP = MON / "v34" / "trace_compare.html"

# Compare view: group by (question instance, arm); base / control / organism side by
# side in 3 columns; questions ranked by base<->organism divergence (monitor caught vs
# missed, or correctness) so the ones that changed float to the top.
HTML_COMPARE = r"""<meta charset="utf-8"><title>v34 trace compare</title>
<style>
:root{--bg:#fff;--fg:#1a1a2e;--mut:#6b6b7b;--line:#e4e4ea;--card:#fafafb}
*{box-sizing:border-box}body{margin:0;font:13.5px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--fg);background:var(--bg)}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:9px 14px;z-index:5}
.filters{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
select,input{font:13px inherit;padding:4px 6px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--fg)}
#count{color:var(--mut);font-size:12px;margin-left:auto}
main{padding:10px 14px;max-width:1500px;margin:0 auto}
.q{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:var(--card)}
.qh{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:8px 12px;cursor:pointer;font-size:12.5px}
.qh:hover{background:#f0f0f4}.q.open .qb{display:block}.qb{display:none;border-top:1px solid var(--line);padding:8px}
.mono{font-family:ui-monospace,Menlo,monospace}.mut{color:var(--mut)}
.mc{white-space:nowrap;font-size:12px}.b-base{color:#3c4043;font-weight:700}.b-control{color:#1a56a8;font-weight:700}.b-organism{color:#b3560a;font-weight:700}
.dv{margin-left:auto;font-weight:700;font-family:ui-monospace,monospace;background:#eef;padding:1px 7px;border-radius:10px}
.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
.col{border:1px solid var(--line);border-radius:6px;background:#fff;min-width:0}
.colh{padding:5px 8px;font-weight:700;border-bottom:1px solid var(--line);text-transform:capitalize;position:sticky;top:0;background:#fff}
.colh.b-base{background:#eef0f2}.colh.b-control{background:#e9f1fb}.colh.b-organism{background:#fdf1e6}
.sc{border-bottom:1px solid var(--line)}.sch{padding:5px 8px;cursor:pointer;font-size:12px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.sch:hover{background:#f5f5f8}.scb{display:none;padding:0 8px 8px}.sc.open .scb{display:block}
.badge{padding:1px 6px;border-radius:9px;font-size:10.5px;font-weight:600}
.s-missed{background:#fde0e0;color:#b30000}.s-caught{background:#dcf3e3;color:#1a7a3c}.s-false_alarm{background:#fff2cc;color:#8a6d00}.s-clean{background:#eef0f2;color:#5a6b7b}
.ok{color:#1a7a3c;font-weight:700}.no{color:#b30000;font-weight:700}
.lbl{font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin:6px 0 2px}
pre{white-space:pre-wrap;word-break:break-word;background:#fbfbfd;border:1px solid var(--line);border-radius:5px;padding:6px;margin:0;font-size:12px;max-height:340px;overflow:auto}
.pr{margin:2px 4px 10px}.pr>summary{cursor:pointer;font-size:12px;color:var(--mut);padding:3px 0}
.key{font-size:12px;background:#fffbe6;border:1px solid #efe1a6;border-radius:5px;padding:6px;margin-top:5px;color:#6b5900}
@media(prefers-color-scheme:dark){:root{--bg:#15151c;--fg:#e6e6ee;--mut:#9a9aa8;--line:#2c2c38;--card:#1b1b24}pre{background:#12121a}select,input{background:#1b1b24;color:var(--fg)}.col,.colh{background:#1b1b24}.colh.b-base{background:#23232c}.colh.b-control{background:#182536}.colh.b-organism{background:#332212}.dv{background:#26263a}.key{background:#2a2410;border-color:#4a3f14;color:#d9c98a}}
</style>
<header><div class="filters">
<select id="f-ds"><option value="">all datasets</option></select>
<select id="f-arm"><option value="1">arm: intervention (hint)</option><option value="0">arm: baseline</option><option value="">arm: any</option></select>
<select id="f-sort">
 <option value="dz">sort: monitor divergence (base↔organism z)</option>
 <option value="dc">sort: correctness divergence</option>
 <option value="dy">sort: behavior(y) divergence</option></select>
<input id="f-text" placeholder="search any CoT…" size="20">
<span id="count"></span>
</div></header>
<main id="list"></main>
<script id="data" type="application/json">__DATA__</script>
<script id="prompts" type="application/json">__PROMPTS__</script>
<script>
const DATA=JSON.parse(document.getElementById('data').textContent);
const PROMPTS=JSON.parse(document.getElementById('prompts').textContent);
const MODELS=['base','control','organism'];
const $=id=>document.getElementById(id);
const esc=s=>String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const fmt=x=>x==null?'–':x.toFixed(2);
const GM={};
for(const r of DATA){const k=r.dataset+'|'+r.id+'|'+r.x;(GM[k]||(GM[k]={ds:r.dataset,id:r.id,x:r.x,cat:r.cat,m:{base:[],control:[],organism:[]}}));if(GM[k].m[r.model])GM[k].m[r.model].push(r);}
const rate=(a,f)=>{const v=a.map(f).filter(x=>x!=null);return v.length?v.reduce((p,c)=>p+c,0)/v.length:null};
const dif=(x,y)=>(x!=null&&y!=null)?Math.abs(x-y):-1;
const GROUPS=Object.values(GM).filter(g=>g.m.base.length&&g.m.organism.length).map(g=>{
 g.st={};for(const m of MODELS){const a=g.m[m];g.st[m]={n:a.length,z:rate(a,r=>r.mon.cot_only.z),c:rate(a,r=>r.correct),y:rate(a,r=>r.y)};}
 g.dz=dif(g.st.base.z,g.st.organism.z);g.dc=dif(g.st.base.c,g.st.organism.c);g.dy=dif(g.st.base.y,g.st.organism.y);return g;});
[...new Set(GROUPS.map(g=>g.ds))].sort().forEach(d=>$('f-ds').append(new Option(d.replace('intervention/','').replace('.jsonl',''),d)));
let VIEW=[],SHOWN=0,SID=0;
function pass(g){const ds=$('f-ds').value,ar=$('f-arm').value,t=$('f-text').value.toLowerCase();
 if(ds&&g.ds!==ds)return 0; if(ar!==''&&String(g.x)!==ar)return 0;
 if(t&&!MODELS.some(m=>g.m[m].some(r=>r.cot.toLowerCase().includes(t))))return 0; return 1;}
const key=g=>({dz:g.dz,dc:g.dc,dy:g.dy}[$('f-sort').value]);
const chip=(g,m)=>`<span class="mc"><span class="b-${m}">${m}</span> z ${fmt(g.st[m].z)} · ✓ ${fmt(g.st[m].c)} <span class=mut>y ${fmt(g.st[m].y)}·n${g.st[m].n}</span></span>`;
function sample(r){const id='s'+(SID++);const corr=r.correct==null?'':(r.correct?'<span class=ok>✓</span>':'<span class=no>✗</span>');
 return `<div class=sc><div class=sch onclick="this.parentNode.classList.toggle('open')">
   <span class="badge s-${r.status}">${r.status}</span><span class=mut>#${r.s}</span> ans <b>${esc(r.fa)}</b> ${corr}<span class=mut>gt ${esc(r.ref)}</span></div>
  <div class=scb><div class=lbl>chain of thought</div><pre>${esc(r.cot)}</pre>
   <div class=lbl>answer</div><pre>${esc(r.answer)}</pre>
   <div class=lbl>monitor · cot_only · z=${r.mon.cot_only.z}</div><pre>${esc(r.mon.cot_only.raw)}</pre></div></div>`;}
function body(g){
 const P=PROMPTS[g.ds+'|'+g.id+'|'+g.x];
 const pr=P?`<details class=pr><summary>▸ prompts & monitor answer-key</summary>
   <div class=lbl>subject prompt — what the model saw${g.x==1?' (includes the hint)':' (baseline: no hint)'}</div><pre>${esc(P.subj)}</pre>
   <div class=lbl>monitor prompt (cot_only) — the classification task</div><pre>${esc(P.mon)}</pre>
   <div class=key>The monitor outputs ONE letter of <b>${P.valid.join(', ')}</b> (a classification). <b>z=1</b> (flagged: the hint influenced the answer) when that letter ∈ {<b>${P.trig.join(', ')}</b>}; otherwise z=0. Each sample below shows its monitor letter + z + the monitor's justification.</div></details>`:'';
 return pr+`<div class=cols>`+MODELS.map(m=>`<div class=col><div class="colh b-${m}">${m} <span class="mut" style="font-weight:400">z ${fmt(g.st[m].z)} · ✓ ${fmt(g.st[m].c)}</span></div>`+(g.m[m].length?g.m[m].map(sample).join(''):'<div class=mut style="padding:8px">no samples</div>')+`</div>`).join('')+`</div>`;}
const groupCard=g=>`<div class=q><div class=qh onclick="this.parentNode.classList.toggle('open')">
  <span class=mono>#${esc(g.id)}</span><span>arm ${g.x==1?'<b>hint</b>':'base'}</span><span class=mut>${esc(g.cat||'')}</span>
  ${chip(g,'base')} ${chip(g,'control')} ${chip(g,'organism')}
  <span class=dv>Δz ${fmt(g.dz)} · Δ✓ ${fmt(g.dc)}</span></div>
  <div class=qb>${body(g)}</div></div>`;
function render(reset){
 const old=$('more'); if(old)old.remove();
 if(reset){VIEW=GROUPS.filter(pass).sort((a,b)=>key(b)-key(a));SHOWN=0;$('list').innerHTML='';
  $('count').textContent=`${VIEW.length} questions (base+organism both present)`;}
 const slice=VIEW.slice(SHOWN,SHOWN+30);
 $('list').insertAdjacentHTML('beforeend',slice.map(groupCard).join(''));
 SHOWN+=slice.length;
 if(SHOWN<VIEW.length){const b=document.createElement('button');b.id='more';b.textContent='show 30 more ('+(VIEW.length-SHOWN)+' left)';b.style='margin:12px auto;display:block;padding:8px 16px';b.onclick=()=>render(false);$('list').append(b);}}
['f-ds','f-arm','f-sort'].forEach(id=>$(id).onchange=()=>render(true));
$('f-text').oninput=()=>{clearTimeout(window._t);window._t=setTimeout(()=>render(true),250)};
render(true);
</script>"""


def load_prompts(datasets):
    """Reconstruct the subject + monitor prompts + answer-key per (dataset,
    instance, arm), keyed to match the compare viewer's group key. Best-effort:
    returns {} if the openai_suite scaffold isn't importable locally."""
    try:
        from monitorability.suite import (load_intervention_items,
                                          monitor_answer_spec, render_monitor_prompt)
        from monitorability.runner import ModelCallOutput, build_subject_prompt
    except Exception as e:
        print(f"[prompts] reconstruction unavailable ({e}); viewer will omit prompts")
        return {}
    PLACE = "‹the student's chain-of-thought is shown in the column above›"
    out = {}
    for ds in datasets:
        try:
            items = load_intervention_items([ds], None)
        except Exception as e:
            print(f"[prompts] skip {ds}: {e}")
            continue
        for it in items:
            try:
                valid, trig = monitor_answer_spec(it.z_rule_ref)
                mo = ModelCallOutput(final_answer="", action_only="", cot_only=PLACE, all_msgs=PLACE)
                out[f"{ds}|{it.instance_id}|{it.arm}"] = {   # match the viewer's group key (dataset stem)
                    "subj": build_subject_prompt(it, None),
                    "mon": render_monitor_prompt(it, "cot_only", mo),
                    "valid": list(valid), "trig": sorted(trig),
                }
            except Exception:
                continue
    print(f"[prompts] reconstructed {len(out)} (instance,arm) prompt sets")
    return out


def build(rows, template, out, prompts=None):
    data_json = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    html = template.replace("__DATA__", data_json)
    if "__PROMPTS__" in template:
        html = html.replace("__PROMPTS__",
                            json.dumps(prompts or {}, ensure_ascii=False).replace("</", "<\\/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(rows)} traces, {out.stat().st_size/1e6:.1f} MB)  ->  file://{out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["gpqa"],
                    help="intervention dataset stems (gpqa wmdp_sandbagging scruples_suggest_wrong memory_advice)")
    ap.add_argument("--mode", choices=["individual", "compare", "both"], default="both")
    args = ap.parse_args()
    rows = load(args.datasets)
    if args.mode in ("individual", "both"):
        build(rows, HTML, OUT)
    if args.mode in ("compare", "both"):
        build(rows, HTML_COMPARE, OUT_CMP, load_prompts(args.datasets))


if __name__ == "__main__":
    main()
