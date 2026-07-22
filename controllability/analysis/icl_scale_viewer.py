"""Generate an interactive HTML dashboard for the ICL example-LENGTH scaling sweep.

An on-page version of analysis/icl/icl_scale.png where you can TOGGLE which control-types,
which versions (V1/V2/V4), and which lengths are shown — the lines recompute as a macro-average
over the selected types. Also embeds example responses (question + graded <think> + score) per
(model, type, version, length) so you can read what the model actually produced (e.g. omit_word
V2 @ length 10). Self-contained (no server / no CDN); reads results/iclscale_*.csv.

  python analysis/icl_scale_viewer.py   &&   open analysis/icl/icl_scale_viewer.html
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
OUT = HERE / "icl"
sys.path.insert(0, str(HERE.parent))          # controllability/  (icl_experiment, prefill_experiment)
sys.path.insert(0, str(HERE.parent.parent))   # repo root         (lib.*)
import prefill_experiment as pe                # noqa: E402
import icl_experiment as icl                   # noqa: E402

MODELS = [
    ("qwen3-32b", "iclscale_qwen3_32b_icl_per_sample.csv", "base_qwen3_32b_prefill_per_sample.csv"),
    ("glm-5.2",   "iclscale_glm_5_2_icl_per_sample.csv",   "base_glm_5_2_prefill_per_sample.csv"),
]
TYPES = ["leetspeak", "spanish", "chinese", "uppercase_thinking", "prefix_sentences",
         "no_first_person", "omit_word", "pirate", "yoda", "jar_jar_binks"]
LEN_ORDER = ["1", "3", "5", "10", "50pct", "100pct"]
LEN_LABEL = {"1": "1", "3": "3", "5": "5", "10": "10", "50pct": "50%", "100pct": "full"}
# validated blue / orange / green trio (passes CVD light + dark; see dataviz validator)
VERSIONS = [
    {"key": "icl_user_plain", "label": "V1 user", "light": "#2a78d6", "dark": "#3987e5"},
    {"key": "icl_user_think", "label": "V2 user+tag", "light": "#eb6834", "dark": "#d95926"},
    {"key": "icl_assistant_visible", "label": "V4 asst-vis", "light": "#008300", "dark": "#008300"},
]
N_EXAMPLES = 3      # per (model,type,version,length): lowest / median / highest compliance
Q_CAP, R_CAP, I_CAP, A_CAP, MSG_CAP = 400, 1800, 300, 500, 2800


def _s(x, cap=None):
    """nan/None-safe string (pandas stores missing text as float nan -> 'nan')."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x)
    if s == "nan":
        return ""
    return s[:cap] if cap else s


def _build_cell_index():
    """Rebuild the exact cells the run used (deterministic: default seed, FULL_TYPES, 30 q) so
    we can reconstruct the full messages each model saw. Returns (idx[ctype][qhash]->Cell, spec map)."""
    cfg = pe.PrefillConfig()
    for k, v in icl.FULL_TYPES.items():
        setattr(cfg, k, v)
    cfg.n_questions = 30
    idx: dict = {}
    for c in pe.build_all_cells(cfg):
        idx.setdefault(c.ctype, {})[pe._qhash(c.question)] = c
    spec = {t: (t, knd, val) for (t, knd, val) in icl.SCALE_LENGTHS}
    return idx, spec


def _clean(d: pd.DataFrame) -> pd.DataFrame:
    if "empty_thinking" in d:
        d = d[~d["empty_thinking"].astype(bool)]
    return d


def _reconstruct_msgs(row, ctype, vkey, len_tag, idx, spec) -> list[dict]:
    """Rebuild the full messages the model saw (ICL example + instruction + real question).
    Guards correctness: an ICL example is only shown when the rebuilt donor's instruction matches
    the one stored at run time (else the offline cell-build drifted -> return [] -> UI fallback)."""
    tgt = idx.get(ctype, {}).get(pe._qhash(str(row["target_question"])))
    if tgt is None:
        return []
    instr = _s(row.get("instruction"))
    if vkey == "instruction_only":  # no example; real Q only — always reconstructable from the target
        return [{"role": "user", "content": pe.assemble_prompt(tgt.question, tgt.options, instr)[:MSG_CAP]}]
    don = idx.get(ctype, {}).get(_s(row.get("donor_question_hash")))
    if don is None or don.instruction != instr:   # unfindable / drifted donor -> UI shows fallback
        return []
    sp = spec.get(len_tag, ("100pct", "frac", 1.0))
    try:
        mm = icl.build_icl_messages(tgt, don, vkey, sp)
    except Exception:
        return []
    return [{"role": m["role"], "content": m["content"][:MSG_CAP]} for m in mm]


def _examples(sub: pd.DataFrame, ctype: str, vkey: str, len_tag: str, idx, spec) -> list[dict]:
    if not len(sub):
        return []
    s = sub.sort_values("compliance_score")
    idxs = sorted({0, len(s) // 2, len(s) - 1})
    out = []
    for i in idxs[:N_EXAMPLES]:
        r = s.iloc[i]
        out.append({"q": _s(r.get("target_question"), Q_CAP),
                    "r": _s(r.get("reasoning"), R_CAP),
                    "instr": _s(r.get("instruction"), I_CAP),
                    "a": _s(r.get("response"), A_CAP),
                    "pred": _s(r.get("predicted")) or None,
                    "cl": _s(r.get("correct_letter")),
                    "fr": _s(r.get("finish_reason")),
                    "msgs": _reconstruct_msgs(r, ctype, vkey, len_tag, idx, spec),
                    "s": round(float(r["compliance_score"]), 2),
                    "c": int(r.get("correct", 0))})
    return out


def build() -> dict:
    data = {"types": TYPES, "lenOrder": LEN_ORDER, "lenLabel": LEN_LABEL,
            "versions": VERSIONS, "models": [], "means": {}, "baseline": {},
            "prefill": {}, "examples": {}}
    idx, spec = _build_cell_index()
    for short, icl_csv, pf_csv in MODELS:
        p = RESULTS / icl_csv
        if not p.exists():
            continue
        d = _clean(pd.read_csv(p))
        d["length"] = d["length"].astype(str)
        data["models"].append(short)
        data["means"][short] = {}
        data["baseline"][short] = {}
        data["prefill"][short] = {}
        data["examples"][short] = {}
        # prefill K5+i per type
        pf = RESULTS / pf_csv
        if pf.exists():
            pdf = _clean(pd.read_csv(pf))
            pdf = pdf[(pdf.k == 5) & (pdf.condition == "prefill_plus_instruction")]
        else:
            pdf = None
        for ct in TYPES:
            dc = d[d.ctype == ct]
            # baseline
            b = dc[dc.variant == "instruction_only"]["compliance_score"]
            data["baseline"][short][ct] = round(float(b.mean()), 4) if len(b) else None
            # prefill
            if pdf is not None:
                pc = pdf[pdf.ctype == ct]["compliance_score"]
                data["prefill"][short][ct] = round(float(pc.mean()), 4) if len(pc) else None
            else:
                data["prefill"][short][ct] = None
            data["means"][short][ct] = {}
            data["examples"][short][ct] = {}
            for v in VERSIONS:
                data["means"][short][ct][v["key"]] = {}
                data["examples"][short][ct][v["key"]] = {}
                vc = dc[dc.variant == v["key"]]
                for L in LEN_ORDER:
                    vl = vc[vc.length == L]
                    m = vl["compliance_score"]
                    data["means"][short][ct][v["key"]][L] = (
                        {"m": round(float(m.mean()), 4), "n": int(len(m))} if len(m) else None)
                    data["examples"][short][ct][v["key"]][L] = _examples(vl, ct, v["key"], L, idx, spec)
            # baseline examples (length-independent)
            data["examples"][short][ct]["instruction_only"] = {
                "na": _examples(dc[dc.variant == "instruction_only"], ct, "instruction_only", "na", idx, spec)}
    return data


# --------------------------------------------------------------------------- #
CSS = """
:root{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,.10);
  --v1:#2a78d6; --v2:#eb6834; --v4:#008300;
}
:root[data-theme=dark], html[data-theme=dark]{
  --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
  --v1:#3987e5; --v2:#d95926; --v4:#008300;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme=light]){
    --surface:#1a1a19; --plane:#0d0d0d; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,.10);
    --v1:#3987e5; --v2:#d95926; --v4:#008300;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:18px 22px 8px}
h1{font-size:18px;margin:0 0 2px}
.sub{color:var(--ink2);font-size:13px;margin:0}
.wrap{padding:0 22px 40px;max-width:1240px}
.controls{display:flex;flex-wrap:wrap;gap:16px;background:var(--surface);border:1px solid var(--border);
  border-radius:12px;padding:14px 16px;margin:14px 0}
.ctl{display:flex;flex-direction:column;gap:6px}
.ctl h4{margin:0;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:700}
.chips{display:flex;flex-wrap:wrap;gap:6px;max-width:520px}
.chip{border:1px solid var(--border);border-radius:999px;padding:3px 10px;font-size:12px;cursor:pointer;
  background:transparent;color:var(--ink2);user-select:none;white-space:nowrap}
.chip.on{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.chip.von{color:#fff;border-color:transparent}
.chip[data-v=icl_user_plain].von{background:var(--v1)}
.chip[data-v=icl_user_think].von{background:var(--v2)}
.chip[data-v=icl_assistant_visible].von{background:var(--v4)}
.mini{font-size:11px;color:var(--muted);cursor:pointer;text-decoration:underline;background:none;border:0;padding:0}
.panels{display:flex;gap:18px;flex-wrap:wrap}
.panel{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:12px 12px 6px;
  flex:1 1 440px;position:relative}
.panel h3{margin:2px 6px 0;font-size:14px}
.legend{display:flex;gap:14px;margin:2px 8px 6px;flex-wrap:wrap;font-size:12px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:5px}
.legend i{width:12px;height:3px;border-radius:2px;display:inline-block}
.legend i.v1{background:var(--v1)}.legend i.v2{background:var(--v2)}.legend i.v4{background:var(--v4)}
.legend i.ref{background:var(--muted);height:2px}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:7px 9px;font-size:12px;box-shadow:0 4px 14px rgba(0,0,0,.14);z-index:5;
  min-width:130px;display:none}
.tip b{display:block;margin-bottom:3px}
.tip .row{display:flex;justify-content:space-between;gap:12px}
.tip .row i{width:9px;height:9px;border-radius:2px;display:inline-block;margin-right:5px}
tspan{font-variant-numeric:tabular-nums}
.resp{margin-top:20px}
.resp .bar{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;margin-bottom:10px}
.resp select{padding:5px 8px;border-radius:8px;border:1px solid var(--border);background:var(--surface);
  color:var(--ink);font:13px system-ui}
.resp label{font-size:11px;color:var(--muted);display:flex;flex-direction:column;gap:3px;text-transform:uppercase;letter-spacing:.05em}
.card{border:1px solid var(--border);border-radius:10px;margin:8px 0;overflow:hidden;background:var(--surface)}
.card .hd{display:flex;justify-content:space-between;padding:6px 12px;font-size:12px;color:var(--ink2);
  border-bottom:1px solid var(--border)}
.badge{font-weight:700;padding:1px 8px;border-radius:999px;font-size:11px}
.card pre{margin:0;padding:10px 12px;white-space:pre-wrap;word-break:break-word;font-size:12.5px;
  max-height:280px;overflow:auto}
.card .q{padding:8px 12px;font-size:12.5px;color:var(--ink2);border-bottom:1px solid var(--border)}
.card .meta{padding:8px 12px;font-size:12.5px;border-bottom:1px solid var(--border);background:rgba(127,127,127,.06)}
.card .meta b{color:var(--ink)}
.chan{padding:8px 12px 0;font-size:10px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);font-weight:700}
.card pre.ans{background:rgba(127,127,127,.05);border-top:1px solid var(--border);max-height:none}
.card .meta.warn{background:rgba(208,59,59,.10);color:#d03b3b;font-weight:600}
.conv{padding:0 12px}
.conv .m{border:1px solid var(--border);border-radius:8px;margin:8px 0;overflow:hidden}
.conv .m .role{padding:4px 10px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.05em;
  color:var(--muted);border-bottom:1px solid var(--border)}
.conv .m.user .role{background:rgba(42,120,214,.12)}
.conv .m.assistant .role{background:rgba(0,131,0,.12)}
.conv .m pre{max-height:340px}
.resp label.chk{flex-direction:row;align-items:center;gap:6px;text-transform:none;letter-spacing:normal;font-size:12.5px;color:var(--ink2)}
.resp label.chk input{margin:0}
.themed{position:absolute;top:14px;right:22px}
.themed button{background:var(--surface);border:1px solid var(--border);color:var(--ink2);border-radius:8px;
  padding:4px 10px;font-size:12px;cursor:pointer}
"""

JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const LEN = D.lenOrder, LL = D.lenLabel, VER = D.versions;
const cssv = {icl_user_plain:'--v1', icl_user_think:'--v2', icl_assistant_visible:'--v4'};
const cvar = k => getComputedStyle(document.documentElement).getPropertyValue(cssv[k]).trim();
const state = {
  types:new Set(D.types), vers:new Set(VER.map(v=>v.key)), lens:new Set(LEN),
  baseline:true, prefill:true
};

function macro(model, key, L){ // mean over selected types of per-type mean
  let vals=[];
  for(const ct of D.types){ if(!state.types.has(ct)) continue;
    const cell = (key==='__base') ? D.baseline[model][ct]
               : (key==='__pref') ? D.prefill[model][ct]
               : (D.means[model][ct][key][L]||{}).m;
    if(cell!=null && !isNaN(cell)) vals.push(cell);
  }
  return vals.length ? vals.reduce((a,b)=>a+b,0)/vals.length : null;
}

function drawPanel(model){
  const lens = LEN.filter(l=>state.lens.has(l));
  const W=520,H=360,m={l:44,r:40,t:14,b:34}, pw=W-m.l-m.r, ph=H-m.t-m.b;
  const xs = i => lens.length<2 ? m.l+pw/2 : m.l + i*pw/(lens.length-1);
  const ys = v => m.t + (1-v)*ph;
  const NS='http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS,'svg');
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.setAttribute('width','100%');
  const el=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);return e;};
  const ink=getComputedStyle(document.documentElement).getPropertyValue('--ink').trim();
  const muted=getComputedStyle(document.documentElement).getPropertyValue('--muted').trim();
  const grid=getComputedStyle(document.documentElement).getPropertyValue('--grid').trim();
  const axis=getComputedStyle(document.documentElement).getPropertyValue('--axis').trim();
  // y grid + labels
  for(let y=0;y<=1.0001;y+=0.2){
    svg.appendChild(el('line',{x1:m.l,x2:m.l+pw,y1:ys(y),y2:ys(y),stroke:grid,'stroke-width':1}));
    const t=el('text',{x:m.l-7,y:ys(y)+3,'text-anchor':'end','font-size':10,fill:muted});
    t.textContent=y.toFixed(1); svg.appendChild(t);
  }
  // x ticks
  lens.forEach((L,i)=>{const t=el('text',{x:xs(i),y:H-m.b+16,'text-anchor':'middle','font-size':11,fill:muted});
    t.textContent=LL[L]; svg.appendChild(t);});
  svg.appendChild(el('line',{x1:m.l,x2:m.l+pw,y1:ys(0),y2:ys(0),stroke:axis,'stroke-width':1.5}));
  // reference lines
  const ref=(key,dash,lab)=>{const v=macro(model,key,null); if(v==null)return;
    svg.appendChild(el('line',{x1:m.l,x2:m.l+pw,y1:ys(v),y2:ys(v),stroke:muted,'stroke-width':1.3,'stroke-dasharray':dash}));
    const t=el('text',{x:m.l+pw,y:ys(v)-4,'text-anchor':'end','font-size':10,fill:muted});
    t.textContent=`${lab} ${v.toFixed(2)}`; svg.appendChild(t);};
  if(state.baseline) ref('__base','4 3','instr');
  if(state.prefill)  ref('__pref','1 3','prefill K5');
  // series
  let endLabels=[];
  VER.forEach(v=>{ if(!state.vers.has(v.key)) return;
    const col=cvar(v.key); let pts=[], d='';
    lens.forEach((L,i)=>{const val=macro(model,v.key,L); if(val==null){return;}
      pts.push([xs(i),ys(val),val,L]); d+=(d?'L':'M')+xs(i)+' '+ys(val)+' ';});
    if(pts.length>1) svg.appendChild(el('path',{d,fill:'none',stroke:col,'stroke-width':2}));
    pts.forEach(p=>svg.appendChild(el('circle',{cx:p[0],cy:p[1],r:4,fill:col})));
    if(pts.length){const last=pts[pts.length-1]; endLabels.push({y:last[1],val:last[2],col});}
  });
  endLabels.sort((a,b)=>a.y-b.y);           // de-collide vertically in the right margin
  for(let i=1;i<endLabels.length;i++)
    if(endLabels[i].y-endLabels[i-1].y<12) endLabels[i].y=endLabels[i-1].y+12;
  endLabels.forEach(L=>{const t=el('text',{x:m.l+pw+5,y:L.y+3,'font-size':11,fill:L.col,'font-weight':600});
    t.textContent=L.val.toFixed(2); svg.appendChild(t);});
  // hover
  const over=el('rect',{x:m.l,y:m.t,width:pw,height:ph,fill:'transparent'});
  svg.appendChild(over);
  const cross=el('line',{y1:m.t,y2:m.t+ph,stroke:axis,'stroke-width':1,'stroke-dasharray':'3 3',opacity:0});
  svg.appendChild(cross);
  const tip=document.getElementById('tip-'+model);
  over.addEventListener('mousemove',ev=>{
    const r=svg.getBoundingClientRect(), sx=(ev.clientX-r.left)*W/r.width;
    let bi=0,bd=1e9; lens.forEach((L,i)=>{const dd=Math.abs(xs(i)-sx);if(dd<bd){bd=dd;bi=i;}});
    cross.setAttribute('x1',xs(bi));cross.setAttribute('x2',xs(bi));cross.setAttribute('opacity',1);
    let html=`<b>length = ${LL[lens[bi]]}</b>`;
    VER.forEach(v=>{if(!state.vers.has(v.key))return;const val=macro(model,v.key,lens[bi]);
      if(val!=null)html+=`<div class=row><span><i style="background:${cvar(v.key)}"></i>${v.label}</span><span>${val.toFixed(2)}</span></div>`;});
    if(state.baseline){const b=macro(model,'__base',null); if(b!=null)html+=`<div class=row><span>instr-only</span><span>${b.toFixed(2)}</span></div>`;}
    tip.innerHTML=html; tip.style.display='block';
    const px=xs(bi)*r.width/W; tip.style.left=Math.min(px+14,r.width-150)+'px'; tip.style.top='16px';
  });
  over.addEventListener('mouseleave',()=>{tip.style.display='none';cross.setAttribute('opacity',0);});
  return svg;
}

function renderCharts(){
  D.models.forEach(model=>{
    const host=document.getElementById('chart-'+model); host.innerHTML='';
    host.appendChild(drawPanel(model));
  });
  // stat line
  const nt=state.types.size;
  document.getElementById('selinfo').textContent =
    `${nt} of ${D.types.length} types averaged · ${state.vers.size} versions · ${state.lens.size} lengths`;
}

/* ---- controls ---- */
function chip(label, on, cls, ds){
  const b=document.createElement('button'); b.className='chip'+(on?(cls||' on'):''); b.textContent=label;
  if(ds) for(const k in ds) b.dataset[k]=ds[k];
  return b;
}
function buildControls(){
  const tc=document.getElementById('typeChips');
  D.types.forEach(ct=>{const b=chip(ct.replace('_thinking',''),true,' on');
    b.onclick=()=>{state.types.has(ct)?state.types.delete(ct):state.types.add(ct);
      b.classList.toggle('on',state.types.has(ct)); renderCharts();};
    b.dataset.ct=ct; tc.appendChild(b);});
  const vc=document.getElementById('verChips');
  VER.forEach(v=>{const b=chip(v.label,true,' von',{v:v.key});
    b.onclick=()=>{state.vers.has(v.key)?state.vers.delete(v.key):state.vers.add(v.key);
      b.classList.toggle('von',state.vers.has(v.key)); renderCharts();};
    vc.appendChild(b);});
  const lc=document.getElementById('lenChips');
  LEN.forEach(L=>{const b=chip(LL[L],true,' on');
    b.onclick=()=>{state.lens.has(L)?state.lens.delete(L):state.lens.add(L);
      b.classList.toggle('on',state.lens.has(L)); renderCharts();};
    lc.appendChild(b);});
  document.getElementById('refBase').onclick=e=>{state.baseline=!state.baseline;
    e.target.classList.toggle('on',state.baseline);renderCharts();};
  document.getElementById('refPref').onclick=e=>{state.prefill=!state.prefill;
    e.target.classList.toggle('on',state.prefill);renderCharts();};
  document.getElementById('onlyOmit').onclick=()=>{
    state.types=new Set(['omit_word']);
    document.querySelectorAll('#typeChips .chip').forEach(b=>b.classList.toggle('on',b.dataset.ct==='omit_word'));
    renderCharts();};
  document.getElementById('allTypes').onclick=()=>{
    state.types=new Set(D.types);
    document.querySelectorAll('#typeChips .chip').forEach(b=>b.classList.add('on')); renderCharts();};
}

/* ---- responses ---- */
function fill(sel,opts){opts.forEach(o=>{const e=document.createElement('option');
  e.value=o[0];e.textContent=o[1];sel.appendChild(e);});}
function buildResp(){
  fill(rm, D.models.map(m=>[m,m]));
  fill(rt, D.types.map(t=>[t,t]));
  const vs=VER.map(v=>[v.key,v.label]); vs.push(['instruction_only','baseline (no example)']);
  fill(rv, vs);
  fill(rl, LEN.map(L=>[L,LL[L]]));
  rt.value='omit_word'; rv.value='icl_user_think'; rl.value='10';
  [rm,rt,rv,rl].forEach(s=>s.onchange=renderResp);
  rfull.onchange=renderResp;
  rv.addEventListener('change',()=>{rl.disabled = rv.value==='instruction_only';});
  renderResp();
}
function renderResp(){
  const model=rm.value,ct=rt.value,v=rv.value, full=rfull.checked;
  const L = v==='instruction_only' ? 'na' : rl.value;
  rl.disabled = v==='instruction_only';
  const ex = (((D.examples[model]||{})[ct]||{})[v]||{})[L] || [];
  const host=document.getElementById('cards'); host.innerHTML='';
  const badge=s=>s>=0.75?'#0ca30c':(s>=0.4?'#eda100':'#d03b3b');
  if(!ex.length){host.innerHTML='<p style="color:var(--muted)">no saved examples for this cell</p>';return;}
  // cell stat: the number the chart plots (mean over ALL samples in the cell), so the 3
  // example cards below are visibly a subset, not the population the mean is over.
  const st = v==='instruction_only' ? (D.baseline[model]||{})[ct]
                                    : ((((D.means[model]||{})[ct]||{})[v]||{})[L]||null);
  const mm = (st&&st.m!==undefined)?st:(typeof st==='number'?{m:st}:null);
  if(mm) host.innerHTML += `<p style="color:var(--muted);font-size:12.5px;margin:0 0 10px">`+
    `cell mean <b style="color:var(--text)">${mm.m.toFixed(2)}</b>`+
    (mm.n?` over n=${mm.n} samples (30 questions × 5 samples each)`:'')+
    ` — this is the value the chart plots. The 3 cards below are illustrative picks `+
    `(lowest / median / highest compliance), not the whole cell.</p>`;
  const lab = v==='instruction_only'?'baseline':`${(VER.find(x=>x.key===v)||{}).label} @ len ${LL[L]||L}`;
  const roleLab={user:'user',assistant:'assistant (prior turn)'};
  ex.forEach((e,i)=>{const c=document.createElement('div');c.className='card';
    const pred = e.pred? `pred ${esc(e.pred)}/${esc(e.cl)}` : `no answer/${esc(e.cl)}`;
    let html=`<div class=hd><span>${model} · ${ct} · ${lab} · ${['lowest','median','highest'][i]||''} compliance</span>`+
      `<span class=badge style="background:${badge(e.s)};color:#fff">compliance ${e.s.toFixed(2)} · ${pred} · ${e.c?'✓':'✗'}</span></div>`;
    html += `<div class=meta><b>Instruction:</b> ${esc(e.instr)||'<em>(none)</em>'}</div>`;
    if(e.fr==='length')
      html += `<div class="meta warn">⚠ generation hit the token limit — the &lt;think&gt; may loop/repeat and the final answer can be missing</div>`;
    if(full){
      html += `<div class=chan>conversation — everything the model saw</div>`;
      if((e.msgs||[]).length){ html += `<div class=conv>`;
        e.msgs.forEach(m=>{html+=`<div class="m ${m.role}"><div class=role>${esc(roleLab[m.role]||m.role)}</div><pre>${esc(m.content)}</pre></div>`;});
        html += `</div>`;
      } else {  // donor drifted offline — still show the real question + note
        html += `<div class=q><b>Q:</b> ${esc(e.q)}</div>`+
          `<p style="color:var(--muted);padding:2px 12px 6px;font-size:12px">(in-context example couldn't be reconstructed offline; instruction + question + output shown)</p>`;
      }
      html += `<div class=chan>model &lt;think&gt; (graded output)</div><pre>${esc(e.r)||'<em>(empty)</em>'}</pre>`;
      html += `<div class=chan>model answer</div><pre class=ans>${esc(e.a)||'<em>(no answer emitted)</em>'}</pre>`;
    } else {
      html += `<div class=q><b>Q:</b> ${esc(e.q)}</div>`;
      const r = esc(e.r).slice(0,650)+(e.r.length>650?' …':'');
      html += `<div class=chan>model &lt;think&gt; (graded)</div><pre>${r||'<em>(empty)</em>'}</pre>`;
    }
    c.innerHTML=html; host.appendChild(c);});
}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

/* ---- theme ---- */
document.getElementById('themeBtn').onclick=()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  const next = cur==='dark'?'light':(cur==='light'?'dark':(matchMedia('(prefers-color-scheme:dark)').matches?'light':'dark'));
  document.documentElement.setAttribute('data-theme',next); renderCharts();
};
buildControls(); buildResp(); renderCharts();
"""


def make_html(data: dict) -> str:
    panels = "".join(
        f'<div class="panel"><h3>{m}</h3><div class="tip" id="tip-{m}"></div>'
        f'<div id="chart-{m}"></div></div>' for m in data["models"])
    swcls = {"icl_user_plain": "v1", "icl_user_think": "v2", "icl_assistant_visible": "v4"}
    legend = ("".join(
        f'<span><i class="{swcls[v["key"]]}"></i>{v["label"]}</span>' for v in VERSIONS)
        + '<span><i class="ref"></i>instr-only (dashed)</span>'
        + '<span><i class="ref"></i>prefill K5 (dotted)</span>')
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>ICL length-scaling viewer</title><style>{CSS}</style></head><body>
<div class=themed><button id=themeBtn>◐ theme</button></div>
<header><h1>ICL example-length scaling</h1>
<p class=sub>Avg compliance of the model's own &lt;think&gt; vs how many sentences of the controlled
example are shown. Toggle control-types, versions, and lengths — lines are the macro-average over the
selected types. Grades native &lt;think&gt;.</p></header>
<div class=wrap>
  <div class=controls>
    <div class=ctl><h4>Control types <button class=mini id=onlyOmit>omit_word only</button>
      <button class=mini id=allTypes>all</button></h4><div class=chips id=typeChips></div></div>
    <div class=ctl><h4>Versions</h4><div class=chips id=verChips></div></div>
    <div class=ctl><h4>Lengths (x-axis)</h4><div class=chips id=lenChips></div></div>
    <div class=ctl><h4>References</h4><div class=chips>
      <button class="chip on" id=refBase>instr-only</button>
      <button class="chip on" id=refPref>prefill K5</button></div>
      <span style="font-size:11px;color:var(--muted)" id=selinfo></span></div>
  </div>
  <div class=legend>{legend}</div>
  <div class=panels>{panels}</div>

  <div class=resp>
    <h3 style="margin:6px 0">Responses — read what the model produced</h3>
    <div class=bar>
      <label>model<select id=rm></select></label>
      <label>control type<select id=rt></select></label>
      <label>version<select id=rv></select></label>
      <label>length<select id=rl></select></label>
      <label class=chk><span>&nbsp;</span><span><input type=checkbox id=rfull> full exchange (Q + A)</span></label>
    </div>
    <div id=cards></div>
  </div>
</div>
<script id=data type="application/json">{json.dumps(data, ensure_ascii=False)}</script>
<script>{JS}</script></body></html>"""


def main() -> None:
    data = build()
    if not data["models"]:
        print("no iclscale_*.csv found")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "icl_scale_viewer.html"
    out.write_text(make_html(data), encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB; models={data['models']})")


if __name__ == "__main__":
    main()
