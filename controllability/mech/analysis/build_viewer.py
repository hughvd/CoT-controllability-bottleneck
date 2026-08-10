"""Emit a self-contained local HTML viewer for the OLMo RL-axis trends.

Inlines fig_data/viewer.json into a single HTML file (opens over file:// with no
server / no external deps). Lets you flip the metric (controllability / capability /
monitorability) and the facet (overall / by modality / by control-type / by mon
dataset) to see, e.g., transforms rising while language falls across the RL axis.

  python analysis/build_viewer.py   ->  analysis/olmo_viewer.html
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "olmo_rl" / "fig_data" / "viewer.json"
OUT = HERE / "olmo_viewer.html"

HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OLMo-3-32B RL-axis trend viewer</title>
<style>
:root{--ink:#1a1a2e;--muted:#6b6b7b;--grid:#e4e4ea;--bg:#ffffff;--card:#f7f7fa;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
color:var(--ink);background:var(--bg);padding:24px 28px;}
h1{font-size:18px;margin:0 0 2px}
.sub{color:var(--muted);font-size:12.5px;margin-bottom:18px}
.controls{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-end;margin-bottom:14px}
.ctl label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-bottom:4px}
select{font:13px inherit;padding:6px 10px;border:1px solid var(--grid);border-radius:7px;background:#fff;color:var(--ink);min-width:170px}
.wrap{display:flex;gap:20px;flex-wrap:wrap}
.chartbox{flex:1 1 620px;min-width:340px;background:var(--card);border:1px solid var(--grid);border-radius:12px;padding:14px 16px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:8px;font-size:12px}
.legend span{display:inline-flex;align-items:center;gap:6px;cursor:pointer;opacity:1}
.legend span.off{opacity:.3}
.swatch{width:12px;height:12px;border-radius:3px;flex:none}
table{border-collapse:collapse;font-size:12px;margin-top:4px}
th,td{padding:3px 9px;text-align:right;border-bottom:1px solid var(--grid)}
th:first-child,td:first-child{text-align:left}
#tt{position:fixed;pointer-events:none;background:var(--ink);color:#fff;font-size:12px;
padding:7px 10px;border-radius:7px;opacity:0;transition:opacity .08s;max-width:280px;z-index:9}
#tt b{color:#fff}
.note{color:var(--muted);font-size:11.5px;margin-top:6px;max-width:640px}
</style></head><body>
<h1>OLMo-3-32B along the RL axis</h1>
<div class="sub">5 canonical checkpoints · 26-type controllability · gpt-5-mini monitor · answered-only capability</div>
<div class="controls">
 <div class="ctl"><label>Metric</label>
  <select id="metric">
   <option value="ctrl_soft">Controllability (soft)</option>
   <option value="ctrl_hard">Controllability (hard ≥0.95)</option>
   <option value="capability">Capability (answered)</option>
   <option value="mon">Monitorability (gmean)</option>
  </select></div>
 <div class="ctl" id="facetbox"><label>Break down by</label>
  <select id="facet"></select></div>
 <div class="ctl" id="scopebox" style="display:none"><label>Monitor scope</label>
  <select id="scope">
   <option value="cot_only">cot_only</option>
   <option value="answer_only">answer_only</option>
   <option value="all_messages">all_messages</option>
   <option value="cot_lift">CoT lift (cot_only − answer_only)</option>
  </select></div>
</div>
<div class="wrap">
 <div class="chartbox"><svg id="chart" viewBox="0 0 640 400" style="width:100%;height:auto"></svg>
  <div class="legend" id="legend"></div>
  <div class="note" id="note"></div></div>
 <div class="chartbox" style="flex:0 1 320px"><div id="tablewrap"></div></div>
</div>
<div id="tt"></div>
<script>
const DB = __DATA__;
const MOD_LABEL = {family:"transform / encoding", language:"language", style:"style", suppression:"suppression"};
const PAL = ["#3b6fb6","#e8710a","#2a9d5c","#8e44ad","#c0392b","#16a085","#d4a017","#7f8c8d",
             "#2c3e50","#e84393","#00838f","#5d4037"];
const CK = DB.checkpoints.filter(c=>c.rl_index>=0).sort((a,b)=>a.rl_index-b.rl_index);
const XL = CK.map(c=>c.label), XI = CK.map(c=>c.rl_index);
const $ = id => document.getElementById(id);
let hidden = new Set();

// ---- build the series set for the current metric/facet/scope ----
function series(){
  const metric=$("metric").value, facet=$("facet").value;
  const byck = k => CK.map(c=>c.tag).indexOf(k); // unused
  const at = (rows, pred) => { const m={}; rows.filter(pred).forEach(r=>m[r.checkpoint]=r); return m; };
  const line = (name,color,valfn,ci) => {
    const pts = CK.map(c=>{const v=valfn(c.label); return v==null?null:v;});
    return {name,color,pts,ci};
  };
  if(metric==="mon"){
    const scope=$("scope").value;
    if(facet==="dataset"){
      const dss=[...new Set(DB.mon_by_dataset.map(r=>r.dataset))];
      return dss.map((ds,i)=>{
        const rows = DB.mon_by_dataset.filter(r=>r.dataset===ds);
        return line(ds, PAL[i%PAL.length], lab=>{
          if(scope==="cot_lift"){
            const a=rows.find(r=>r.checkpoint===lab&&r.scope==="cot_only");
            const b=rows.find(r=>r.checkpoint===lab&&r.scope==="answer_only");
            return (a&&b)?a.gmean-b.gmean:null;
          }
          const r=rows.find(r=>r.checkpoint===lab&&r.scope===scope); return r?r.gmean:null;
        });
      });
    } else { // by archetype (mean over datasets)
      const ars=[...new Set(DB.mon_by_dataset.map(r=>r.archetype))];
      return ars.map((ar,i)=>line(ar,PAL[i%PAL.length],lab=>{
        const rows=DB.mon_by_dataset.filter(r=>r.archetype===ar&&r.checkpoint===lab&&
          (scope==="cot_lift"?true:r.scope===scope));
        if(!rows.length) return null;
        if(scope==="cot_lift"){
          const co=DB.mon_by_dataset.filter(r=>r.archetype===ar&&r.checkpoint===lab&&r.scope==="cot_only");
          const ao=DB.mon_by_dataset.filter(r=>r.archetype===ar&&r.checkpoint===lab&&r.scope==="answer_only");
          const m=a=>a.reduce((s,x)=>s+x.gmean,0)/a.length;
          return (co.length&&ao.length)?m(co)-m(ao):null;
        }
        return rows.reduce((s,x)=>s+x.gmean,0)/rows.length;
      }));
    }
  }
  // controllability / capability
  const key = metric;
  if(facet==="overall"){
    const ov=at(DB.overall,()=>true);
    const cikey = metric==="ctrl_soft"?["ctrl_soft_lo","ctrl_soft_hi"]:
                  metric==="capability"?["cap_lo","cap_hi"]:null;
    return [line(metric.replace("_"," "),PAL[0],lab=>ov[lab]?ov[lab][key]:null,
      cikey?CK.map(c=>ov[c.label]?[ov[c.label][cikey[0]],ov[c.label][cikey[1]]]:null):null)];
  }
  if(facet==="modality"){
    const mods=[...new Set(DB.ctrl_by_type.map(r=>r.modality))];
    return mods.map((m,i)=>{
      const rows=DB.ctrl_by_type.filter(r=>r.modality===m);
      return line(MOD_LABEL[m]||m, PAL[i%PAL.length], lab=>{
        const rs=rows.filter(r=>r.checkpoint===lab);
        return rs.length? rs.reduce((s,x)=>s+x[key],0)/rs.length : null;
      });
    });
  }
  // by family (26), colored by modality
  const mods=[...new Set(DB.ctrl_by_type.map(r=>r.modality))];
  const fams=[...new Set(DB.ctrl_by_type.map(r=>r.family))];
  return fams.map(f=>{
    const rows=DB.ctrl_by_type.filter(r=>r.family===f);
    const mod=rows[0].modality, ci=mods.indexOf(mod);
    return {...line(f.replace(/^.*:/,""),PAL[ci%PAL.length],lab=>{
      const r=rows.find(r=>r.checkpoint===lab); return r?r[key]:null;},null), mod};
  });
}

function draw(){
  const S=series();
  const metric=$("metric").value;
  const W=640,H=400,ml=54,mr=16,mt=18,mb=44, pw=W-ml-mr, ph=H-mt-mb;
  const all=S.flatMap(s=>s.pts.filter(v=>v!=null)).concat(
            S.flatMap(s=>s.ci?s.ci.filter(Boolean).flat():[]));
  let lo=Math.min(...all,0), hi=Math.max(...all);
  if(metric==="capability"){lo=0.3;hi=0.6;}
  else if(metric==="mon"&&$("scope").value!=="cot_lift"){lo=Math.min(0.4,lo);hi=Math.max(hi,1.0);}
  else {lo=0; hi=hi*1.15||1;}
  const X=i=>ml+(XI.length<2?pw/2:pw*i/(XI.length-1));
  const Y=v=>mt+ph-ph*(v-lo)/(hi-lo);
  let g="";
  // gridlines + y ticks
  const nt=5;
  for(let t=0;t<=nt;t++){const v=lo+(hi-lo)*t/nt,y=Y(v);
    g+=`<line x1=${ml} y1=${y.toFixed(1)} x2=${W-mr} y2=${y.toFixed(1)} stroke="#e4e4ea" stroke-width="1"/>`;
    g+=`<text x=${ml-8} y=${(y+3).toFixed(1)} font-size=10 fill="#6b6b7b" text-anchor="end">${v.toFixed(2)}</text>`;}
  // x labels
  XL.forEach((l,i)=>{g+=`<text x=${X(i).toFixed(1)} y=${H-mb+18} font-size=11 fill="#1a1a2e" text-anchor="middle">${l}</text>`;});
  g+=`<text x=${ml+pw/2} y=${H-6} font-size=11 fill="#6b6b7b" text-anchor="middle">RL step</text>`;
  // CI bands
  S.forEach(s=>{ if(s.ci&&!hidden.has(s.name)){
    let up="",dn=""; s.ci.forEach((c,i)=>{if(c){up+=`${X(i).toFixed(1)},${Y(c[1]).toFixed(1)} `;}});
    for(let i=s.ci.length-1;i>=0;i--){if(s.ci[i])dn+=`${X(i).toFixed(1)},${Y(s.ci[i][0]).toFixed(1)} `;}
    if(up)g+=`<polygon points="${up}${dn}" fill="${s.color}" opacity="0.13"/>`;}});
  // lines + points
  S.forEach(s=>{ if(hidden.has(s.name))return;
    let d="",started=false;
    s.pts.forEach((v,i)=>{if(v==null){started=false;return;}
      d+=(started?"L":"M")+X(i).toFixed(1)+" "+Y(v).toFixed(1)+" ";started=true;});
    const thin = S.length>8;
    g+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${thin?1.4:2.2}" opacity="${thin?0.85:1}"/>`;
    s.pts.forEach((v,i)=>{if(v!=null)g+=`<circle cx=${X(i).toFixed(1)} cy=${Y(v).toFixed(1)} r=${thin?2.4:4} fill="${s.color}"/>`;});
  });
  $("chart").innerHTML=g;
  // legend
  $("legend").innerHTML=S.map(s=>`<span data-n="${s.name}" class="${hidden.has(s.name)?'off':''}">
    <span class="swatch" style="background:${s.color}"></span>${s.name}</span>`).join("");
  $("legend").querySelectorAll("span[data-n]").forEach(el=>el.onclick=()=>{
    const n=el.dataset.n; hidden.has(n)?hidden.delete(n):hidden.add(n); draw();});
  // table
  let th=`<tr><th>series</th>${XL.map(l=>`<th>${l}</th>`).join("")}</tr>`;
  let tb=S.map(s=>`<tr><td>${s.name}</td>${s.pts.map(v=>`<td>${v==null?"–":v.toFixed(3)}</td>`).join("")}</tr>`).join("");
  $("tablewrap").innerHTML=`<table>${th}${tb}</table>`;
  // hover
  const svg=$("chart"), tt=$("tt");
  svg.onmousemove=e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;
    let bi=0,bd=1e9; XI.forEach((_,i)=>{const dd=Math.abs(X(i)-px);if(dd<bd){bd=dd;bi=i;}});
    let html=`<b>${XL[bi]}</b><br>`+S.filter(s=>!hidden.has(s.name)&&s.pts[bi]!=null)
      .sort((a,b)=>b.pts[bi]-a.pts[bi])
      .map(s=>`<span style="color:${s.color}">●</span> ${s.name}: <b>${s.pts[bi].toFixed(3)}</b>`).join("<br>");
    tt.innerHTML=html;tt.style.opacity=1;tt.style.left=(e.clientX+14)+"px";tt.style.top=(e.clientY+14)+"px";};
  svg.onmouseleave=()=>{tt.style.opacity=0;};
  // note
  $("note").innerHTML = metric.startsWith("ctrl")||metric==="capability"
    ? "Controllability = mean compliance over the 26-type diverse universe. Capability = accuracy among answered traces (excludes ~30% truncated at 8192 tok). SFT omitted (different eval universe)."
    : "gmean of the OpenAI monitorability suite (gpt-5-mini monitor). intervention = mean over gpqa/wmdp_sandbagging/scruples/memory_advice. 'CoT lift' = how much the CoT adds over the answer alone.";
}

function refreshFacet(){
  const metric=$("metric").value, f=$("facet");
  $("scopebox").style.display = metric==="mon"?"block":"none";
  let opts;
  if(metric==="mon") opts=[["dataset","by dataset"],["archetype","by archetype"]];
  else opts=[["overall","overall"],["modality","by modality"],["family","by control-type (26)"]];
  f.innerHTML=opts.map(([v,l])=>`<option value="${v}">${l}</option>`).join("");
}
$("metric").onchange=()=>{hidden.clear();refreshFacet();draw();};
$("facet").onchange=()=>{hidden.clear();draw();};
$("scope").onchange=draw;
refreshFacet();draw();
</script></body></html>"""


def main():
    data = json.load(open(DATA))
    OUT.write_text(HTML.replace("__DATA__", json.dumps(data)))
    print(f"wrote {OUT}  ({OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
