'use strict';
function averageRanks(values) {
  const sorted=values.map((v,i)=>[v,i]).sort((a,b)=>a[0]-b[0]); const ranks=Array(values.length);
  for(let i=0;i<sorted.length;){let j=i+1;while(j<sorted.length && sorted[j][0]===sorted[i][0])j++;
    for(let k=i;k<j;k++)ranks[sorted[k][1]]=(i+j+1)/2;i=j;}return ranks;
}
function pearson(x,y){const n=x.length,mx=x.reduce((a,b)=>a+b,0)/n,my=y.reduce((a,b)=>a+b,0)/n;
  let cov=0,vx=0,vy=0;for(let i=0;i<n;i++){const a=x[i]-mx,b=y[i]-my;cov+=a*b;vx+=a*a;vy+=b*b;}
  return vx>0 && vy>0 ? Math.max(-1,Math.min(1,cov/Math.sqrt(vx*vy))) : null;}
function stats(points){const p=points.filter(d=>Number.isFinite(d.x)&&Number.isFinite(d.y));const x=p.map(d=>d.x),y=p.map(d=>d.y);
  const xs=p.length?Math.max(...x)-Math.min(...x):null,ys=p.length?Math.max(...y)-Math.min(...y):null;
  const valid=p.length>=3 && xs>1e-12 && ys>1e-12;
  return {n:p.length,pearson:valid?pearson(x,y):null,spearman:valid?pearson(averageRanks(x),averageRanks(y)):null,x_span:xs,y_span:ys,
    low:y.filter(v=>v<=.05).length,high:y.filter(v=>v>=.95).length};}
if(typeof module!=='undefined')module.exports={averageRanks,pearson,stats};
if(typeof document!=='undefined'){
const DATA=JSON.parse(document.getElementById('viewer-data').textContent);
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pretty=s=>s.replace('family:','').replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());
const colors=['#295ca8','#ad5934','#7a54a1','#328775','#a47c24','#bd4e7b','#43859d','#787047','#624694','#47754b','#986362','#555f70'];
const color=Object.fromEntries(DATA.models.map((m,i)=>[m.model,colors[i%colors.length]]));
let tab='plots',active=new Set(DATA.models.map(m=>m.model));
const fmt=x=>x===null?'—':(x>=0?'+':'')+x.toFixed(2);
function points(control,ev){return DATA.models.filter(m=>active.has(m.model)).map(m=>({model:m.model,x:m.control[control][$('metric').value],y:m.monitor[ev],
  eligibility:m.monitor[ev.replace('mon_gmean_','mon_eligfrac_')]})).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));}
function notice(s){if(s.n<3)return 'Too few paired models: correlation undefined.';
  if(s.x_span<=1e-12 || s.y_span<=1e-12)return 'Constant axis: correlation undefined.';
  const bits=[];if(s.x_span<.1)bits.push('Restricted x range');if(s.y_span<.1)bits.push('Restricted y range');
  bits.push(`y ≤ 0.05: ${s.low}/${s.n} · y ≥ 0.95: ${s.high}/${s.n}`);return bits.join(' · ');}
function clearGrid(id){const parent=$(id);parent.querySelectorAll('.plot').forEach(p=>Plotly.purge(p));parent.replaceChildren();}
function chart(parent,control,ev,family=false){const p=points(control,ev.key),s=stats(p);const card=document.createElement('article');card.className='card';
  card.innerHTML=`<div class="card-head"><h3>${esc(family?pretty(control):ev.name)}<span class="scope">${esc(ev.scope)}</span></h3><div class="members">${esc(family?DATA.families[control.slice(7)].map(pretty).join(' + '):ev.metric)}</div><div class="stat">ρ ${fmt(s.spearman)} · r ${fmt(s.pearson)} · n = ${s.n} models${s.y_span===null?'':` · y span ${s.y_span.toFixed(3)}`}</div></div><div class="plot"></div><div class="notice">${esc(notice(s))}</div>`;
  parent.append(card);const plot=card.querySelector('.plot');if(!p.length){plot.innerHTML='<p class="empty">Select models with paired data to show this plot.</p>';return;}
  const trace={type:'scatter',mode:$('labels').checked?'markers+text':'markers',x:p.map(d=>d.x),y:p.map(d=>d.y),text:p.map(d=>d.model),textposition:'top center',textfont:{size:9},
    marker:{size:11,color:p.map(d=>color[d.model]),line:{color:'#fff',width:1}},
    customdata:p.map(d=>[d.model,Number.isFinite(d.eligibility)?`<br>Eligible fraction: ${(100*d.eligibility).toFixed(1)}%`:'']),
    hovertemplate:'<b>%{customdata[0]}</b><br>Control: %{x:.3f}<br>Monitorability: %{y:.3f}%{customdata[1]}<extra></extra>'};
  const zoom=$('zoom').checked;Plotly.newPlot(plot,[trace],{margin:{l:60,r:22,t:20,b:64},font:{family:'system-ui, sans-serif',size:11,color:'#43516a'},paper_bgcolor:'#fff',plot_bgcolor:'#fff',showlegend:false,
    xaxis:{title:{text:`${pretty(control)} · ${$('metric').value} compliance`,font:{size:11}},range:zoom?undefined:[-.025,1.025],autorange:zoom,gridcolor:'#edf0f5',zeroline:false,tickformat:'.2f'},
    yaxis:{title:{text:family?`${ev.name} · ${ev.metric}`:`Monitorability · ${ev.metric}`,font:{size:11}},range:zoom?undefined:[-.025,1.025],autorange:zoom,gridcolor:'#edf0f5',zeroline:false,tickformat:'.2f'},hovermode:'closest'},
    {responsive:true,displaylogo:false,scrollZoom:false,modeBarButtonsToRemove:['select2d','lasso2d'],toImageButtonOptions:{filename:`${control.replace(':','_')}_${ev.key}`,format:'png',scale:2}});}
function ranking(ev){const coeff=$('coefficient').value,absolute=$('absolute').checked;
  return DATA.types.map(control=>({control,...stats(points(control,ev.key))})).sort((a,b)=>{
    if(a[coeff]===null)return b[coeff]===null?a.control.localeCompare(b.control):1;if(b[coeff]===null)return -1;
    return (absolute?Math.abs(b[coeff])-Math.abs(a[coeff]):b[coeff]-a[coeff])||a.control.localeCompare(b.control);});}
function render(){ $('model-count').textContent=`· ${active.size} of ${DATA.models.length} selected`;
  for(const id of ['plots','families','rankings'])$(id).hidden=tab!==id;
  document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));
  if(tab==='plots'){clearGrid('plots-grid');for(const ev of DATA.evals)chart($('plots-grid'),$('control').value,ev);}
  if(tab==='families'){clearGrid('family-grid');const ev=DATA.evals.find(e=>e.key===$('family-eval').value);for(const f of Object.keys(DATA.families))chart($('family-grid'),'family:'+f,ev,true);}
  if(tab==='rankings'){const parent=$('rank-grid');parent.replaceChildren();for(const ev of DATA.evals){const card=document.createElement('article');card.className='card rank-card';
    card.innerHTML=`<h3>${esc(ev.name)}<span class="scope">${esc(ev.scope)}</span></h3><p class="subtle">${esc(ev.metric)} · ${$('metric').value} compliance</p><table><thead><tr><th>#</th><th>Control type</th><th>ρ</th><th>r</th><th>n</th></tr></thead><tbody>${ranking(ev).map((r,i)=>`<tr title="${esc(notice(r))}"><td>${r[$('coefficient').value]===null?'—':i+1}</td><td><button class="link-button" data-control="${esc(r.control)}">${esc(pretty(r.control))}</button></td><td class="number">${fmt(r.spearman)}</td><td class="number">${fmt(r.pearson)}</td><td class="number">${r.n}</td></tr>`).join('')}</tbody></table><p class="rank-note">${esc(notice({...stats(points(DATA.types[0],ev.key)),x_span:1}))} · Rankings are exploratory.</p>`;parent.append(card);}
    parent.querySelectorAll('[data-control]').forEach(b=>b.onclick=()=>{$('control').value=b.dataset.control;tab='plots';render();$('plots').scrollIntoView({block:'start'});});}}
function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csv(rows){return rows.map(row=>row.map(v=>'"'+String(v??'').replaceAll('"','""')+'"').join(',')).join('\r\n');}
for(const [f,types] of Object.entries(DATA.families)){
  const group=document.createElement('optgroup');group.label=pretty(f);for(const c of types){const opt=new Option(pretty(c),c);group.append(opt);}$('control').append(group);}
const fg=document.createElement('optgroup');fg.label='Category averages';for(const f of Object.keys(DATA.families))fg.append(new Option(pretty(f),'family:'+f));$('control').append(fg);
for(const e of DATA.evals)$('family-eval').append(new Option(`${e.name} · ${e.scope}`,e.key));
for(const m of DATA.models){const label=document.createElement('label');label.innerHTML=`<input type="checkbox" checked value="${esc(m.model)}"><span class="dot" style="background:${color[m.model]}"></span>${esc(m.model)}`;label.querySelector('input').onchange=e=>{e.target.checked?active.add(m.model):active.delete(m.model);render();};$('models').append(label);}
for(const id of ['metric','labels','zoom','control','family-eval','coefficient','absolute'])$(id).onchange=render;
for(const b of document.querySelectorAll('[data-tab]'))b.onclick=()=>{tab=b.dataset.tab;render();};
$('all-models').onclick=()=>{active=new Set(DATA.models.map(m=>m.model));$('models').querySelectorAll('input').forEach(i=>i.checked=true);render();};
$('no-models').onclick=()=>{active.clear();$('models').querySelectorAll('input').forEach(i=>i.checked=false);render();};
$('download-data').onclick=()=>download('control_monitorability_data.json',JSON.stringify(DATA,null,2),'application/json');
$('download-rankings').onclick=()=>{const rows=[['evaluation','scope','control','compliance_metric','rank_by','absolute_sort','rank','spearman','pearson','n','x_span','y_span','included_models']];
  for(const ev of DATA.evals)ranking(ev).forEach((r,i)=>rows.push([ev.name,ev.scope,r.control,$('metric').value,$('coefficient').value,$('absolute').checked,r[$('coefficient').value]===null?'':i+1,r.spearman,r.pearson,r.n,r.x_span,r.y_span,[...active].join(';')]));download('control_monitorability_rankings.csv',csv(rows),'text/csv');};
$('download-points').onclick=()=>{const rows=[['model','control','evaluation','scope','compliance_metric','controllability','monitorability']];for(const c of [...DATA.types,...Object.keys(DATA.families).map(f=>'family:'+f)])for(const ev of DATA.evals)for(const p of points(c,ev.key))rows.push([p.model,c,ev.name,ev.scope,$('metric').value,p.x,p.y]);download('control_monitorability_points.csv',csv(rows),'text/csv');};
$('coverage').innerHTML='<table><thead><tr><th>Model</th><th>Source run</th><th>Valid responses / type</th><th>Questions / type</th><th>Empty reasoning / type</th></tr></thead><tbody>'+DATA.models.map(m=>{const c=DATA.counts.filter(c=>c.model===m.model);const ns=c.map(d=>d.valid_samples),qs=c.map(d=>d.questions),es=c.map(d=>d.empty_reasoning||0);return `<tr><td>${esc(m.model)}</td><td>${esc(m.run)}</td><td>${Math.min(...ns)}–${Math.max(...ns)}</td><td>${Math.min(...qs)}–${Math.max(...qs)}</td><td>${Math.min(...es)}–${Math.max(...es)}</td></tr>`;}).join('')+'</tbody></table>';
window.controlViewer={stats,data:DATA,render};render();
}
