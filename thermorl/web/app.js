"use strict";
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const state = {
  catalog: null, data: null, context: null, frame: 1, busy: true, playing: false, timer: null
};
const fmt = (v, digits=2) => Number(v).toFixed(digits);
const clock = m => `${String(Math.floor(m/60)).padStart(2,"0")}:${String(m%60).padStart(2,"0")}`;
function el(tag, value, className) {
  const n=document.createElement(tag);
  if(value!==undefined)n.textContent=value;
  if(className)n.className=className;
  return n;
}
function set(s,v) {
  $(s).textContent=v;
}
function table(target, headers, rows) {
  const t=el('table'),head=el('tr');
  headers.forEach(h=>head.append(el('th',h)));
  const thead=el('thead');
  thead.append(head);
  t.append(thead);
  const body=el('tbody');
  rows.forEach(row=> {
    const tr=el('tr');row.forEach(v=>tr.append(el('td',v)));body.append(tr);
  });
  t.append(body);
  $(target).replaceChildren(t);
}
async function json(url) {
  const response=await fetch(url);
  const data=await response.json();
  if(!response.ok)throw Error(data.error||'Request failed');
  return data;
}
function error(e) {
  set('#error',e.message);
  $('#error').hidden=false;
  set('#status','Request failed. Check the settings and Python terminal.');
}
function busy(value,message) {
  state.busy=value;
  for(const id of ['generate','scenario','controller','seed','compare'])$('#'+id).disabled=value;
  for(const id of ['restart','back','step','play','timeline','download'])$('#'+id).disabled=value||!state.data;
  $$('[data-jump]').forEach(b=>b.disabled=value||!state.data);
  if(message)set('#status',message);
}
function selection() {
  const raw=$('#seed').value.trim();
  if(!/^\d+$/.test(raw)||Number(raw)>4294967295)throw Error('Day seed must be a whole number between 0 and 4294967295.');
  return {
    scenario:$('#scenario').value,controller:$('#controller').value,seed:Number(raw)
  };
}
function url(s,controller=s.controller) {
  return '/api/simulate?'+new URLSearchParams( {
    ...s,controller
  });
}
function sameSettings() {
  if(!state.context)return false;
  try {
    return JSON.stringify(selection())===JSON.stringify(state.context);
  }catch {
    return false;
  }
}
function show(view) {
  if(view!=='step')pause();
  $$('.view').forEach(s=>s.hidden=s.id!=='view-'+view);
  $$('[data-view]').forEach(b=> {
    const active=b.dataset.view===view;b.classList.toggle('active',active);if(active)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');
  });
}
async function generate() {
  if(state.busy)return;
  pause();
  $('#error').hidden=true;
  try {
    const chosen=selection();
    busy(true,'Generating a full-day Python rollout…');
    const data=await json(url(chosen));
    state.data=data;
    state.context=chosen;
    state.frame=Math.min(state.frame,data.frames.length-1);
    $('#timeline').max=data.frames.length-1;
    $('#comparison').replaceChildren();
    $('#pending').hidden=true;
    set('#scenario-description',state.catalog.scenarios.find(s=>s.id===chosen.scenario).description);
    render();
    set('#status','Ready · Python-generated rollout. Playback does not train the agent.');
  }catch(e) {
    error(e);
  }finally {
    busy(false);
  }
}
function render() {
  if(!state.data)return;
  const f=state.data.frames[state.frame],c=state.catalog.config;
  const policy=state.catalog.controllers.find(p=>p.id===state.context.controller);
  set('#clock',clock(f.minute));
  set('#step-number',`${state.frame} / ${state.data.frames.length-1}`);
  $('#timeline').value=state.frame;
  $('#timeline').setAttribute('aria-valuetext',clock(f.minute));
  set('#policy-note',policy.kind==='rl'?`${policy.label} · ${policy.episodes??'Unknown'} training episodes · greedy evaluation (ε = 0).`:`Baseline: ${policy.label}. This controller uses fixed rules, not learned Q-values.`);
  const initial=f.action===null;
  set('#state-context',initial?'Initial state, before the first action. Advance to inspect a transition.':`Inputs at ${clock(f.interval_start_minute)}. Outdoor ${fmt(f.outdoor,1)}°C; solar ${fmt(f.solar,0)} W/m².`);
  const before=initial?f.temperature:f.temperature_before;
  table('#state-table',['State input','Room 1','Room 2'],[['Air temperature (°C)',...before.map(v=>fmt(v))],['Students present',...f.occupancy],['Next class in (min)',...f.next_class_minutes.map((v,i)=>f.next_class_occupancy[i]?(v===240?'≥240':v):'None')],['Next class size',...f.next_class_occupancy]]);
  set('#observation',initial?'No action has been selected.':JSON.stringify(f.observation.map(v=>Number(fmt(v,4)))));
  set('#chosen-action',initial?'No action yet.':`aₜ = ${f.action}: Room 1 ${['off','fan','AC','AC + fan'][f.equipment[0][0]*2+f.equipment[0][1]]}; Room 2 ${['off','fan','AC','AC + fan'][f.equipment[1][0]*2+f.equipment[1][1]]}.`);
  set('#power',`${fmt(f.power_kw)} / ${fmt(f.cap_kw)} kW · ${f.feasible_actions} feasible joint actions`);
  const grid=$('#action-grid');
  grid.replaceChildren();
  for(let a=0;a<16;a++) {
    const valid=initial?null:f.action_mask[a];
    const n=el('div',undefined,'action-cell'+(valid===false?' masked':'')+(a===f.action?' selected':''));
    n.append(el('span',`A${a} · (${Math.floor(a/4)}, ${a%4})`));
    n.append(el('small',initial?'—':!valid?'masked':f.q_values?`Q = ${fmt(f.q_values[a],3)}`:'feasible'));
    grid.append(n);
  }set('#action-note',initial?'Advance one step to see the action mask.':f.q_values?(f.state_seen===false?'Unseen tabular state: Q-values default to zero. Greedy ties choose the first feasible action.':'Select the largest estimated Q-value among feasible actions. Values are estimates, not guaranteed returns.'):'Highlighted = chosen; crossed out = over the power cap. Room modes: 0 off, 1 fan, 2 AC, 3 AC + fan.');
  table('#transition',['Transition','Room 1','Room 2'],[['Air before (°C)',...before.map(v=>fmt(v))],['Air after (°C)',...f.temperature.map(v=>fmt(v))],['ΔT (°C)',...f.temperature.map((v,i)=>fmt(v-before[i],3))],['Comfort proxy (°C)',...f.effective_temperature.map(v=>fmt(v))]]);
  set('#transition-note',initial?'Initial temperatures; no transition has occurred.':`Action interval ${clock(f.interval_start_minute)}–${clock(f.minute)} · Δt = ${state.data.step_minutes} min. Proxy = air − ${c.fan_comfort_offset_c}°C when fan is on.`);
  const heatRows=initial?[]:Object.entries(f.heat_terms_kw).map(([k,v])=>[k,...v.map(x=>fmt(x,3))]);
  if(!initial)heatRows.push(['Total',...[0,1].map(i=>fmt(Object.values(f.heat_terms_kw).reduce((sum,v)=>sum+v[i],0),3))]);
  table('#heat-table',['Net heat (kW)','Room 1','Room 2'],heatRows);
  table('#reward-table',['Weighted penalty','Value'],Object.entries(f.reward_terms).map(([k,v])=>[k,fmt(v,4)]));
  set('#reward-value',`rₜ = −Σ penalties = ${fmt(f.reward,4)}`);
  const metrics=$('#running-metrics');
  metrics.replaceChildren();
  for(const [label,value] of [['Energy so far',fmt(f.energy_kwh)+' kWh'],['Occupied comfort so far',f.comfort_pct===null?'No occupied time':fmt(f.comfort_pct,1)+'%'],['Worst-room discomfort',fmt(f.worst_room_discomfort_c)+'°C'],['AC switches / cap violations',`${f.ac_switches} / ${f.cap_violations}`]]) {
    const n=el('div',label);
    n.append(el('strong',value));
    metrics.append(n);
  }renderRooms(f,c,policy);
  drawTemperature();
  set('#comparison-context',`${state.catalog.scenarios.find(s=>s.id===state.context.scenario).label} · day seed ${state.context.seed}. Full-day comparison, independent of the playback position.`);
}
function svgNode(tag,attributes,value) {
  const n=document.createElementNS('http://www.w3.org/2000/svg',tag);
  Object.entries(attributes).forEach(([k,v])=>n.setAttribute(k,v));
  if(value!==undefined)n.textContent=value;
  return n;
}
// The schematic only renders recorded transitions; Python owns all physics.
function createRooms() {
  const container=$('#rooms');
  for(let i=0;i<2;i++) {
    const room=el('section',undefined,`room room-${i+1}`);
    room.id=`room-${i+1}`;
    room.setAttribute('aria-label',`Classroom ${i+1}`);
    const heading=el('div',undefined,'room-heading');
    heading.append(el('h4',`Classroom ${i+1}`),el('span','','comfort-status'));
    const temperature=el('div',undefined,'room-temperature');
    temperature.append(el('strong','','air-reading'),el('span','','temperature-change'));
    const svg=svgNode('svg',{viewBox:'0 0 320 205',class:'room-plan',role:'img'});
    svg.append(svgNode('title',{},`Classroom ${i+1} equipment and students`));
    svg.append(svgNode('rect',{x:9,y:9,width:302,height:187,rx:12,class:'room-floor'}));
    svg.append(svgNode('rect',{x:292,y:70,width:7,height:62,rx:3,class:'room-window'}));
    svg.append(svgNode('text',{x:285,y:153,'text-anchor':'end',class:'plan-label'},'WINDOW'));
    svg.append(svgNode('rect',{x:107,y:23,width:102,height:5,rx:2,class:'board'}));
    svg.append(svgNode('text',{x:158,y:42,'text-anchor':'middle',class:'plan-label'},'TEACHING BOARD'));
    const ac=svgNode('g',{class:'ac-unit'});
    ac.append(svgNode('rect',{x:23,y:21,width:55,height:23,rx:5}),svgNode('path',{d:'M31 37H70'}),svgNode('text',{x:50,y:32,'text-anchor':'middle'},'AC'));
    const breeze=svgNode('g',{class:'airflow'});
    [34,50,66].forEach(x=>breeze.append(svgNode('path',{d:`M${x} 48q-7 9 0 18t0 18`})));
    svg.append(ac,breeze);
    const fan=svgNode('g',{class:'fan-rotor'});
    for(let a=0;a<3;a++)fan.append(svgNode('path',{d:'M261 37C241 29 247 12 255 18C261 22 264 29 261 37Z',transform:`rotate(${a*120} 261 37)`}));
    fan.append(svgNode('circle',{cx:261,cy:37,r:4}));
    svg.append(fan);
    // Twelve shared desks, each with five seat markers (60 student capacity).
    for(let d=0;d<12;d++) {
      const x=89+(d%4)*47,y=76+Math.floor(d/4)*40;
      svg.append(svgNode('rect',{x:x-15,y:y-4,width:30,height:11,rx:3,class:'desk'}));
      for(let s=0;s<5;s++)svg.append(svgNode('circle',{cx:x-12+s*6,cy:y+15,r:2.5,class:'student'}));
    }
    const equipment=el('div',undefined,'equipment-status');
    equipment.append(el('span','','ac-status'),el('span','','fan-status'));
    const stats=el('div',undefined,'room-stats');
    stats.append(el('span','','room-occupancy'),el('span','','room-proxy'));
    room.append(heading,el('p','AIR TEMPERATURE AFTER ACTION','air-label'),temperature,svg,equipment,stats,el('p','','room-next'));
    container.append(room);
  }
}
function renderRooms(f,c,policy) {
  const initial=f.action===null;
  set('#scene-interval',initial?'00:00 · initial state':`${clock(f.interval_start_minute)} → ${clock(f.minute)}`);
  set('#scene-weather',`Outdoors ${fmt(f.outdoor,1)}°C  /  Solar input ${fmt(f.solar,0)} W/m²`);
  set('#scene-band',`Comfort proxy target ${c.comfort_low_c}–${c.comfort_high_c}°C`);
  const powers=[];
  for(let i=0;i<2;i++) {
    const room=$(`#room-${i+1}`),[ac,fan]=f.equipment[i],t=f.temperature[i],proxy=f.effective_temperature[i];
    const occupied=f.occupancy[i]>0;
    const status=!occupied?'empty':proxy>c.comfort_high_c?'warm':proxy<c.comfort_low_c?'cool':'comfortable';
    room.dataset.comfort=status;
    room.querySelector('.air-label').textContent=initial?'INITIAL AIR TEMPERATURE':'AIR TEMPERATURE AFTER ACTION';
    room.classList.toggle('ac-on',Boolean(ac));
    room.classList.toggle('fan-on',Boolean(fan));
    room.querySelector('.comfort-status').textContent={empty:'Unoccupied',warm:'Too warm',cool:'Too cool',comfortable:'Comfortable'}[status];
    room.querySelector('.air-reading').textContent=`${fmt(t,1)}°C`;
    const change=initial?0:t-f.temperature_before[i];
    room.querySelector('.temperature-change').textContent=initial?'Initial air temperature':`${fmt(f.temperature_before[i],1)}°C before · ${change>0?'+':''}${fmt(change,2)}°C`;
    room.querySelector('.ac-status').textContent=`AC ${ac?'ON':'OFF'}`;
    room.querySelector('.fan-status').textContent=`Fan ${fan?'ON':'OFF'}`;
    room.querySelectorAll('.student').forEach((n,j)=>n.classList.toggle('present',j<f.occupancy[i]));
    room.querySelector('.room-occupancy').textContent=`${f.occupancy[i]} students`;
    room.querySelector('.room-proxy').textContent=`Proxy ${fmt(proxy,1)}°C`;
    room.querySelector('.room-next').textContent=f.next_class_occupancy[i]?`Next class: ${f.next_class_minutes[i]>=240?'≥240':f.next_class_minutes[i]} min · ${f.next_class_occupancy[i]} students (at interval start)`:'No upcoming class today';
    room.querySelector('svg').setAttribute('aria-label',`Classroom ${i+1}: ${f.occupancy[i]} students, AC ${ac?'on':'off'}, fan ${fan?'on':'off'}, air ${fmt(t,1)} degrees, ${status}.`);
    powers.push(ac*c.ac_electric_kw+fan*c.fan_electric_kw);
  }
  set('#scene-power',`${fmt(f.power_kw)} kW`);
  set('#scene-cap',`of ${fmt(f.cap_kw)} kW available`);
  const meter=$('#power-meter'),breakdown=$('#power-breakdown');
  meter.replaceChildren();breakdown.replaceChildren();
  const entries=[['Classroom 1',powers[0],'room-1'],['Classroom 2',powers[1],'room-2'],['Unused capacity',Math.max(0,f.cap_kw-f.power_kw),'unused']];
  entries.forEach(([label,value,className])=> {
    const segment=el('span',undefined,className);
    segment.style.width=`${f.cap_kw>0?value/f.cap_kw*100:0}%`;
    meter.append(segment);
    const row=el('div',undefined,'power-row '+className);
    row.append(el('span',label),el('strong',`${fmt(value)} kW`));
    breakdown.append(row);
  });
  meter.setAttribute('aria-label',entries.map(([label,value])=>`${label} ${fmt(value)} kilowatts`).join(', '));
  set('#scene-feasible',`${f.feasible_actions} / 16 actions fit this interval’s power cap.`);
  set('#scene-reward',initial?'—':fmt(f.reward,3));
  set('#scene-policy',policy.kind==='rl'?'Frozen learned policy · greedy action':'Rule-based controller · fixed policy');
  set('#scene-decision',initial?'Start with Next decision to see the controller choose equipment and the rooms respond.':`Action ${f.action} → equipment runs for ${state.data.step_minutes} minutes → air temperatures above are the outcome. ${f.state_seen===false?'This tabular state is unseen: all Q-values are zero, so the greedy tie chooses the first feasible action.':'Inspect the action values and reward breakdown below to explain this decision.'}`);
}
function plot(target, series, xmax, labels, band) {
  const svg=$(target);
  svg.replaceChildren();
  const points=series.flatMap(s=>s.points);
  if(!points.length) {
    svg.append(svgNode('text', {
      x:30,y:75,fill:'#586572','font-size':14
    },'No saved training log available for this policy.'));
    return;
  }const all=points.map(p=>p[1]).concat(band||[]);
  let lo=Math.min(...all),hi=Math.max(...all);
  const pad=Math.max((hi-lo)*.12,1);
  lo-=pad;
  hi+=pad;
  const x=v=>48+v/Math.max(xmax,1)*890,y=v=>174-(v-lo)/(hi-lo)*155;
  if(band)svg.append(svgNode('rect', {
    x:48,y:y(band[1]),width:890,height:y(band[0])-y(band[1]),fill:'#e9f5ef'
  }));
  for(let i=0;i<5;i++) {
    const v=lo+(hi-lo)*i/4;
    svg.append(svgNode('line', {
      x1:48,x2:938,y1:y(v),y2:y(v),stroke:'#e2e6ea'
    }),svgNode('text', {
      x:40,y:y(v)+4,'text-anchor':'end',fill:'#586572','font-size':10
    },fmt(v,1)));
  }labels.forEach(([v,label])=>svg.append(svgNode('text', {
    x:x(v),y:198,'text-anchor':'middle',fill:'#586572','font-size':10
  },label)));
  for(const s of series) {
    svg.append(svgNode('path', {
      d:s.points.map((p,i)=>(i?'L':'M')+x(p[0])+','+y(p[1])).join(' '),fill:'none',stroke:s.color,'stroke-width':1.8,...(s.dash? {
        'stroke-dasharray':'4 4'
      }: {
      })
    }));
  }
}
function drawTemperature() {
  const frames=state.data.frames.slice(0,state.frame+1),key=$('#chart-mode').value,c=state.catalog.config;
  plot('#chart',[ {
    points:frames.map(f=>[f.minute,f[key][0]]),color:'#315bdd'
  }, {
    points:frames.map(f=>[f.minute,f[key][1]]),color:'#8761cb'
  }, {
    points:frames.map(f=>[f.minute,f.outdoor]),color:'#8b939b',dash:true
  }],1440,[0,6,12,18,24].map(h=>[h*60,`${String(h).padStart(2,'0')}:00`]),[c.comfort_low_c,c.comfort_high_c]);
}
function pause() {
  clearTimeout(state.timer);
  state.playing=false;
  set('#play','Play');
  $('#play').setAttribute('aria-pressed','false');
}
function seek(index) {
  if(state.busy||!state.data)return;
  pause();
  state.frame=Math.max(0,Math.min(index,state.data.frames.length-1));
  render();
}
function tick() {
  if(!state.playing)return;
  if(state.frame>=state.data.frames.length-1) {
    pause();
    return;
  }state.frame++;
  render();
  state.timer=setTimeout(tick,Number($('#speed').value));
}
function play() {
  if(state.playing) {
    pause();
    return;
  }if(!state.data||state.busy)return;
  if(state.frame===state.data.frames.length-1)state.frame=0;
  state.playing=true;
  set('#play','Pause');
  $('#play').setAttribute('aria-pressed','true');
  tick();
}
function training() {
  const model=state.catalog.controllers.find(p=>p.id===$('#training-model').value);
  if(!model) {
    set('#training-info','No learned policies loaded. Train a run with train.py, then restart the GUI. Baselines are still available.');
    plot('#training-chart',[],1,[]);
    return;
  }const history=model.training_history||[],h=model.hyperparameters;
  set('#training-info',`${model.label} · ${model.episodes??'Unknown'} episodes · training seed ${model.training_seed??'unknown'} · ${Object.entries(h||{}).map(([k,v])=>k+' = '+v).join(', ')}${history.length?' · last ε = '+fmt(history.at(-1).epsilon,3)+' · last logged loss = '+fmt(history.at(-1).mean_loss,3):''}`);
  const max=history.at(-1)?.episode||1;
  plot('#training-chart',[ {
    points:history.map(r=>[r.episode,r.return]),color:'#315bdd'
  }],max,[1,Math.round(max/2),max].map(n=>[n,`Episode ${n}`]));
}
async function compare() {
  if(state.busy)return;
  if(!sameSettings()) {
    error(Error('Generate day on the decision screen to apply the selected scenario, controller and seed before comparing.'));
    return;
  }pause();
  $('#error').hidden=true;
  busy(true,'Evaluating all controllers on the same day…');
  try {
    const results=await Promise.all(state.catalog.controllers.map(async p=>( {
      p,data:await json(url(state.context,p.id))
    })));
    table('#comparison',['Controller','Energy (kWh)','Comfort violations (%)','Class-start violations (%)','Worst-room discomfort (°C)','Cap violations'],results.map(( {
      p,data
    })=>[p.label+(p.kind==='rl'?' · learned':' · rule'),...['energy_kwh','comfort_violation_pct','class_start_violation_pct','worst_room_mean_discomfort_c','cap_violations'].map(k=>fmt(data.metrics[k],k==='cap_violations'?0:2))]));
    set('#status','Same-day comparison complete. Use held-out evaluation for final claims.');
  }catch(e) {
    error(e);
  }finally {
    busy(false);
  }
}
function download() {
  const headers=['minute','air_1','air_2','proxy_1','proxy_2','students_1','students_2','ac_1','fan_1','ac_2','fan_2','power_kw','cap_kw','cumulative_energy_kwh','reward'];
  const rows=state.data.frames.map(f=>[f.minute,...f.temperature,...f.effective_temperature,...f.occupancy,...f.equipment.flat(),f.power_kw,f.cap_kw,f.energy_kwh,f.reward]);
  const u=URL.createObjectURL(new Blob([[headers,...rows].map(r=>r.join(',')).join('\n')], {
    type:'text/csv;charset=utf-8'
  }));
  const a=el('a');
  a.href=u;
  a.download=`thermorl_${state.context.controller}_${state.context.scenario}_${state.context.seed}.csv`;
  a.click();
  setTimeout(()=>URL.revokeObjectURL(u),1000);
}
async function init() {
  createRooms();
  busy(true);
  $$('[data-view]').forEach(b=>b.onclick=()=>show(b.dataset.view));
  $$('[data-open]').forEach(b=>b.onclick=()=>show(b.dataset.open));
  $('#settings').onsubmit=e=> {
    e.preventDefault();
    generate();
  };
  $('#restart').onclick=()=>seek(0);
  $('#back').onclick=()=>seek(state.frame-1);
  $('#step').onclick=()=>seek(state.frame+1);
  $('#play').onclick=play;
  $('#timeline').oninput=e=>seek(Number(e.target.value));
  $$('[data-jump]').forEach(b=>b.onclick=()=>seek(Math.round(Number(b.dataset.jump)/state.data.step_minutes)));
  $('#chart-mode').onchange=()=> {
    if(state.data)drawTemperature();
  };
  $('#training-model').onchange=training;
  $('#compare').onclick=compare;
  $('#download').onclick=download;
  for(const id of ['scenario','controller','seed'])$('#'+id).oninput=()=> {
    $('#pending').hidden=false;
    pause();
  };
  try {
    state.catalog=await json('/api/catalog');
    for(const [id,items] of [['scenario',state.catalog.scenarios],['controller',state.catalog.controllers],['training-model',state.catalog.controllers.filter(p=>p.kind==='rl')]]) {
      const s=$('#'+id);
      for(const item of items) {
        const o=el('option',item.label);
        o.value=item.id;
        s.append(o);
      }
    }$('#scenario').value='restricted';
    $('#controller').value=state.catalog.controllers.find(p=>p.algorithm==='q_learning')?.id||'timetable';
    const c=state.catalog.config;
    table('#parameters',['Parameter','Loaded value'],Object.entries(c).map(([k,v])=>[k,Array.isArray(v)?v.join(', '):v]));
    set('#reward-equation',`Loaded weights: energy ${c.energy_weight}, discomfort ${c.discomfort_weight}, worst room ${c.fairness_weight}, switching ${c.switching_weight}. Comfort band ${c.comfort_low_c}–${c.comfort_high_c}°C; fan offset ${c.fan_comfort_offset_c}°C.`);
    training();
    if(state.catalog.warnings.length)error(Error(state.catalog.warnings.join(' ')));
    busy(false);
    await generate();
  }catch(e) {
    error(e);
    busy(false);
  }
}
init();
