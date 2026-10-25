"use strict";
const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const state = {
  catalog: null, data: null, context: null, runs: new Map(), frame: null, busy: true, playing: false, timer: null
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
  for(const id of ['generate','scenario','controller','seed'])$('#'+id).disabled=value;
  for(const id of ['restart','back','step','play','timeline','download'])$('#'+id).disabled=value||!state.data;
  $('#controller').disabled=value||!state.data;
  $$('#comparison button').forEach(b=>b.disabled=value);
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
const scenarioQuestions = {
  normal: 'Test question: how does each policy behave when both ACs can run?',
  restricted: 'Test question: how does each policy share cooling when both ACs cannot run together?',
  unequal: 'Test question: how does each policy handle unequal heat exposure?',
  schedule_changes: 'Test question: how does each policy respond to announced timetable changes?',
  heatwave: 'Test question: how does each policy behave in hotter conditions withheld from training?'
};
function describeConditions() {
  const scenario=state.catalog.scenarios.find(s=>s.id===$('#scenario').value);
  set('#scenario-description',scenario.description);
  set('#scenario-purpose',scenarioQuestions[scenario.id]);
}
function inspectPolicy(id) {
  if(state.busy||!state.runs.has(id))return;
  pause();
  $('#controller').value=id;
  state.context={...state.context,controller:id};
  state.data=state.runs.get(id);
  render();
  renderComparison();
}
function show(view) {
  if(view!=='step')pause();
  $$('.view').forEach(s=>s.hidden=s.id!=='view-'+view);
  $$('[data-view]').forEach(b=> {
    const active=b.dataset.view===view;b.classList.toggle('active',active);if(active)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');
  });
  window.scrollTo({top:0,behavior:'instant'});
}
async function generate() {
  if(state.busy)return;
  pause();
  $('#error').hidden=true;
  try {
    const chosen=selection();
    busy(true,'Running all policies on identical day inputs…');
    const results=await Promise.all(state.catalog.controllers.map(async policy=>({
      policy,data:await json(url(chosen,policy.id))
    })));
    // Commit the experiment only after every policy has finished successfully.
    state.runs=new Map(results.map(({policy,data})=>[policy.id,data]));
    state.data=state.runs.get(chosen.controller);
    state.context=chosen;
    state.frame=Math.min(state.frame??Math.round(720/state.data.step_minutes),state.data.frames.length-1);
    $('#timeline').max=state.data.frames.length-1;
    $('#pending').hidden=true;
    const scenario=state.catalog.scenarios.find(s=>s.id===chosen.scenario);
    set('#experiment-title',`${scenario.label} · day seed ${chosen.seed}`);
    describeConditions();
    render();
    renderComparison();
    set('#status',`Experiment ready · ${results.length} policies · identical day inputs · frozen evaluation.`);
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
  set('#policy-note',policy.kind==='rl'?`${policy.label} · ${policy.initialization?'fine-tuned':'trained'} ${policy.episodes??'Unknown'} ${policy.initialization?'additional ':''}episodes · selected checkpoint ${policy.selected_episode??'unknown'} · greedy evaluation (ε = 0).`:`Baseline: ${policy.label}. This controller uses fixed rules, not learned Q-values.`);
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
  }set('#action-note',initial?'Advance one step to see the action mask.':f.q_values?(f.state_seen===false?'Unseen tabular state: Q-values default to zero. Greedy ties choose the first feasible action.':`Select the largest estimated Q-value among feasible actions. Values are estimates, not guaranteed returns.${policy.reward_scale>1?' DQN values use scaled training-reward units; the displayed reward stays unscaled.':''}`):'Highlighted = chosen; crossed out = over the power cap. Room modes: 0 off, 1 fan, 2 AC, 3 AC + fan.');
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
    const arrival=el('p','','room-arrival');
    arrival.hidden=true;
    room.append(heading,el('p','AIR TEMPERATURE AFTER ACTION','air-label'),temperature,svg,equipment,stats,arrival,el('p','','room-next'));
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
    const arrival=room.querySelector('.room-arrival');
    arrival.hidden=initial||!f.class_start[i]||!occupied;
    arrival.textContent=arrival.hidden?'':`Class arrival ${clock(f.interval_start_minute)} · Entry proxy ${fmt(f.arrival_comfort_proxy[i],1)}°C · ${f.arrival_violation[i]>0?'Outside comfort band':'Within comfort band'} (before action)`;
    room.querySelector('.room-next').textContent=f.next_class_occupancy[i]?`Next class: ${f.next_class_minutes[i]>=240?'≥240':f.next_class_minutes[i]} min · ${f.next_class_occupancy[i]} students (at interval start)`:'No upcoming class today';
    room.querySelector('svg').setAttribute('aria-label',`Classroom ${i+1}: ${f.occupancy[i]} students, AC ${ac?'on':'off'}, fan ${fan?'on':'off'}, air ${fmt(t,1)} degrees, ${status}.`);
    powers.push(ac*c.ac_electric_kw+fan*c.fan_electric_kw);
  }
  set('#scene-power',`${fmt(f.power_kw)} kW`);
  set('#scene-cap',`Allowed limit: ${fmt(f.cap_kw)} kW`);
  const meter=$('#power-meter'),breakdown=$('#power-breakdown');
  meter.replaceChildren();breakdown.replaceChildren();
  const entries=[['Classroom 1',powers[0],'room-1'],['Classroom 2',powers[1],'room-2'],['Remaining capacity',Math.max(0,f.cap_kw-f.power_kw),'unused']];
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
  const selectionReason=policy.kind==='baseline'?policy.description:f.state_seen===false?'Unseen tabular state: all Q-values are zero; the greedy tie selects the first feasible action.':'The policy selects a highest estimated Q-value among actions that fit the power limit.';
  set('#scene-decision',initial?'Initial state: no action has run. Advance one decision to see the policy act.':`Action ${f.action} · ${selectionReason} Equipment runs for ${state.data.step_minutes} minutes; the rooms above show the outcome. Open “Explain this decision” for the recorded inputs, values and reward.`);
}
function plot(target, series, xmax, labels, band, fixedRange) {
  const svg=$(target);
  svg.replaceChildren();
  const points=series.flatMap(s=>s.points);
  if(!points.length) {
    svg.append(svgNode('text', {
      x:30,y:75,fill:'#586572','font-size':14
    },'No saved training log available for this policy.'));
    return;
  }const all=points.map(p=>p[1]).concat(band||[]);
  let lo=fixedRange?fixedRange[0]:Math.min(...all),hi=fixedRange?fixedRange[1]:Math.max(...all);
  const pad=fixedRange?0:Math.max((hi-lo)*.12,1);
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
    x:x(v),y:198,'text-anchor':v>=xmax?'end':v===0?'start':'middle',fill:'#586572','font-size':10
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
    $('#training-facts').replaceChildren();
    set('#validation-info','No learned policy loaded.');
    plot('#validation-chart',[],1,[]);
    return;
  }const history=model.training_history||[],h=model.hyperparameters;
  const facts=$('#training-facts');
  facts.replaceChildren();
  const last=history.at(-1);
  for(const [label,value] of [[model.initialization?'Additional episodes':'Training episodes',model.episodes??'Unknown'],['Selected checkpoint',model.selected_episode??'Unknown'],['Last logged ε',last?fmt(last.epsilon,3):'Unavailable'],[model.algorithm==='q_learning'?'Last logged states':'Last logged loss',last?(model.algorithm==='q_learning'?last.visited_states:fmt(last.mean_loss,3)):'Unavailable']]) {
    const item=el('div',label);
    item.append(el('strong',String(value)));
    facts.append(item);
  }
  set('#training-info',`${model.label} · ${model.episodes??'Unknown'} episodes · training seed ${model.training_seed??'unknown'} · ${Object.entries(h||{}).map(([k,v])=>k+' = '+v).join(', ')}${history.length?' · last ε = '+fmt(history.at(-1).epsilon,3)+' · last logged loss = '+fmt(history.at(-1).mean_loss,3):''}`);
  if(model.initialization)$('#training-info').append(document.createTextNode(` Fine-tuned from a previously trained checkpoint (episode ${model.initialization.source_selected_episode}); replay and optimizer restarted.`));
  const architectureModel=model.algorithm==='dqn'?model:state.catalog.controllers.find(p=>p.algorithm==='dqn');
  set('#network-description',`${architectureModel?.network_inputs??21} inputs → 64 ReLU → 64 ReLU → 16 Q-values\nNew comfort_v1 runs derive 21 features from 17 raw observations.`);
  const validation=model.validation_history||[];
  const floorNote=model.selection_mode==='comfort_energy'?` Validation comfort target ${model.comfort_floor}%: ${model.selection_comfort_floor_met?'met':'not met'}.`:'';
  set('#validation-info',`Training seed ${model.training_seed??'unknown'} · selected episode ${model.selected_episode??'unknown'} of ${model.episodes??'unknown'}. ${model.selection}.${floorNote}`);
  const validationMax=validation.at(-1)?.episode||1;
  plot('#validation-chart',validation.length?[{points:validation.map(r=>[r.episode,r.mean_comfort_pct]),color:'#315bdd'}]:[],validationMax,[0,Math.round(validationMax/2),validationMax].map(n=>[n,`Episode ${n}`]),undefined,[0,100]);
  if(validation.length&&model.selected_episode!==undefined) {
    const x=48+model.selected_episode/validationMax*890;
    $('#validation-chart').append(svgNode('line',{x1:x,x2:x,y1:19,y2:174,stroke:'#315bdd','stroke-dasharray':'4 4'}),svgNode('text',{x:x+4,y:15,fill:'#315bdd','font-size':10,'text-anchor':x>800?'end':'start'},`Selected: ${model.selected_episode}`));
  }
  const max=history.at(-1)?.episode||1;
  plot('#training-chart',[ {
    points:history.map(r=>[r.episode,r.return]),color:'#315bdd'
  }],max,[1,Math.round(max/2),max].map(n=>[n,`Episode ${n}`]));
}
function renderComparison() {
  if(!state.context)return;
  const results=state.catalog.controllers.map(policy=>({policy,data:state.runs.get(policy.id)}));
  const t=el('table'),head=el('tr'),thead=el('thead'),body=el('tbody');
  ['Policy','Arrival discomfort (%) ↓','First-15-minute discomfort (%) ↓','Whole-day comfortable (%) ↑','Energy (kWh) ↓','Inspect'].forEach(label=>head.append(el('th',label)));
  thead.append(head);t.append(thead,body);
  results.forEach(({policy,data},index)=> {
    const row=el('tr');
    row.classList.toggle('inspected',policy.id===state.context.controller);
    const name=el('td'),title=el('strong',`${index+1}. ${policy.label}`);
    name.append(title,el('small',policy.kind==='rl'?'Learned policy':'Rule-based baseline'));
    const action=el('td'),button=el('button',policy.id===state.context.controller?'Viewing':'Inspect');
    button.type='button';
    button.setAttribute('aria-label',`Inspect ${policy.label}`);
    button.setAttribute('aria-pressed',String(policy.id===state.context.controller));
    button.disabled=state.busy;
    button.onclick=()=>inspectPolicy(policy.id);
    action.append(button);
    const m=data.metrics;
    const arrival=el('td',!m.class_arrivals?'Not applicable':m.arrival_discomfort_pct===null?'Unavailable':fmt(m.arrival_discomfort_pct));
    arrival.title=`${m.uncomfortable_arrivals??'Unknown'} uncomfortable / ${m.class_arrivals} class arrivals; ${m.arrival_measurements} measured`;
    const early=el('td',m.early_class_occupied_minutes?fmt(m.class_start_violation_pct):'Not applicable');
    early.title=`${fmt(m.early_class_uncomfortable_minutes)} uncomfortable / ${fmt(m.early_class_occupied_minutes)} early-class room-minutes`;
    const whole=el('td',m.occupied_room_minutes?fmt(m.comfort_pct):'Not applicable');
    whole.title=`${fmt(m.comfortable_room_minutes)} comfortable / ${fmt(m.occupied_room_minutes)} occupied room-minutes`;
    row.append(name,arrival,early,whole,el('td',fmt(m.energy_kwh)),action);
    body.append(row);
  });
  $('#comparison').replaceChildren(t);
  table('#comparison-extra',['Policy','Too hot (%)','Too cold (%)','Worst-room discomfort (°C)','AC switches','Cap violations'],results.map(({policy,data})=>[policy.label,...['overheating_pct','overcooling_pct','worst_room_mean_discomfort_c','ac_switches','cap_violations'].map(k=>!data.metrics.occupied_room_minutes&&['overheating_pct','overcooling_pct','worst_room_mean_discomfort_c'].includes(k)?'Not applicable':fmt(data.metrics[k],k==='ac_switches'||k==='cap_violations'?0:2))]));
  table('#comparison-counts',['Policy','Uncomfortable / total arrivals','Uncomfortable / early room-minutes','Comfortable / occupied room-minutes'],results.map(({policy,data:{metrics:m}})=>[policy.label,`${m.uncomfortable_arrivals??'Unavailable'} / ${m.class_arrivals}`,`${fmt(m.early_class_uncomfortable_minutes)} / ${fmt(m.early_class_occupied_minutes)}`,`${fmt(m.comfortable_room_minutes)} / ${fmt(m.occupied_room_minutes)}`]));
  drawComparison(results);
}
function drawComparison(results) {
  const svg=$('#comparison-chart');
  svg.replaceChildren();
  const energyMax=Math.max(1,...results.map(r=>r.data.metrics.energy_kwh))*1.12;
  const violationMax=100;
  const x=v=>58+v/energyMax*418,y=v=>228-v/violationMax*191;
  for(let i=0;i<=4;i++) {
    const energy=energyMax*i/4,violation=violationMax*i/4;
    svg.append(svgNode('line',{x1:58,x2:476,y1:y(violation),y2:y(violation),stroke:'#e4e9f1'}));
    svg.append(svgNode('text',{x:49,y:y(violation)+4,'text-anchor':'end',class:'axis-label'},fmt(violation,0)));
    svg.append(svgNode('text',{x:x(energy),y:249,'text-anchor':'middle',class:'axis-label'},fmt(energy,0)));
  }
  svg.append(svgNode('text',{x:58,y:17,class:'axis-title'},'Occupied time comfortable (%) ↑'));
  svg.append(svgNode('text',{x:267,y:274,'text-anchor':'middle',class:'axis-title'},'Energy consumed (kWh) →'));
  results.forEach(({policy,data},i)=> {
    if(!data.metrics.occupied_room_minutes)return;
    const selected=policy.id===state.context.controller;
    const group=svgNode('g',{});
    group.append(svgNode('title',{},`${i+1}. ${policy.label}: ${fmt(data.metrics.energy_kwh)} kWh, ${fmt(data.metrics.comfort_pct)}% occupied time comfortable`));
    group.append(svgNode('circle',{cx:x(data.metrics.energy_kwh),cy:y(data.metrics.comfort_pct),r:selected?12:10,fill:selected?'#315bdd':policy.kind==='rl'?'#6574ac':'#8b9aad',stroke:'#fff','stroke-width':2}));
    group.append(svgNode('text',{x:x(data.metrics.energy_kwh),y:y(data.metrics.comfort_pct)+4,'text-anchor':'middle',fill:'#fff','font-size':11,'font-weight':600},String(i+1)));
    svg.append(group);
  });
}
function download() {
  const headers=['minute','interval_start_minute','air_1','air_2','proxy_1','proxy_2','students_1','students_2','ac_1','fan_1','ac_2','fan_2','power_kw','cap_kw','cumulative_energy_kwh','reward','class_start_1','class_start_2','arrival_proxy_1','arrival_proxy_2','arrival_violation_1','arrival_violation_2'];
  const rows=state.data.frames.map(f=>[f.minute,f.interval_start_minute??'',...f.temperature,...f.effective_temperature,...f.occupancy,...f.equipment.flat(),f.power_kw,f.cap_kw,f.energy_kwh,f.reward,...f.class_start.map(Number),...f.arrival_comfort_proxy,...f.arrival_violation]);
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
  $('#download').onclick=download;
  $('#controller').onchange=()=>inspectPolicy($('#controller').value);
  for(const id of ['scenario','seed'])$('#'+id).oninput=()=> {
    $('#pending').hidden=false;
    describeConditions();
    pause();
  };
  try {
    state.catalog=await json('/api/catalog');
    for(const [id,items] of [['scenario',state.catalog.scenarios],['controller',state.catalog.controllers],['training-model',state.catalog.controllers.filter(p=>p.kind==='rl')]]) {
      const s=$('#'+id);
      if(id==='training-model')s.disabled=items.length===0;
      for(const item of items) {
        const o=el('option',item.label);
        o.value=item.id;
        s.append(o);
      }
    }$('#scenario').value='restricted';
    $('#controller').value=state.catalog.controllers.filter(p=>p.algorithm==='dqn').sort((a,b)=>(b.selected_validation?.mean_return??-Infinity)-(a.selected_validation?.mean_return??-Infinity))[0]?.id||state.catalog.controllers.find(p=>p.algorithm==='q_learning')?.id||'timetable';
    const c=state.catalog.config;
    table('#parameters',['Parameter','Loaded value'],Object.entries(c).map(([k,v])=>[k,Array.isArray(v)?v.join(', '):v]));
    set('#reward-equation',`Loaded weights: energy ${c.energy_weight}, discomfort ${c.discomfort_weight}, worst room ${c.fairness_weight}, switching ${c.switching_weight}. Comfort band ${c.comfort_low_c}–${c.comfort_high_c}°C; fan offset ${c.fan_comfort_offset_c}°C.`);
    const study=state.catalog.study;
    if(study) {
      $('#saved-study').hidden=false;
      set('#study-context',`${study.partition} partition · ${study.scenario_count} scenarios × ${study.days_per_scenario} synthetic days = ${study.total_days} days per policy · reference checkpoint ${study.reference_checkpoint} · independent training seeds ${(study.training_seeds||[]).join(', ')}. Values below are saved results, unaffected by the demo seed.`);
      set('#study-status',study.status);
      table('#study-results',['Policy / recipe','Arrival discomfort (%)','First 15 min discomfort (%)','Whole-day comfort (%)','Energy (kWh/day)'],
        (study.rows||[]).map(r=>[r.label,...['mean_arrival_discomfort_pct','mean_class_start_violation_pct','mean_comfort_pct','mean_energy_kwh'].map(k=>r.score[k]===null?'Unavailable':fmt(r.score[k]))]));
    }
    training();
    if(state.catalog.warnings.length) {
      set('#model-warnings',state.catalog.warnings.join(' '));
      $('#model-warnings').hidden=false;
    }
    busy(false);
    await generate();
  }catch(e) {
    error(e);
    busy(false);
  }
}
init();
