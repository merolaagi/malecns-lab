'use strict';
const $ = id => document.getElementById(id);
const names={intact:'Intact subset',silence_vnc:'VNC silenced',silence_left_dn:'Left DN silenced',no_feedback:'Feedback removed',no_stimulus:'No stimulus',shuffled:'Shuffled wiring'};
const legNames=['LF','LM','LH','RF','RM','RH'];
let circuit,run,playing=false,frame=0,previous=0,history=[];
const colors={dn:'#c2e899',vnc:'#6fc7bc',mn:'#e9be75',grid:'#243229',muted:'#99aa9c',text:'#e7eee5'};
for(const key of ['drive','bias','gain','feedback']) $(key).oninput=()=>$(key+'Value').textContent=Number($(key).value).toFixed(1);
$('legbars').innerHTML=legNames.map((l,i)=>`<div class="bar"><span>${l}</span><div class="track"><div class="fill" id="bar${i}" style="width:0"></div></div><span id="hz${i}">—</span></div>`).join('');
function parameters(){const p={};for(const k of ['mode','condition'])p[k]=$(k).value;for(const k of ['drive','bias','gain','feedback','duration','seed'])p[k]=Number($(k).value);return p;}
function line(ctx,points,color,width=1){if(!points.length)return;ctx.beginPath();ctx.moveTo(...points[0]);for(const p of points.slice(1))ctx.lineTo(...p);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
function arena(){
 const canvas=$('arena'),ctx=canvas.getContext('2d'),W=canvas.width,H=canvas.height;
 ctx.clearRect(0,0,W,H);
 const rows=run?.trace||[], current=rows[frame]||{x:0,y:0,heading:0,phases:[0,Math.PI,0,Math.PI,0,Math.PI],strides:[0,0,0,0,0,0],motor:[0,0,0,0,0,0],t:0};
 let extent=18;for(const q of rows)extent=Math.max(extent,Math.abs(q.x)+4,Math.abs(q.y)+4);
 const scale=Math.min(W,H)/(2*extent),point=(x,y)=>[W/2+x*scale,H/2-y*scale];
 const spacing=extent>35?10:5;
 for(let n=-Math.ceil(extent*W/H/spacing)*spacing;n<=extent*W/H;n+=spacing){const [x]=point(n,0);line(ctx,[[x,0],[x,H]],colors.grid);}
 for(let n=-Math.ceil(extent/spacing)*spacing;n<=extent;n+=spacing){const [,y]=point(0,n);line(ctx,[[0,y],[W,y]],colors.grid);}
 ctx.fillStyle=colors.muted;ctx.font='13px monospace';ctx.fillText(`Grid ${spacing} mm · top view`,24,H-24);ctx.fillText('x →    y ↑',W-135,H-24);
 if(run?.parameters.mode==='target'){
  const [tx,ty]=point(12,6);ctx.beginPath();ctx.arc(tx,ty,9,0,2*Math.PI);ctx.strokeStyle=colors.dn;ctx.lineWidth=2;ctx.stroke();ctx.fillStyle=colors.dn;ctx.font='13px sans-serif';ctx.fillText('Synthetic target cue',tx+16,ty+4);
 }
 line(ctx,rows.slice(0,frame+1).map(r=>point(r.x,r.y)),'#6fc7bc88',2);
 const [x,y]=point(current.x,current.y);
 ctx.save();ctx.translate(x,y);ctx.rotate(-current.heading);const size=Math.max(12,scale*1.1);ctx.scale(size,size);
 // Six phase-animated legs are a visual proxy, not an inferred muscle simulation.
 for(let k=0;k<6;k++){
  const side=k<3?-1:1,index=k%3,hipX=.55-index*.55;
  const swing=Math.sin(current.phases[k])*.5*current.strides[k];
  const knee=[hipX+.2+swing,side*.88],foot=[hipX+.45+swing,side*1.5];
  line(ctx,[[hipX,side*.25],knee,foot],Math.sin(current.phases[k])<=0?colors.dn:'#6c806f',.075);
 }
 ctx.fillStyle='#c2e899';ctx.beginPath();ctx.ellipse(-.65,0,.8,.38,0,0,2*Math.PI);ctx.fill();
 ctx.fillStyle='#e7eee5';ctx.beginPath();ctx.ellipse(.15,0,.52,.35,0,0,2*Math.PI);ctx.fill();
 ctx.fillStyle='#c2e899';ctx.beginPath();ctx.ellipse(.72,0,.28,.35,0,0,2*Math.PI);ctx.fill();
 ctx.fillStyle='#132018';for(const a of [-.22,.22]){ctx.beginPath();ctx.arc(.84,a,.09,0,Math.PI*2);ctx.fill();}
 line(ctx,[[.91,-.12],[1.22,-.35]],colors.dn,.045);line(ctx,[[.91,.12],[1.22,.35]],colors.dn,.045);ctx.restore();
 ctx.fillStyle=colors.muted;ctx.font='13px sans-serif';ctx.fillText('Engineered gait · neural motor-rate readout',24,30);
 $('time').textContent=current.t.toFixed(2)+' s';
 for(let k=0;k<6;k++){$('bar'+k).style.width=Math.min(100,current.motor[k]*2)+'%';$('hz'+k).textContent=current.motor[k].toFixed(0);}
}
function activity(){
 const c=$('activity'),ctx=c.getContext('2d'),W=c.width,H=c.height,pad={l:65,r:25,t:25,b:50};ctx.clearRect(0,0,W,H);
 if(!run){ctx.fillStyle=colors.muted;ctx.font='15px sans-serif';ctx.fillText('Run an experiment to inspect the circuit response.',30,65);return;}
 const rows=run.trace,max=Math.max(10,...rows.flatMap(r=>[r.dn_hz,r.vnc_hz,r.motor_hz]))*1.1;
 const x=t=>pad.l+t/run.parameters.duration*(W-pad.l-pad.r),y=v=>H-pad.b-v/max*(H-pad.t-pad.b);
 ctx.font='12px monospace';ctx.fillStyle=colors.muted;
 for(let i=0;i<=4;i++){const val=max*i/4;line(ctx,[[pad.l,y(val)],[W-pad.r,y(val)]],colors.grid);ctx.fillText(val.toFixed(0),18,y(val)+4);}
 for(let i=0;i<=4;i++){const t=run.parameters.duration*i/4;ctx.fillText(t.toFixed(1),x(t)-9,H-25);}
 ctx.fillText('Hz',18,16);ctx.fillText('Simulation time (s)',W/2-60,H-5);
 for(const [key,color] of [['dn_hz',colors.dn],['vnc_hz',colors.vnc],['motor_hz',colors.mn]])line(ctx,rows.map(r=>[x(r.t),y(r[key])]),color,2.5);
 line(ctx,[[x(rows[frame].t),pad.t],[x(rows[frame].t),H-pad.b]],'#e7eee555',1);
}
function showResult(result){
 run=result;frame=0;playing=true;previous=0;$('play').disabled=false;$('play').textContent='Pause';$('export').disabled=false;$('scrub').disabled=false;$('scrub').max=run.trace.length-1;$('scrub').value=0;
 const m=run.metrics;$('path').textContent=m.path_mm.toFixed(2)+' mm';$('motor').textContent=m.motor_mean_hz.toFixed(1)+' Hz';$('active').textContent=m.active_neurons+' / '+circuit.nodes.length;$('spikes').textContent=m.spikes.toLocaleString();
 history.unshift(run);const tbody=$('history');tbody.replaceChildren();
 for(const r of history){const tr=document.createElement('tr'),p=r.parameters;for(const text of [names[p.condition],`${p.mode} · ${p.duration}s · seed ${p.seed} · drive ${p.drive} · bias ${p.bias} · gain ${p.gain} · fb ${p.feedback}`,r.metrics.motor_mean_hz.toFixed(2),r.metrics.path_mm.toFixed(2),r.metrics.turn_deg.toFixed(1),(100*r.metrics.clipped_motor_fraction).toFixed(0)+'%']){const td=document.createElement('td');td.textContent=text;tr.append(td);}tbody.append(tr);}
 arena();activity();
}
$('run').onclick=async()=>{
 const button=$('run');button.disabled=true;$('status').textContent='Simulating neural activity and body feedback…';playing=false;
 try{const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(parameters())});const result=await response.json();if(!response.ok)throw Error(result.error||'Simulation failed');showResult(result);$('status').textContent='Experiment complete. Replay or change one condition.';}
 catch(error){$('status').textContent=error.message;}finally{button.disabled=false;}
};
$('play').onclick=()=>{if(!run)return;if(frame>=run.trace.length-1)frame=0;playing=!playing;previous=0;$('play').textContent=playing?'Pause':'Play';};
$('scrub').oninput=()=>{frame=Number($('scrub').value);playing=false;$('play').textContent='Play';arena();activity();};
$('export').onclick=()=>{if(!run)return;const result={...run,provenance:{dataset:circuit.dataset,files:circuit.files,selection:circuit.selection},neuron_order:circuit.nodes.map(n=>({bodyId:n.bodyId,type:n.type,instance:n.instance}))};const blob=new Blob([JSON.stringify(result)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=`malecns-${run.parameters.condition}-seed${run.parameters.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
function animate(now){if(playing&&run){if(!previous)previous=now;const delta=now-previous;if(delta>=20){frame=Math.min(run.trace.length-1,frame+Math.floor(delta/20));previous=now;$('scrub').value=frame;arena();activity();if(frame===run.trace.length-1){playing=false;$('play').textContent='Replay';}}}requestAnimationFrame(animate);}
async function init(){arena();activity();try{const response=await fetch('/api/circuit');if(!response.ok)throw Error('Unable to load measured circuit');circuit=await response.json();$('counts').textContent=`${circuit.nodes.length.toLocaleString()} neurons · ${circuit.edges.length.toLocaleString()} measured edges`;$('run').click();}catch(e){$('status').textContent=e.message;$('run').disabled=true;}}
init();requestAnimationFrame(animate);
