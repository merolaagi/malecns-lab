'use strict';
const el=id=>document.getElementById('lc-'+id);
const labels={intact:'Intact circuit',no_plasticity:'No plasticity',silence_kc:'KC silenced',no_reward:'No rewards',no_dopamine:'No PAM gate',shuffled:'Shuffled PN → KC'};
let result=null,data=null,trialIndex=0,playing=false,last=0;
const palette={text:'#e7eee5',muted:'#99aa9c',a:'#c2e899',b:'#6fc7bc',reward:'#e9be75',line:'#2b372e'};
for(const [key,n] of [['rate',2],['temp',2],['retain',3]])el(key).oninput=()=>el(key+'-out').textContent=Number(el(key).value).toFixed(n);
const percent=v=>(v*100).toFixed(1)+'%';
function params(){return {condition:el('condition').value,rewarded_odor:el('odor').value,seed:Number(el('seed').value),train_trials:Number(el('train').value),probe_trials:Number(el('probe').value),learning_rate:Number(el('rate').value),temperature:Number(el('temp').value),retention:Number(el('retain').value),delay:Number(el('delay').value)};}
function stroke(ctx,points,color,width=1){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
function arena(){
 const c=el('arena'),g=c.getContext('2d'),w=c.width,h=c.height;g.clearRect(0,0,w,h);
 const r=result?.trials[trialIndex];g.font='15px sans-serif';g.fillStyle=palette.muted;
 if(!r){g.fillText('Run the experiment to replay odor choices.',40,60);return;}
 el('phase').textContent=r.phase;el('trial').textContent=`Trial ${r.trial} / ${result.trials.length}`;
 const aLeft=r.odor_a_side==='left',chosenLeft=r.choice_side==='left';
 stroke(g,[[500,385],[500,245],[260,145]],palette.line,68);stroke(g,[[500,245],[740,145]],palette.line,68);
 stroke(g,[[500,385],[500,245],[chosenLeft?260:740,145]],r.choice==='A'?palette.a:palette.b,3);
 for(const [x,odor,side] of [[260,aLeft?'A':'B','left'],[740,aLeft?'B':'A','right']]){
  g.beginPath();g.arc(x,120,60,0,Math.PI*2);g.fillStyle='#18221b';g.fill();g.strokeStyle=odor==='A'?palette.a:palette.b;g.lineWidth=2;g.stroke();
  g.fillStyle=odor==='A'?palette.a:palette.b;g.font='24px Georgia';g.textAlign='center';g.fillText('Odor '+odor,x,126);
  g.font='13px sans-serif';g.fillStyle=palette.muted;g.fillText(side,x,208);
 }
 const x=chosenLeft?325:675,y=174;g.save();g.translate(x,y);g.rotate(chosenLeft?-.95:.95);
 g.fillStyle=palette.text;g.beginPath();g.ellipse(0,0,9,21,0,0,Math.PI*2);g.fill();g.beginPath();g.arc(0,-24,7,0,Math.PI*2);g.fill();
 for(let i=-1;i<=1;i++){stroke(g,[[0,i*10],[-18,i*15+5]],palette.a,2);stroke(g,[[0,i*10],[18,i*15+5]],palette.a,2);}g.restore();
 g.textAlign='left';g.font='15px sans-serif';g.fillStyle=palette.text;g.fillText(`Choice: odor ${r.choice} · ${r.choice_side}`,35,35);
 g.textAlign='right';g.fillStyle=r.reward?palette.reward:palette.muted;g.fillText(r.training?(r.reward?'Sugar reward delivered':'No reward on this choice'):'Unrewarded test · learning frozen',w-35,35);
 g.textAlign='center';g.fillStyle=palette.muted;g.font='14px sans-serif';g.fillText(`P(A) ${percent(r.p_a)}    ·    P(B) ${percent(1-r.p_a)}`,500,430);
 g.textAlign='left';g.font='12px sans-serif';g.fillText('Locations change between trials. The path illustrates the sampled choice.',35,h-13);
 for(const b of el('phases').children)b.classList.toggle('selected',b.textContent===r.phase);
}
function chart(){
 const c=el('chart'),g=c.getContext('2d'),W=c.width,H=c.height;g.clearRect(0,0,W,H);if(!result)return;
 const x=t=>60+(t-1)/(result.trials.length-1)*(W-85),y=p=>H-50-p*(H-82);
 g.font='12px monospace';g.fillStyle=palette.muted;
 for(let i=0;i<=4;i++){const p=i/4;stroke(g,[[60,y(p)],[W-25,y(p)]],palette.line);g.fillText(String(i*25),22,y(p)+4);}
 let offset=0;
 result.summary.forEach((s,i)=>{if(s.training){g.fillStyle='#c2e89908';g.fillRect(x(offset+1),25,x(offset+s.trials)-x(offset+1),H-75);}if(offset)stroke(g,[[x(offset+1),25],[x(offset+1),H-50]],'#75857955');g.fillStyle=palette.muted;g.font='11px sans-serif';g.fillText(s.phase,x(offset+1)+5,16);offset+=s.trials;});
 stroke(g,result.trials.map(r=>[x(r.trial),y(r.p_a)]),palette.a,2.5);
 for(const r of result.trials){g.fillStyle='#e9be7566';g.beginPath();g.arc(x(r.trial),y(r.choice==='A'?1:0),2,0,Math.PI*2);g.fill();}
 stroke(g,[[x(trialIndex+1),25],[x(trialIndex+1),H-50]],palette.text,1);
 g.fillStyle=palette.muted;g.font='12px monospace';for(let i=0;i<=4;i++){const t=Math.round(1+i*(result.trials.length-1)/4);g.fillText(String(t),x(t)-8,H-28);}g.fillText('Trial number',W/2-35,H-7);g.fillText('P(A) %',7,15);
}
function populate(r){
 result=r;trialIndex=0;playing=true;last=0;el('scrub').max=r.trials.length-1;el('scrub').value=0;el('scrub').disabled=false;el('play').disabled=false;el('play').textContent='Pause';el('export').disabled=false;
 for(const [id,index] of [['before',0],['learned',2],['memory',3],['reverse',5]])el(id).textContent=percent(r.summary[index].mean_probability);
 const table=el('summary');table.replaceChildren();r.summary.forEach((s,i)=>{const row=document.createElement('tr');for(const val of [s.phase,s.target_odor,percent(s.mean_probability),`${Math.round(s.choice_fraction*s.trials)} / ${s.trials}`,s.reward_count,r.checkpoints[i].weights_changed?'Updated':'Frozen']){const td=document.createElement('td');td.textContent=String(val);row.append(td);}table.append(row);});
 el('finding').textContent=`${labels[r.parameters.condition]}: preference for the original rewarded odor changes from ${percent(r.summary[0].mean_probability)} to ${percent(r.summary[2].mean_probability)}. After reversal, preference for the newly rewarded odor is ${percent(r.summary[5].mean_probability)}. These are model probabilities; actual sampled choices are shown below.`;
 el('phases').replaceChildren();let offset=0;for(const s of r.summary){const start=offset,b=document.createElement('button');b.textContent=s.phase;b.onclick=()=>{trialIndex=start;playing=false;el('play').textContent='Play';el('scrub').value=start;arena();chart();};el('phases').append(b);offset+=s.trials;}arena();chart();
}
el('run').onclick=async()=>{el('run').disabled=true;playing=false;el('status').textContent='Running acquisition, probes, delay and reversal…';try{const response=await fetch('/api/learn',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(params())});const r=await response.json();if(!response.ok)throw Error(r.error);populate(r);el('status').textContent='Complete. Select a phase or compare an ablation.';}catch(e){el('status').textContent=e.message;}finally{el('run').disabled=false;}};
el('play').onclick=()=>{if(trialIndex===result.trials.length-1)trialIndex=0;playing=!playing;last=0;el('play').textContent=playing?'Pause':'Play';};
el('scrub').oninput=()=>{playing=false;trialIndex=Number(el('scrub').value);el('play').textContent='Play';arena();chart();};
el('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));a.download=`odor-learning-${result.parameters.condition}-seed${result.parameters.seed}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);};
function tick(now){if(playing&&result){if(!last)last=now;if(now-last>220){trialIndex=Math.min(trialIndex+1,result.trials.length-1);last=now;el('scrub').value=trialIndex;arena();chart();if(trialIndex===result.trials.length-1){playing=false;el('play').textContent='Replay';}}}requestAnimationFrame(tick);}
async function init(){arena();try{const response=await fetch('/api/learning-circuit');if(!response.ok)throw Error('Could not load learning circuit');data=await response.json();el('counts').textContent=`${data.nodes.length.toLocaleString()} cells · ${data.edges.length.toLocaleString()} measured edges`;const groups={};for(const n of data.nodes)groups[n.role]=(groups[n.role]||0)+1;el('structure').textContent=`${groups.PN} projection neurons, ${groups.KC.toLocaleString()} Kenyon cells, ${groups.MBON} output neurons and ${groups.PAM} PAM dopamine neurons. Anatomical synapse counts constrain edge support and initial weight scaling.`;el('run').click();
 const br=await fetch('/api/learning-benchmark');if(!br.ok)throw Error('Saved benchmark unavailable');const bench=await br.json();for(const condition of bench.conditions){const runs=bench.runs.filter(r=>r.parameters.condition===condition);const tr=document.createElement('tr');const vals=[labels[condition],...[2,3,5].map(index=>percent(runs.reduce((s,r)=>s+r.summary[index].mean_probability,0)/runs.length))];for(const v of vals){const td=document.createElement('td');td.textContent=v;tr.append(td);}el('benchmark').append(tr);}}
 catch(e){el('status').textContent=e.message;}}
init();requestAnimationFrame(tick);
