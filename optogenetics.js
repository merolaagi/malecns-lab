'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH'];
let meta = null;

function ctx(id) { const cv = $(id), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1; cv.width = w * d; cv.height = h * d; const g = cv.getContext('2d'); g.scale(d, d); g.font = '11px sans-serif'; return [g, w, h]; }
const tagClass = r => r === 'match' || r === 'proxy match' ? 'match' : ['opposite', 'mismatch', 'proxy mismatch', 'no TTM activity'].includes(r) ? 'opposite' : 'other';

async function loadMeta() {
  meta = await (await fetch('/api/optogenetics-summary')).json();
  $('experiment').innerHTML = Object.entries(meta.experiments).map(([k, e]) =>
    `<option value="${esc(k)}"${meta.available[k] ? '' : ' disabled'}>${esc(k)}${meta.available[k] ? ` (${meta.available[k]} cells)` : ' (not in circuit yet)'}</option>`).join('');
  $('condition').innerHTML = Object.entries(meta.conditions).map(([k, v]) => `<option value="${esc(k)}">${esc(v)}</option>`).join('');
  $('sourcenote').textContent = meta.source === 'optogenetics' ? 'Circuit: the optogenetics build.' : 'Circuit: the original steering circuit (DNa01, DNa02, DNg13). Run build_optogenetics.py for MDN, P9 and the giant fiber.';
  showExpect();
}
function showExpect() { const e = meta.experiments[$('experiment').value]; $('expect').innerHTML = `Real flies: <b>${esc(e.expect)}</b>${e.reference ? ` <span class="muted">(${esc(e.reference)})</span>` : ''}`; }
$('experiment').addEventListener('change', showExpect);
$('light').addEventListener('input', () => { $('lightv').textContent = (+$('light').value).toFixed(1); });
$('resting').addEventListener('input', () => { $('restv').textContent = (+$('resting').value).toFixed(2) + ' of threshold'; });

function drawPath(trace) {
  const [g, w, h] = ctx('path');
  const xs = trace.map(t => t.x), ys = trace.map(t => t.y);
  const span = Math.max(4, ...xs.map(Math.abs), ...ys.map(Math.abs)) * 1.15;
  const k = Math.min(w, h) / (2 * span), X = x => w / 2 + x * k, Y = y => h / 2 - y * k;
  g.strokeStyle = '#1e2a21';
  for (let v = -Math.floor(span); v <= span; v++) { g.beginPath(); g.moveTo(X(v), 0); g.lineTo(X(v), h); g.stroke(); g.beginPath(); g.moveTo(0, Y(v)); g.lineTo(w, Y(v)); g.stroke(); }
  trace.forEach((t, i) => { if (!i) return; const p = trace[i - 1]; g.strokeStyle = t.t >= 0.5 ? '#c2e899' : '#55635a'; g.lineWidth = 2; g.beginPath(); g.moveTo(X(p.x), Y(p.y)); g.lineTo(X(t.x), Y(t.y)); g.stroke(); });
  const last = trace[trace.length - 1];
  g.fillStyle = '#f2c46f'; g.save(); g.translate(X(last.x), Y(last.y)); g.rotate(-last.heading);
  g.beginPath(); g.moveTo(10, 0); g.lineTo(-6, 5); g.lineTo(-6, -5); g.closePath(); g.fill(); g.restore();
  g.fillStyle = '#99aa9c'; g.lineWidth = 1; g.fillText('start: facing right; 1 grid square = 1 mm', 8, h - 8);
}
function drawLegs(legs) {
  const [g, w, h] = ctx('legs'); const max = Math.max(5, ...Object.values(legs));
  const bw = (w - 60) / 6;
  LEGS.forEach((leg, i) => {
    const v = legs[leg] || 0, bh = (h - 40) * v / max, x = 40 + i * bw;
    g.fillStyle = leg[0] === 'L' ? '#6fc7bc' : '#c2e899'; g.fillRect(x + 6, h - 24 - bh, bw - 12, bh);
    g.fillStyle = '#99aa9c'; g.fillText(leg, x + bw / 2 - 7, h - 8); g.fillText(v.toFixed(1), x + bw / 2 - 9, h - 28 - bh);
  });
  g.fillText('Hz', 6, 16);
}
$('run').addEventListener('click', async () => {
  $('run').disabled = true; $('status').textContent = 'Simulating…'; $('status').className = 'wb-status';
  try {
    const body = { experiment: $('experiment').value, side: $('side').value, light: +$('light').value, duration: +$('duration').value, seed: +$('seed').value, condition: $('condition').value, resting: +$('resting').value, gap_junctions: $('gaps').checked };
    const r = await fetch('/api/optogenetics', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const d = await r.json(); if (!r.ok) throw new Error(d.error);
    const m = d.metrics;
    drawPath(d.trace); drawLegs(m.leg_motor_hz);
    const turn = m.turn_deg, dir = Math.abs(turn) < 20 ? 'no clear turn' : turn > 0 ? 'turned left' : 'turned right';
    const toward = body.side === 'both' || Math.abs(turn) < 20 ? '' : ((turn < 0) === (body.side === 'right') ? ', toward the lit side' : ', away from the lit side');
    let h = `<b>${d.lit_cells} cells lit</b>, firing ${m.target_hz.toFixed(0)} Hz. The model ${dir} (${turn >= 0 ? '+' : ''}${turn.toFixed(0)}°)${toward}, at ${m.speed_mm_s.toFixed(1)} mm/s.`;
    if (m.lit_side_share != null) h += ` ${Math.round(m.lit_side_share * 100)}% of leg motor activity is on the lit side.`;
    if (m.ttm_cells) h += ` Jump-muscle motor neuron: ${m.ttm_hz.toFixed(1)} Hz${m.ttm_latency_s != null ? `, first spike ${(m.ttm_latency_s * 1000).toFixed(0)} ms after light` : ', silent'}.`;
    $('verdict').innerHTML = h;
    const muscles = Object.entries(m.muscles).sort((a, b) => b[1] - a[1]).slice(0, 12);
    $('muscles').innerHTML = muscles.length ? '<table class="bench"><tr><th>Leg and muscle group</th><th></th><th>Motor neuron rate</th></tr>' + muscles.map(([k, v]) => `<tr><td>${esc(k.replace('_', ' '))}</td><td></td><td>${v.toFixed(1)} Hz</td></tr>`).join('') + '</table>' : '<p class="hint">No leg muscle group above 0.5 Hz.</p>';
    $('status').textContent = '';
  } catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; }
  $('run').disabled = false;
});

async function score() {
  try {
    const r = await fetch('/api/optogenetics-benchmark'); if (!r.ok) throw new Error('No scorecard yet: python3 optogenetics.py');
    const b = await r.json();
    const variants = b.physiology_variants ? Object.keys(b.physiology_variants) : null;
    if (!variants) throw new Error('This scorecard predates the physiology variants; rerun python3 optogenetics.py');
    let h = `<table class="bench"><tr><th>Cell type</th><th>Real flies</th>${variants.map(v => `<th>${esc(b.physiology_variants[v])}</th>`).join('')}</tr>`;
    for (const [name, row] of Object.entries(b.experiments)) {
      if (!row.available) { h += `<tr><td>${esc(name)}</td><td colspan="${variants.length + 1}" class="muted">not in this circuit yet</td></tr>`; continue; }
      h += `<tr><td>${esc(name)}</td><td>${esc(row.expect)}</td>` + variants.map(v => {
        const c = row.physiology[v];
        return `<td><span class="tag ${tagClass(c.intact.result)}">${esc(c.intact.result)}</span><br><span class="muted" style="font-size:10px">shuffled: ${esc(c.shuffled.result)}; sham: ${esc(c.sham.result)}</span></td>`;
      }).join('') + '</tr>';
      const last = row.physiology[variants[variants.length - 1]].intact, first = row.physiology[variants[0]].intact;
      h += `<tr><td></td><td colspan="${variants.length + 1}" class="muted" style="text-align:left">Connectome only: ${esc(first.detail)}${last.detail !== first.detail ? `<br>With both: ${esc(last.detail)}` : ''}</td></tr>`;
    }
    $('score').innerHTML = h + '</table>';
  } catch (e) { $('score').innerHTML = `<p class="hint">${esc(e.message)}</p>`; }
}
// --- physics body --------------------------------------------------------------------------------------------
function drawBody(trace, base) {
  const [g, w, h] = ctx('bpath');
  const pts = trace.concat(base || []);
  const x0 = trace[0].x, y0 = trace[0].y;
  const span = Math.max(3, ...pts.map(p => Math.abs(p.x - x0)), ...pts.map(p => Math.abs(p.y - y0))) * 1.15;
  const k = Math.min(w, h) / (2 * span), X = x => w / 2 + (x - x0) * k, Y = y => h / 2 - (y - y0) * k;
  g.strokeStyle = '#e9e9e7';
  const grid = span > 12 ? 5 : 1;
  for (let v = -Math.floor(span / grid) * grid; v <= span; v += grid) { g.beginPath(); g.moveTo(X(x0 + v), 0); g.lineTo(X(x0 + v), h); g.stroke(); g.beginPath(); g.moveTo(0, Y(y0 + v)); g.lineTo(w, Y(y0 + v)); g.stroke(); }
  for (const [tr, colour, width] of [[base, '#9a9a9a', 1.5], [trace, '#2f5fd0', 2.4]]) {
    if (!tr) continue; g.strokeStyle = colour; g.lineWidth = width; g.beginPath();
    tr.forEach((p, i) => i ? g.lineTo(X(p.x), Y(p.y)) : g.moveTo(X(p.x), Y(p.y))); g.stroke();
  }
  const on = trace.find(p => p.t >= 0.5); if (on) { g.fillStyle = '#e9a23b'; g.beginPath(); g.arc(X(on.x), Y(on.y), 4, 0, 7); g.fill(); }
  g.fillStyle = '#6e6e6e'; g.lineWidth = 1; g.fillText(`grid ${grid} mm`, 8, h - 8);
}
function drawLegs2(legs) {
  const [g, w, h] = ctx('blegs'); if (!legs || !legs.length) return;
  const last = legs[legs.length - 1], names = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH'], bw = (w - 60) / 6;
  names.forEach((n, i) => {
    const a = last.amp[i], d = last.dir[i], bh = (h - 50) * a, x = 40 + i * bw;
    g.fillStyle = d >= 0 ? '#2f5fd0' : '#d0524a'; g.fillRect(x + 8, h - 26 - bh, bw - 16, bh);
    g.fillStyle = '#6e6e6e'; g.fillText(n, x + bw / 2 - 7, h - 8);
    g.fillText(`${(last.freq[i] || 0).toFixed(0)} Hz`, x + bw / 2 - 12, h - 30 - bh);
  });
}
let bodyBench = null;
function showEffect(e, setup) {
  $('beffect').innerHTML = `<p class="note">Setup: ${esc(setup)}. Against the comparison run: turns ${e.yaw_deg >= 0 ? '+' : ''}${e.yaw_deg.toFixed(0)}° ${Math.abs(e.yaw_deg) < 1 ? '' : e.yaw_deg > 0 ? '(left)' : '(right)'}, forward ${e.forward_mm >= 0 ? '+' : ''}${e.forward_mm.toFixed(2)} mm (travel during light ${e.forward_mm_lit.toFixed(2)} mm), sideways ${e.lateral_mm >= 0 ? '+' : ''}${e.lateral_mm.toFixed(2)} mm, rises ${e.rise_mm >= 0 ? '+' : ''}${e.rise_mm.toFixed(3)} mm.</p>`;
}
function showSaved() {
  if (!bodyBench) return;
  const name = $('bexp').value, saved = bodyBench.traces[`${name}|${$('bside').value}`];
  if (saved) { drawBody(saved.trace, bodyBench.comparisons[name]); drawLegs2(saved.legs); }
}
async function bodyScore() {
  try {
    const r = await fetch('/api/body-benchmark'); if (!r.ok) throw new Error('No physics scorecard yet: .venv/bin/python bodysim.py (needs flygym)');
    const b = bodyBench = await r.json();
    if (!b.ablations) throw new Error('This physics scorecard predates the closed-loop model; rerun .venv/bin/python bodysim.py');
    $('bexp').innerHTML = Object.keys(b.experiments).map(k => `<option value="${esc(k)}">${esc(k)}</option>`).join('');
    $('babl').innerHTML = Object.entries(b.ablations).map(([k, v]) => `<option value="${esc(k)}">${esc(v)}</option>`).join('');
    const abl = Object.keys(b.ablations);
    let h = `<table class="bench"><tr><th>Cell type</th><th>Real flies</th><th>Setup</th>${abl.map(a => `<th>${esc(b.ablations[a])}</th>`).join('')}</tr>`;
    for (const [name, row] of Object.entries(b.experiments)) {
      h += `<tr><td>${esc(name)}</td><td>${esc(row.expect)}</td><td>${esc(row.setup)}</td>` + abl.map(a => {
        const c = row.ablations[a]; if (!c) return '<td>–</td>';
        const ctl = c.shuffled ? `<br><span class="muted" style="font-size:10px">shuffled: ${esc(c.shuffled.result)}; sham: ${esc(c.sham.result)}</span>` : '';
        return `<td><span class="tag ${tagClass(c.intact.result)}">${esc(c.intact.result)}</span>${ctl}</td>`;
      }).join('') + '</tr>';
      h += `<tr><td></td><td colspan="${abl.length + 2}" class="muted" style="text-align:left">Everything on: ${esc(row.ablations.full.intact.detail)}</td></tr>`;
    }
    $('bscore').innerHTML = `<div class="tw">${h}</table></div>` + `<p class="hint">${b.mirror_added == null ? 'This circuit was built before left-right completion; rebuild it with build_optogenetics.py.' : `Left-right completion added ${b.mirror_added} cells.`} Pruning to exact counts per side would drop ${b.matching ? b.matching.dropped : '–'} cells (the “pruned” column). Verdicts use the mean of ${b.seeds.length} seeds; the detail line says how many seeds agree.</p>`;
    showSaved();
  } catch (e) { $('bscore').innerHTML = `<p class="hint">${esc(e.message)}</p>`; }
}
for (const id of ['bexp', 'bside']) $(id).addEventListener('change', showSaved);
$('brun').addEventListener('click', async () => {
  $('brun').disabled = true; $('bstatus').textContent = 'Simulating circuit and body in closed loop (10–30 s)…'; $('bstatus').className = 'wb-status';
  try {
    const body = { experiment: $('bexp').value, side: $('bside').value, ablation: $('babl').value, seed: 1, condition: $('condition').value };
    const r = await fetch('/api/body-run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const d = await r.json(); if (!r.ok) throw new Error(d.error);
    drawBody(d.trace, d.comparison); drawLegs2(d.legs); showEffect(d.effect, d.setup); $('bstatus').textContent = '';
  } catch (e) { $('bstatus').textContent = e.message; $('bstatus').className = 'wb-status err'; }
  $('brun').disabled = false;
});
async function diagnosis() {
  try {
    const r = await fetch('/api/diagnosis'); if (!r.ok) throw new Error('No diagnosis yet: .venv/bin/python diagnose.py');
    const d = await r.json();
    const cell = x => x.mean_amp < 0.05 ? '<span class="muted">silent</span>' : `${x.mean_amp.toFixed(2)} <span class="muted" style="font-size:11px">${x.rear_dir >= 0 ? 'fwd' : 'back'} ${Math.abs(x.rear_dir).toFixed(2)} / front ${x.front_dir >= 0 ? '+' : ''}${x.front_dir.toFixed(2)}</span>`;
    let h = `<h3 style="font-size:13px;margin:6px 0">1. The circuit alone: how strongly the legs would step, and which way, by light strength</h3>
      <div class="tw"><table class="bench"><tr><th>Glutamate</th><th>Cell type, side</th>${d.lights.map(l => `<th>light ${l}</th>`).join('')}</tr>`;
    for (const glu of Object.keys(d.glutamate)) for (const [e, side] of d.cases) {
      const row = d.stage1.filter(x => x.glutamate === glu && x.experiment === e && x.side === side);
      if (row.length) h += `<tr><td>${esc(glu)}</td><td>${esc(e)}, ${esc(side)}</td>${row.map(x => `<td>${cell(x)}</td>`).join('')}</tr>`;
    }
    h += '</table></div><p class="hint">Mean step amplitude over the six legs (0 to 1), the middle and hind legs\' direction (fwd or back) and the front legs\' direction (−1 backward to +1 forward), in the second half of the light.</p>';
    if (d.stage2) {
      h += `<h3 style="font-size:13px;margin:14px 0 6px">2. In the body, at the light where bilateral P9 first makes the legs step (${Object.entries(d.chosen_light).map(([g, l]) => `${esc(g)} glutamate: ${l}`).join('; ')})</h3>
        <div class="tw"><table class="bench"><tr><th>Front legs</th><th>Cell type</th><th>Measured wiring</th><th>Shuffled</th><th>Sham</th></tr>`;
      const by = {}; d.stage2.forEach(x => { (by[`${x.front}|${x.experiment}`] ||= {})[x.condition] = x; });
      for (const [k, c] of Object.entries(by)) {
        const [front, e] = k.split('|');
        h += `<tr><td>${esc(front === 'own' ? 'own direction' : 'follow the others')}</td><td>${esc(e)}</td>${['intact', 'shuffled', 'sham'].map(cn => c[cn] ? `<td><span class="tag ${tagClass(c[cn].result)}">${esc(c[cn].result)}</span></td>` : '<td>–</td>').join('')}</tr>`;
        h += `<tr><td></td><td colspan="4" class="muted" style="text-align:left">${esc(c.intact.detail)}</td></tr>`;
      }
      h += '</table></div>';
    }
    if (d.stage3) {
      h += `<h3 style="font-size:13px;margin:14px 0 6px">3. Coordination: bilateral P9, front legs following, how far the body walks in 1.5 s</h3>
        <table class="bench"><tr><th>Rhythm</th><th></th><th>Forward travel</th><th>Turn</th></tr>${d.stage3.map(x => `<tr><td>${esc(x.text)}</td><td></td><td>${x.forward_mm.toFixed(2)} mm</td><td>${x.yaw_deg >= 0 ? '+' : ''}${x.yaw_deg.toFixed(0)}°</td></tr>`).join('')}</table>`;
    }
    $('diag').innerHTML = h;
  } catch (e) { $('diag').innerHTML = `<p class="hint">${esc(e.message)}</p>`; }
}
async function coordination() {
  try {
    const r = await fetch('/api/coordination'); if (!r.ok) throw new Error('No coordination results yet: .venv/bin/python coordination.py');
    const d = await r.json();
    const circuits = Object.keys(d.circuits);
    let h = `<p class="note">${circuits.map(c => `${esc(c)} circuit: ${d.circuits[c].neurons.toLocaleString()} neurons${d.circuits[c].added ? ' (added: ' + Object.entries(d.circuits[c].added).map(([k, v]) => `${v} ${esc(k)}`).join(', ') + ')' : ''}`).join(' · ')}${circuits.includes('coordination') ? '' : ' · The coordination circuit has not been built yet: build_coordination.py.'}</p>`;
    h += `<h3 style="font-size:13px;margin:10px 0 6px">1. The circuit alone: does the drive reach every leg, evenly?</h3><div class="tw"><table class="bench"><tr><th>Circuit</th><th>Light on</th><th>Light</th><th>Front-leg step size</th><th>Middle and hind</th><th>Front direction</th><th>Middle and hind direction</th><th>Frequency spread</th></tr>`;
    for (const x of d.stage_a) h += `<tr><td>${esc(x.circuit)}</td><td>${esc(x.experiment)}, ${esc(x.side)}</td><td>${x.light}</td><td>${x.front_amp.toFixed(2)}</td><td>${x.rear_amp.toFixed(2)}</td><td>${x.front_dir >= 0 ? '+' : ''}${x.front_dir.toFixed(2)}</td><td>${x.rear_dir >= 0 ? '+' : ''}${x.rear_dir.toFixed(2)}</td><td>${x.frequency_spread == null ? '–' : x.frequency_spread.toFixed(2)}</td></tr>`;
    h += '</table></div><p class="hint">Step size 0 to 1; direction −1 backward to +1 forward; frequency spread is the standard deviation over the mean of the stepping legs\' frequencies (0 = all legs at one speed).</p>';
    const names = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH'];
    for (const [c, K] of Object.entries(d.coupling_matrices)) {
      const max = Math.max(...K.flat());
      h += `<h3 style="font-size:13px;margin:12px 0 6px">Measured coupling, ${esc(c)} circuit (row: receiving leg; mean = uniform strength)</h3><table class="bench" style="width:auto"><tr><th></th>${names.map(n => `<th>${n}</th>`).join('')}</tr>${K.map((row, i) => `<tr><td>${names[i]}</td>${row.map(v => `<td style="background:rgba(47,95,208,${(0.08 + 0.5 * v / max).toFixed(2)})">${v.toFixed(1)}</td>`).join('')}</tr>`).join('')}</table>`;
    }
    if (d.stage_b) {
      h += `<h3 style="font-size:13px;margin:14px 0 6px">2. In the body (light: ${Object.entries(d.chosen_light).map(([c, l]) => `${esc(c)} ${l}`).join(', ')})</h3><div class="tw"><table class="bench"><tr><th>Circuit</th><th>Coupling</th><th>Cell type</th><th>Measured wiring</th><th>Shuffled</th><th>Sham</th></tr>`;
      const by = {}; d.stage_b.forEach(x => { (by[`${x.circuit}|${x.coupling}|${x.experiment}`] ||= {})[x.condition] = x; });
      for (const [k, c] of Object.entries(by)) {
        const [circ, coup, e] = k.split('|');
        h += `<tr><td>${esc(circ)}</td><td>${esc(coup)}</td><td>${esc(e)}</td>${['intact', 'shuffled', 'sham'].map(cn => c[cn] ? `<td><span class="tag ${tagClass(c[cn].result)}">${esc(c[cn].result)}</span></td>` : '<td>–</td>').join('')}</tr>`;
        h += `<tr><td></td><td colspan="5" class="muted" style="text-align:left">${esc(c.intact.detail)}</td></tr>`;
      }
      h += '</table></div>';
    }
    $('coord').innerHTML = h;
  } catch (e) { $('coord').innerHTML = `<p class="hint">${esc(e.message)}</p>`; }
}
loadMeta(); score(); bodyScore(); diagnosis(); coordination();
