'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
let data = null;
function ctx(id) { const cv = $(id), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1; cv.width = w * d; cv.height = h * d; const g = cv.getContext('2d'); g.scale(d, d); g.font = '11px sans-serif'; return [g, w, h]; }
function axes(g, w, h, lo, hi, xlabels) {
  const Y = v => 14 + (hi - v) / (hi - lo) * (h - 40);
  g.strokeStyle = '#2b372e'; g.fillStyle = '#99aa9c';
  for (const v of [lo, 0, hi]) { g.beginPath(); g.moveTo(48, Y(v)); g.lineTo(w - 12, Y(v)); g.stroke(); g.fillText(v.toFixed(2), 4, Y(v) + 4); }
  xlabels.forEach(([x, label]) => g.fillText(label, x - 10, h - 8));
  return Y;
}
function drawForget() {
  const [g, w, h] = ctx('forget'); if (!data) return;
  const series = [['punishment|massed', '#f28b82', [5, 4]], ['punishment|spaced', '#f28b82', []], ['reward|massed', '#6fc7bc', [5, 4]], ['reward|spaced', '#6fc7bc', []]];
  const all = series.flatMap(([k]) => data.forgetting[k].curve.map(c => c.preference));
  const hi = Math.max(0.05, ...all), lo = Math.min(-0.05, ...all);
  const hours = data.forgetting['punishment|massed'].curve.map(c => c.hours);
  const X = i => 56 + i * (w - 80) / (hours.length - 1);
  const Y = axes(g, w, h, lo, hi, hours.map((hr, i) => [X(i), hr + ' h']));
  series.forEach(([key, colour, dash], row) => {
    g.strokeStyle = colour; g.setLineDash(dash); g.lineWidth = 1.8; g.beginPath();
    data.forgetting[key].curve.forEach((c, i) => i ? g.lineTo(X(i), Y(c.preference)) : g.moveTo(X(i), Y(c.preference)));
    g.stroke(); g.setLineDash([]); g.lineWidth = 1;
    g.fillStyle = colour; g.fillText(key.replace('|', ', '), w - 130, 18 + row * 14);
  });
}
function drawExtinct() {
  const [g, w, h] = ctx('extinct'); if (!data) return;
  const trace = data.extinction.intact.trace;
  const values = trace.flatMap(t => [t.preference, t.control]);
  const hi = Math.max(0.05, ...values), lo = Math.min(-0.05, ...values);
  const X = i => 56 + i * (w - 80) / (trace.length - 1);
  const Y = axes(g, w, h, lo, hi, [[X(0), 'trained'], [X(trace.length - 1), trace[trace.length - 1].step]]);
  for (const [field, colour] of [['control', '#99aa9c'], ['preference', '#c2e899']]) {
    g.strokeStyle = colour; g.lineWidth = 1.8; g.beginPath();
    trace.forEach((t, i) => i ? g.lineTo(X(i), Y(t[field])) : g.moveTo(X(i), Y(t[field]))); g.stroke(); g.lineWidth = 1;
  }
}
function tables() {
  const s = data.summary;
  $('summary').innerHTML = `<p class="note">${Object.entries(s.cells).map(([k, v]) => `${v.toLocaleString()} ${k}`).join(' · ')}. ${s.compartments.length} compartments: ${s.compartments.map(esc).join(', ')}. ${s.approach_outputs} approach-promoting and ${s.avoid_outputs} avoidance-promoting output neurons (${s.unassigned_outputs} unassigned). ${s.loop_synapses.toLocaleString()} synapses from output neurons back onto dopamine neurons.</p>`;
  const ext = data.extinction;
  let h = '';
  for (const reinforcer of ['punishment', 'reward']) {
    const rows = data.ablation[reinforcer].rows;
    h += `<h3 style="font-size:12px;margin:12px 0 6px">${reinforcer} memory, full circuit ${rows[0].memory}</h3><table class="bench"><tr><th>Blocked</th><th>Memory left</th><th>Share lost</th><th>Teachers</th><th>Readers</th></tr>`;
    for (const r of rows.slice(1)) h += `<tr><td>${esc(r.blocked)}</td><td>${r.memory}</td><td class="${(r.share_lost || 0) > 0.2 ? 'hi' : ''}">${r.share_lost == null ? '–' : Math.round(r.share_lost * 100) + '%'}</td><td>${r.teachers.map(esc).join(', ')}</td><td>${r.readers.map(esc).join(', ')}</td></tr>`;
    h += '</table>';
  }
  $('ablation').innerHTML = h;
  $('controls').innerHTML = '<table class="bench"><tr><th>Condition</th><th>Aversive memory right after spaced training</th></tr>'
    + Object.entries(data.controls).map(([k, v]) => `<tr><td>${esc(data.conditions[k])}</td><td>${v}</td></tr>`).join('') + '</table>'
    + `<p class="note">Extinction removes ${Math.round(ext.intact.extinction_fraction * 100)}% of the aversive memory with the loops intact and ${Math.round(ext.no_loops.extinction_fraction * 100)}% with them cut. Spontaneous recovery after the delay: ${ext.intact.spontaneous_recovery ? 'yes' : 'no'}.</p>`
    + '<h3 style="font-size:12px;margin:14px 0 6px">Loop models</h3><table class="bench"><tr><th>Model</th><th>Extinction</th><th>γ1 share of aversive memory</th></tr>'
    + Object.entries(data.loop_models).map(([k, v]) => `<tr><td>${esc(v.text)}</td><td>${Math.round(v.extinction * 100)}%</td><td>${Math.round((v.gamma1_share || 0) * 100)}%</td></tr>`).join('') + '</table>'
    + '<p class="hint">Signed loops reproduce extinction with recovery; neither model gives γ1 the weight fly experiments do. The γ1 output neuron inhibits approach and avoidance outputs about equally, and its strongest outputs go to CRE neurons outside the output layer, which this readout does not include.</p>';
}
async function go() {
  $('run').disabled = true; $('status').textContent = 'Running experiments…'; $('status').className = 'wb-status';
  try {
    const response = await fetch('/api/memory', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ seed: +$('seed').value }) });
    const d = await response.json();
    if (!response.ok) throw new Error(d.error || response.status);
    data = d; drawForget(); drawExtinct(); tables(); $('status').textContent = 'Done.';
  } catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; $('summary').innerHTML = `<p class="hint">${esc(e.message)}</p>`; }
  finally { $('run').disabled = false; }
}
$('run').onclick = go;
window.addEventListener('resize', () => { drawForget(); drawExtinct(); });
go();
