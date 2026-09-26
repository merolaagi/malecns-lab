'use strict';
const $ = id => document.getElementById(id);
const COLOUR = { acetylcholine: '#6fc7bc', gaba: '#e9be75', glutamate: '#f28b82', histamine: '#8ab4f8' };
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
let result = null, cells = [], body = 10360;

function ctx(id) { const cv = $(id), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1; cv.width = w * d; cv.height = h * d; const g = cv.getContext('2d'); g.scale(d, d); g.font = '11px sans-serif'; return [g, w, h]; }
function draw() {
  if (!result) return;
  const [g, w, h] = ctx('voltage');
  const lo = -82, hi = Math.max(-38, ...result.voltage_mv);
  const X = t => 52 + t / result.parameters.duration_ms * (w - 68);
  const Y = mv => h - 24 - (mv - lo) / (hi - lo) * (h - 40);
  g.strokeStyle = '#2b372e'; g.fillStyle = '#99aa9c';
  for (const mv of [-80, -70, -60, -50, -40, 0].filter(v => v >= lo && v <= hi)) {
    g.beginPath(); g.moveTo(52, Y(mv)); g.lineTo(w - 14, Y(mv)); g.stroke(); g.fillText(mv + ' mV', 4, Y(mv) + 4);
  }
  g.setLineDash([5, 4]); g.strokeStyle = '#c2e899';
  g.beginPath(); g.moveTo(52, Y(result.membrane.threshold_mv)); g.lineTo(w - 14, Y(result.membrane.threshold_mv)); g.stroke(); g.setLineDash([]);
  g.strokeStyle = '#e7eee5'; g.lineWidth = 1.5; g.beginPath();
  result.voltage_mv.forEach((mv, i) => i ? g.lineTo(X(result.time_ms[i]), Y(mv)) : g.moveTo(X(result.time_ms[i]), Y(mv)));
  g.stroke(); g.lineWidth = 1;
  g.fillStyle = '#99aa9c'; g.fillText('0 ms', 52, h - 6); g.fillText(result.parameters.duration_ms + ' ms', w - 60, h - 6);

  const [gc, cw, ch] = ctx('conductance');
  const series = Object.entries(result.conductance_ns);
  const peak = Math.max(1e-6, ...series.flatMap(([, v]) => v));
  gc.strokeStyle = '#2b372e'; gc.beginPath(); gc.moveTo(52, ch - 24); gc.lineTo(cw - 14, ch - 24); gc.stroke();
  gc.fillStyle = '#99aa9c'; gc.fillText(peak.toFixed(2) + ' nS', 4, 16); gc.fillText('0', 38, ch - 20);
  series.forEach(([transmitter, values], row) => {
    gc.strokeStyle = COLOUR[transmitter] || '#d4a5f5'; gc.lineWidth = 1.5; gc.beginPath();
    values.forEach((value, i) => {
      const x = 52 + result.time_ms[i] / result.parameters.duration_ms * (cw - 68), y = ch - 24 - value / peak * (ch - 44);
      i ? gc.lineTo(x, y) : gc.moveTo(x, y);
    });
    gc.stroke(); gc.lineWidth = 1;
    gc.fillStyle = COLOUR[transmitter] || '#d4a5f5'; gc.fillText(transmitter, 90 + row * 110, 16);
  });
}
function show(data) {
  result = data; body = data.bodyId;
  $('title').textContent = `${data.name} · body ${data.bodyId}`;
  $('summary').textContent = `${data.partner_count} measured inputs, ${data.synapses_total} synapses. `
    + data.groups.map(g => `${g.transmitter}: ${g.sources} cells, ${g.weight_ns} nS total, peak ${g.peak_conductance_ns} nS`).join(' · ')
    + `. Mean voltage ${data.mean_voltage_mv} mV, ${data.spike_rate_hz} spikes/s.`;
  $('partners').innerHTML = '<tr><th>Source</th><th>Synapses</th><th>Transmitter</th><th>Receptor</th><th>Top probabilities</th></tr>'
    + data.partners.map(p => `<tr><td>${esc(p.name)}</td><td>${p.synapses}</td><td>${esc(p.transmitter || 'unknown')}</td><td>${esc(p.receptor)}</td><td>${Object.entries(p.probabilities).slice(0, 3).map(([k, v]) => `${k.slice(0, 4)} ${v}`).join(', ')}</td></tr>`).join('');
  $('receptors').innerHTML = '<table class="kv">' + Object.entries(data.receptors).map(([k, r]) =>
    `<tr><td style="color:${COLOUR[k]}">${k}</td><td>${r.reversal} mV</td><td>${r.rise}/${r.decay} ms</td></tr>`).join('') + '</table>';
  draw();
}
async function post(url, payload) {
  const response = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.status);
  return data;
}
const settings = () => ({ body_id: body, rate_hz: +$('rate').value, excitation_scale: +$('exc').value, inhibition_scale: +$('inh').value,
                          duration_ms: +$('duration').value, mode: $('mode').value, spiking: $('spiking').checked, seed: +$('seed').value });
async function probe() {
  $('run').disabled = true; $('status').textContent = 'Simulating conductances…'; $('status').className = 'wb-status';
  try { show(await post('/api/synapse', settings())); $('status').textContent = 'Measured wiring, assumed physiology.'; }
  catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; }
  finally { $('run').disabled = false; }
}
$('run').onclick = probe;
$('shunt').onclick = async () => {
  $('shunt').disabled = true; $('status').textContent = 'Comparing the same excitation with and without inhibition…';
  try {
    const s = settings(), d = await post('/api/synapse-shunting', { body_id: s.body_id, rate_hz: s.rate_hz, duration_ms: s.duration_ms, excitation_scale: s.excitation_scale, mode: s.mode, seed: s.seed });
    $('status').textContent = `Excitation alone: ${d.depolarisation_alone_mv} mV. With inhibition: ${d.depolarisation_with_inhibition_mv} mV. Ratio ${d.ratio}. ${d.note}`;
  } catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; }
  finally { $('shunt').disabled = false; }
};
function hits(query) {
  const q = query.trim().toLowerCase();
  const matches = cells.filter(c => String(c.bodyId).includes(q) || (c.instance || c.type || '').toLowerCase().includes(q)).slice(0, 12);
  $('hits').innerHTML = matches.map(c => `<li style="grid-template-columns:1fr auto"><b><a href="#" data-id="${c.bodyId}" style="color:var(--text)">${esc(c.instance || c.type || c.bodyId)}</a></b><span>${c.bodyId}</span></li>`).join('')
    || '<li><span>No match in the locomotion circuit.</span></li>';
}
$('go').onclick = () => hits($('search').value);
$('search').addEventListener('keydown', e => { if (e.key === 'Enter') hits($('search').value); });
$('hits').addEventListener('click', e => {
  const link = e.target.closest('a[data-id]'); if (!link) return;
  e.preventDefault(); body = Number(link.dataset.id); probe();
});
for (const [id, suffix] of [['rate', ' Hz'], ['exc', ''], ['inh', ''], ['duration', ' ms']]) $(id).oninput = () => $(id + 'Out').textContent = $(id).value + suffix;
$('mode').onchange = probe;
window.addEventListener('resize', draw);
fetch('/api/circuit').then(r => r.json()).then(d => { cells = d.nodes; hits('DN'); }).catch(() => {});
probe();
