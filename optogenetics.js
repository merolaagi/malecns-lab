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
    const body = { experiment: $('experiment').value, side: $('side').value, light: +$('light').value, duration: +$('duration').value, seed: +$('seed').value, condition: $('condition').value };
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
    const b = await r.json(); const conds = Object.keys(b.conditions);
    let h = `<table class="bench"><tr><th>Cell type</th><th>Real flies</th>${conds.map(c => `<th>${esc(c)}</th>`).join('')}<th>Share on the lit side (intact / shuffled / sham)</th></tr>`;
    for (const [name, row] of Object.entries(b.experiments)) {
      if (!row.available) { h += `<tr><td>${esc(name)}</td><td colspan="${conds.length + 2}" class="muted">not in this circuit yet</td></tr>`; continue; }
      const c = row.conditions;
      h += `<tr><td>${esc(name)}</td><td>${esc(row.expect)}</td>${conds.map(k => `<td><span class="tag ${tagClass(c[k].result)}">${esc(c[k].result)}</span></td>`).join('')}`
        + `<td>${conds.map(k => c[k].lit_side_share == null ? '–' : Math.round(c[k].lit_side_share * 100) + '%').join(' / ')}</td></tr>`;
      h += `<tr><td></td><td colspan="${conds.length + 2}" class="muted" style="text-align:left">${esc(c.intact.detail)}</td></tr>`;
    }
    $('score').innerHTML = h + '</table>';
  } catch (e) { $('score').innerHTML = `<p class="hint">${esc(e.message)}</p>`; }
}
loadMeta(); score();
