'use strict';
const $ = id => document.getElementById(id);
const C = { text: '#e7eee5', muted: '#99aa9c', line: '#2b372e', good: '#c2e899', bad: '#f28b82', mid: '#3a473d', bar: '#6fc7bc', held: '#e9be75' };
const pct = v => v === null || v === undefined ? '–' : Math.round(v * 100) + '%';

function canvas(id) {
  const cv = $(id), w = cv.clientWidth, h = cv.clientHeight || 240, d = devicePixelRatio || 1;
  cv.width = w * d; cv.height = h * d; cv.style.height = h + 'px';
  const c = cv.getContext('2d'); c.scale(d, d); c.font = '11px sans-serif'; return [c, w, h];
}
function mix(a, b, t) { const p = x => [1, 3, 5].map(i => parseInt(x.slice(i, i + 2), 16)); const A = p(a), B = p(b); return 'rgb(' + A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',') + ')'; }
function pairColor(v) { return v >= .5 ? mix(C.mid, C.good, (v - .5) * 2) : mix(C.mid, C.bad, (.5 - v) * 2); }

function drawValues(r) {
  const [c, w, h] = canvas('cValues'), vals = r.values, n = vals.length;
  const lo = Math.min(-1.2, ...vals.map(v => v.mean)), hi = Math.max(1.2, ...vals.map(v => v.mean));
  const Y = v => 16 + (hi - v) / (hi - lo) * (h - 40), bw = (w - 50) / n;
  c.strokeStyle = C.line; c.beginPath(); c.moveTo(36, Y(0)); c.lineTo(w - 6, Y(0)); c.stroke();
  c.fillStyle = C.muted; c.fillText('+1', 4, Y(1) + 4); c.fillText('0', 4, Y(0) + 4); c.fillText('−1', 4, Y(-1) + 4);
  vals.forEach((v, i) => {
    const x = 40 + i * bw + bw * .15, bwid = bw * .7, y0 = Y(0), y1 = Y(v.mean);
    if (v.held_out) { c.strokeStyle = C.held; c.lineWidth = 1.5; c.strokeRect(x, Math.min(y0, y1), bwid, Math.abs(y1 - y0) || 1); }
    else { c.fillStyle = C.bar; c.fillRect(x, Math.min(y0, y1), bwid, Math.abs(y1 - y0) || 1); }
    c.fillStyle = v.held_out ? C.held : C.text; c.textAlign = 'center'; c.fillText(v.number, x + bwid / 2, h - 6); c.textAlign = 'left';
  });
}
function drawMatrix(id, M, held, fmt, colorFor) {
  const [c, w, h] = canvas(id), n = M.length, s = Math.min((w - 40) / n, (h - 30) / n), ox = 30, oy = 6;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const v = M[i][j];
    c.fillStyle = v === null ? '#18221b' : colorFor(v); c.fillRect(ox + j * s + 1, oy + i * s + 1, s - 2, s - 2);
    if (v !== null && s > 26) { c.fillStyle = C.text; c.textAlign = 'center'; c.fillText(fmt(v), ox + j * s + s / 2, oy + i * s + s / 2 + 4); }
  }
  c.textAlign = 'center';
  for (let k = 0; k < n; k++) {
    c.fillStyle = held.includes(k + 1) ? C.held : C.muted;
    c.fillText(k + 1, ox + k * s + s / 2, oy + n * s + 14); c.textAlign = 'right'; c.fillText(k + 1, ox - 6, oy + k * s + s / 2 + 4); c.textAlign = 'center';
  }
  c.textAlign = 'left';
}
function drawDistance(r) {
  const [c, w, h] = canvas('cDist'), d = r.distance_curve;
  if (!d.length) return;
  const maxD = Math.max(...d.map(x => x.distance)), X = k => 40 + (k - 1) / Math.max(1, maxD - 1) * (w - 60), Y = v => 12 + (1 - v) * (h - 40);
  c.strokeStyle = C.line; [0.5, 1].forEach(v => { c.beginPath(); c.moveTo(36, Y(v)); c.lineTo(w - 10, Y(v)); c.stroke(); });
  c.fillStyle = C.muted; c.fillText('100%', 2, Y(1) + 4); c.fillText('50%', 6, Y(.5) + 4);
  c.strokeStyle = C.good; c.lineWidth = 2; c.beginPath(); d.forEach((p, i) => i ? c.lineTo(X(p.distance), Y(p.p_larger)) : c.moveTo(X(p.distance), Y(p.p_larger))); c.stroke();
  d.forEach(p => { c.fillStyle = C.good; c.beginPath(); c.arc(X(p.distance), Y(p.p_larger), 3, 0, 7); c.fill(); c.fillStyle = C.muted; c.textAlign = 'center'; c.fillText(p.distance, X(p.distance), h - 6); });
  c.textAlign = 'left';
}
function show(r) {
  $('aT').textContent = pct(r.accuracy.trained_pairs); $('aO').textContent = pct(r.accuracy.one_novel); $('aB').textContent = pct(r.accuracy.both_novel);
  $('aE').textContent = r.weights.nonzero_edges.toLocaleString();
  drawValues(r);
  drawMatrix('cMatrix', r.p_larger, r.parameters.held_out, v => Math.round(v * 100), pairColor);
  drawMatrix('cJac', r.kc_jaccard.map((row, i) => row.map((v, j) => i === j ? null : v)), r.parameters.held_out, v => v.toFixed(2), v => mix(C.mid, C.bar, Math.min(1, v * 2)));
  drawDistance(r);
}
let last = null;
async function run() {
  const body = { condition: $('condition').value, seed: +$('seed').value, train_trials: +$('trials').value, held_out: $('design').value.split(',').map(Number) };
  $('numRun').disabled = true; $('status').textContent = 'Running…'; $('status').className = 'wb-status';
  try {
    const res = await fetch('/api/numerosity', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const r = await res.json(); if (!res.ok) throw new Error(r.error || 'HTTP ' + res.status);
    last = r; show(r); $('status').textContent = r.label + ', seed ' + r.parameters.seed;
  } catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; }
  finally { $('numRun').disabled = false; }
}
async function bench() {
  try {
    const d = await (await fetch('/api/numerosity-benchmark')).json();
    const cell = x => { const cls = x.mean > .65 ? 'hi' : x.mean < .35 ? 'lo' : ''; return '<td class="' + cls + '">' + Math.round(x.mean * 100) + '% ± ' + Math.round(x.sd * 100) + '</td>'; };
    let html = '<table class="bench"><tr><th>Held out</th><th>Condition</th><th>Trained pairs</th><th>One held out</th><th>Both held out</th></tr>';
    for (const [design, conds] of Object.entries(d.summary)) for (const [cond, a] of Object.entries(conds))
      html += '<tr><td>' + d.designs[design].join(', ') + '</td><td>' + d.labels[cond] + '</td>' + cell(a.trained_pairs) + cell(a.one_novel) + cell(a.both_novel) + '</tr>';
    $('bench').innerHTML = html + '</table>';
  } catch (e) { $('bench').innerHTML = '<p class="hint">No saved benchmark. Run numerosity.py --benchmark.</p>'; }
}
$('numRun').onclick = run;
window.addEventListener('resize', () => last && show(last));
run(); bench();
