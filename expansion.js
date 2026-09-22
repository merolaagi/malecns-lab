'use strict';
const $ = id => document.getElementById(id);
const FAMILY = { fly_measured: '#c2e899', fly_shuffled: '#a9d68a', random_sparse: '#8fc47c', dense_topk: '#6fb07a', no_sparsity: '#6fc7bc', pixels_associative: '#8ab4f8', fly_delta: '#e9be75', pixels_delta: '#f0a07a', mlp_backprop: '#f28b82' };
const SHORT = { fly_measured: 'Measured', fly_shuffled: 'Shuffled', random_sparse: 'Random sparse', dense_topk: 'Dense random', no_sparsity: 'Not sparse', pixels_associative: 'Pixels, associative', fly_delta: 'Error-driven on code', pixels_delta: 'Pixels, error-driven', mlp_backprop: 'MLP backprop' };
const C = { text: '#e7eee5', muted: '#99aa9c', line: '#2b372e' };
let last = null;
function canvas(id) { const cv = $(id), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1; cv.width = w * d; cv.height = h * d; const g = cv.getContext('2d'); g.scale(d, d); g.font = '11px sans-serif'; return [g, w, h]; }
function bars(id, rows, max, lines = []) {
  const [g, w, h] = canvas(id), left = 128, bh = Math.min(24, (h - 20) / rows.length - 4);
  lines.forEach(([v, dash]) => { const x = left + v / max * (w - left - 40); g.setLineDash(dash); g.strokeStyle = C.muted; g.beginPath(); g.moveTo(x, 4); g.lineTo(x, h - 4); g.stroke(); g.setLineDash([]); });
  rows.forEach(([k, v], i) => {
    const y = 8 + i * (bh + 4); g.fillStyle = C.muted; g.textAlign = 'right'; g.fillText(SHORT[k], left - 8, y + bh * .7);
    g.fillStyle = FAMILY[k]; g.fillRect(left, y, Math.max(1, v / max * (w - left - 40)), bh);
    g.fillStyle = C.text; g.textAlign = 'left'; g.fillText(Math.round(v * 100) + '%', left + v / max * (w - left - 40) + 6, y + bh * .7);
  });
}
function curve(r) {
  const [g, w, h] = canvas('cCurve'), keys = ['fly_measured', 'pixels_associative', 'fly_delta', 'mlp_backprop'], n = r.tasks.length;
  const X = i => 44 + i * (w - 70) / Math.max(1, n - 1), Y = v => 12 + (1 - v) * (h - 44);
  g.strokeStyle = C.line; [0, .5, 1].forEach(v => { g.beginPath(); g.moveTo(40, Y(v)); g.lineTo(w - 20, Y(v)); g.stroke(); g.fillStyle = C.muted; g.fillText(Math.round(v * 100) + '%', 4, Y(v) + 4); });
  r.tasks.forEach((t, i) => { g.fillStyle = C.muted; g.textAlign = 'center'; g.fillText('after ' + t.join('+'), X(i), h - 8); }); g.textAlign = 'left';
  keys.forEach((k, j) => {
    const s = r.models[k].seen_accuracy_by_task; g.strokeStyle = FAMILY[k]; g.lineWidth = 2; g.beginPath(); s.forEach((v, i) => i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))); g.stroke(); g.lineWidth = 1;
    g.fillStyle = FAMILY[k]; g.fillText(SHORT[k], 52, h - 96 + j * 15);
  });
}
function codes(r) {
  const k = $('codeKind').value, M = r.class_code_overlap[k], [g, w, h] = canvas('cCode'); if (!M) return;
  const n = M.length, s = Math.min((w - 40) / n, (h - 24) / n);
  M.forEach((row, i) => row.forEach((v, j) => { const t = i === j ? 1 : Math.min(1, v * 1.5); g.fillStyle = i === j ? '#18221b' : `rgb(${Math.round(24 + 170 * t)},${Math.round(34 + 198 * t)},${Math.round(27 + 126 * t)})`; g.fillRect(30 + j * s + 1, 4 + i * s + 1, s - 2, s - 2);
    if (i !== j && s > 24) { g.fillStyle = C.text; g.textAlign = 'center'; g.fillText(v.toFixed(2), 30 + j * s + s / 2, 4 + i * s + s / 2 + 4); } }));
  g.textAlign = 'center'; for (let i = 0; i < n; i++) { g.fillStyle = C.muted; g.fillText(i, 30 + i * s + s / 2, 4 + n * s + 14); g.textAlign = 'right'; g.fillText(i, 24, 4 + i * s + s / 2 + 4); g.textAlign = 'center'; }
}
function show(r) {
  const order = Object.keys(r.models);
  bars('cAcc', order.map(k => [k, r.models[k].final_accuracy]), 1, [[r.joint_upper_bound, [6, 4]], [r.chance, [2, 3]]]);
  bars('cForget', order.map(k => [k, r.models[k].forgetting]), 1);
  curve(r); codes(r);
}
async function run() {
  const body = { task: $('task').value, per_class: +$('perClass').value, seed: +$('seed').value };
  $('expRun').disabled = true; $('status').textContent = 'Running…'; $('status').className = 'wb-status';
  try { const res = await fetch('/api/expansion', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); const r = await res.json(); if (!res.ok) throw new Error(r.error || res.status);
    last = r; show(r); $('status').textContent = `${r.train_size} training and ${r.test_size} test images, seed ${r.parameters.seed}`; }
  catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; }
  finally { $('expRun').disabled = false; }
}
async function bench() {
  try {
    const d = await (await fetch('/api/expansion-benchmark')).json(), names = { digits: 'Digits', digits_5_per_class: 'Digits, 5 per class', two_shape: 'Two-shape classes' };
    let h = '<table class="bench"><tr><th>Model</th><th>Params</th>' + Object.keys(d.conditions).map(c => `<th>${names[c]}<br>accuracy</th><th>forgetting</th>`).join('') + '</tr>';
    for (const k of Object.keys(d.labels)) {
      h += `<tr><td>${d.labels[k]}</td><td>${d.conditions.digits.parameters[k].toLocaleString()}</td>`;
      for (const c of Object.values(d.conditions)) { const a = c.summary[k].final_accuracy, f = c.summary[k].forgetting; h += `<td class="${a.mean > .8 ? 'hi' : a.mean < .4 ? 'lo' : ''}">${Math.round(a.mean * 100)}% ± ${Math.round(a.sd * 100)}</td><td class="${f.mean > .5 ? 'lo' : ''}">${Math.round(f.mean * 100)}%</td>`; }
      h += '</tr>';
    }
    h += '<tr><td>MLP trained on all classes at once (upper bound)</td><td></td>' + Object.values(d.conditions).map(c => `<td>${Math.round(c.joint_upper_bound * 100)}%</td><td></td>`).join('') + '</tr>';
    $('bench').innerHTML = h + '</table>';
  } catch (e) { $('bench').innerHTML = '<p class="hint">No saved benchmark.</p>'; }
}
$('expRun').onclick = run; $('codeKind').onchange = () => last && codes(last);
window.addEventListener('resize', () => last && show(last));
run(); bench();
