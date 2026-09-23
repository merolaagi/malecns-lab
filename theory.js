'use strict';
const $ = id => document.getElementById(id);
const COL = { delta: '#f28b82', delta_normalised: '#6fc7bc', assoc: '#c2e899', dense_topk: '#8ab4f8', fly_measured: '#c2e899', exact: '#e7eee5', asym: '#e9be75', text: '#e7eee5', muted: '#99aa9c', line: '#2b372e' };
let last = null;
function frame(id, { xlo, xhi, ylo = 0, yhi = 1, xlog = false, xlabel = '', ylabel = '', xticks = [] }) {
  const cv = $(id), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1;
  cv.width = w * d; cv.height = h * d; const g = cv.getContext('2d'); g.scale(d, d); g.font = '11px sans-serif';
  const L = 52, B = 34, tx = v => L + ((xlog ? Math.log10(v / xlo) / Math.log10(xhi / xlo) : (v - xlo) / (xhi - xlo))) * (w - L - 18);
  const ty = v => h - B - ((v - ylo) / (yhi - ylo)) * (h - B - 14);
  g.strokeStyle = COL.line; g.fillStyle = COL.muted;
  for (let i = 0; i <= 4; i++) { const v = ylo + (yhi - ylo) * i / 4; g.beginPath(); g.moveTo(L, ty(v)); g.lineTo(w - 14, ty(v)); g.stroke(); g.textAlign = 'right'; g.fillText(v.toFixed(v < 1 ? 2 : 0), L - 6, ty(v) + 4); }
  g.textAlign = 'center'; xticks.forEach(v => g.fillText(String(v), tx(v), h - 12));
  g.fillText(xlabel, (L + w) / 2, h - 1); g.save(); g.translate(12, h / 2); g.rotate(-Math.PI / 2); g.fillText(ylabel, 0, 0); g.restore();
  return { g, tx, ty, w, h };
}
const dot = (p, x, y, c, r = 3.5) => { p.g.fillStyle = c; p.g.beginPath(); p.g.arc(p.tx(x), p.ty(y), r, 0, 7); p.g.fill(); };
const line = (p, pts, c, dash = []) => { p.g.strokeStyle = c; p.g.lineWidth = 1.8; p.g.setLineDash(dash); p.g.beginPath(); pts.forEach(([x, y], i) => i ? p.g.lineTo(p.tx(x), p.ty(y)) : p.g.moveTo(p.tx(x), p.ty(y))); p.g.stroke(); p.g.setLineDash([]); p.g.lineWidth = 1; };
const key = (p, items, x0 = 70) => items.forEach(([label, c], i) => { p.g.fillStyle = c; p.g.textAlign = 'left'; p.g.fillText('■ ' + label, x0, 22 + i * 15); });

function draw(r) {
  const pts = r.P1.points, ov = pts.map(p => p.cross_task_overlap);
  let p = frame('c1', { xlo: Math.min(...ov) - .03, xhi: Math.max(...ov) + .03, ylo: 0, yhi: 1, xlabel: 'overlap between old- and new-class codes', ylabel: 'forgetting', xticks: [.5, .6, .7, .8, .9] });
  pts.forEach(q => { dot(p, q.cross_task_overlap, q.delta_forgetting, COL.delta); dot(p, q.cross_task_overlap, q.delta_normalised_forgetting, COL.delta_normalised); });
  line(p, [[0, r.P1.associative_forgetting_mean], [1, r.P1.associative_forgetting_mean]], COL.assoc, [5, 4]);
  key(p, [['error-driven', COL.delta], ['error-driven, normalised at test', COL.delta_normalised], ['associative rule', COL.assoc]]);
  $('t1').textContent = r.verdicts.P1.text;

  p = frame('c2', { xlo: 0.004, xhi: 0.5, ylo: .5, yhi: 1, xlog: true, xlabel: 'sparsity f (fraction of units active)', ylabel: 'accuracy', xticks: [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4] });
  const styles = { 'digits|dense_topk': [COL.dense_topk, [4, 3]], 'digits|fly_measured': [COL.fly_measured, [4, 3]], 'two_shape|dense_topk': [COL.dense_topk, []], 'two_shape|fly_measured': [COL.fly_measured, []] };
  for (const [k, c] of Object.entries(r.P2.curves)) { const [col, dash] = styles[k]; line(p, c.map(x => [x.sparsity, x.accuracy]), col, dash); const pk = r.P2.peaks[k]; if (k.startsWith('two_shape')) dot(p, pk.sparsity, pk.accuracy, col, 5); }
  key(p, [['two-shape classes (solid)', COL.muted], ['digits (dashed)', COL.muted], ['dense random', COL.dense_topk], ['measured PN→KC', COL.fly_measured]]);
  $('t2').textContent = r.verdicts.P2.text;

  p = frame('c3', { xlo: 0, xhi: Math.PI / 2, ylo: 0, yhi: 1, xlabel: 'angle between inputs (radians)', ylabel: 'code overlap / k', xticks: [0, 0.5, 1, 1.5] });
  for (const [w, v] of Object.entries(r.P3.wirings)) v.points.forEach(q => dot(p, q.theta, q.overlap, COL[w], 1.6));
  line(p, r.P3.theory.map(t => [t.theta, t.exact]), COL.exact);
  line(p, r.P3.theory.map(t => [t.theta, t.asymptotic]), COL.asym, [5, 4]);
  key(p, [['exact bivariate-normal curve', COL.exact], ['f^tan²(θ/2)', COL.asym], ['dense random codes', COL.dense_topk], ['measured PN→KC codes', COL.fly_measured]], p.w - 250);
  $('t3').textContent = r.verdicts.P3.text;

  p = frame('c4', { xlo: 9, xhi: 440, ylo: 0, yhi: 1, xlog: true, xlabel: 'number of classes stored', ylabel: 'accuracy', xticks: [10, 20, 50, 100, 200, 400] });
  const cols = [COL.dense_topk, COL.assoc, COL.asym];
  Object.entries(r.P4.curves).forEach(([f, v], i) => { line(p, v.points.map(q => [q.classes, q.predicted]), cols[i], [5, 4]); v.points.forEach(q => dot(p, q.classes, q.simulated, cols[i])); });
  key(p, Object.keys(r.P4.curves).map((f, i) => [`f = ${f} (dots simulated, dashed predicted)`, cols[i]]), p.w - 300);
  $('t4').textContent = r.verdicts.P4.text;

  for (const k of ['P1', 'P2', 'P3', 'P4']) { const e = $('v' + k); e.textContent = r.verdicts[k].supported ? 'supported' : 'not supported'; e.className = 'verdict ' + (r.verdicts[k].supported ? 'y' : 'n'); }
}
async function load(recompute) {
  $('thRun').disabled = true; $('status').textContent = recompute ? 'Running…' : 'Loading…'; $('status').className = 'wb-status';
  try {
    const res = recompute ? await fetch('/api/theory', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ seed: +$('seed').value }) })
                          : await fetch('/api/theory-benchmark');
    const r = await res.json(); if (!res.ok) throw new Error(r.error || res.status);
    last = r; draw(r); $('status').textContent = `seed ${r.parameters.seed}` + (recompute ? '' : ' (saved run)');
  } catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; }
  finally { $('thRun').disabled = false; }
}
$('thRun').onclick = () => load(true);
window.addEventListener('resize', () => last && draw(last));
load(false);
