'use strict';
const $ = id => document.getElementById(id);
let D = null, order = [], sel = null, cell = 10, pad = 110, hover = null;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = n => Math.round(n).toLocaleString();

fetch('/api/regions').then(async r => { const d = await r.json(); if (!r.ok) throw new Error(d.error || r.status); return d; }).then(d => {
  D = d;
  $('sub').textContent = `${d.rois.length} primary regions, ${fmt(d.traced_neurons)} traced neurons, dataset ${d.dataset}. Region profiles found for ${d.lab_cells_found} of ${d.lab_cells_total} lab cells.`;
  draw();
}).catch(e => { $('sub').textContent = 'No region data yet: ' + e.message; });

function busiest() {
  const tot = D.roi_totals.map((t, i) => [i, t.input_connections + t.output_connections]).sort((a, b) => b[1] - a[1]);
  const n = $('scope').value === 'all' ? tot.length : Math.min(+$('scope').value, tot.length);
  return tot.slice(0, n).map(x => x[0]).sort((a, b) => D.rois[a].localeCompare(D.rois[b]));
}
function color(v, max) {
  if (v <= 0) return '#18221b';
  const t = Math.max(0, Math.min(1, Math.log10(1 + v) / Math.log10(1 + max)));
  const a = [24, 34, 27], b = [194, 232, 153], c = [233, 190, 117];
  const mix = t < .7 ? a.map((x, i) => x + (b[i] - x) * t / .7) : b.map((x, i) => x + (c[i] - x) * (t - .7) / .3);
  return `rgb(${mix.map(Math.round).join(',')})`;
}
function draw() {
  if (!D) return;
  order = busiest();
  const n = order.length, cv = $('matrix');
  cell = Math.max(4, Math.min(14, Math.floor(700 / n)));
  const W = pad + n * cell + 10, H = pad + n * cell + 10, dpr = devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
  const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const nodiag = $('nodiag').checked;
  let max = 0; for (const a of order) for (const b of order) if (!(nodiag && a === b)) max = Math.max(max, D.flow[a][b]);
  order.forEach((a, i) => order.forEach((b, j) => { g.fillStyle = nodiag && a === b ? '#121914' : color(D.flow[a][b], max); g.fillRect(pad + j * cell, pad + i * cell, cell - (cell > 5), cell - (cell > 5)); }));
  g.font = Math.min(11, cell) + 'px sans-serif'; g.fillStyle = '#99aa9c';
  if (cell >= 7) order.forEach((a, i) => {
    const on = sel === a;
    g.fillStyle = on ? '#c2e899' : '#99aa9c';
    g.textAlign = 'right'; g.fillText(D.rois[a], pad - 4, pad + i * cell + cell - 2);
    g.save(); g.translate(pad + i * cell + cell - 2, pad - 4); g.rotate(-Math.PI / 2); g.textAlign = 'left'; g.fillText(D.rois[a], 0, 0); g.restore();
  });
  if (sel !== null && order.includes(sel)) {
    const k = order.indexOf(sel); g.strokeStyle = '#c2e899'; g.lineWidth = 1;
    g.strokeRect(pad, pad + k * cell, n * cell, cell); g.strokeRect(pad + k * cell, pad, cell, n * cell);
  }
  g.fillStyle = '#99aa9c'; g.textAlign = 'left'; g.fillText('input region ↓   output region →', 4, 14);
}
function at(e) {
  const r = $('matrix').getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const i = Math.floor((y - pad) / cell), j = Math.floor((x - pad) / cell), n = order.length;
  if (x < pad && i >= 0 && i < n) return { row: order[i] };
  if (y < pad && j >= 0 && j < n) return { row: order[j] };
  if (i >= 0 && j >= 0 && i < n && j < n) return { row: order[i], col: order[j] };
  return null;
}
$('matrix').addEventListener('click', e => { const h = at(e); if (!h) return; sel = h.row; draw(); panel(h); });
$('matrix').addEventListener('mousemove', e => { const h = at(e); $('matrix').title = h && h.col !== undefined ? `${D.rois[h.row]} → ${D.rois[h.col]}: ${fmt(D.flow[h.row][h.col])}` : h ? D.rois[h.row] : ''; });
$('scope').onchange = draw; $('nodiag').onchange = draw;

function panel(h) {
  const a = h.row, name = D.rois[a], t = D.roi_totals[a];
  const outs = D.rois.map((r, j) => [r, D.flow[a][j]]).filter(x => x[0] !== name && x[1] > 0).sort((x, y) => y[1] - x[1]).slice(0, 10);
  const ins = D.rois.map((r, j) => [r, D.flow[j][a]]).filter(x => x[0] !== name && x[1] > 0).sort((x, y) => y[1] - x[1]).slice(0, 10);
  const lab = Object.entries(D.profiles).filter(([, p]) => p.input.some(x => x.roi === name && x.fraction >= .1) || p.output.some(x => x.roi === name && x.fraction >= .1));
  const list = rows => '<ul class="inlist">' + rows.map(([r, v]) => `<li style="grid-template-columns:1fr auto;cursor:default"><b>${esc(r)}</b><span>${fmt(v)}</span></li>`).join('') + '</ul>';
  $('panel').innerHTML = `<h2>${esc(name)}</h2><table class="kv"><tr><td>Input connections</td><td>${fmt(t.input_connections)}</td></tr><tr><td>Output connections</td><td>${fmt(t.output_connections)}</td></tr><tr><td>Neurons with input here</td><td>${fmt(t.neurons_with_input)}</td></tr>`
    + (h.col !== undefined ? `<tr><td>Flow to ${esc(D.rois[h.col])}</td><td>${fmt(D.flow[a][h.col])}</td></tr>` : '') + '</table>'
    + `<h2 style="margin-top:16px">Drives output in</h2>${list(outs)}<h2 style="margin-top:16px">Driven by input in</h2>${list(ins)}`
    + `<h2 style="margin-top:16px">Lab cells working here</h2><p class="hint">Cells with at least 10% of their input or output in ${esc(name)}.</p>`
    + (lab.length ? '<ul class="inlist">' + lab.slice(0, 40).map(([id, p]) => `<li style="grid-template-columns:1fr auto"><b><a style="color:var(--text)" href="${p.circuit === 'locomotion' ? '/atlas#cell=' + id + '&view=anatomy' : '/neuron?circuit=learning&id=' + id}">${id}</a></b><span>${p.circuit}</span></li>`).join('') + '</ul>' + (lab.length > 40 ? `<p class="hint">and ${lab.length - 40} more</p>` : '') : '<p class="hint">None.</p>');
}
