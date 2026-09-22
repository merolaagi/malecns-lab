'use strict';
const $ = id => document.getElementById(id);
let D = null, order = [], groupOf = {}, sel = null, cell = 10, pad = 120, hov = null, max = 1;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmt = n => Math.round(n).toLocaleString();
const pct = x => (x * 100).toFixed(x < 0.01 ? 2 : 1) + '%';

fetch('/api/regions').then(async r => { const d = await r.json(); if (!r.ok) throw new Error(d.error || r.status); return d; }).then(d => {
  D = d;
  D.rowSum = D.flow.map(r => r.reduce((a, b) => a + b, 0));
  D.colSum = D.rois.map((_, j) => D.flow.reduce((a, r) => a + r[j], 0));
  hierarchyOrder();
  $('sub').textContent = `${d.rois.length} primary regions, ${fmt(d.traced_neurons)} traced neurons, dataset ${d.dataset}. Region profiles for ${d.lab_cells_found} of ${d.lab_cells_total} lab cells.`
    + (d.neurons ? '' : ' Neuron counts per connection need a rebuild with the current build_regions.py.');
  draw();
}).catch(e => { $('sub').textContent = 'No region data yet: ' + e.message; });

// Primary regions in the order they appear in neuPrint's hierarchy, grouped by top-level division.
function hierarchyOrder() {
  const rank = {}; let k = 0;
  const walk = (h, top) => { for (const [name, sub] of Object.entries(h || {})) { const clean = name.replace(/\*$/, ''); if (!(clean in rank)) { rank[clean] = k++; groupOf[clean] = top || clean; } walk(sub, top || (name === 'CNS' ? null : clean)); } };
  const root = D.hierarchy && D.hierarchy.CNS ? D.hierarchy.CNS : D.hierarchy;
  walk(root, null);
  D.hrank = D.rois.map(r => r in rank ? rank[r] : 1e6);
}
function current() {
  const busy = D.rois.map((_, i) => [i, D.roi_totals[i].input_connections + D.roi_totals[i].output_connections]).sort((a, b) => b[1] - a[1]);
  const n = $('scope').value === 'all' ? busy.length : Math.min(+$('scope').value, busy.length);
  const idx = busy.slice(0, n).map(x => x[0]), how = $('order').value;
  if (how === 'name') idx.sort((a, b) => D.rois[a].localeCompare(D.rois[b]));
  else if (how === 'hierarchy') idx.sort((a, b) => D.hrank[a] - D.hrank[b] || D.rois[a].localeCompare(D.rois[b]));
  return idx;
}
function shade(v) {
  if (v <= 0) return '#18221b';
  const t = $('scale').value === 'log' ? Math.log10(1 + v) / Math.log10(1 + max) : v / max;
  const x = Math.max(0, Math.min(1, t)), a = [24, 34, 27], b = [111, 199, 188], c = [194, 232, 153], d = [233, 190, 117];
  const seg = x < .45 ? [a, b, x / .45] : x < .8 ? [b, c, (x - .45) / .35] : [c, d, (x - .8) / .2];
  return `rgb(${seg[0].map((u, i) => Math.round(u + (seg[1][i] - u) * seg[2])).join(',')})`;
}
function draw() {
  if (!D) return;
  order = current();
  const n = order.length, cv = $('matrix'), nodiag = $('nodiag').checked;
  cell = Math.max(8, Math.min(14, Math.floor(1150 / n)));
  const W = pad + n * cell + 10, H = pad + n * cell + 10, dpr = devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
  const g = cv.getContext('2d'); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  max = 0; for (const a of order) for (const b of order) if (!(nodiag && a === b)) max = Math.max(max, D.flow[a][b]);
  const gap = cell > 6 ? 1 : 0;
  order.forEach((a, i) => order.forEach((b, j) => { g.fillStyle = nodiag && a === b ? '#121914' : shade(D.flow[a][b]); g.fillRect(pad + j * cell, pad + i * cell, cell - gap, cell - gap); }));
  if ($('order').value === 'hierarchy') {           // dividers between top-level divisions
    g.strokeStyle = '#6b7c6e'; g.lineWidth = 1;
    for (let i = 1; i < n; i++) if (groupOf[D.rois[order[i]]] !== groupOf[D.rois[order[i - 1]]]) {
      g.beginPath(); g.moveTo(pad + i * cell - .5, pad); g.lineTo(pad + i * cell - .5, pad + n * cell); g.moveTo(pad, pad + i * cell - .5); g.lineTo(pad + n * cell, pad + i * cell - .5); g.stroke();
    }
  }
  const hl = hov ? [hov.row, hov.col] : [];
  if (hov) {                                         // crosshair on the hovered row and column
    const i = order.indexOf(hov.row), j = order.indexOf(hov.col);
    g.fillStyle = 'rgba(231,238,229,.08)'; if (i >= 0) g.fillRect(pad, pad + i * cell, n * cell, cell); if (j >= 0) g.fillRect(pad + j * cell, pad, cell, n * cell);
    if (i >= 0 && j >= 0) { g.strokeStyle = '#e7eee5'; g.lineWidth = 1.5; g.strokeRect(pad + j * cell - .5, pad + i * cell - .5, cell, cell); }
  }
  g.font = Math.min(11, cell) + 'px sans-serif';
  if (cell >= 6) order.forEach((a, i) => {
    const color = a === sel ? '#c2e899' : '#99aa9c';
    g.fillStyle = hl[0] === a ? '#e7eee5' : color; g.textAlign = 'right'; g.fillText(D.rois[a], pad - 4, pad + i * cell + cell - 2);
    g.save(); g.translate(pad + i * cell + cell - 2, pad - 4); g.rotate(-Math.PI / 2); g.textAlign = 'left'; g.fillStyle = hl[1] === a ? '#e7eee5' : color; g.fillText(D.rois[a], 0, 0); g.restore();
  });
  g.fillStyle = '#99aa9c'; g.textAlign = 'left'; g.font = '11px sans-serif'; g.fillText('input region ↓   output region →', 4, 14);
}
function at(e) {
  const r = $('matrix').getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, n = order.length;
  const i = Math.floor((y - pad) / cell), j = Math.floor((x - pad) / cell);
  if (x < pad && i >= 0 && i < n) return { row: order[i], x, y };
  if (y < pad && j >= 0 && j < n) return { row: order[j], x, y };
  if (i >= 0 && j >= 0 && i < n && j < n) return { row: order[i], col: order[j], x, y };
  return null;
}
function tip(h) {
  const t = $('tip');
  if (!h || h.col === undefined) { t.hidden = true; return; }
  const a = h.row, b = h.col, A = esc(D.rois[a]), B = esc(D.rois[b]), f = D.flow[a][b];
  const nrn = D.neurons ? D.neurons[a][b] : null;
  t.innerHTML = `<div style="font-size:14px;font-weight:600;margin-bottom:4px">${A} → ${B}</div>`
    + `<div>Flow <b style="float:right;margin-left:16px">${fmt(f)}</b></div>`
    + (nrn !== null ? `<div>Neurons <b style="float:right;margin-left:16px">${fmt(nrn)}</b></div>` : '')
    + `<div style="color:var(--muted)">${pct(D.rowSum[a] ? f / D.rowSum[a] : 0)} of ${A}-driven output lands in ${B}</div>`
    + `<div style="color:var(--muted)">${pct(D.colSum[b] ? f / D.colSum[b] : 0)} of ${B}'s output is driven from ${A}</div>`
    + (a !== b ? `<div style="color:var(--muted)">Reverse ${B} → ${A}: ${fmt(D.flow[b][a])}${D.neurons ? ' · ' + fmt(D.neurons[b][a]) + ' neurons' : ''}</div>` : '<div style="color:var(--muted)">Within-region flow</div>');
  t.hidden = false;
  const wrap = $('wrap'), cvr = $('matrix').getBoundingClientRect(), wr = wrap.getBoundingClientRect();
  let left = cvr.left - wr.left + wrap.scrollLeft + h.x + 16, top = cvr.top - wr.top + wrap.scrollTop + h.y + 16;
  if (left + t.offsetWidth > wrap.scrollLeft + wrap.clientWidth) left -= t.offsetWidth + 32;
  t.style.left = left + 'px'; t.style.top = top + 'px';
}
$('matrix').addEventListener('mousemove', e => {
  const h = at(e), key = h ? h.row + ':' + h.col : '', old = hov ? hov.row + ':' + hov.col : '';
  hov = h && h.col !== undefined ? h : (h ? { row: h.row, col: undefined } : null);
  if (key !== old) draw();
  tip(h);
});
$('matrix').addEventListener('mouseleave', () => { hov = null; tip(null); draw(); });
$('matrix').addEventListener('click', e => { const h = at(e); if (!h) return; sel = h.row; draw(); panel(h); });
for (const id of ['scope', 'order', 'scale', 'nodiag']) $(id).onchange = draw;

function panel(h) {
  const a = h.row, name = D.rois[a], t = D.roi_totals[a];
  const outs = D.rois.map((r, j) => [r, D.flow[a][j], D.neurons ? D.neurons[a][j] : null]).filter(x => x[0] !== name && x[1] > 0).sort((x, y) => y[1] - x[1]).slice(0, 10);
  const ins = D.rois.map((r, j) => [r, D.flow[j][a], D.neurons ? D.neurons[j][a] : null]).filter(x => x[0] !== name && x[1] > 0).sort((x, y) => y[1] - x[1]).slice(0, 10);
  const lab = Object.entries(D.profiles).filter(([, p]) => p.input.some(x => x.roi === name && x.fraction >= .1) || p.output.some(x => x.roi === name && x.fraction >= .1));
  const list = rows => '<ul class="inlist">' + rows.map(([r, v, n]) => `<li style="grid-template-columns:1fr auto;cursor:default"><b>${esc(r)}</b><span>${fmt(v)}${n !== null ? ' · ' + fmt(n) + ' n' : ''}</span></li>`).join('') + '</ul>';
  $('panel').innerHTML = `<h2>${esc(name)}</h2><p class="hint">${esc(groupOf[name] || '')}</p><table class="kv"><tr><td>Input connections</td><td>${fmt(t.input_connections)}</td></tr><tr><td>Output connections</td><td>${fmt(t.output_connections)}</td></tr><tr><td>Neurons with input here</td><td>${fmt(t.neurons_with_input)}</td></tr>`
    + (h.col !== undefined ? `<tr><td>Flow to ${esc(D.rois[h.col])}</td><td>${fmt(D.flow[a][h.col])}</td></tr>` : '') + '</table>'
    + `<h2 style="margin-top:16px">Drives output in</h2>${list(outs)}<h2 style="margin-top:16px">Driven by input in</h2>${list(ins)}`
    + `<h2 style="margin-top:16px">Lab cells working here</h2><p class="hint">Cells with at least 10% of their input or output in ${esc(name)}.</p>`
    + (lab.length ? '<ul class="inlist">' + lab.slice(0, 40).map(([id, p]) => `<li style="grid-template-columns:1fr auto"><b><a style="color:var(--text)" href="${p.circuit === 'locomotion' ? '/atlas#cell=' + id + '&view=anatomy' : '/neuron?circuit=learning&id=' + id}">${id}</a></b><span>${p.circuit}</span></li>`).join('') + '</ul>' + (lab.length > 40 ? `<p class="hint">and ${lab.length - 40} more</p>` : '') : '<p class="hint">None.</p>');
}
