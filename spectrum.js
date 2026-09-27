'use strict';
const $ = id => document.getElementById(id);
let result = null, saved = null;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function ctx(id) { const cv = $(id), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1; cv.width = w * d; cv.height = h * d; const g = cv.getContext('2d'); g.scale(d, d); g.font = '11px sans-serif'; return [g, w, h]; }
function plane() {
  const [g, w, h] = ctx('plane'); if (!result) return;
  const conditions = result.conditions, other = conditions[$('against').value] || conditions.degree_shuffle;
  const all = [...conditions.measured.eigenvalues, ...(other.eigenvalues || [])];
  const span = Math.max(1.05, ...all.map(([x, y]) => Math.hypot(x, y))) * 1.1;
  const X = x => w / 2 + x / span * (Math.min(w, h) / 2 - 16), Y = y => h / 2 - y / span * (Math.min(w, h) / 2 - 16);
  g.strokeStyle = '#2b372e'; g.beginPath(); g.moveTo(X(-span), Y(0)); g.lineTo(X(span), Y(0)); g.moveTo(X(0), Y(-span)); g.lineTo(X(0), Y(span)); g.stroke();
  g.strokeStyle = '#6b7c6e'; g.setLineDash([4, 4]); g.beginPath(); g.arc(X(0), Y(0), Math.abs(X(1) - X(0)), 0, 7); g.stroke(); g.setLineDash([]);
  for (const [values, colour] of [[other.eigenvalues || [], 'rgba(153,170,156,.65)'], [conditions.measured.eigenvalues, 'rgba(194,232,153,.85)']]) {
    g.fillStyle = colour;
    for (const [x, y] of values) { g.beginPath(); g.arc(X(x), Y(y), 2, 0, 7); g.fill(); }
  }
  g.fillStyle = '#99aa9c'; g.fillText('radius 1', X(0) + Math.abs(X(1) - X(0)) - 30, Y(0) - 6);
}
function bars() {
  const [g, w, h] = ctx('bars'); if (!result) return;
  const rows = Object.entries(result.conditions);
  const max = Math.max(...rows.map(([, v]) => v.radius), 1);
  const bh = Math.min(28, (h - 16) / rows.length - 6);
  rows.forEach(([key, v], i) => {
    const y = 8 + i * (bh + 6), width = v.radius / max * (w - 190);
    g.fillStyle = '#99aa9c'; g.textAlign = 'right'; g.fillText(key.replace('_', ' '), 132, y + bh * 0.7);
    g.fillStyle = key === 'measured' ? '#c2e899' : (key === 'weight_shuffle' || key === 'sign_permute' ? '#6fc7bc' : '#8ab4f8');
    g.fillRect(140, y, Math.max(1, width), bh);
    g.fillStyle = '#e7eee5'; g.textAlign = 'left'; g.fillText(v.radius.toFixed(3), 146 + width, y + bh * 0.7);
  });
  g.fillStyle = '#6fc7bc'; g.fillText('teal: measured topology kept', 140, h - 4);
}
function detail() {
  const c = result.conditions.measured;
  let h = '<table class="kv">'
    + `<tr><td>Graph</td><td>${esc(result.label)} · ${result.cells} cells, ${result.edges.toLocaleString()} edges</td></tr>`
    + `<tr><td>Normalisation</td><td>${esc(result.normalisation_text)}</td></tr>`
    + `<tr><td>Measured radius</td><td>${c.radius.toFixed(4)} · ${result.radius_ratio_vs_rewired}× the rewired controls, ${result.radius_ratio_vs_weight_shuffle}× the weight-shuffled one</td></tr>`
    + `<tr><td>Non-normality</td><td>${c.henrici ?? 'not computed for this size'}</td></tr>`
    + `<tr><td>Transient gain</td><td>${c.transient_gain ?? 'not computed'}</td></tr>`
    + `<tr><td>Reciprocal edges</td><td>${(result.reciprocity * 100).toFixed(1)}%</td></tr>`
    + `<tr><td>Leading eigenvector spread</td><td>${c.participation_ratio ?? '–'} (1 means spread over every cell)</td></tr>`
    + `<tr><td>Cells fed only from their own loop</td><td>${result.closed.self_contained_cells} of ${result.closed.cells_with_input}</td></tr></table>`;
  if (c.leading_named) h += '<p class="note">Cells carrying the leading eigenvector: ' + c.leading_named.slice(0, 6).map(r => `${esc(r.cell)} (${(r.share * 100).toFixed(1)}%)`).join(', ') + '.</p>';
  h += `<p class="hint">${esc(result.note)}</p>`;
  $('detail').innerHTML = h;
}
async function go() {
  $('run').disabled = true; $('status').textContent = 'Computing eigenvalues…'; $('status').className = 'wb-status';
  try {
    const response = await fetch('/api/spectral', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: $('graph').value, normalisation: $('normalisation').value }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.status);
    result = data;
    $('against').innerHTML = Object.keys(data.conditions).filter(k => k !== 'measured').map(k => `<option value="${k}">${k.replace('_', ' ')}</option>`).join('');
    plane(); bars(); detail();
    $('status').textContent = 'Done.';
  } catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; }
  finally { $('run').disabled = false; }
}
async function controls() {
  try {
    saved = await (await fetch('/api/spectral-benchmark')).json();
    $('graph').innerHTML = Object.entries(saved.graphs).map(([k, v]) => `<option value="${k}">${v.label} (${v.cells} cells)</option>`).join('');
    let h = '<p class="note">Synapse-count threshold: the measured radius against randomisations that move edges, and against one that only moves weights.</p><table class="bench"><tr><th>Minimum synapses</th><th>Edges</th><th>Measured</th><th>Rewired</th><th>Weights shuffled</th><th>Measured ÷ rewired</th></tr>';
    for (const row of saved.thresholds.locomotion.rows)
      h += `<tr><td>${row.threshold}</td><td>${row.edges.toLocaleString()}</td><td>${row.measured_radius}</td><td>${Math.max(row.randomised_radius.degree_shuffle, row.randomised_radius.rewire_targets).toFixed(3)}</td><td>${row.randomised_radius.weight_shuffle}</td><td class="${row.ratio_vs_rewired > 1.5 ? 'hi' : ''}">${row.ratio_vs_rewired}</td></tr>`;
    h += '</table>';
    if (saved.coverage && saved.coverage.rows) {
      h += '<p class="note" style="margin-top:14px">Input coverage: cells with almost no observed input are the ones that can look self-contained inside a subset.</p><table class="bench"><tr><th>Minimum coverage</th><th>Cells kept</th><th>Measured</th><th>Shuffled</th><th>Cells fed only from their own loop</th></tr>';
      for (const row of saved.coverage.rows)
        h += `<tr><td>${(row.min_coverage * 100).toFixed(0)}%</td><td>${row.cells_kept}</td><td>${row.measured_radius}</td><td>${row.shuffled_radius}</td><td>${row.closed_cells}</td></tr>`;
      h += `</table><p class="hint">${esc(saved.coverage.note)}</p>`;
    }
    $('controls').innerHTML = h;
    go();
  } catch (e) { $('controls').innerHTML = '<p class="hint">No saved benchmark.</p>'; go(); }
}
$('run').onclick = go;
$('against').onchange = plane;
window.addEventListener('resize', () => { plane(); bars(); });
controls();
