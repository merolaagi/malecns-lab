'use strict';
const $ = id => document.getElementById(id);
let result = null, saved = null;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
function draw() {
  if (!result) return;
  const cv = $('memory'), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1;
  cv.width = w * d; cv.height = h * d; const g = cv.getContext('2d'); g.scale(d, d); g.font = '11px sans-serif';
  const values = result.memory_by_delay, n = values.length;
  const X = i => 44 + i * (w - 60) / Math.max(1, n - 1), Y = v => h - 26 - v * (h - 42);
  g.strokeStyle = '#2b372e'; g.fillStyle = '#99aa9c';
  for (const v of [0, 0.5, 1]) { g.beginPath(); g.moveTo(44, Y(v)); g.lineTo(w - 14, Y(v)); g.stroke(); g.fillText(v.toFixed(1), 8, Y(v) + 4); }
  g.strokeStyle = '#c2e899'; g.lineWidth = 1.8; g.beginPath();
  values.forEach((v, i) => i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))); g.stroke(); g.lineWidth = 1;
  g.fillStyle = '#c2e899'; values.forEach((v, i) => { g.beginPath(); g.arc(X(i), Y(v), 2.5, 0, 7); g.fill(); });
  g.fillStyle = '#99aa9c'; g.fillText('delay 1', 40, h - 8); g.fillText('delay ' + n, w - 60, h - 8);
}
function show(data) {
  result = data;
  $('detail').innerHTML = '<table class="kv">'
    + `<tr><td>Placement</td><td>${esc(data.placement_text)}</td></tr>`
    + `<tr><td>Network</td><td>${data.cells} cells, ${data.edges.toLocaleString()} edges</td></tr>`
    + `<tr><td>Inhibitory</td><td>${data.inhibitory_cells} cells (${(data.inhibitory_share * 100).toFixed(1)}%)</td></tr>`
    + `<tr><td>Natural radius</td><td>${data.natural_radius} · after scaling ${data.effective_radius}</td></tr>`
    + `<tr><td>Memory capacity</td><td>${data.memory_capacity}</td></tr>`
    + `<tr><td>Integration</td><td>${data.integration_score}</td></tr>`
    + `<tr><td>Activity</td><td>mean ${data.mean_activity}, saturated ${(data.saturated_fraction * 100).toFixed(1)}%</td></tr></table>`;
  draw();
}
async function go() {
  $('run').disabled = true; $('status').textContent = 'Running…'; $('status').className = 'wb-status';
  try {
    const response = await fetch('/api/inhibition', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ placement: $('placement').value, scaling: $('scaling').value, cells: +$('cells').value, seed: +$('seed').value }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.status);
    show(data); $('status').textContent = 'Done.';
  } catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; }
  finally { $('run').disabled = false; }
}
async function benchmark() {
  try {
    saved = await (await fetch('/api/inhibition-benchmark')).json();
    const block = saved.shared;
    $('placement').innerHTML = Object.entries(block.placements).map(([k, v]) => `<option value="${k}">${v}</option>`).join('');
    let h = `<p class="hint">${esc(block.scaling_text)}. ${block.seeds} seeds.</p><table class="bench"><tr><th>Placement</th><th>Inhibitory cells</th><th>Natural radius</th><th>Memory capacity</th><th>Integration</th><th>Activity</th></tr>`;
    for (const [name, row] of Object.entries(block.summary))
      h += `<tr><td>${esc(row.text)}</td><td>${row.inhibitory_cells}</td><td class="${name === 'measured' ? 'hi' : ''}">${row.natural_radius}</td><td>${row.memory_capacity} ± ${row.memory_sd}</td><td>${row.integration_score}</td><td>${row.mean_activity}</td></tr>`;
    $('summary').innerHTML = h + '</table>';
    const paired = saved.paired;
    let p = `<p class="note">Measured placement against each alternative, ${Object.values(paired.rows)[0].seeds} seeds, same subsample per seed.</p><table class="bench"><tr><th>Comparison</th><th>Memory difference</th><th>Seeds favouring measured</th><th>t</th></tr>`;
    for (const [name, row] of Object.entries(paired.rows))
      p += `<tr><td>measured − ${name.replace('_', ' ')}</td><td class="${row.t > 2 ? 'hi' : ''}">${row.mean_difference >= 0 ? '+' : ''}${row.mean_difference} ± ${row.sd}</td><td>${row.favours_reference} of ${row.seeds}</td><td>${row.t ?? '–'}</td></tr>`;
    $('paired').innerHTML = p + `</table><p class="hint">${esc(paired.note)}</p>`;
  } catch (e) { $('summary').innerHTML = '<p class="hint">No saved benchmark.</p>'; }
  go();
}
$('run').onclick = go;
window.addEventListener('resize', draw);
benchmark();
