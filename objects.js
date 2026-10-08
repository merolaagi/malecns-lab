'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const CLASSES = ['paper cup', 'mug', 'bottle', 'wine glass', 'can'];
const CONDITIONS = { intact: 'Inputs as built, adapting eye', raw_pixels: 'No lamina adaptation', shuffled: 'Shuffled visual inputs', random: 'Random visual inputs', olfactory_kcs: 'Moved onto olfactory KCs', assumed_features: 'Measured wiring, assumed features' };
const f2 = v => v == null ? '–' : (+v).toFixed(2);

function ctx(id) { const cv = $(id), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1; cv.width = w * d; cv.height = h * d; const g = cv.getContext('2d'); g.scale(d, d); g.font = '11px sans-serif'; return [g, w, h]; }
function fill(sel, items, value) { $(sel).innerHTML = Object.entries(items).map(([k, v]) => `<option value="${esc(k)}"${k === value ? ' selected' : ''}>${esc(v)}</option>`).join(''); }
const asMap = list => Object.fromEntries(list.map(x => [x, x]));
fill('cls', asMap(CLASSES), 'paper cup'); fill('morph_to', asMap(CLASSES), 'bottle'); fill('trained', asMap(CLASSES), 'paper cup');
fill('contrast', asMap(CLASSES), 'mug'); fill('condition', CONDITIONS, 'intact');

async function post(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json(); if (!r.ok) throw new Error(j.error || r.statusText); return j;
}

// --- the view -------------------------------------------------------------------------------------------------
function drawImage(img) {
  const [g, w, h] = ctx('human'); const side = Math.min(w, h), ox = (w - side) / 2;
  const cell = side / img.width; const max = Math.max(1e-6, ...img.values);
  for (let i = 0; i < img.height; i++) for (let j = 0; j < img.width; j++) {
    const v = Math.min(1, img.values[i * img.width + j] / max) * 255 | 0;
    g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(ox + j * cell, i * cell, cell + 0.6, cell + 0.6);
  }
}
function drawHex(id, columns, values, diverging) {
  const [g, w, h] = ctx(id); const extent = 72, side = Math.min(w, h), ox = w / 2, oy = h / 2, k = side / (2 * extent);
  const r = 5 / Math.sqrt(3) * k * 1.02; const max = Math.max(1e-6, ...values.map(Math.abs));
  columns.forEach(([x, y], i) => {
    const v = values[i] / max;
    if (diverging) g.fillStyle = v >= 0 ? `rgba(194,232,153,${Math.min(1, v * 1.4)})` : `rgba(242,139,130,${Math.min(1, -v * 1.4)})`;
    else { const c = Math.max(0, v) * 255 | 0; g.fillStyle = `rgb(${c},${c},${c})`; }
    g.beginPath();
    for (let a = 0; a < 6; a++) { const t = Math.PI / 3 * a; const px = ox + x * k + r * Math.cos(t), py = oy - y * k + r * Math.sin(t); a ? g.lineTo(px, py) : g.moveTo(px, py); }
    g.closePath(); g.fill();
  });
}
let viewTimer = null, viewBusy = false, viewAgain = false;
function labels() {
  $('lightv').textContent = (+$('light').value).toFixed(2); $('azv').textContent = $('light_az').value + '°';
  $('distv').textContent = $('distance').value + ' cm'; $('rotv').textContent = $('rotation').value + '°'; $('morphv').textContent = (+$('morph').value).toFixed(2);
}
async function refreshView() {
  if (viewBusy) { viewAgain = true; return; }
  viewBusy = true; labels();
  try {
    const v = await post('/api/objects-view', { cls: $('cls').value, light: +$('light').value, light_az: +$('light_az').value, distance: +$('distance').value,
      rotation: +$('rotation').value, morph: +$('morph').value, morph_to: $('morph_to').value, seed: +$('seed').value, trained: $('trained').value });
    drawImage(v.image); drawHex('fly', v.columns, v.photoreceptors, false); drawHex('lamina', v.columns, v.contrast, true);
    setSource(v.source);
    $('viewstats').innerHTML = `<span>Visual inputs active <b>${v.vpn_active} / ${v.vpn_total}</b></span><span>Kenyon cells active <b>${v.kc_active}</b></span>`
      + `<span>Memory response after learning “${esc(v.trained)}” with reward <b>${v.score.toFixed(3)}</b></span>`;
  } catch (e) { $('viewstats').textContent = e.message.includes('already running') ? 'Busy with an experiment; try again in a moment.' : e.message; }
  viewBusy = false;
  if (viewAgain) { viewAgain = false; refreshView(); }
}
for (const id of ['cls', 'light', 'light_az', 'distance', 'rotation', 'morph', 'morph_to']) $(id).addEventListener('input', () => { labels(); clearTimeout(viewTimer); viewTimer = setTimeout(refreshView, 120); });
function setSource(source) { const b = $('source'); b.textContent = source === 'measured' ? 'measured visual inputs' : 'synthetic visual inputs'; b.className = 'badge ' + source; }

// --- plots ------------------------------------------------------------------------------------------------------
function linePlot(id, xs, series, lo, hi, xfmt) {
  const [g, w, h] = ctx(id); if (!xs.length) return;
  const X = i => 46 + i * (w - 64) / (xs.length - 1), Y = v => 12 + (hi - v) / (hi - lo) * (h - 36);
  g.strokeStyle = '#2b372e'; g.fillStyle = '#99aa9c';
  for (const v of [lo, (lo + hi) / 2, hi]) { g.beginPath(); g.moveTo(46, Y(v)); g.lineTo(w - 12, Y(v)); g.stroke(); g.fillText(v.toFixed(2), 4, Y(v) + 4); }
  xs.forEach((x, i) => { if (i % Math.ceil(xs.length / 9) === 0 || i === xs.length - 1) g.fillText(xfmt(x), X(i) - 8, h - 8); });
  for (const [values, colour, dash] of series) {
    g.strokeStyle = colour; g.setLineDash(dash || []); g.lineWidth = 1.8; g.beginPath();
    values.forEach((v, i) => i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))); g.stroke(); g.setLineDash([]); g.lineWidth = 1;
  }
}
function heat(id, names, m, lo, hi) {
  const [g, w, h] = ctx(id); const n = names.length, left = 70, top = 8, size = Math.min((w - left - 8) / n, (h - top - 60) / n);
  m.forEach((row, i) => row.forEach((v, j) => {
    const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
    g.fillStyle = `rgb(${20 + t * 174 | 0},${30 + t * 202 | 0},${24 + t * 129 | 0})`; g.fillRect(left + j * size, top + i * size, size - 1, size - 1);
    g.fillStyle = t > 0.6 ? '#0d130f' : '#d6e2d8'; g.fillText(v.toFixed(2), left + j * size + size / 2 - 11, top + i * size + size / 2 + 4);
  }));
  g.fillStyle = '#99aa9c';
  names.forEach((nm, i) => { g.fillText(nm, 2, top + i * size + size / 2 + 4); g.save(); g.translate(left + i * size + size / 2, top + n * size + 6); g.rotate(Math.PI / 5); g.fillText(nm, 0, 4); g.restore(); });
}
function showSimilarity(sim) {
  const off = [];
  sim.vpn.forEach((r, i) => r.forEach((v, j) => off.push(v)));
  heat('simvpn', sim.classes, sim.vpn, Math.min(...off), 1); heat('simkc', sim.classes, sim.kc, 0, Math.max(...sim.kc.flat()));
  const lc = sim.light_change;
  $('lightnote').textContent = `Same object and view, only the light level changed: visual-input codes stay ${f2(lc['0.4 vs 1.0'].vpn)} / ${f2(lc['2.0 vs 1.0'].vpn)} alike (0.4× and 2× light against 1×), Kenyon-cell codes ${f2(lc['0.4 vs 1.0'].kc)} / ${f2(lc['2.0 vs 1.0'].kc)}. The diagonal is two different exemplars of the same class.`;
}
function showRun(r) {
  setSource(r.source);
  const sets = Object.entries(r.test_sets), others = Object.keys(sets[0][1].pairwise_auc);
  let h = `<p class="note">${esc(r.condition_text)}. ${esc(r.protocol_text)}. ${r.circuit.visual_inputs} inputs onto ${r.circuit.receiving_kcs} Kenyon cells, ${r.circuit.active_kcs} active per view. Measured feature mix for ${r.circuit.measured_features ?? 0}, measured receptive field for ${r.circuit.measured_positions}; mean visual share ${f2(r.circuit.mean_visual_share)}.</p>`;
  h += `<table class="bench"><tr><th>Test views</th><th>Recognised, one look (AUC)</th><th>Eight glimpses</th>${others.map(c => `<th>vs ${esc(c)}</th>`).join('')}</tr>`;
  for (const [k, v] of sets) h += `<tr><td>${esc(k)}</td><td class="${v.auc_one_look > 0.8 ? 'hi' : v.auc_one_look < 0.6 ? 'lo' : ''}">${f2(v.auc_one_look)}</td><td>${f2(v.auc_all_glimpses)}</td>${others.map(c => `<td>${f2(v.pairwise_auc[c])}</td>`).join('')}</tr>`;
  $('result').innerHTML = h + '</table><p class="hint">AUC: chance that a test view of the trained object gets a stronger memory response than a view of another object. 0.5 is chance, 1 is perfect. Familiar views use the training ranges with new draws; each other row moves one factor outside them.</p>';
  const rows = r.morph.rows;
  linePlot('morphplot', rows.map(x => x.morph), [[rows.map(x => x.relative), '#c2e899'], [rows.map(x => x.vpn_overlap), '#99aa9c', [4, 3]], [rows.map(x => x.kc_overlap), '#f2c46f', [2, 3]]],
    0, Math.max(1.05, ...rows.map(x => x.relative)), x => x.toFixed(2));
  linePlot('glimpseplot', r.glimpses.map(x => x.glimpses), [[r.glimpses.map(x => x.auc), '#6fc7bc']], 0.4, 1, x => String(x));
  showSimilarity(r.similarity);
}
$('run').addEventListener('click', async () => {
  $('run').disabled = true; $('status').textContent = 'Training and testing (about ten seconds)…'; $('status').className = 'wb-status';
  try {
    showRun(await post('/api/objects', { seed: +$('seed').value, trained: $('trained').value, protocol: $('protocol').value, contrast: $('contrast').value, condition: $('condition').value }));
    $('status').textContent = 'Done';
  } catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; }
  $('run').disabled = false;
});

async function bench() {
  try {
    const r = await fetch('/api/objects-benchmark'); if (!r.ok) throw new Error('No saved benchmark yet: python3 objects.py --benchmark');
    const b = await r.json(); setSource(b.source);
    const sets = Object.keys(b.summary.intact.test_sets);
    let h = `<table class="bench"><tr><th>Circuit</th><th></th>${sets.map(s => `<th>${esc(s)}</th>`).join('')}<th>8 glimpses</th></tr>`;
    const row = (label, s) => `<tr><td>${esc(label)}</td><td></td>${sets.map(k => `<td class="${s.test_sets[k].auc_one_look > 0.8 ? 'hi' : s.test_sets[k].auc_one_look < 0.6 ? 'lo' : ''}">${f2(s.test_sets[k].auc_one_look)}</td>`).join('')}<td>${f2(s.glimpses[s.glimpses.length - 1].auc)}</td></tr>`;
    for (const [c, s] of Object.entries(b.summary)) h += row(b.conditions[c], s);
    h += row('Cup rewarded, mug punished (differential)', b.differential);
    h += '</table><h3 style="font-size:12px;margin:14px 0 6px">Paired against the measured circuit, one look (mean AUC difference, seeds better)</h3>';
    h += `<table class="bench"><tr><th>Circuit</th><th></th>${sets.map(s => `<th>${esc(s)}</th>`).join('')}</tr>`;
    for (const [c, p] of Object.entries(b.paired_vs_intact)) h += `<tr><td>${esc(b.conditions[c])}</td><td></td>${sets.map(k => `<td class="${Math.abs(p[k].mean) > 2 * p[k].sd && Math.abs(p[k].mean) > 0.03 ? (p[k].mean > 0 ? 'hi' : 'lo') : ''}">${p[k].mean >= 0 ? '+' : ''}${f2(p[k].mean)} (${p[k].better}/${p[k].n})</td>`).join('')}</tr>`;
    const cupmug = s => f2(s.test_sets.familiar.pairwise_auc.mug), dm = b.differential.test_sets;
    h += `</table><p class="note">The differential row's AUC against all four objects is low because untrained bottles and glasses now score like cups: the memory becomes “not a mug”. Cup against mug, familiar views: ${cupmug(b.summary.intact)} with reward only, ${cupmug(b.differential)} after differential training. Dim light: ${f2(b.summary.intact.test_sets['dim light'].pairwise_auc.mug)} vs ${f2(dm['dim light'].pairwise_auc.mug)}.</p>`;
    $('bench').innerHTML = h;
    if (!$('result').querySelector('table')) {
      const s = b.summary.intact;
      linePlot('morphplot', s.morph.map(x => x.morph), [[s.morph.map(x => x.relative), '#c2e899'], [s.morph.map(x => x.vpn_overlap), '#99aa9c', [4, 3]], [s.morph.map(x => x.kc_overlap), '#f2c46f', [2, 3]]], 0, 1.05, x => x.toFixed(2));
      linePlot('glimpseplot', s.glimpses.map(x => x.glimpses), [[s.glimpses.map(x => x.auc), '#6fc7bc']], 0.4, 1, x => String(x));
      showSimilarity(s.similarity);
    }
  } catch (e) { $('bench').innerHTML = `<p class="hint">${esc(e.message)}</p>`; }
}
labels(); refreshView(); bench();
addEventListener('resize', () => { clearTimeout(viewTimer); viewTimer = setTimeout(refreshView, 200); });
