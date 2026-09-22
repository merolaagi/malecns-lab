'use strict';
const NC = window.NeuronCore;
const $ = id => document.getElementById(id);
const COL = { exc: '#6fc7bc', inh: '#f28b82', zero: '#5b6a5e', on: '#e9be75', idle: '#3a473d', text: '#e7eee5', muted: '#99aa9c', line: '#2b372e' };
const SHOWN = 16;
let D = null, channels = [], active = [], result = null, raf = null, probToken = 0;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const status = (t, err) => { $('status').textContent = t; $('status').className = 'wb-status' + (err ? ' err' : ''); };

function params() { return { gmax: +$('gmax').value, rate: +$('rate').value }; }

async function load(circuit, id, extra = '') {
  status('Loading…');
  try {
    const r = await fetch('/api/neuron?circuit=' + encodeURIComponent(circuit) + '&id=' + encodeURIComponent(id) + extra);
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'HTTP ' + r.status);
    D = d;
    const w = NC.weights(d.inputs.map(x => x.synapses));
    channels = d.inputs.map((x, i) => ({ weight: w[i], sign: x.sign, active: false }));
    active = channels.map(() => false);
    const top = Math.min(3, channels.length);
    for (let i = 0, k = 0; i < channels.length && k < top; i++) if (channels[i].sign > 0) { active[i] = true; k++; }
    $('circuit').value = d.circuit; $('bid').value = d.cell.bodyId;
    history.replaceState(null, '', '/neuron?circuit=' + d.circuit + '&id=' + d.cell.bodyId);
    renderInfo(); renderNumbers(); run(); probability();
    status('Loaded ' + d.inputs.length + ' measured inputs');
  } catch (e) { status(e.message, true); }
}

function cellTitle() { return D.cell.instance || D.cell.type || String(D.cell.bodyId); }
function renderInfo() {
  const c = D.cell, isKC = c.role === 'KC';
  $('name').textContent = cellTitle();
  $('subtitle').textContent = [c.type, c.superclass || c.class || c.role, 'body ' + c.bodyId].filter(Boolean).join(', ');
  const exc = D.inputs.filter(x => x.sign > 0).length, inh = D.inputs.filter(x => x.sign < 0).length, zero = D.inputs.length - exc - inh;
  const row = (k, v) => v === undefined || v === null || v === '' ? '' : '<tr><td>' + k + '</td><td>' + esc(v) + '</td></tr>';
  $('info').innerHTML = '<h2>This cell</h2><table class="kv">' + row('Body ID', c.bodyId) + row('Type', c.type) + row('Class', c.superclass || c.class)
    + row('Side', c.rootSide || c.somaSide) + row('Transmitter', c.nt) + row('Inputs', D.inputs.length + ' (' + exc + ' excitatory, ' + inh + ' inhibitory' + (zero ? ', ' + zero + ' zero' : '') + ')')
    + row('Output partners', D.output_partners)
    + (D.quality ? row('Input in this subset', D.quality.coverage_in === null ? 'unknown' : Math.round(D.quality.coverage_in * 100) + '% of ' + D.quality.post_total + ' synapses') : '')
    + (D.quality && D.quality.nt ? row('Per-synapse transmitter', Object.entries(D.quality.nt.top_share).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([k, v]) => k + ' ' + Math.round(v * 100) + '%').join(', ')) : '')
    + (D.regions ? row('Input regions', D.regions.input.slice(0, 3).map(x => x.roi + ' ' + Math.round(x.fraction * 100) + '%').join(', ')) + row('Output regions', D.regions.output.slice(0, 3).map(x => x.roi + ' ' + Math.round(x.fraction * 100) + '%').join(', ')) : '')
    + '</table>'
    + (D.quality && D.quality.coverage_in !== null && D.quality.coverage_in < 0.25 ? '<p class="warn">Only ' + Math.round(D.quality.coverage_in * 100) + '% of this cell’s input is in the lab’s subset. The simulation below is driven by a small slice of its real input.</p>' : '')
    + '<p style="display:flex;gap:14px;flex-wrap:wrap;font-size:12px">' + (D.circuit === 'locomotion' ? '<a href="/atlas#cell=' + c.bodyId + '&view=anatomy" style="color:var(--accent)">Show in circuit atlas</a>' : '')
    + '<a href="/3d?cells=' + c.bodyId + '" style="color:var(--accent)">View in 3D</a><a target="_blank" rel="noopener" href="' + esc(NG.url({ cells: [c.bodyId], position: c.somaLocation || null, title: cellTitle() })) + '" style="color:var(--accent)">Neuroglancer ↗</a></p>'
    + '<h2 style="margin-top:18px">Measured inputs</h2><p class="hint">Click to switch an input on or off. Sorted by synapse count.</p><ul class="inlist" id="inlist"></ul>'
    + (D.modulatory.length ? '<p class="note">' + D.modulatory.length + ' PAM dopamine inputs (' + D.modulatory.reduce((a, b) => a + b.synapses, 0) + ' synapses) are listed in the data but not simulated here. In the network model they only gate plasticity.</p>' : '')
    + (D.circuit === 'learning' ? (D.inputs.some(x => x.note && x.note.startsWith('Sign from the MaleCNS consensus')) ? '<p class="note">Input signs come from the MaleCNS consensus transmitter of each input cell.</p>' : '<p class="warn">The learning dataset has no transmitter predictions, so every PN input is treated as excitatory. Some PN classes are GABAergic in reality. Build data/quality.json to replace this assumption.</p>') : '')
    + (zero ? '<p class="warn">' + zero + ' inputs have unclear transmitter and carry zero weight, matching the network model.</p>' : '')
    + (isKC ? '<p class="note">Kenyon cells receive PN input on claws in the calyx. As far as published recordings show, several claws must be active together to fire a Kenyon cell.</p>' : '');
  renderList();
}
function renderList() {
  $('inlist').innerHTML = D.inputs.map((x, i) => '<li data-i="' + i + '" class="' + (active[i] ? 'on' : '') + '"><span class="sign ' + (x.sign > 0 ? 'p">+' : x.sign < 0 ? 'n">−' : 'z">0') + '</span><b title="' + esc(x.name) + '">' + esc(x.name) + '</b><span>' + x.synapses + '</span></li>').join('');
}
$('info').addEventListener('click', e => { const li = e.target.closest('li[data-i]'); if (!li) return; const i = +li.dataset.i; active[i] = !active[i]; renderList(); run(); });

function renderNumbers() {
  const box = $('numRows');
  if (!D.number_codes) { box.innerHTML = ''; return; }
  const n = D.number_codes.codes.scalar.length;
  const row = (cond, label) => '<div class="wb-row"><span class="lab">' + label + '</span>' + Array.from({ length: n }, (_, k) => {
    const hits = D.number_codes.codes[cond][k].length;
    return '<button class="chip" data-num="' + cond + ':' + k + '" title="' + hits + ' of this cell’s inputs are active for ' + (k + 1) + '">' + (k + 1) + (hits ? ' · ' + hits : '') + '</button>';
  }).join('') + '</div>';
  box.innerHTML = row('scalar', 'Scalar number code') + row('onehot', 'One-hot number code')
    + '<p class="hint">These are the codes the numerosity experiment feeds to PNs (seed ' + D.number_codes.seed + '). The small count is how many of this cell’s inputs each number switches on.</p>';
}
document.addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b || !D) return;
  if (b.dataset.num) { const [cond, k] = b.dataset.num.split(':'); const on = new Set(D.number_codes.codes[cond][+k]); active = channels.map((_, i) => on.has(i)); renderList(); run(); }
  if (b.dataset.preset) {
    const p = b.dataset.preset;
    active = channels.map(ch => p === 'all' ? true : p === 'exc' ? ch.sign > 0 : p === 'random' ? Math.random() < 0.2 : false);
    renderList(); run();
  }
});

function run() {
  if (!D) return;
  const ch = channels.map((c, i) => ({ ...c, active: active[i] }));
  result = NC.simulate({ channels: ch, ...params() });
  const a = active.filter(Boolean).length;
  $('mActive').textContent = a + ' of ' + channels.length;
  $('mSpikes').textContent = result.spikes.length;
  $('mPeak').textContent = Math.round(result.peakVd) + ' mV';
  drawMorph(null); animate();
}
function probability() {
  const token = ++probToken; $('mProb').textContent = '…';
  setTimeout(() => {
    if (token !== probToken || !D) return;
    const p = NC.fireProbability(channels, { ...params(), trials: channels.length > 60 ? 30 : 60 });
    if (token === probToken) $('mProb').textContent = Math.round(p * 100) + '%';
  }, 30);
}

/* ---------- morphology schematic ---------- */
function vcol(v) { const x = Math.max(0, Math.min(1, (v + 70) / 100)); return x < 0.18 ? COL.idle : x < 0.5 ? COL.on : '#e8907a'; }
function drawMorph(frame) {
  const n = Math.min(SHOWN, channels.length), extra = channels.length - n, rows = n + (extra ? 1 : 0);
  const H = Math.max(240, rows * 26 + 50), cy = H / 2, svg = $('morph');
  svg.setAttribute('viewBox', '0 0 680 ' + H);
  const vd = frame ? frame.vd : -65, v = frame ? frame.v : -65, recent = frame ? frame.recent : new Set();
  let s = '';
  for (let r = 0; r < rows; r++) {
    const y = 30 + r * 26, isExtra = r === n, idx = isExtra ? null : r;
    const on = isExtra ? active.slice(n).some(Boolean) : active[idx], sign = isExtra ? 1 : channels[idx].sign;
    const stroke = on ? (sign < 0 ? COL.inh : vcol(vd)) : COL.line;
    const w = isExtra ? 1.5 : 1 + channels[idx].weight * 3;
    s += '<path d="M200 ' + y + ' Q250 ' + y + ' 290 ' + cy + '" fill="none" stroke="' + stroke + '" stroke-width="' + w + '"/>';
    const fill = on ? (sign < 0 ? COL.inh : sign > 0 ? COL.on : COL.zero) : 'transparent';
    const rr = !isExtra && recent.has(idx) ? 11 : 8;
    const label = isExtra ? extra + ' weaker inputs' : (D.inputs[idx].name || '').slice(0, 20) + ', ' + D.inputs[idx].synapses;
    s += '<g data-ch="' + (isExtra ? 'extra' : idx) + '" style="cursor:pointer"><circle cx="190" cy="' + y + '" r="' + rr + '" fill="' + fill + '" stroke="' + (sign < 0 ? COL.inh : sign > 0 ? COL.exc : COL.zero) + '"/>'
      + '<text x="172" y="' + (y + 4) + '" text-anchor="end" fill="' + COL.muted + '" font-size="11">' + esc(label) + '</text></g>';
  }
  s += '<path d="M290 ' + cy + ' L440 ' + cy + '" stroke="' + vcol(vd) + '" stroke-width="5" stroke-linecap="round"/>';
  s += '<path d="M350 ' + cy + ' L350 ' + (cy + 60) + '" stroke="' + COL.line + '" stroke-width="2"/><circle cx="350" cy="' + (cy + 74) + '" r="13" fill="none" stroke="' + COL.muted + '"/>';
  s += '<text x="372" y="' + (cy + 78) + '" fill="' + COL.muted + '" font-size="11">Soma, off the signal path</text>';
  s += '<rect x="440" y="' + (cy - 10) + '" width="34" height="20" rx="3" fill="' + vcol(v) + '"/><text x="457" y="' + (cy - 18) + '" text-anchor="middle" fill="' + COL.muted + '" font-size="11">Spike initiation zone</text>';
  s += '<path d="M474 ' + cy + ' L630 ' + cy + '" stroke="' + (v > -20 ? '#e8907a' : COL.line) + '" stroke-width="3" stroke-linecap="round"/>';
  s += '<text x="630" y="' + (cy + 20) + '" text-anchor="end" fill="' + COL.muted + '" font-size="11">' + D.output_partners + ' output partners</text>';
  s += '<text x="300" y="' + (cy - 12) + '" fill="' + COL.muted + '" font-size="11">Dendrite</text>';
  svg.innerHTML = s;
}
$('morph').addEventListener('click', e => {
  const g = e.target.closest('g[data-ch]'); if (!g) return;
  if (g.dataset.ch === 'extra') { const on = !active.slice(SHOWN).some(Boolean); for (let i = SHOWN; i < active.length; i++) active[i] = on; }
  else active[+g.dataset.ch] = !active[+g.dataset.ch];
  renderList(); run();
});

/* ---------- traces ---------- */
function draw(upto) {
  const cv = $('trace'), w = cv.clientWidth, H = cv.clientHeight, d = devicePixelRatio || 1;
  cv.width = w * d; cv.height = H * d; const c = cv.getContext('2d'); c.scale(d, d);
  const tr = result.trace, N = Math.min(upto, tr.t.length), T = tr.t.at(-1) || 200, X = t => 46 + t / T * (w - 56);
  const bands = [[4, 80, -90, -20, 'Dendrite', 'mV'], [86, 166, -80, 50, 'Spike zone', 'mV'], [172, 246, 0, 1, 'Gates', ''], [252, 326, -900, 900, 'Currents', 'µA/cm²']];
  c.font = '10px sans-serif';
  bands.forEach(([y0, y1, lo, hi, l, u]) => {
    c.strokeStyle = COL.line; c.lineWidth = 1; c.beginPath(); c.moveTo(46, y1); c.lineTo(w - 10, y1); c.stroke();
    c.fillStyle = COL.muted; c.fillText(l, 4, y0 + 10); c.fillText(hi + (u ? '' : ''), 4, y0 + 22); c.fillText(String(lo), 4, y1 - 2);
  });
  const Y = (b, v) => { const [y0, y1, lo, hi] = bands[b]; return y1 - (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * (y1 - y0 - 4); };
  const line = (b, arr, col, width = 1.3) => { c.strokeStyle = col; c.lineWidth = width; c.beginPath(); for (let i = 0; i < N; i++) { const x = X(tr.t[i]), y = Y(b, arr[i]); i ? c.lineTo(x, y) : c.moveTo(x, y); } c.stroke(); };
  line(0, tr.vd, COL.exc); line(1, tr.v, '#e8907a'); line(2, tr.m, '#8ab4f8'); line(2, tr.h, COL.muted); line(2, tr.n, COL.on);
  line(3, tr.ina.map(x => -x), '#8ab4f8', 1); line(3, tr.ik.map(x => -x), COL.on, 1);
  const tNow = tr.t[N - 1] ?? 0;
  c.fillStyle = COL.on; for (const [t, j] of result.events) { if (t > tNow) break; c.fillRect(X(t), 78, 1, 3 + channels[j].weight * 3); }
  [['m', '#8ab4f8'], ['h', COL.muted], ['n', COL.on]].forEach(([l, col], i) => { c.fillStyle = col; c.fillText(l, w - 60 + i * 16, 184); });
  [['Na⁺ inward', '#8ab4f8'], ['K⁺ outward', COL.on]].forEach(([l, col], i) => { c.fillStyle = col; c.fillText(l, w - 150 + i * 72, 264); });
  c.fillStyle = COL.muted; c.fillText('0', 46, H - 2); c.textAlign = 'right'; c.fillText(Math.round(T) + ' ms', w - 10, H - 2); c.textAlign = 'left';
  if (N > 0) {
    const recent = new Set(result.events.filter(([t]) => t <= tNow && t > tNow - 3).map(e => e[1]));
    drawMorph({ vd: tr.vd[N - 1], v: tr.v[N - 1], recent });
  }
}
function animate() {
  cancelAnimationFrame(raf);
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) { draw(result.trace.t.length); return; }
  let i = 0; const step = () => { i += 14; draw(i); if (i < result.trace.t.length) raf = requestAnimationFrame(step); }; step();
}
window.addEventListener('resize', () => result && draw(result.trace.t.length));

for (const [id, dp] of [['gmax', 3], ['rate', 0]]) {
  $(id).addEventListener('input', () => { $(id + 'Out').textContent = (+$(id).value).toFixed(dp); });
  $(id).addEventListener('change', () => { run(); probability(); });
}
$('load').onclick = () => load($('circuit').value, $('bid').value);
$('bid').addEventListener('keydown', e => { if (e.key === 'Enter') $('load').click(); });
$('randomKc').onclick = () => load('learning', 'random', '&seed=' + Math.floor(Math.random() * 1e6));
const q = new URLSearchParams(location.search);
load(q.get('circuit') || 'learning', q.get('id') || '57729');
