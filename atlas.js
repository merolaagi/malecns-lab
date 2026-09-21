'use strict';
const C = window.AtlasCore;
const COLOR = { dn: '#c2e899', in: '#6fc7bc', mn: '#e9be75', sn: '#e8907a' };
const IN_EDGE = '#8ab4f8', OUT_EDGE = '#f28b82';
const VOXEL_NM = 8; // MaleCNS voxel size, used only for the scale bar.
const LAYER_TEXT = {
  dn: { m: 'Receive the stimulus current after 0.5 s; left/right bias sets which side. Same LIF dynamics as every other cell.',
        w: 'Brain-side inputs (visual, steering) are not in the subset, so these cells respond only to injected current and weak VNC feedback.' },
  in: { m: 'Colored groups are developmental hemilineages, read from type names such as IN19A016. ACh +1, GABA and glutamate −1, unclear 0.',
        w: 'The computational core and the largest recurrent block, yet the README and motion UI count only descending, motor and sensory cells.' },
  mn: { m: 'Pooled by side and leg subclass into six mean rates. Rate/50 sets oscillator speed and stride. Muscles are not modeled.',
        w: '59 cells exit through non-leg nerves but are still pooled into front or hind stride. 126 have unclear transmitter and act as silent sources.' },
  sn: { m: 'Chordotonal organ and campaniform sensilla afferents, all ACh. They receive a hand-designed stance × stride current.',
        w: 'Their somata sit in the legs, so positions are estimated from partners. Front legs have 11 cells against 40 each for middle and hind.' },
};

let A = null, Q = null;
fetch('/api/quality').then(r => r.ok ? r.json() : null).then(q => { Q = q ? q.cells : null; if (A && S.sel?.kind === 'cell') renderPanel(); }).catch(() => {});
const S = { view: 'circuit', layer: null, group: null, sel: null, hover: null, cam: { s: 1, x: 0, y: 0 },
            regionMode: false, threshold: 0, drag: null, rect: null };
const cv = document.getElementById('map'), ctx = cv.getContext('2d');
const tip = document.getElementById('tip'), panel = document.getElementById('panel');
let W = 0, H = 0, hits = [], layerCache = {}, bounds = null;

const fmt = n => Math.round(n).toLocaleString();
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const cellName = i => A.nodes[i].raw.instance || A.nodes[i].raw.type || String(A.nodes[i].raw.bodyId);
const findGroup = (layer, key) => A.groups[layer].find(g => g.key === key);
const shortGroup = k => k.replace(/^Hemilineage /, '').replace(/ MN$/, '');

fetch('/api/circuit').then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); }).then(data => {
  A = C.build(data);
  // Fit the VNC. The six descending somata sit in the brain, far anterior; an indicator points to them.
  const pts = A.nodes.filter(n => n.x !== null && n.layer !== 'dn');
  bounds = { x0: Math.min(...pts.map(n => n.x)), x1: Math.max(...pts.map(n => n.x)), y0: Math.min(...pts.map(n => n.y)), y1: Math.max(...pts.map(n => n.y)) };
  fromHash(); resize(); render();
}).catch(e => { panel.innerHTML = '<h2>Circuit unavailable</h2><p class="sub">Couldn’t load /api/circuit (' + esc(e.message) + '). Start the lab with server.py and reload.</p>'; });

/* ---------- navigation ---------- */
function go(view, opts = {}) {
  Object.assign(S, { view }, opts);
  if (view !== 'anatomy') S.rect = null;
  render();
}
function select(sel) { S.sel = sel; toHash(); render(); }
function toHash() {
  const s = S.sel;
  const h = s?.kind === 'cell' ? 'cell=' + A.nodes[s.i].raw.bodyId : s?.kind === 'group' ? 'group=' + s.layer + ':' + encodeURIComponent(s.key)
    : s?.kind === 'layer' ? 'layer=' + s.layer : '';
  history.replaceState(null, '', h ? '#' + h + '&view=' + S.view : location.pathname);
}
function fromHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get('cell')) { const n = A.nodes.find(n => String(n.raw.bodyId) === p.get('cell')); if (n) { S.sel = { kind: 'cell', i: n.i }; S.layer = n.layer; S.group = n.group; } }
  else if (p.get('group')) { const [l, k] = p.get('group').split(':'); if (A.groups[l] && findGroup(l, decodeURIComponent(k))) { S.layer = l; S.group = decodeURIComponent(k); S.sel = { kind: 'group', layer: l, key: S.group }; } }
  else if (p.get('layer') && A.groups[p.get('layer')]) { S.layer = p.get('layer'); S.sel = { kind: 'layer', layer: S.layer }; }
  const v = p.get('view'); S.view = v === 'anatomy' ? 'anatomy' : S.layer && v === 'layer' ? 'layer' : S.layer && S.sel?.kind !== 'layer' ? 'layer' : 'circuit';
}

/* ---------- geometry ---------- */
function resize() {
  const dpr = window.devicePixelRatio || 1;
  W = cv.clientWidth; H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', () => { resize(); draw(); });
function fitScale() { return Math.min((W - 80) / (bounds.x1 - bounds.x0), (H - 80) / (bounds.y1 - bounds.y0)); }
// Dorsal view, anterior up. Fly's left (larger x) on the viewer's left.
function toScreen(n) {
  const f = fitScale() * S.cam.s;
  return [W / 2 + ((bounds.x0 + bounds.x1) / 2 - n.x) * f + S.cam.x, H / 2 + (n.y - (bounds.y0 + bounds.y1) / 2) * f + S.cam.y];
}

function arrow(x1, y1, cx, cy, x2, y2, width, color, alpha) {
  ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.quadraticCurveTo(cx, cy, x2, y2); ctx.stroke();
  const a = Math.atan2(y2 - cy, x2 - cx), h = 5 + width;
  ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - h * Math.cos(a - .4), y2 - h * Math.sin(a - .4)); ctx.lineTo(x2 - h * Math.cos(a + .4), y2 - h * Math.sin(a + .4)); ctx.fill();
  ctx.globalAlpha = 1;
}
// Curved edge between two circles, bowed to one side so A→B and B→A separate.
function linkCircles(a, b, width, color, alpha) {
  const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1, nx = -dy / d, ny = dx / d, bow = Math.min(60, d * .18);
  const cx = (a.x + b.x) / 2 + nx * bow, cy = (a.y + b.y) / 2 + ny * bow;
  const sa = Math.atan2(cy - a.y, cx - a.x), ea = Math.atan2(cy - b.y, cx - b.x);
  arrow(a.x + a.r * Math.cos(sa), a.y + a.r * Math.sin(sa), cx, cy, b.x + (b.r + 2) * Math.cos(ea), b.y + (b.r + 2) * Math.sin(ea), width, color, alpha);
}
function selfLoop(a, width, color, alpha) {
  const ang = Math.atan2(a.y - H / 2, a.x - W / 2) || -Math.PI / 2, ox = Math.cos(ang), oy = Math.sin(ang);
  const x1 = a.x + a.r * Math.cos(ang - .5), y1 = a.y + a.r * Math.sin(ang - .5), x2 = a.x + a.r * Math.cos(ang + .5), y2 = a.y + a.r * Math.sin(ang + .5);
  arrow(x1, y1, a.x + ox * (a.r + 42), a.y + oy * (a.r + 42), x2, y2, width, color, alpha);
}
function label(text, x, y, font, color, align = 'center') {
  ctx.font = font; ctx.textAlign = align; ctx.lineJoin = 'round';
  ctx.strokeStyle = '#121914'; ctx.lineWidth = 4; ctx.strokeText(text, x, y);
  ctx.fillStyle = color; ctx.fillText(text, x, y);
}
const widthFor = syn => Math.max(.8, Math.log10(syn + 1) * 2.1);

/* ---------- views ---------- */
function drawCircuit() {
  const k = Math.min(W, H) / 640, P = { dn: [.5, .15], in: [.5, .5], mn: [.78, .82], sn: [.22, .82] };
  const node = {};
  for (const L of C.LAYERS) {
    const n = A.groups[L.key].reduce((a, g) => a + g.members.length, 0);
    node[L.key] = { x: P[L.key][0] * W, y: P[L.key][1] * H, r: (10 + Math.sqrt(n) * 2.4) * k, n };
  }
  const focus = S.hover?.kind === 'layer' ? S.hover.layer : S.sel?.kind === 'layer' ? S.sel.layer : null;
  for (const [key, v] of Object.entries(A.layerEdges)) {
    const [a, b] = key.split('>'), on = !focus || a === focus || b === focus;
    if (a === b) selfLoop(node[a], widthFor(v.synapses), COLOR[a], on ? .6 : .08);
    else linkCircles(node[a], node[b], widthFor(v.synapses), COLOR[a], on ? .6 : .08);
  }
  for (const L of C.LAYERS) {
    const d = node[L.key], sel = S.sel?.kind === 'layer' && S.sel.layer === L.key;
    ctx.fillStyle = COLOR[L.key] + '33'; ctx.strokeStyle = COLOR[L.key]; ctx.lineWidth = sel ? 3 : 1.2;
    ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, 7); ctx.fill(); ctx.stroke();
    label(L.name, d.x, d.y + d.r + 18, '14px Georgia', '#e7eee5');
    label(fmt(d.n) + ' cells', d.x, d.y + d.r + 33, '11px sans-serif', '#99aa9c');
    hits.push({ kind: 'layer', layer: L.key, x: d.x, y: d.y, r: d.r });
  }
  hint('Click a layer for its summary. Double-click to open it.');
}

function layerData(k) {
  if (layerCache[k]) return layerCache[k];
  const gi = new Map(A.groups[k].map((g, j) => [g.key, j]));
  const ref = i => { const n = A.nodes[i]; return n.layer === k ? 'g' + gi.get(n.group) : 'L' + n.layer; };
  const m = new Map();
  for (let e = 0; e < A.E; e++) {
    const a = ref(A.pre[e]), b = ref(A.post[e]);
    if (a[0] === 'L' && b[0] === 'L') continue;
    const key = a + '>' + b; m.set(key, (m.get(key) || 0) + A.w[e]);
  }
  return (layerCache[k] = [...m.entries()].map(([key, syn]) => { const [a, b] = key.split('>'); return { a, b, syn }; }).sort((x, y) => y.syn - x.syn));
}
function drawLayer() {
  const k = S.layer, G = A.groups[k], sc = Math.min(W, H) / 640, R = Math.min(W, H) * .33, cx = W / 2, cy = H / 2;
  const pos = {};
  G.forEach((g, j) => {
    const t = -Math.PI / 2 + j / G.length * Math.PI * 2;
    pos['g' + j] = { x: cx + R * Math.cos(t), y: cy + R * Math.sin(t), r: (4 + Math.sqrt(g.members.length) * 2.3) * sc, t, g };
  });
  const corners = [[.08, .1], [.92, .1], [.92, .9], [.08, .9]];
  C.LAYERS.filter(L => L.key !== k).forEach((L, j) => {
    pos['L' + L.key] = { x: corners[j][0] * W, y: corners[j][1] * H, r: 16 * sc + 6, layer: L.key };
  });
  const focus = S.hover?.kind === 'group' ? S.hover.key : S.sel?.kind === 'group' && S.sel.layer === k ? S.sel.key : null;
  const fref = focus ? 'g' + G.findIndex(g => g.key === focus) : null;
  const all = layerData(k), edges = fref ? all.filter(e => e.a === fref || e.b === fref) : all.slice(0, 140);
  if (fref) for (const e of all.slice(0, 140)) if (e.a !== e.b && e.a !== fref && e.b !== fref) linkCircles(pos[e.a], pos[e.b], widthFor(e.syn) * .7, '#6fc7bc', .04);
  for (const e of edges) {
    if (e.a === e.b) continue;
    const src = pos[e.a], col = fref ? (e.b === fref ? IN_EDGE : OUT_EDGE) : src.layer ? COLOR[src.layer] : COLOR[k];
    linkCircles(src, pos[e.b], widthFor(e.syn) * .7, col, fref ? .8 : .28);
  }
  for (const [key, p] of Object.entries(pos)) {
    const isSel = p.g && focus === p.g.key;
    const col = p.layer ? COLOR[p.layer] : COLOR[k];
    ctx.fillStyle = col + (p.layer ? '22' : '55'); ctx.strokeStyle = col; ctx.lineWidth = isSel ? 3 : 1;
    ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 7); ctx.fill(); ctx.stroke();
    if (p.layer) {
      label(C.LAYER_NAME[p.layer], p.x, p.y + p.r + 14, '11px sans-serif', '#99aa9c');
      hits.push({ kind: 'layer', layer: p.layer, x: p.x, y: p.y, r: p.r });
    } else {
      if (G.length <= 40 || p.g.members.length >= 6 || isSel) {
        const lx = cx + (R + p.r + 10) * Math.cos(p.t), ly = cy + (R + p.r + 10) * Math.sin(p.t);
        label(shortGroup(p.g.key), lx, ly + 4, '11px sans-serif', '#e7eee5', Math.cos(p.t) > .2 ? 'left' : Math.cos(p.t) < -.2 ? 'right' : 'center');
      }
      hits.push({ kind: 'group', layer: k, key: p.g.key, x: p.x, y: p.y, r: Math.max(p.r, 6) });
    }
  }
  const self = all.filter(e => e.a === e.b && e.a[0] === 'g' && (!fref || e.a === fref)).reduce((a, e) => a + e.syn, 0);
  hint(G.length + ' groups. ' + (fref ? 'Blue edges are inputs to the selected group, red are outputs.' : 'Showing the 140 strongest group links. Select a group to see all of its links.') + (fref ? ' Synapses within this group: ' : ' Synapses within groups: ') + fmt(self) + '. Double-click a group to see it in anatomy.');
}

function focusSet() {
  const s = S.sel;
  if (!s) return null;
  if (s.kind === 'cell') {
    const set = new Set([s.i]);
    A.outs[s.i].forEach(e => set.add(A.post[e])); A.ins[s.i].forEach(e => set.add(A.pre[e]));
    return set;
  }
  if (s.kind === 'group') return new Set(findGroup(s.layer, s.key).members);
  if (s.kind === 'layer') return new Set(A.groups[s.layer].flatMap(g => g.members));
  if (s.kind === 'region') return new Set(s.cells);
  return null;
}
function drawAnatomy() {
  const f = fitScale() * S.cam.s, focus = focusSet(), pts = A.nodes.map(n => n.x === null ? null : toScreen(n));
  if (S.threshold > 0) {
    ctx.lineWidth = .6;
    for (let e = 0; e < A.E; e++) {
      if (A.w[e] < S.threshold) continue;
      const a = A.pre[e], b = A.post[e];
      if (!pts[a] || !pts[b] || (focus && !focus.has(a) && !focus.has(b))) continue;
      ctx.globalAlpha = .25; ctx.strokeStyle = COLOR[A.nodes[a].layer];
      ctx.beginPath(); ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
  if (S.sel?.kind === 'cell') {
    const i = S.sel.i, p = pts[i];
    if (p) {
      for (const e of A.ins[i]) if (pts[A.pre[e]]) line(pts[A.pre[e]], p, IN_EDGE, A.w[e]);
      for (const e of A.outs[i]) if (pts[A.post[e]]) line(p, pts[A.post[e]], OUT_EDGE, A.w[e]);
    }
  }
  const r = Math.max(2, Math.min(7, 2.4 * Math.sqrt(S.cam.s)));
  for (const n of A.nodes) {
    const p = pts[n.i]; if (!p) continue;
    const on = !focus || focus.has(n.i);
    ctx.globalAlpha = on ? .95 : .12;
    if (n.placed) { ctx.strokeStyle = COLOR[n.layer]; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 7); ctx.stroke(); }
    else { ctx.fillStyle = COLOR[n.layer]; ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 7); ctx.fill(); }
    hits.push({ kind: 'cell', i: n.i, x: p[0], y: p[1], r: Math.max(r, 5) });
  }
  ctx.globalAlpha = 1;
  if (S.sel?.kind === 'cell' && pts[S.sel.i]) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(...pts[S.sel.i], r + 4, 0, 7); ctx.stroke(); }
  const rc = S.rect || (S.sel?.kind === 'region' ? S.sel.screen : null);
  if (rc && S.rect) { ctx.setLineDash([5, 4]); ctx.strokeStyle = '#c2e899'; ctx.lineWidth = 1; ctx.strokeRect(rc.x, rc.y, rc.w, rc.h); ctx.setLineDash([]); }
  // Orientation and scale.
  ctx.fillStyle = '#99aa9c'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
  ctx.fillText('Anterior', W / 2, 16); ctx.fillText('Posterior', W / 2, H - 30);
  const offDN = A.nodes.filter(n => n.layer === 'dn' && pts[n.i] && pts[n.i][1] < 0).length;
  if (offDN) { ctx.fillStyle = COLOR.dn; ctx.fillText('↑ ' + offDN + ' descending somata in the brain. Scroll out to see them.', W / 2, 32); ctx.fillStyle = '#99aa9c'; }
  ctx.textAlign = 'left'; ctx.fillText('L', 10, H / 2); ctx.textAlign = 'right'; ctx.fillText('R', W - 10, H / 2);
  const bar = 100000 / VOXEL_NM * f; // 100 µm
  if (bar > 20 && bar < W / 2) {
    ctx.strokeStyle = '#99aa9c'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(W - 20 - bar, H - 28); ctx.lineTo(W - 20, H - 28); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText('100 µm', W - 20, H - 36);
  }
  const legend = C.LAYERS.map(L => L.key);
  legend.forEach((k, j) => { ctx.fillStyle = COLOR[k]; ctx.beginPath(); ctx.arc(22, 30 + j * 18, 4, 0, 7); ctx.fill(); ctx.fillStyle = '#99aa9c'; ctx.textAlign = 'left'; ctx.fillText(C.LAYER_NAME[k], 32, 34 + j * 18); });
  ctx.strokeStyle = '#99aa9c'; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(22, 30 + 4 * 18, 4, 0, 7); ctx.stroke(); ctx.fillText('Position estimated from partners', 32, 34 + 4 * 18);
  hint((S.regionMode ? 'Drag to select a region.' : 'Drag to pan, scroll to zoom, shift-drag to select a region.') + ' Dorsal view of somata.');
}
function line(a, b, color, syn) {
  ctx.globalAlpha = .7; ctx.strokeStyle = color; ctx.lineWidth = Math.max(.6, Math.log10(syn + 1) * 1.3);
  ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); ctx.globalAlpha = 1;
}
function hint(t) { document.getElementById('hint').textContent = t; }

function draw() {
  if (!A) return;
  hits = []; ctx.clearRect(0, 0, W, H);
  if (S.view === 'circuit') drawCircuit(); else if (S.view === 'layer') drawLayer(); else drawAnatomy();
}

/* ---------- panels ---------- */
function comp(entries, colorFor) {
  const tot = entries.reduce((a, e) => a + e[1], 0) || 1;
  return '<div class="comp">' + entries.map(([k, v]) => '<div style="width:' + (v / tot * 100).toFixed(2) + '%;background:' + colorFor(k) + '"></div>').join('') + '</div>'
    + '<div class="keys">' + entries.map(([k, v]) => '<span><span class="sw" style="background:' + colorFor(k) + '"></span>' + esc(k) + ' ' + fmt(v) + '</span>').join('') + '</div>';
}
const NT_COLOR = { acetylcholine: '#6fc7bc', gaba: '#f28b82', glutamate: '#e9be75', unclear: '#5b6a5e', unknown: '#5b6a5e' };
function summaryHtml(s, withGroups = true) {
  const rows = l => l.length ? '<ul class="list">' + l.map(([k, v]) => '<li><b>' + esc(k) + '</b><span>' + fmt(v) + '</span></li>').join('') + '</ul>' : '<p class="sub">None within this subset.</p>';
  return '<div class="stats"><div class="stat"><small>Cells</small><strong>' + fmt(s.cells) + '</strong></div>'
    + '<div class="stat"><small>Edges inside</small><strong>' + fmt(s.internal.edges) + '</strong></div>'
    + '<div class="stat"><small>Synapses in</small><strong>' + fmt(s.input.synapses) + '</strong></div>'
    + '<div class="stat"><small>Synapses out</small><strong>' + fmt(s.output.synapses) + '</strong></div></div>'
    + (s.layers.length > 1 ? '<h3>Layers</h3>' + comp(s.layers.map(([k, v]) => [C.LAYER_NAME[k], v]), k => COLOR[C.LAYERS.find(L => L.name === k).key]) : '')
    + '<h3>Predicted transmitter</h3>' + comp(s.nt, k => NT_COLOR[k] || '#5b6a5e')
    + (withGroups ? '<h3>Cell groups</h3>' + rows(s.groups.slice(0, 8)) : '')
    + '<h3>Strongest inputs, by synapses</h3>' + rows(s.topInputs)
    + '<h3>Strongest outputs, by synapses</h3>' + rows(s.topOutputs);
}
function chips(idx, max = 60) {
  return '<div class="chips">' + idx.slice(0, max).map(i => '<button data-cell="' + i + '">' + esc(cellName(i)) + '</button>').join('')
    + (idx.length > max ? '<span class="sub">and ' + (idx.length - max) + ' more</span>' : '') + '</div>';
}
function panelCircuit() {
  const all = A.nodes.map(n => n.i), s = C.summarize(A, all);
  return '<h2>Locomotion circuit</h2><p class="sub">MaleCNS v1.0 subset. ' + fmt(A.nodes.length) + ' cells, ' + fmt(A.E) + ' edges, ' + fmt(A.w.reduce((a, b) => a + b, 0)) + ' synapses.</p>'
    + '<h3>Layers</h3><ul class="list">' + C.LAYERS.map(L => '<li><button class="link" data-layer="' + L.key + '"><span class="sw" style="background:' + COLOR[L.key] + '"></span>' + L.name + '</button><span>' + fmt(A.groups[L.key].reduce((a, g) => a + g.members.length, 0)) + '</span></li>').join('') + '</ul>'
    + '<h3>Layer to layer</h3><ul class="list">' + Object.entries(A.layerEdges).sort((a, b) => b[1].synapses - a[1].synapses).map(([k, v]) => { const [a, b] = k.split('>'); return '<li><b>' + C.LAYER_NAME[a] + ' → ' + C.LAYER_NAME[b] + '</b><span>' + fmt(v.synapses) + '</span></li>'; }).join('') + '</ul>'
    + '<h3>Predicted transmitter</h3>' + comp(s.nt, k => NT_COLOR[k] || '#5b6a5e');
}
function panelLayer(k) {
  const idx = A.groups[k].flatMap(g => g.members), s = C.summarize(A, idx), t = LAYER_TEXT[k];
  return '<h2>' + C.LAYER_NAME[k] + '</h2><p class="sub">' + A.groups[k].length + ' groups</p>'
    + '<div class="actions">' + (S.view === 'layer' && S.layer === k ? '' : '<button data-act="openLayer" data-layer="' + k + '">Open layer</button>')
    + (S.view === 'anatomy' ? '' : '<button class="secondary" data-act="anatomyLayer" data-layer="' + k + '">Show in anatomy</button>') + '</div>'
    + '<h3>In the model</h3><p class="model">' + t.m + '</p><p class="flag">' + t.w + '</p>'
    + summaryHtml(s, false)
    + '<h3>Groups</h3><ul class="list">' + A.groups[k].map(g => '<li><button class="link" data-group="' + esc(g.key) + '" data-glayer="' + k + '">' + esc(g.key) + '</button><span>' + g.members.length + '</span></li>').join('') + '</ul>';
}
function panelGroup(layer, key) {
  const g = findGroup(layer, key), s = C.summarize(A, g.members);
  return '<h2>' + esc(key) + '</h2><p class="sub">' + C.LAYER_NAME[layer] + '</p>'
    + '<div class="actions">' + (S.view === 'anatomy' ? '' : '<button data-act="anatomyGroup">Show in anatomy</button>') + '<button class="secondary" data-act="openLayer" data-layer="' + layer + '">Open layer wiring</button></div>'
    + summaryHtml(s, false) + '<h3>Cells</h3>' + chips(g.members);
}
function qualityHtml(bodyId, aggregateNt) {
  if (!Q) return '<h3>Data quality</h3><p class="sub">Not built yet. Run build_quality.py to see input coverage and per-synapse transmitters.</p>';
  const q = Q[String(bodyId)];
  if (!q) return '';
  const pct = v => v === null || v === undefined ? 'unknown' : Math.round(v * 100) + '%';
  let h = '<h3>Data quality</h3><table class="kv">'
    + '<tr><td>Input in this subset</td><td>' + pct(q.coverage_in) + (q.post_total ? ' of ' + fmt(q.post_total) + ' synapses' : '') + '</td></tr>'
    + '<tr><td>Output in this subset</td><td>' + pct(q.coverage_out) + (q.downstream_total ? ' of ' + fmt(q.downstream_total) : '') + '</td></tr>';
  if (q.nt) {
    const shares = Object.entries(q.nt.top_share).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([k, v]) => k + ' ' + Math.round(v * 100) + '%').join(', ');
    h += '<tr><td>Per-synapse transmitter</td><td>' + esc(shares) + ' of ' + fmt(q.nt.tbars) + ' synapses</td></tr>'
      + '<tr><td>Sign probability</td><td>+ ' + Math.round(q.nt.p_positive * 100) + '%, − ' + Math.round(q.nt.p_negative * 100) + '%</td></tr>';
  }
  h += '</table>';
  if (q.coverage_in !== null && q.coverage_in < 0.25) h += '<p class="flag">The subset contains under a quarter of this cell’s input, so its simulated activity leaves out most of what drives it.</p>';
  const cons = q.consensus_nt ?? aggregateNt;
  if (q.nt && cons && !['unclear', 'unknown'].includes(cons) && q.nt.dominant !== cons)
    h += '<p class="flag">Per-synapse predictions mostly say ' + esc(q.nt.dominant) + ', but the consensus transmitter is ' + esc(cons) + '. The model uses the consensus; raw per-synapse calls are unreliable for some classes, including motor neurons.</p>';
  else if (q.nt && q.nt.dominant_share < 0.6) h += '<p class="flag">Mixed transmitter evidence across its synapses; treat its sign as uncertain.</p>';
  return h;
}
function partnerList(i, dir) {
  const p = C.partners(A, i, dir, 12);
  return '<p class="sub">' + fmt(p.total) + ' partners, ' + fmt(p.synapses) + ' synapses. Strongest:</p><ul class="list">'
    + p.top.map(x => '<li><button class="link" data-cell="' + x.cell + '"><span class="sw" style="background:' + COLOR[A.nodes[x.cell].layer] + '"></span>' + esc(cellName(x.cell)) + '</button><span>' + fmt(x.synapses) + '</span></li>').join('') + '</ul>';
}
function panelCell(i) {
  const n = A.nodes[i], r = n.raw, t = C.treatment(A, i);
  const row = (k, v) => v === null || v === undefined || v === '' ? '' : '<tr><td>' + k + '</td><td>' + esc(v) + '</td></tr>';
  return '<h2>' + esc(cellName(i)) + '</h2><p class="sub"><span class="sw" style="background:' + COLOR[n.layer] + '"></span>' + C.LAYER_NAME[n.layer] + '. ' + esc(n.group) + '</p>'
    + '<table class="kv">' + row('Body ID', r.bodyId) + row('Type', r.type) + row('Class', [r.superclass, r.subclass].filter(Boolean).join(' / '))
    + row('Side', r.rootSide || r.somaSide) + row('Transmitter', r.nt ? r.nt + (r.nt_confidence ? ' (' + Math.round(r.nt_confidence * 100) + '% confidence)' : '') : null)
    + row('Entry nerve', r.entryNerve) + row('Exit nerve', r.exitNerve) + row('Soma (voxels)', r.somaLocation ? r.somaLocation.join(', ') : 'Not annotated')
    + row('Model leg pool', n.layer === 'mn' || n.layer === 'sn' ? C.legText(n.leg) : null) + row('Status', r.status) + '</table>'
    + '<div class="actions"><button data-act="anatomyCell">' + (S.view === 'anatomy' ? 'Center on cell' : 'Show in anatomy') + '</button><button class="secondary" data-group="' + esc(n.group) + '" data-glayer="' + n.layer + '">Open group</button><a class="wb-link" href="/neuron?circuit=locomotion&id=' + r.bodyId + '">Open in neuron workbench</a></div>'
    + '<h3>In the model</h3><div class="model">' + t.lines.join('<br>') + '</div>' + t.flags.map(f => '<p class="flag">' + esc(f) + '</p>').join('')
    + qualityHtml(r.bodyId, r.nt)
    + '<h3><span class="sw" style="background:' + IN_EDGE + '"></span>Inputs</h3>' + partnerList(i, 'in')
    + '<h3><span class="sw" style="background:' + OUT_EDGE + '"></span>Outputs</h3>' + partnerList(i, 'out');
}
function panelRegion(sel) {
  if (!sel.cells.length) return '<h2>Empty region</h2><p class="sub">No cells fall inside that rectangle. Drag a larger area.</p>';
  const s = C.summarize(A, sel.cells);
  return '<h2>Selected region</h2><p class="sub">Summary of every cell drawn inside the rectangle. Estimated positions count too.</p>'
    + summaryHtml(s) + '<h3>Cells</h3>' + chips(sel.cells);
}
function renderPanel() {
  const s = S.sel;
  panel.innerHTML = !s ? (S.view === 'layer' ? panelLayer(S.layer) : panelCircuit())
    : s.kind === 'layer' ? panelLayer(s.layer) : s.kind === 'group' ? panelGroup(s.layer, s.key)
    : s.kind === 'cell' ? panelCell(s.i) : panelRegion(s);
  panel.scrollTop = 0;
}
function renderCrumbs() {
  const parts = ['<button data-crumb="circuit">Circuit</button>'];
  const layer = S.sel?.kind === 'cell' ? A.nodes[S.sel.i].layer : S.sel?.kind === 'region' ? null : S.sel?.layer || S.layer;
  if (layer) parts.push('<button data-crumb="layer" data-layer="' + layer + '">' + C.LAYER_NAME[layer] + '</button>');
  if (S.sel?.kind === 'group') parts.push('<span class="here">' + esc(S.sel.key) + '</span>');
  if (S.sel?.kind === 'cell') parts.push('<button data-group="' + esc(A.nodes[S.sel.i].group) + '" data-glayer="' + layer + '">' + esc(A.nodes[S.sel.i].group) + '</button><span class="sep">›</span><span class="here">' + esc(cellName(S.sel.i)) + '</span>');
  if (S.sel?.kind === 'region') parts.push('<span class="here">Region, ' + S.sel.cells.length + ' cells</span>');
  document.getElementById('crumbs').innerHTML = parts.join('<span class="sep">›</span>');
  document.getElementById('vWiring').setAttribute('aria-pressed', S.view !== 'anatomy');
  document.getElementById('vAnatomy').setAttribute('aria-pressed', S.view === 'anatomy');
  document.getElementById('tools').hidden = S.view !== 'anatomy';
}
function render() { if (!A) return; renderCrumbs(); renderPanel(); draw(); }

/* ---------- panel and toolbar actions ---------- */
function centerOn(i) {
  const n = A.nodes[i]; if (n.x === null) return;
  S.cam.s = Math.max(S.cam.s, 2.5); S.cam.x = 0; S.cam.y = 0;
  const [x, y] = toScreen(n); S.cam.x = W / 2 - x; S.cam.y = H / 2 - y;
}
document.addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  if (b.dataset.cell) { const i = +b.dataset.cell; S.layer = A.nodes[i].layer; select({ kind: 'cell', i }); return; }
  if (b.dataset.group) { S.layer = b.dataset.glayer; S.group = b.dataset.group; if (S.view === 'circuit') S.view = 'layer'; select({ kind: 'group', layer: b.dataset.glayer, key: b.dataset.group }); return; }
  if (b.dataset.crumb === 'circuit') { S.layer = null; S.view = 'circuit'; select(null); return; }
  if (b.dataset.crumb === 'layer') { S.layer = b.dataset.layer; if (S.view !== 'anatomy') S.view = 'layer'; select({ kind: 'layer', layer: b.dataset.layer }); return; }
  if (b.dataset.layer && !b.dataset.act) { select({ kind: 'layer', layer: b.dataset.layer }); return; }
  const act = b.dataset.act;
  if (act === 'openLayer') { S.layer = b.dataset.layer; S.view = 'layer'; select(null); }
  else if (act === 'anatomyLayer') { S.view = 'anatomy'; select({ kind: 'layer', layer: b.dataset.layer }); }
  else if (act === 'anatomyGroup') { S.view = 'anatomy'; render(); }
  else if (act === 'anatomyCell') { S.view = 'anatomy'; centerOn(S.sel.i); render(); }
});
document.getElementById('vWiring').onclick = () => go(S.layer ? 'layer' : 'circuit');
document.getElementById('vAnatomy').onclick = () => go('anatomy');
document.getElementById('regionBtn').onclick = e => { S.regionMode = !S.regionMode; e.currentTarget.setAttribute('aria-pressed', S.regionMode); draw(); };
document.getElementById('resetBtn').onclick = () => { S.cam = { s: 1, x: 0, y: 0 }; draw(); };
document.getElementById('thresh').oninput = e => { S.threshold = +e.target.value; document.getElementById('threshOut').textContent = S.threshold || 'off'; draw(); };
document.addEventListener('keydown', e => { if (e.key === 'Escape' && S.sel) { select(null); } });

/* ---------- canvas interaction ---------- */
function hitAt(x, y) {
  let best = null, bd = Infinity;
  for (const h of hits) { const d = Math.hypot(h.x - x, h.y - y); if (d <= h.r + 3 && d < bd) { best = h; bd = d; } }
  return best;
}
function local(e) { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
cv.addEventListener('pointerdown', e => {
  const [x, y] = local(e); cv.setPointerCapture(e.pointerId);
  S.drag = { x, y, x0: x, y0: y, cam: { ...S.cam }, region: S.view === 'anatomy' && (S.regionMode || e.shiftKey), moved: false };
});
cv.addEventListener('pointermove', e => {
  const [x, y] = local(e);
  if (S.drag) {
    const d = S.drag; if (Math.hypot(x - d.x0, y - d.y0) > 4) d.moved = true;
    if (S.view === 'anatomy' && d.moved) {
      if (d.region) S.rect = { x: Math.min(x, d.x0), y: Math.min(y, d.y0), w: Math.abs(x - d.x0), h: Math.abs(y - d.y0) };
      else { S.cam.x = d.cam.x + x - d.x0; S.cam.y = d.cam.y + y - d.y0; }
      draw();
    }
    return;
  }
  const h = hitAt(x, y), key = h ? h.kind + (h.i ?? h.key ?? h.layer) : null, old = S.hover ? S.hover.kind + (S.hover.i ?? S.hover.key ?? S.hover.layer) : null;
  S.hover = h; cv.style.cursor = h ? 'pointer' : S.view === 'anatomy' ? (S.regionMode ? 'crosshair' : 'grab') : 'default';
  if (h) {
    tip.hidden = false; tip.style.left = Math.min(x + 14, W - 270) + 'px'; tip.style.top = (y + 14) + 'px';
    tip.innerHTML = h.kind === 'cell' ? '<b>' + esc(cellName(h.i)) + '</b><br>' + C.LAYER_NAME[A.nodes[h.i].layer] + ', ' + esc(A.nodes[h.i].group) + '<br>' + esc(A.nodes[h.i].raw.nt || 'unknown')
      : h.kind === 'group' ? '<b>' + esc(h.key) + '</b><br>' + findGroup(h.layer, h.key).members.length + ' cells'
      : '<b>' + C.LAYER_NAME[h.layer] + '</b>';
  } else tip.hidden = true;
  if (key !== old && S.view !== 'anatomy') draw();
});
cv.addEventListener('pointerleave', () => { tip.hidden = true; if (S.hover) { S.hover = null; draw(); } });
cv.addEventListener('pointerup', e => {
  const d = S.drag; S.drag = null; if (!d) return;
  const [x, y] = local(e);
  if (d.region && d.moved && S.rect) {
    const r = S.rect, cells = A.nodes.filter(n => { if (n.x === null) return false; const [px, py] = toScreen(n); return px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h; }).map(n => n.i);
    S.rect = null; S.regionMode = false; document.getElementById('regionBtn').setAttribute('aria-pressed', false);
    select({ kind: 'region', cells }); return;
  }
  if (d.moved) return;
  const h = hitAt(x, y);
  if (!h) { if (S.sel && S.view !== 'circuit') { select(S.view === 'layer' ? null : S.sel.kind === 'cell' ? null : S.sel); } return; }
  if (h.kind === 'layer') select({ kind: 'layer', layer: h.layer });
  else if (h.kind === 'group') { S.group = h.key; select({ kind: 'group', layer: h.layer, key: h.key }); }
  else { S.layer = A.nodes[h.i].layer; select({ kind: 'cell', i: h.i }); }
});
cv.addEventListener('dblclick', e => {
  const h = hitAt(...local(e)); if (!h) return;
  if (h.kind === 'layer') { S.layer = h.layer; S.view = 'layer'; select(null); }
  else if (h.kind === 'group') { S.view = 'anatomy'; select({ kind: 'group', layer: h.layer, key: h.key }); }
  else if (h.kind === 'cell') { centerOn(h.i); draw(); }
});
cv.addEventListener('wheel', e => {
  if (S.view !== 'anatomy') return;
  e.preventDefault();
  const [x, y] = local(e), k = Math.exp(-e.deltaY * .0015), s = Math.min(40, Math.max(.5, S.cam.s * k)), r = s / S.cam.s;
  S.cam.x = x - W / 2 - (x - W / 2 - S.cam.x) * r; S.cam.y = y - H / 2 - (y - H / 2 - S.cam.y) * r; S.cam.s = s; draw();
}, { passive: false });
