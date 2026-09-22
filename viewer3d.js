'use strict';
const $ = id => document.getElementById(id);
// Region names come from each layer's segment properties; meshes come simplified from /api/mesh/<key>/<id>.
const ROI_LAYERS = [['rois/fullbrain-roi-v5', 'roi'], ['rois/malecns-vnc-neuropil-roi-v0', 'vnc-roi']];
const SHELLS = [['brain-shell', [1, 2, 3]], ['vnc-shell', [1]]];
const PALETTE = [0xc2e899, 0x6fc7bc, 0xe9be75, 0xf28b82, 0x8ab4f8, 0xd4a5f5, 0xf5c16c, 0x7fd1ae];
const PRESETS = { dns: [10360, 523769, 10442, 10760, 11074, 512006], dna02: [10360, 523769], kc: [57729],
                  steerR: ['LAL(R)', 'VES(R)', 'IPS(R)', 'SPS(R)', 'GNG'], steerL: ['LAL(L)', 'VES(L)', 'IPS(L)', 'SPS(L)', 'GNG'] };
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const status = t => { $('status').textContent = t; };

let names = {}, regionIds = {}, regionLayerOf = {}, infoCache = {};
const cells = new Map(), regions = new Map(), shells = [];

/* ---------- three.js scene with a small orbit control ---------- */
const stage = $('stage'), renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio || 1); stage.prepend(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0d130f);
const camera = new THREE.PerspectiveCamera(40, 1, 1, 20000);
scene.add(new THREE.AmbientLight(0xffffff, 0.55)); const sun = new THREE.DirectionalLight(0xffffff, 0.75); scene.add(sun);
const world = new THREE.Group(); scene.add(world);
let orbit = { theta: 0.6, phi: 1.2, dist: 1200, target: new THREE.Vector3() };
function place() {
  const { theta, phi, dist, target } = orbit;
  camera.position.set(target.x + dist * Math.sin(phi) * Math.cos(theta), target.y + dist * Math.cos(phi), target.z + dist * Math.sin(phi) * Math.sin(theta));
  camera.up.set(0, -1, 0); camera.lookAt(target); sun.position.copy(camera.position);
}
function resize() { const w = stage.clientWidth, h = stage.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
window.addEventListener('resize', resize);
let drag = null;
renderer.domElement.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey }; renderer.domElement.setPointerCapture(e.pointerId); });
renderer.domElement.addEventListener('pointermove', e => {
  if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) { const s = orbit.dist / 900, right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0), up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1); orbit.target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s); }
  else { orbit.theta += dx * 0.008; orbit.phi = Math.max(0.05, Math.min(Math.PI - 0.05, orbit.phi - dy * 0.008)); }
  place();
});
renderer.domElement.addEventListener('pointerup', () => { drag = null; });
renderer.domElement.addEventListener('wheel', e => { e.preventDefault(); orbit.dist = Math.max(20, Math.min(8000, orbit.dist * Math.exp(e.deltaY * 0.001))); place(); }, { passive: false });
(function loop() { renderer.render(scene, camera); requestAnimationFrame(loop); })();
function fit() {
  const box = new THREE.Box3(); world.children.forEach(o => { if (o.visible) box.expandByObject(o); });
  if (box.isEmpty()) return;
  box.getCenter(orbit.target); orbit.dist = box.getSize(new THREE.Vector3()).length() * 1.1 + 50; place();
}

/* ---------- data access through the lab's bucket proxy ---------- */
async function bytes(path) { const r = await fetch('/api/gcs/' + path); if (!r.ok) { let m = r.status; try { m = (await r.json()).error; } catch (e) {} throw new Error(m); } return r.arrayBuffer(); }
async function json(path) { return JSON.parse(new TextDecoder().decode(await bytes(path))); }
async function info(layer) { return infoCache[layer] || (infoCache[layer] = json(layer + '/info')); }
async function loadMesh(key, id) {
  const r = await fetch(`/api/mesh/${key}/${id}`);
  if (!r.ok) { let m = r.status; try { m = (await r.json()).error; } catch (e) {} throw new Error(m); }
  const meta = JSON.parse(r.headers.get('X-Mesh-Info') || '{}'), m = NG.decodeLegacy(await r.arrayBuffer());
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(NG.toMicrometres(m.vertices).vertices, 3));
  g.setIndex(new THREE.BufferAttribute(m.indices, 1)); g.computeVertexNormals(); g.userData = meta;
  return g;
}
const meshNote = g => g.userData && g.userData.triangles ? (g.userData.triangles < g.userData.triangles_original
  ? `Simplified from ${g.userData.triangles_original.toLocaleString()} to ${g.userData.triangles.toLocaleString()} triangles` : `${g.userData.triangles.toLocaleString()} triangles`) : '';
async function loadSkeleton(id) {
  const r = await fetch('/api/skeleton?id=' + id), d = await r.json(); if (!r.ok) throw new Error(d.error || r.status);
  const byId = new Map(d.points.map(p => [p[0], p])), pos = [];
  for (const p of d.points) { const q = byId.get(p[4]); if (q) pos.push(p[1], p[2], p[3], q[1], q[2], q[3]); }
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); return g;
}

/* ---------- neurons ---------- */
async function addCell(id) {
  id = String(id).trim(); if (!/^\d+$/.test(id) || cells.has(id)) return;
  const color = PALETTE[cells.size % PALETTE.length], item = { id, color, obj: null, state: 'loading', note: '' };
  cells.set(id, item); render();
  try {
    const g = await loadMesh('neuron', id);
    item.obj = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color })); item.state = 'mesh'; item.note = meshNote(g);
  } catch (e) {
    try { item.obj = new THREE.LineSegments(await loadSkeleton(id), new THREE.LineBasicMaterial({ color })); item.state = 'skeleton'; item.note = 'Mesh unavailable (' + e.message + ')'; }
    catch (e2) { item.state = 'failed'; item.note = e.message + '; skeleton: ' + e2.message; }
  }
  if (item.obj && cells.has(id)) { world.add(item.obj); fit(); }
  render();
}
/* ---------- regions ---------- */
async function regionIndex() {
  for (const [layer, key] of ROI_LAYERS) {
    try {
      const i = await info(layer); if (typeof i.segment_properties !== 'string') continue;
      const map = NG.labelsToIds(await json(layer + '/' + i.segment_properties + '/info'));
      for (const [label, id] of Object.entries(map)) if (!(label in regionIds)) { regionIds[label] = id; regionLayerOf[label] = key; }
    } catch (e) { status('Region names unavailable for ' + layer + ': ' + e.message); }
  }
}
async function addRegion(name) {
  name = name.trim(); if (!name || regions.has(name)) return;
  const item = { name, obj: null, state: 'loading', note: '', color: 0x8ab4f8 };
  regions.set(name, item); render();
  try {
    if (!(name in regionIds)) await regionIndex();
    if (!(name in regionIds)) throw new Error('no region shape named ' + name);
    const g = await loadMesh(regionLayerOf[name], regionIds[name]);
    item.obj = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: item.color, transparent: true, opacity: 0.18, depthWrite: false })); item.state = 'mesh'; item.note = meshNote(g);
    world.add(item.obj); fit();
  } catch (e) { item.state = 'failed'; item.note = e.message; }
  render();
}
async function loadShells() {
  for (const [layer, ids] of SHELLS) for (const id of ids) {
    try { const g = await loadMesh(layer, id); const m = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ color: 0xffffff, transparent: true, opacity: 0.05, depthWrite: false })); m.visible = $('shells').checked; shells.push(m); world.add(m); }
    catch (e) { status('Outline ' + layer + ' ' + id + ' unavailable: ' + e.message); }
  }
  fit();
}

/* ---------- UI ---------- */
function render() {
  const row = (label, it, key) => `<li><span class="sw3" style="background:#${it.color.toString(16).padStart(6, '0')}"></span><b title="${esc(it.note)}">${esc(label)}</b>`
    + `<span class="st ${it.state === 'mesh' ? 'ok' : it.state === 'skeleton' ? 'fb' : it.state === 'failed' ? 'err' : ''}" title="${esc(it.note)}">${it.state}</span>`
    + `<span><button data-toggle="${key}">${it.obj && !it.obj.visible ? 'show' : 'hide'}</button><button data-remove="${key}">×</button></span></li>`;
  $('cellList').innerHTML = [...cells.values()].map(c => row((names[c.id] ? names[c.id] + ' · ' : '') + c.id, c, 'c:' + c.id)).join('');
  $('regionList').innerHTML = [...regions.values()].map(r => row(r.name, r, 'r:' + r.name)).join('');
  $('ngLink').href = NG.url({ cells: [...cells.keys()], regions: [...regions.keys()], regionIds });
  const q = new URLSearchParams(); if (cells.size) q.set('cells', [...cells.keys()].join(',')); if (regions.size) q.set('regions', [...regions.keys()].join(','));
  history.replaceState(null, '', '/3d' + (q.toString() ? '?' + q : ''));
  const failed = [...cells.values(), ...regions.values()].filter(x => x.state === 'failed');
  if (failed.length) status(failed.map(x => (x.id || x.name) + ': ' + x.note).join(' · '));
}
document.addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  const key = b.dataset.toggle || b.dataset.remove;
  if (key) {
    const [kind, id] = [key.slice(0, 1), key.slice(2)], map = kind === 'c' ? cells : regions, it = map.get(id); if (!it) return;
    if (b.dataset.toggle && it.obj) it.obj.visible = !it.obj.visible;
    if (b.dataset.remove) { if (it.obj) world.remove(it.obj); map.delete(id); }
    render(); return;
  }
  const p = PRESETS[b.dataset.preset]; if (p) p.forEach(x => typeof x === 'number' ? addCell(x) : addRegion(x));
});
$('addCells').onclick = () => { $('cellIn').value.split(/[\s,]+/).filter(Boolean).forEach(addCell); $('cellIn').value = ''; };
$('cellIn').addEventListener('keydown', e => { if (e.key === 'Enter') $('addCells').click(); });
$('addRegion').onclick = () => { addRegion($('regionIn').value); $('regionIn').value = ''; };
$('regionIn').addEventListener('keydown', e => { if (e.key === 'Enter') $('addRegion').click(); });
$('shells').onchange = () => shells.forEach(m => { m.visible = $('shells').checked; });
$('reset').onclick = fit;

resize(); place(); loadShells();
fetch('/api/explorer').then(r => r.json()).then(g => { for (const n of g.nodes) names[n.bodyId] = n.instance || n.type || ''; render(); }).catch(() => {});
fetch('/api/regions').then(r => r.ok ? r.json() : null).then(d => { if (d) $('regionNames').innerHTML = d.rois.map(r => `<option value="${esc(r)}">`).join(''); }).catch(() => {});
const q = new URLSearchParams(location.search);
(q.get('cells') || '').split(',').filter(Boolean).forEach(addCell);
(q.get('regions') || '').split(',').filter(Boolean).forEach(addRegion);
render();
