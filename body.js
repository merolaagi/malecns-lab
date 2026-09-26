'use strict';
const $ = id => document.getElementById(id);
const { build, LEGS, PART_NAMES, BRAIN_DRIVEN, mat } = FlyMesh;
const deg = d => d * Math.PI / 180;
const manual = { coxa: 0, femur: 0, knee: 0, tarsus: 0, wing: 12, stroke: 0, haltere: 0, headYaw: 0, headPitch: 0, antenna: 0, proboscis: 0, abdomen: 0 };
const legPoses = Object.fromEntries(LEGS.map(k => [k, { coxa: 0, femur: 0, knee: 0, tarsus: 0 }]));
const partPoses = {};
let run = null, frame = 0, playing = true, selected = 'thorax';

/* ---------- scene ---------- */
const stage = $('stage'), renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
renderer.setClearColor(0x0b110d);
stage.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.02, 200);
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const sun = new THREE.DirectionalLight(0xffffff, 0.85); sun.position.set(4, 8, 6); scene.add(sun);
const fill = new THREE.DirectionalLight(0x8ecedf, 0.35); fill.position.set(-5, 3, -4); scene.add(fill);
const grid = new THREE.GridHelper(12, 24, 0x53695a, 0x2a3a31); grid.position.y = -0.85; scene.add(grid);
const fly = build(0xc2e899);
scene.add(fly.root);
let orbit = { theta: 0.85, phi: 1.15, dist: 8, target: new THREE.Vector3(0, 0, 0) };
function place() {
  const { theta, phi, dist, target } = orbit;
  camera.position.set(target.x + dist * Math.sin(phi) * Math.cos(theta), target.y + dist * Math.cos(phi), target.z + dist * Math.sin(phi) * Math.sin(theta));
  camera.lookAt(target); sun.position.copy(camera.position).add(new THREE.Vector3(0, 6, 0));
}
function resetCamera() {                       // frame whatever the body currently occupies
  apply(0); fly.root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(fly.root);
  if (box.isEmpty()) { orbit.dist = 8; orbit.target.set(0, 0, 0); return place(); }
  box.getCenter(orbit.target);
  orbit.dist = box.getSize(new THREE.Vector3()).length() * 1.15 + 0.5;
  place();
}
function resize() { const w = stage.clientWidth, h = stage.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
window.addEventListener('resize', resize);
let drag = null;
renderer.domElement.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey, moved: 0 }; renderer.domElement.setPointerCapture(e.pointerId); });
renderer.domElement.addEventListener('pointermove', e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY; drag.moved += Math.abs(dx) + Math.abs(dy);
  if (drag.pan) { const s = orbit.dist / 900, right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0), up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1); orbit.target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s); }
  else { orbit.theta += dx * 0.008; orbit.phi = Math.max(0.08, Math.min(Math.PI - 0.08, orbit.phi - dy * 0.008)); }
  place();
});
renderer.domElement.addEventListener('pointerup', e => {
  if (drag && drag.moved < 6) pick(e);
  drag = null;
});
renderer.domElement.addEventListener('wheel', e => { e.preventDefault(); orbit.dist = Math.max(1.2, Math.min(60, orbit.dist * Math.exp(e.deltaY * 0.001))); place(); }, { passive: false });
function pick(e) {
  const rect = renderer.domElement.getBoundingClientRect(), ray = new THREE.Raycaster();
  ray.setFromCamera(new THREE.Vector2((e.clientX - rect.left) / rect.width * 2 - 1, -(e.clientY - rect.top) / rect.height * 2 + 1), camera);
  for (const hit of ray.intersectObjects(scene.children, true)) {
    let p = hit.object; while (p && !p.userData.part) p = p.parent;
    if (p) { selectPart(p.userData.part); return; }
  }
}

/* ---------- posing ---------- */
function currentLegAngles(name) {
  const pose = legPoses[name];
  const f = run ? run.trace[Math.min(frame, run.trace.length - 1)].agents[0] : null;
  const measured = f ? f.legs[name] : null;
  return {
    coxa: deg(pose.coxa) + (measured ? measured.coxa : 0),
    femur: deg(pose.femur) + (measured ? measured.femur : 0),
    knee: deg(pose.knee) + (measured ? measured.tibia - 0.6 : 0),
    tarsus: deg(pose.tarsus) + (measured ? measured.tarsus : 0),
    planted: measured ? measured.planted : null,
  };
}
function apply(time) {
  const explode = Number($('explode').value);
  for (const [name, part] of Object.entries(fly.parts)) {
    part.rotation.set(0, 0, 0);
    part.position.copy(part.userData.base).multiplyScalar(1 + explode);
    const pose = partPoses[name];
    part.visible = pose ? pose.visible : true;
    if (pose) { part.rotation.y = deg(pose.yaw); part.rotation.z = deg(pose.pitch); }
    part.traverse(o => { if (o.isMesh && o.material) { o.material.opacity = $('ghost').checked && name !== selected ? 0.25 : (o.material.userData?.opacity ?? o.material.opacity); o.material.transparent = o.material.opacity < 1; } });
  }
  for (const name of LEGS) {
    const rig = fly.legs[name], a = currentLegAngles(name), side = rig.side;
    rig.coxa.rotation.y = side * (Math.PI / 2) - a.coxa;       // protraction swings the leg forward
    rig.coxa.rotation.z = -0.12 + a.femur * 0.35;              // trochanter levation lifts it
    rig.femur.rotation.z = -0.45 + a.knee * 0.45;
    rig.tibia.rotation.z = -0.25 - a.tarsus * 0.25;
    rig.foot.material.color.setHex(a.planted === false ? 0x44523f : 0xc2e899);
    rig.foot.scale.setScalar(a.planted === false ? 0.8 : 1.2);
  }
  const f = run ? run.trace[Math.min(frame, run.trace.length - 1)].agents[0] : null;
  const stroke = $('flap').checked ? Math.sin(time * 0.012) * (manual.stroke || 0.6) : manual.stroke;
  for (const side of ['left', 'right']) {
    const sign = side === 'left' ? -1 : 1;
    fly.parts[side + '_wing'].rotation.y += sign * (deg(manual.wing) + 0.9 * stroke + (f && f.singing && side === 'left' ? 0.9 : 0));
    fly.parts[side + '_haltere'].rotation.z += manual.haltere * 0.6 * Math.sin(time * 0.02);
    fly.parts[side + '_antenna'].rotation.y += sign * deg(manual.antenna);
  }
  fly.parts.head.rotation.y += deg(manual.headYaw);
  fly.parts.head.rotation.z += deg(manual.headPitch);
  fly.parts.proboscis.scale.y = 1 + manual.proboscis * 1.5;
  fly.parts.abdomen.rotation.y += deg(manual.abdomen);
  fly.root.scale.setScalar(Number($('zoom').value));
  grid.visible = $('grid').checked;
  // Stand on the lowest foot so the body sits on the floor at any pose.
  fly.root.position.y = 0; fly.root.updateMatrixWorld(true);
  const v = new THREE.Vector3(); let lowest = Infinity;
  for (const name of LEGS) { fly.legs[name].foot.getWorldPosition(v); lowest = Math.min(lowest, v.y); }
  fly.root.position.y = -lowest - 0.85;
}
(function loop(t) { apply(t || 0); renderer.render(scene, camera); requestAnimationFrame(loop); })(0);

/* ---------- controls ---------- */
for (const [key, label] of Object.entries(PART_NAMES)) $('part').add(new Option(label, key));
function selectPart(name) {
  selected = name; $('part').value = name;
  const pose = partPoses[name] || (partPoses[name] = { yaw: 0, pitch: 0, visible: true });
  $('yaw').value = pose.yaw; $('yawOut').textContent = pose.yaw + '°';
  $('pitch').value = pose.pitch; $('pitchOut').textContent = pose.pitch + '°';
  $('visible').checked = pose.visible;
  const tag = $('partTag');
  tag.textContent = BRAIN_DRIVEN.has(name) ? 'brain-driven legs' : 'manual only';
  tag.className = 'tag' + (BRAIN_DRIVEN.has(name) ? ' brain' : '');
  if (LEGS.includes(name)) { $('leg').value = name; syncLegSliders(); }
}
function syncLegSliders() {
  const pose = legPoses[$('leg').value === 'all' ? 'LF' : $('leg').value];
  for (const id of ['coxa', 'femur', 'knee', 'tarsus']) { $(id).value = pose[id]; $(id + 'Out').textContent = pose[id] + '°'; }
}
$('part').onchange = () => selectPart($('part').value);
for (const id of ['yaw', 'pitch']) $(id).oninput = () => { partPoses[selected][id] = Number($(id).value); $(id + 'Out').textContent = $(id).value + '°'; };
$('visible').onchange = () => { partPoses[selected].visible = $('visible').checked; };
for (const id of ['coxa', 'femur', 'knee', 'tarsus']) {
  $(id).oninput = () => {
    const value = Number($(id).value); $(id + 'Out').textContent = value + '°';
    if ($('leg').value === 'all') { manual[id] = value; for (const leg of LEGS) legPoses[leg][id] = value; }
    else legPoses[$('leg').value][id] = value;
  };
}
$('leg').onchange = syncLegSliders;
for (const [id, unit] of [['wing', '°'], ['stroke', ''], ['haltere', ''], ['headYaw', '°'], ['headPitch', '°'], ['antenna', '°'], ['proboscis', ''], ['abdomen', '°']]) {
  $(id).oninput = () => { manual[id] = Number($(id).value); $(id + 'Out').textContent = $(id).value + unit; };
}
$('explode').oninput = () => $('explodeOut').textContent = $('explode').value;
$('zoom').oninput = () => { $('zoomOut').textContent = $('zoom').value + '×'; resetCamera(); };
$('reset').onclick = resetCamera;
$('resetPose').onclick = () => {
  for (const k of Object.keys(partPoses)) delete partPoses[k];
  for (const leg of LEGS) for (const j of ['coxa', 'femur', 'knee', 'tarsus']) legPoses[leg][j] = 0;
  for (const id of ['coxa', 'femur', 'knee', 'tarsus', 'headYaw', 'headPitch', 'antenna', 'abdomen']) { manual[id] = 0; $(id).value = 0; $(id + 'Out').textContent = '0°'; }
  for (const id of ['stroke', 'haltere', 'proboscis']) { manual[id] = 0; $(id).value = 0; $(id + 'Out').textContent = '0'; }
  manual.wing = 12; $('wing').value = 12; $('wingOut').textContent = '12°';
  $('explode').value = 0; $('explodeOut').textContent = '0';
  selectPart(selected);
};

/* ---------- circuit drive ---------- */
function readout() {
  if (!run) { $('legReadout').textContent = ''; return; }
  const a = run.trace[Math.min(frame, run.trace.length - 1)].agents[0];
  $('legReadout').innerHTML = '<table class="kv" style="margin-top:8px"><tr><td></td><td>coxa</td><td>femur</td><td>tibia</td><td>tarsus</td></tr>'
    + LEGS.map(leg => { const j = a.legs[leg]; return `<tr><td>${leg}${j.planted ? ' ▪' : ''}</td>` + ['coxa', 'femur', 'tibia', 'tarsus'].map(k => `<td>${(j[k] * 180 / Math.PI).toFixed(0)}°</td>`).join('') + '</tr>'; }).join('')
    + '</table>';
}
$('drive').onclick = async () => {
  $('drive').disabled = true; $('status').textContent = 'Running the measured circuit…';
  try {
    const response = await fetch('/api/arena', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scenario: 'rivalry', seed: 7, duration: 12 }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.status);
    run = data; frame = 0; $('frame').max = data.trace.length - 1;
    $('status').textContent = 'Leg joints follow the motor neurons of each joint\u2019s muscles. Sliders add to those angles; other parts stay manual.';
  } catch (e) { $('status').textContent = e.message; }
  finally { $('drive').disabled = false; }
};
$('play').onclick = () => { playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; };
$('frame').oninput = e => { playing = false; $('play').textContent = 'Play'; frame = Number(e.target.value); readout(); };
setInterval(() => {
  if (!run || !playing) return;
  frame = (frame + 1) % run.trace.length; $('frame').value = frame;
  $('clock').textContent = run.trace[frame].t.toFixed(1) + ' s';
  readout();
}, 60);

selectPart('thorax'); resize(); resetCamera();
