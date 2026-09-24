'use strict';
/* Three jointed flies in 3D. Leg joints follow the simulation's motor neurons; head, antennae, wings
   and abdomen have no motor neurons in this subset and are driven by the controls only.
   Segment lengths follow bodyplan.py (coxa .30, femur .75, tibia .75, tarsus .55 mm). */
const $ = id => document.getElementById(id);
const COLOR = { male_1: 0x8ab4f8, male_2: 0x6fc7bc, female: 0xe9be75 };
const LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH'];
const SEG = { coxa: 0.30, femur: 0.75, tibia: 0.75, tarsus: 0.55 };
const HIP = { F: 0.55, M: 0.0, H: -0.55 };
let run = null, frame = 0, playing = true, rigs = {}, manual = { headYaw: 0, headPitch: 0, antenna: 0, abdomen: 0, wingL: 8, wingR: 8 };

/* ---------- scene ---------- */
const stage = $('stage3'), renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(devicePixelRatio || 1); stage.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b110d);
const camera = new THREE.PerspectiveCamera(42, 1, 0.05, 500);
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 0.8); key.position.set(6, 12, 8); scene.add(key);
const world = new THREE.Group(); scene.add(world);
const grid = new THREE.GridHelper(60, 24, 0x2b372e, 0x1a241c); grid.position.y = 0; world.add(grid);
const marker = new THREE.Mesh(new THREE.CircleGeometry(3, 32).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0xc2e899, transparent: true, opacity: 0.18 }));
marker.visible = false; world.add(marker);
let orbit = { theta: 0.9, phi: 1.05, dist: 26, target: new THREE.Vector3() };
function place() {
  const { theta, phi, dist, target } = orbit;
  camera.position.set(target.x + dist * Math.sin(phi) * Math.cos(theta), target.y + dist * Math.cos(phi), target.z + dist * Math.sin(phi) * Math.sin(theta));
  camera.lookAt(target);
}
function resize() { const w = stage.clientWidth, h = stage.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
window.addEventListener('resize', resize);
let drag = null;
renderer.domElement.addEventListener('pointerdown', e => { drag = { x: e.clientX, y: e.clientY, pan: e.shiftKey }; renderer.domElement.setPointerCapture(e.pointerId); });
renderer.domElement.addEventListener('pointermove', e => {
  if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) { const s = orbit.dist / 700, right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0), up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1); orbit.target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s); $('follow').value = ''; }
  else { orbit.theta += dx * 0.008; orbit.phi = Math.max(0.12, Math.min(Math.PI / 2 - 0.02, orbit.phi - dy * 0.006)); }
  place();
});
renderer.domElement.addEventListener('pointerup', () => { drag = null; });
renderer.domElement.addEventListener('wheel', e => { e.preventDefault(); orbit.dist = Math.max(3, Math.min(120, orbit.dist * Math.exp(e.deltaY * 0.001))); place(); }, { passive: false });

/* ---------- fly rig ---------- */
const mat = (colour, opts = {}) => new THREE.MeshLambertMaterial(Object.assign({ color: colour }, opts));
function segment(length, radius, colour) {           // a limb segment lying along +x from its joint
  const g = new THREE.Group();
  const m = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius * 0.75, length, 6), mat(colour));
  m.rotation.z = -Math.PI / 2; m.position.x = length / 2; g.add(m);
  return g;
}
function buildFly(colour) {
  const fly = new THREE.Group(), body = new THREE.Group(); fly.add(body);
  const thorax = new THREE.Mesh(new THREE.SphereGeometry(0.42, 16, 12), mat(colour));
  thorax.scale.set(1.25, 0.95, 1.0); body.add(thorax);
  const abdomenPivot = new THREE.Group(); abdomenPivot.position.x = -0.35; body.add(abdomenPivot);
  const abdomen = new THREE.Mesh(new THREE.SphereGeometry(0.42, 16, 12), mat(colour));
  abdomen.scale.set(1.7, 0.85, 0.85); abdomen.position.x = -0.6; abdomenPivot.add(abdomen);
  const neck = new THREE.Group(); neck.position.x = 0.55; body.add(neck);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.3, 16, 12), mat(0xe7eee5));
  head.scale.set(0.85, 1, 1.05); neck.add(head);
  const eyes = [];
  for (const side of [-1, 1]) {                       // compound eyes: fixed to the head, never moved
    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.19, 12, 10), mat(0x8c2f2a));
    eye.position.set(0.05, 0.02, side * 0.2); eye.scale.set(0.9, 1.1, 0.8); neck.add(eye); eyes.push(eye);
  }
  const antennae = [];
  for (const side of [-1, 1]) {
    const a = new THREE.Group(); a.position.set(0.22, -0.05, side * 0.09); neck.add(a);
    a.add(segment(0.28, 0.035, 0.6 * colour === 0 ? colour : colour));
    a.rotation.z = -0.5; a.rotation.y = side * 0.35; antennae.push({ group: a, side });
  }
  const wings = [];
  for (const side of [-1, 1]) {
    const w = new THREE.Group(); w.position.set(0.15, 0.22, side * 0.12); body.add(w);
    const blade = new THREE.Mesh(new THREE.CircleGeometry(0.95, 16).rotateX(-Math.PI / 2), new THREE.MeshLambertMaterial({ color: 0xe7eee5, transparent: true, opacity: 0.32, side: THREE.DoubleSide }));
    blade.scale.set(1.0, 1, 0.42); blade.position.x = -0.85; w.add(blade); wings.push({ group: w, side });
  }
  const legs = {};
  for (const leg of LEGS) {
    const side = leg[0] === 'L' ? -1 : 1;
    const coxa = new THREE.Group(); coxa.position.set(HIP[leg[1]], -0.18, side * 0.3); body.add(coxa);
    const femur = segment(SEG.femur, 0.045, colour); femur.position.x = SEG.coxa;
    const tibia = segment(SEG.tibia, 0.038, colour); tibia.position.x = SEG.femur;
    const tarsus = segment(SEG.tarsus, 0.03, colour); tarsus.position.x = SEG.tibia;
    const foot = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 6), mat(0xc2e899)); foot.position.x = SEG.tarsus; tarsus.add(foot);
    coxa.add(segment(SEG.coxa, 0.055, colour)); coxa.add(femur); femur.add(tibia); tibia.add(tarsus);
    legs[leg] = { coxa, femur, tibia, tarsus, foot, side };
  }
  return { fly, body, neck, abdomenPivot, wings, antennae, eyes, legs };
}

/* ---------- posing ---------- */
const deg = d => d * Math.PI / 180;
function poseFly(rig, agent, singing) {
  const p = agent;
  rig.fly.position.set(p.x, 0, -p.y);                        // arena x,y (mm) -> world x,z; height set below
  rig.fly.rotation.y = p.heading;
  rig.neck.rotation.y = deg(manual.headYaw); rig.neck.rotation.z = deg(manual.headPitch);
  rig.abdomenPivot.rotation.y = deg(manual.abdomen);
  rig.antennae.forEach(a => { a.group.rotation.y = a.side * (0.35 + deg(manual.antenna)); });
  rig.wings.forEach(w => {
    const base = w.side < 0 ? manual.wingL : manual.wingR;
    const extend = ($('songWing').checked && singing && w.side < 0) ? 70 : base;
    w.group.rotation.y = w.side * deg(extend);
  });
  for (const leg of LEGS) {
    const j = p.legs[leg], rl = rig.legs[leg], side = rl.side;
    // Rest pose: legs splay outward and step down to the ground. Joint angles modulate it:
    // coxa protraction swings the leg forward (yaw), trochanter levation lifts it, tibia extension
    // straightens the knee, tarsus depression plants the foot.
    rl.coxa.rotation.y = side * (Math.PI / 2) - j.coxa;
    rl.coxa.rotation.z = -0.10 + j.femur * 0.35;      // negative pitches the limb downward
    rl.femur.rotation.z = -0.45 + j.tibia * 0.40;
    rl.tibia.rotation.z = -0.25 - j.tarsus * 0.25;
    rl.foot.material.color.setHex(j.planted ? 0xc2e899 : 0x44523f);
    rl.foot.scale.setScalar(j.planted ? 1.3 : 0.8);
  }
  // The body rides on whichever legs are lowest, so planted feet meet the ground instead of the
  // body floating at a fixed height. Purely geometric, like the rest of the drawing.
  rig.fly.updateMatrixWorld(true);
  const v = new THREE.Vector3();
  let lowest = Infinity, lowestPlanted = Infinity;
  for (const leg of LEGS) {
    rig.legs[leg].foot.getWorldPosition(v);
    lowest = Math.min(lowest, v.y);
    if (p.legs[leg].planted) lowestPlanted = Math.min(lowestPlanted, v.y);
  }
  const reference = Number.isFinite(lowestPlanted) ? lowestPlanted : lowest;
  rig.fly.position.y = Math.max(0.5, Math.min(1.6, -reference + 0.06));
}
function ensureRigs() {
  world.children.filter(c => c.userData.isFly).forEach(c => world.remove(c));
  rigs = {};
  for (const a of run.agents) { const rig = buildFly(COLOR[a.id]); rig.fly.userData.isFly = true; world.add(rig.fly); rigs[a.id] = rig; }
}
function draw() {
  if (!run) return;
  const now = run.trace[Math.min(frame, run.trace.length - 1)];
  now.agents.forEach(a => rigs[a.id] && poseFly(rigs[a.id], a, a.singing));
  const followId = $('follow').value;
  if (followId && rigs[followId]) { orbit.target.copy(rigs[followId].fly.position); place(); }
  if (run.food) { marker.visible = true; marker.position.set(run.food.position[0], 0.02, -run.food.position[1]); }
  else if (run.threat && now.t >= run.threat.time) { marker.visible = true; marker.position.set(run.threat.position[0], 0.02, -run.threat.position[1]); }
  else marker.visible = false;
  $('clock3').textContent = now.t.toFixed(1) + ' s';
  readout(now);
}
function readout(now) {
  const a = now.agents.find(x => x.id === $('who').value); if (!a) return;
  let h = '<table class="kv"><tr><td></td><td style="color:var(--muted)">coxa</td><td style="color:var(--muted)">femur</td><td style="color:var(--muted)">tibia</td><td style="color:var(--muted)">tarsus</td></tr>';
  for (const leg of LEGS) {
    const j = a.legs[leg];
    h += `<tr><td>${leg}${j.planted ? ' ▪' : ''}</td>` + ['coxa', 'femur', 'tibia', 'tarsus'].map(k => `<td>${(j[k] * 180 / Math.PI).toFixed(0)}°</td>`).join('') + '</tr>';
  }
  $('legreadout').innerHTML = h + '</table>';
}
(function loop() { renderer.render(scene, camera); requestAnimationFrame(loop); })();
setInterval(() => { if (run && playing) { frame = (frame + 1) % run.trace.length; $('frame3').value = frame; draw(); } }, 60);

/* ---------- controls ---------- */
for (const id of ['headYaw', 'headPitch', 'antenna', 'abdomen', 'wingL', 'wingR']) {
  $(id).addEventListener('input', () => { manual[id] = +$(id).value; $(id + 'Out').textContent = $(id).value + '°'; draw(); });
}
$('play3').onclick = () => { playing = !playing; $('play3').textContent = playing ? 'Pause' : 'Play'; };
$('frame3').oninput = e => { playing = false; $('play3').textContent = 'Play'; frame = +e.target.value; draw(); };
$('who').onchange = () => draw();
$('follow').onchange = () => { if (!$('follow').value) { orbit.target.set(0, 0, 0); place(); } draw(); };
async function go() {
  $('run3').disabled = true; $('status').textContent = 'Running…'; $('status').className = 'wb-status';
  try {
    const res = await fetch('/api/arena', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scenario: $('scenario').value, seed: +$('seed').value }) });
    const r = await res.json(); if (!res.ok) throw new Error(r.error || res.status);
    run = r; frame = 0; $('frame3').max = r.trace.length - 1; ensureRigs(); draw();
    $('status').textContent = `${r.parameters.duration} s, seed ${r.parameters.seed}, ${r.gait[r.agents[0].id].mean_speed_mm_s.toFixed(1)} mm/s`;
  } catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; }
  finally { $('run3').disabled = false; }
}
$('run3').onclick = go;
resize(); place(); go();
