'use strict';
const $ = id => document.getElementById(id);
const COLOR = { male_1: '#8ab4f8', male_2: '#6fc7bc', female: '#e9be75' };
const NEAR = 5, BODY_MM = 2.5;   // a Drosophila body is about 2.5 mm long
const UNITS = 2.45 / BODY_MM;    // drawing units per mm (the body spans 2.45 units)
const LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH'];
let run = null, frame = 0, playing = true, timer = null;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function ctx(id) { const cv = $(id), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1; cv.width = w * d; cv.height = h * d; const g = cv.getContext('2d'); g.scale(d, d); g.font = '11px sans-serif'; return [g, w, h]; }
function bounds() {
  let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  run.trace.forEach(f => f.agents.forEach(a => { x0 = Math.min(x0, a.x); x1 = Math.max(x1, a.x); y0 = Math.min(y0, a.y); y1 = Math.max(y1, a.y); }));
  for (const extra of [run.food && run.food.position, run.threat && run.threat.position]) if (extra) { x0 = Math.min(x0, extra[0]); x1 = Math.max(x1, extra[0]); y0 = Math.min(y0, extra[1]); y1 = Math.max(y1, extra[1]); }
  const pad = 4;
  return { x0: x0 - pad, x1: x1 + pad, y0: y0 - pad, y1: y1 + pad };
}
// Body and legs follow the motion lab's drawing: abdomen, thorax, head, antennae, and six legs
// animated by this agent's own leg oscillators. Wings fold back, and one extends while singing.
function fly(g, a, s, colour) {
  g.save(); g.scale(s * BODY_MM / 2.45, s * BODY_MM / 2.45);
  g.lineCap = 'round';
  LEGS.forEach(leg => {
    const j = a.legs[leg], side = leg[0] === 'L' ? -1 : 1;
    const row = { F: 0.55, M: 0.0, H: -0.55 }[leg[1]];
    const hip = [row * UNITS, side * 0.25], foot = [j.foot[0] * UNITS, j.foot[1] * UNITS];
    // Knee placed outward from the hip-foot line; a lifted leg bends more.
    const mx = hip[0] + 0.45 * (foot[0] - hip[0]), my = hip[1] + 0.45 * (foot[1] - hip[1]);
    const knee = [mx + 0.18 * Math.sin(j.coxa), my + side * (0.3 + 0.25 * Math.max(j.femur, 0))];
    g.strokeStyle = j.planted ? colour : '#6c806f'; g.lineWidth = j.planted ? .085 : .06;
    g.beginPath(); g.moveTo(hip[0], hip[1]); g.lineTo(knee[0], knee[1]); g.lineTo(foot[0], foot[1]); g.stroke();
    if (j.planted) { g.fillStyle = colour; g.beginPath(); g.arc(foot[0], foot[1], .07, 0, 7); g.fill(); }
  });
  const wing = a.singing ? 1.15 : 0.18;                      // extended wing marks the song
  g.strokeStyle = '#e7eee5'; g.globalAlpha = .5; g.lineWidth = .09;
  g.beginPath(); g.moveTo(.1, -.12); g.lineTo(-1.15, -.12 - wing); g.stroke();
  g.beginPath(); g.moveTo(.1, .12); g.lineTo(-1.15, .12 + .18); g.stroke();
  g.globalAlpha = 1;
  g.fillStyle = colour; g.beginPath(); g.ellipse(-.65, 0, .8, .38, 0, 0, 7); g.fill();
  g.fillStyle = '#e7eee5'; g.beginPath(); g.ellipse(.15, 0, .52, .35, 0, 0, 7); g.fill();
  g.fillStyle = colour; g.beginPath(); g.ellipse(.72, 0, .28, .35, 0, 0, 7); g.fill();
  g.fillStyle = '#132018'; for (const y of [-.22, .22]) { g.beginPath(); g.arc(.84, y, .09, 0, 7); g.fill(); }
  g.strokeStyle = colour; g.lineWidth = .045;
  g.beginPath(); g.moveTo(.91, -.12); g.lineTo(1.22, -.35); g.moveTo(.91, .12); g.lineTo(1.22, .35); g.stroke();
  g.restore();
}
function drawArena() {
  const [g, w, h] = ctx('arenaCv'); if (!run) return;
  const b = bounds(), s = Math.min(w / (b.x1 - b.x0), h / (b.y1 - b.y0));
  const X = x => (x - b.x0) * s + (w - (b.x1 - b.x0) * s) / 2, Y = y => h - ((y - b.y0) * s + (h - (b.y1 - b.y0) * s) / 2);
  g.strokeStyle = '#1e2a21';
  for (let m = Math.ceil(b.x0 / 5) * 5; m <= b.x1; m += 5) { g.beginPath(); g.moveTo(X(m), 0); g.lineTo(X(m), h); g.stroke(); }
  for (let m = Math.ceil(b.y0 / 5) * 5; m <= b.y1; m += 5) { g.beginPath(); g.moveTo(0, Y(m)); g.lineTo(w, Y(m)); g.stroke(); }
  const now = run.trace[Math.min(frame, run.trace.length - 1)];
  const odour = (x, y, colour) => {                          // faint field showing what the agents smell
    g.save(); g.globalCompositeOperation = 'lighter';
    for (let i = 8; i >= 1; i--) {                             // layered discs avoid a hard gradient edge
      g.globalAlpha = 0.012 * (9 - i) / 8; g.fillStyle = colour;
      g.beginPath(); g.arc(X(x), Y(y), i * 1.3 * s, 0, 7); g.fill();
    }
    g.restore();
  };
  if (run.food) { const f = run.food.position; odour(f[0], f[1], '#c2e899'); g.fillStyle = '#c2e899'; g.beginPath(); g.arc(X(f[0]), Y(f[1]), 3 * s, 0, 7); g.globalAlpha = .35; g.fill(); g.globalAlpha = 1; g.fillText('food', X(f[0]) + 4 * s, Y(f[1]) - 3 * s); }
  if (run.parameters.scenario === 'rivalry' || run.parameters.scenario === 'courtship') { const f = now.agents.find(a => a.id === 'female'); odour(f.x, f.y, COLOR.female); }
  if (run.threat && now.t >= run.threat.time) { const p = run.threat.position, r = (2 + 6 * Math.min(1, now.t - run.threat.time)) * s; g.strokeStyle = '#f28b82'; g.lineWidth = 2; g.beginPath(); g.arc(X(p[0]), Y(p[1]), r, 0, 7); g.stroke(); g.lineWidth = 1; g.fillStyle = '#f28b82'; g.fillText('threat', X(p[0]) + 6, Y(p[1]) - r - 6); }
  run.agents.forEach(a => {
    g.strokeStyle = COLOR[a.id] + '55'; g.beginPath();
    run.trace.slice(0, frame + 1).forEach((fr, i) => { const p = fr.agents.find(x => x.id === a.id); i ? g.lineTo(X(p.x), Y(p.y)) : g.moveTo(X(p.x), Y(p.y)); }); g.stroke();
  });
  now.agents.forEach(p => {
    const colour = COLOR[p.id];
    g.save(); g.translate(X(p.x), Y(p.y)); g.rotate(-p.heading); fly(g, p, s, colour); g.restore();
    g.fillStyle = colour; g.fillText(p.id.replace('_', ' ') + (p.singing ? ' ♪ song' : ''), X(p.x) + 1.4 * s, Y(p.y) + 2.2 * s);
  });
  g.fillStyle = '#99aa9c';
  g.fillText(`${now.t.toFixed(1)} s`, 8, 16);
  g.fillText(`5 mm grid · flies drawn to scale (${BODY_MM} mm body)`, 8, h - 8);
  $('clock').textContent = now.t.toFixed(1) + ' s';
}
function drawDistances() {
  const [g, w, h] = ctx('distCv'); if (!run) return;
  const pairs = [['male_1', 'male_2', '#a9b6ad'], ['male_1', 'female', COLOR.male_1], ['male_2', 'female', COLOR.male_2]];
  const T = run.trace[run.trace.length - 1].t, maxD = 26, X = t => 40 + t / T * (w - 60), Y = d => h - 22 - (d / maxD) * (h - 34);
  g.strokeStyle = '#2b372e'; [0, 10, 20].forEach(d => { g.beginPath(); g.moveTo(40, Y(d)); g.lineTo(w - 16, Y(d)); g.stroke(); g.fillStyle = '#99aa9c'; g.fillText(d + ' mm', 4, Y(d) + 4); });
  g.setLineDash([3, 3]); g.strokeStyle = '#99aa9c'; g.beginPath(); g.moveTo(40, Y(NEAR)); g.lineTo(w - 16, Y(NEAR)); g.stroke(); g.setLineDash([]);
  pairs.forEach(([a, b, col]) => {
    g.strokeStyle = col; g.lineWidth = 1.6; g.beginPath();
    run.trace.forEach((fr, i) => { const p = fr.agents.find(x => x.id === a), q = fr.agents.find(x => x.id === b); const d = Math.hypot(p.x - q.x, p.y - q.y); i ? g.lineTo(X(fr.t), Y(Math.min(d, maxD))) : g.moveTo(X(fr.t), Y(Math.min(d, maxD))); }); g.stroke(); g.lineWidth = 1;
  });
  const t = run.trace[Math.min(frame, run.trace.length - 1)].t;
  g.strokeStyle = '#e7eee5'; g.beginPath(); g.moveTo(X(t), 6); g.lineTo(X(t), h - 22); g.stroke();
}
function legPanel() {
  if (!run) return;
  const id = $('legAgent').value, now = run.trace[Math.min(frame, run.trace.length - 1)];
  const a = now.agents.find(x => x.id === id); if (!a) return;
  const joints = ['coxa', 'femur', 'tibia', 'tarsus'], span = { coxa: .55, femur: 1, tibia: .6, tarsus: .6 }, rest = { coxa: 0, femur: 0, tibia: .6, tarsus: 0 };
  let h = '<table class="kv"><tr><td></td>' + joints.map(j => `<td style="color:var(--muted)">${j}</td>`).join('') + '</tr>';
  for (const leg of LEGS) {
    const j = a.legs[leg];
    h += `<tr><td>${leg}${j.planted ? ' ▪' : ''}</td>` + joints.map(name => {
      const v = Math.max(-1, Math.min(1, (j[name] - rest[name]) / span[name])), w = Math.abs(v) * 26;
      const colour = v >= 0 ? 'var(--teal)' : 'var(--gold)';
      return `<td><span style="display:inline-block;width:30px;text-align:right"><span style="display:inline-block;height:8px;width:${v < 0 ? w : 0}px;background:${colour}"></span></span><span style="display:inline-block;width:30px"><span style="display:inline-block;height:8px;width:${v > 0 ? w : 0}px;background:${colour}"></span></span></td>`;
    }).join('') + '</tr>';
  }
  $('legs').innerHTML = h + '</table>';
}
function gaitPanel() {
  if (!run || !run.gait) return;
  const id = $('legAgent').value, g = run.gait[id], R = run.reported_ranges;
  const band = (v, [lo, hi]) => v >= lo && v <= hi ? 'var(--teal)' : 'var(--gold)';
  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
  const duty = mean(g.duty_factor_relative), steps = mean(g.steps_per_second);
  $('gait').innerHTML = '<table class="kv">'
    + `<tr><td>Speed</td><td style="color:${band(g.mean_speed_mm_s, R.speed_mm_s)}">${g.mean_speed_mm_s.toFixed(1)} mm/s</td><td style="color:var(--muted)">reported ${R.speed_mm_s.join('–')}</td></tr>`
    + `<tr><td>Steps per leg</td><td style="color:${band(steps, R.step_hz)}">${steps.toFixed(1)} /s</td><td style="color:var(--muted)">reported ${R.step_hz.join('–')}</td></tr>`
    + `<tr><td>Duty factor</td><td style="color:${band(duty, R.duty_factor)}">${duty.toFixed(2)}</td><td style="color:var(--muted)">reported ${R.duty_factor.join('–')}</td></tr>`
    + `<tr><td>Tripod index</td><td style="color:var(--gold)">${g.tripod_index.toFixed(2)}</td><td style="color:var(--muted)">0 = no coordination</td></tr>`
    + `<tr><td>Legs never lifted</td><td>${g.duty_factor_absolute.filter(v => v > 0.9).length} of 6</td><td style="color:var(--muted)">by the absolute rule</td></tr></table>`;
}
function metrics() {
  const p = run.pairs, row = (k, v) => `<tr><td>${k}</td><td>${v}</td></tr>`;
  const name = k => k.replace('male_1', 'male 1').replace('male_2', 'male 2').replace('|', ' ↔ ');
  let h = `<p class="hint">${esc(run.scenario_text)}<br>${esc(run.condition_text)}</p><table class="kv">`;
  for (const [k, v] of Object.entries(p)) h += row(name(k) + ' closest', `${v.min_distance === null ? '–' : v.min_distance.toFixed(2)} mm, near ${v.near_s.toFixed(1)} s`);
  h += row('Both males near her', run.both_males_near_female_s.toFixed(1) + ' s');
  h += row('Song', Object.entries(run.song_seconds).filter(([, v]) => v > 0).map(([k, v]) => `${name(k)} ${v.toFixed(1)} s`).join(', ') || 'none');
  if (run.food) h += row('Reached food', Object.entries(run.food.arrivals).map(([k, v]) => `${name(k)} ${v}s`).join(', ') || 'nobody');
  for (const a of run.agents) h += row(name(a.id) + ' walked', `${a.path_mm.toFixed(1)} mm, ${a.mean_dn_hz.toFixed(1)} Hz descending`);
  $('metrics').innerHTML = h + '</table>';
}
function tick() { if (!run || !playing) return; frame = (frame + 1) % run.trace.length; $('frame').value = frame; drawArena(); drawDistances(); legPanel(); }
async function go() {
  $('arenaRun').disabled = true; $('status').textContent = 'Running…'; $('status').className = 'wb-status';
  try {
    const res = await fetch('/api/arena', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scenario: $('scenario').value, condition: $('condition').value, seed: +$('seed').value, body: $('bodyMode').value }) });
    const r = await res.json(); if (!res.ok) throw new Error(r.error || res.status);
    run = r; frame = 0; $('frame').max = r.trace.length - 1; metrics(); drawArena(); drawDistances(); legPanel(); gaitPanel();
    $('status').textContent = `${r.parameters.duration} s, seed ${r.parameters.seed}`;
  } catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; }
  finally { $('arenaRun').disabled = false; }
}
async function bench() {
  try {
    const d = await (await fetch('/api/arena-benchmark')).json();
    let h = '<table class="bench"><tr><th>Scenario</th><th>Senses</th><th>Male 1 near her</th><th>Closest</th><th>Both males near</th><th>Song</th><th>Mean path</th><th>Reached food</th></tr>';
    for (const [sc, cs] of Object.entries(d.summary)) for (const [c, v] of Object.entries(cs))
      h += `<tr><td>${sc}</td><td>${c.replace('_', ' ')}</td><td>${v.male1_female_near_s.toFixed(1)} s</td><td>${v.male1_female_min_mm.toFixed(1)} mm</td><td>${v.both_males_near_female_s.toFixed(1)} s</td><td>${v.song_s.toFixed(1)} s</td><td>${v.mean_path_mm.toFixed(0)} mm</td><td>${sc === 'food' ? v.food_arrived.toFixed(1) + ' of 3' + (v.food_first_arrival_s ? `, first ${v.food_first_arrival_s.toFixed(1)} s` : '') : '–'}</td></tr>`;
    $('bench').innerHTML = h + '</table>';
  } catch (e) { $('bench').innerHTML = '<p class="hint">No saved benchmark.</p>'; }
}
$('arenaRun').onclick = go;
$('play').onclick = () => { playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; };
$('frame').oninput = e => { playing = false; $('play').textContent = 'Play'; frame = +e.target.value; drawArena(); drawDistances(); legPanel(); };
$('legAgent').onchange = () => { legPanel(); gaitPanel(); };
window.addEventListener('resize', () => { drawArena(); drawDistances(); });
timer = setInterval(tick, 60);
go(); bench();
