'use strict';
const $ = id => document.getElementById(id);
const COLOR = { male_1: '#8ab4f8', male_2: '#6fc7bc', female: '#e9be75' };
const NEAR = 5, VIEW = 30;   // mm half-width of the drawn arena
let run = null, frame = 0, playing = true, timer = null;
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function ctx(id) { const cv = $(id), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1; cv.width = w * d; cv.height = h * d; const g = cv.getContext('2d'); g.scale(d, d); g.font = '11px sans-serif'; return [g, w, h]; }
function drawArena() {
  const [g, w, h] = ctx('arenaCv'); if (!run) return;
  const s = Math.min(w, h) / (2 * VIEW), X = x => w / 2 + x * s, Y = y => h / 2 - y * s;
  g.strokeStyle = '#1e2a21'; for (let m = -VIEW; m <= VIEW; m += 5) { g.beginPath(); g.moveTo(X(m), 0); g.lineTo(X(m), h); g.moveTo(0, Y(m)); g.lineTo(w, Y(m)); g.stroke(); }
  const now = run.trace[Math.min(frame, run.trace.length - 1)];
  if (run.food) { const f = run.food.position; g.fillStyle = 'rgba(194,232,153,.18)'; g.beginPath(); g.arc(X(f[0]), Y(f[1]), 3 * s, 0, 7); g.fill(); g.fillStyle = '#c2e899'; g.fillText('food', X(f[0]) + 4, Y(f[1]) - 4); }
  if (run.threat && now.t >= run.threat.time) { const p = run.threat.position, r = (2 + 6 * Math.min(1, (now.t - run.threat.time))) * s; g.strokeStyle = '#f28b82'; g.lineWidth = 2; g.beginPath(); g.arc(X(p[0]), Y(p[1]), r, 0, 7); g.stroke(); g.fillStyle = '#f28b82'; g.fillText('threat', X(p[0]) + 4, Y(p[1]) - r - 4); g.lineWidth = 1; }
  run.agents.forEach(a => {                                        // trail
    g.strokeStyle = COLOR[a.id] + '55'; g.beginPath();
    run.trace.slice(0, frame + 1).forEach((fr, i) => { const p = fr.agents.find(x => x.id === a.id); i ? g.lineTo(X(p.x), Y(p.y)) : g.moveTo(X(p.x), Y(p.y)); }); g.stroke();
  });
  now.agents.forEach(p => {
    const c = COLOR[p.id]; g.save(); g.translate(X(p.x), Y(p.y)); g.rotate(-p.heading);
    g.fillStyle = c; g.beginPath(); g.moveTo(9, 0); g.lineTo(-6, 5); g.lineTo(-6, -5); g.closePath(); g.fill();
    if (p.singing) { g.strokeStyle = c; g.beginPath(); g.arc(0, 0, 13 + 3 * Math.sin(frame / 2), -0.9, 0.9); g.stroke(); }
    g.restore();
    g.fillStyle = c; g.fillText(p.id.replace('_', ' ') + (p.singing ? ' ♪' : ''), X(p.x) + 10, Y(p.y) + 14);
  });
  g.fillStyle = '#99aa9c'; g.fillText(`${now.t.toFixed(1)} s`, 8, 16); g.fillText('5 mm grid', 8, h - 8);
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
function tick() { if (!run || !playing) return; frame = (frame + 1) % run.trace.length; $('frame').value = frame; drawArena(); drawDistances(); }
async function go() {
  $('arenaRun').disabled = true; $('status').textContent = 'Running…'; $('status').className = 'wb-status';
  try {
    const res = await fetch('/api/arena', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ scenario: $('scenario').value, condition: $('condition').value, seed: +$('seed').value }) });
    const r = await res.json(); if (!res.ok) throw new Error(r.error || res.status);
    run = r; frame = 0; $('frame').max = r.trace.length - 1; metrics(); drawArena(); drawDistances();
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
$('frame').oninput = e => { playing = false; $('play').textContent = 'Play'; frame = +e.target.value; drawArena(); drawDistances(); };
window.addEventListener('resize', () => { drawArena(); drawDistances(); });
timer = setInterval(tick, 60);
go(); bench();
