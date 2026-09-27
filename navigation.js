'use strict';
const $ = id => document.getElementById(id);
let result = null;
function ctx(id) { const cv = $(id), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1; cv.width = w * d; cv.height = h * d; const g = cv.getContext('2d'); g.scale(d, d); g.font = '11px sans-serif'; return [g, w, h]; }
function drawPath() {
  const [g, w, h] = ctx('path'); if (!result) return;
  const xs = result.trace.map(f => f.x), ys = result.trace.map(f => f.y);
  const pad = 12, span = Math.max(20, ...xs.map(Math.abs), ...ys.map(Math.abs)) * 1.15;
  const X = x => w / 2 + x / span * (Math.min(w, h) / 2 - pad), Y = y => h / 2 - y / span * (Math.min(w, h) / 2 - pad);
  g.strokeStyle = '#1e2a21';
  for (let v = -span; v <= span; v += span / 4) { g.beginPath(); g.moveTo(X(v), 0); g.lineTo(X(v), h); g.moveTo(0, Y(v)); g.lineTo(w, Y(v)); g.stroke(); }
  let previous = null;
  for (const f of result.trace) {
    if (previous) {
      g.strokeStyle = f.homing ? '#c2e899' : '#6b7c6e'; g.lineWidth = f.homing ? 1.8 : 1.2;
      g.beginPath(); g.moveTo(X(previous.x), Y(previous.y)); g.lineTo(X(f.x), Y(f.y)); g.stroke();
    }
    previous = f;
  }
  g.lineWidth = 1;
  const last = result.trace[result.trace.length - 1];
  g.setLineDash([4, 4]); g.strokeStyle = '#e9be75';
  g.beginPath(); g.moveTo(X(last.x), Y(last.y)); g.lineTo(X(last.x + last.home_x), Y(last.y + last.home_y)); g.stroke(); g.setLineDash([]);
  g.strokeStyle = '#e7eee5'; g.beginPath(); g.moveTo(X(0) - 6, Y(0)); g.lineTo(X(0) + 6, Y(0)); g.moveTo(X(0), Y(0) - 6); g.lineTo(X(0), Y(0) + 6); g.stroke();
  g.fillStyle = '#99aa9c'; g.fillText(`${Math.round(span)} mm`, 8, h - 8);
}
function drawBump() {
  const [g, w, h] = ctx('bump'); if (!result) return;
  const frames = result.trace, n = frames[0].bump.length;
  const cw = (w - 46) / frames.length, rh = (h - 28) / n;
  frames.forEach((f, i) => {
    const peak = Math.max(...f.bump);
    f.bump.forEach((v, k) => {
      const t = Math.min(1, v / (peak || 1));
      g.fillStyle = `rgb(${Math.round(24 + 170 * t)},${Math.round(34 + 198 * t)},${Math.round(27 + 126 * t)})`;
      g.fillRect(40 + i * cw, 8 + k * rh, Math.max(cw, 1), rh);
    });
  });
  g.strokeStyle = '#e7eee5'; g.lineWidth = 1.2; g.beginPath();
  frames.forEach((f, i) => {
    const angle = ((f.heading % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
    const y = 8 + (angle / (2 * Math.PI)) * (h - 28);
    i ? g.lineTo(40 + i * cw, y) : g.moveTo(40 + i * cw, y);
  });
  g.stroke(); g.lineWidth = 1;
  g.fillStyle = '#99aa9c'; g.fillText('0°', 8, 16); g.fillText('360°', 4, h - 22); g.fillText('time →', w - 50, h - 8);
}
function show(data) {
  result = data;
  const m = data.metrics;
  $('metrics').innerHTML = '<table class="kv">'
    + `<tr><td>Got home</td><td>${m.homed ? 'yes' : 'no'} · closest ${m.closest_during_homing_mm} mm</td></tr>`
    + `<tr><td>Furthest out</td><td>${m.furthest_distance_mm} mm</td></tr>`
    + `<tr><td>Compass error</td><td>${m.mean_heading_error_deg}° mean, ${m.final_heading_error_deg}° at the end</td></tr>`
    + `<tr><td>Home vector error</td><td>${m.home_vector_error_mm} mm</td></tr>`
    + `<tr><td>Bumps at the end</td><td>${m.bumps_at_end}</td></tr>`
    + `<tr><td>Condition</td><td>${data.condition_text}</td></tr>`
    + `<tr><td>Connectivity</td><td>${data.connectivity_text}</td></tr></table>`;
  drawPath(); drawBump();
}
async function go() {
  $('run').disabled = true; $('status').textContent = 'Running…'; $('status').className = 'wb-status';
  try {
    const response = await fetch('/api/navigation', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ connectivity: $('connectivity').value, condition: $('condition').value, compass_noise: +$('noise').value, seed: +$('seed').value }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || response.status);
    show(data); $('status').textContent = data.metrics.homed ? 'Got home.' : 'Did not get home.';
  } catch (e) { $('status').textContent = e.message; $('status').className = 'wb-status err'; }
  finally { $('run').disabled = false; }
}
async function bench() {
  try {
    const d = await (await fetch('/api/navigation-benchmark')).json();
    let h = '<table class="bench"><tr><th>Condition</th><th>Closest to home</th><th>Got home</th><th>Compass error</th><th>Home vector error</th><th>Bumps</th></tr>';
    for (const [condition, v] of Object.entries(d.summary))
      h += `<tr><td>${d.conditions[condition]}</td><td class="${v.closest_during_homing_mm < 15 ? 'hi' : 'lo'}">${v.closest_during_homing_mm} mm</td><td>${Math.round(v.homed * 100)}%</td><td>${v.mean_heading_error_deg}°</td><td>${v.home_vector_error_mm} mm</td><td>${v.bumps_at_end}</td></tr>`;
    $('bench').innerHTML = h + '</table>';
  } catch (e) { $('bench').innerHTML = '<p class="hint">No saved benchmark.</p>'; }
}
$('run').onclick = go;
$('noise').oninput = () => $('noiseOut').textContent = $('noise').value;
window.addEventListener('resize', () => { drawPath(); drawBump(); });
go(); bench();
