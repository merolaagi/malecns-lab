'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = v => v == null ? '–' : Math.round(v * 100) + '%';
const C = ['#2f5fd0', '#d0524a', '#4aa7ee', '#e9a23b', '#3f9a70', '#8150e6', '#2aa0a0', '#8a8a8a'];
const DASH = [[], [7, 4], [2, 4], [9, 4, 2, 4], [4, 3], [1, 3], [6, 2], [3, 6]];
function lines(id, xs, series, xlabel, legendId) {
  const cv = $(id), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1; cv.width = w * d; cv.height = h * d;
  const g = cv.getContext('2d'); g.scale(d, d); g.font = '12px Inter, sans-serif';
  const L = 46, R = 30, T = 12, B = 34, X = i => L + i * (w - L - R) / Math.max(1, xs.length - 1), Y = v => T + (1 - v) * (h - T - B);
  g.strokeStyle = '#e6e6e3'; g.strokeRect(L, T, w - L - R, h - T - B); g.fillStyle = '#6e6e6e';
  for (const v of [0, 0.25, 0.5, 0.75, 1]) { g.strokeStyle = '#efefed'; g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(w - R, Y(v)); g.stroke(); g.fillText(Math.round(v * 100) + '%', 8, Y(v) + 4); }
  xs.forEach((x, i) => g.fillText(String(x), X(i) - 8, h - 16)); g.fillText(xlabel, w / 2 - 40, h - 2);
  series.forEach((s, k) => {
    g.strokeStyle = C[k % C.length]; g.setLineDash(DASH[k % DASH.length]); g.lineWidth = k === 0 ? 2.6 : 1.6; g.beginPath();
    s.values.forEach((v, i) => i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))); g.stroke(); g.setLineDash([]);
    s.values.forEach((v, i) => { g.beginPath(); g.arc(X(i), Y(v), 3, 0, 7); g.fillStyle = k === 0 ? C[0] : '#fff'; g.fill(); g.strokeStyle = k === 0 ? '#fff' : C[k % C.length]; g.lineWidth = 1.3; g.stroke(); });
  });
  $(legendId).innerHTML = series.map((s, k) => `<span><i style="background:${C[k % C.length]}"></i>${esc(s.name)}</span>`).join('');
}
const mean = a => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
const signed = v => (v >= 0 ? '+' : '−') + Math.abs(Math.round(v * 100));

let B = null, D = null;
async function load() {
  try { const r = await fetch('/api/backbone-benchmark'); if (r.ok) B = await r.json(); } catch (e) {}
  try { const r = await fetch('/api/dopamine-loops'); if (r.ok) D = await r.json(); } catch (e) {}
  inside(); dopamine(); summary();
}
const row = (o) => B.rows.find(r => Object.entries(o).every(([k, v]) => r[k] === v));

function inside() {
  if (!B) { for (const id of ['rules', 'biology', 'motor']) $(id).innerHTML = '<p class="hint">No results yet: python3 backbone.py</p>'; return; }
  const wir = Object.keys(B.wirings), tasks = Object.keys(B.tasks);
  const table = (backbone, rules) => tasks.map(t => `<h3 style="font-size:13px;margin:14px 0 6px">${esc(B.tasks[t])}</h3><div class="tw"><table class="bench"><tr><th style="text-align:left">Learning rule</th>${wir.map(w => `<th>${esc(B.wirings[w])}</th>`).join('')}${t === 'continual' ? '<th>Forgetting (measured)</th>' : ''}</tr>
    ${rules.map(rule => { const cells = wir.map(w => row({ task: t, backbone, wiring: w, rule, biology: 'plain' })); const best = Math.max(...cells.map(c => c ? c.final : 0));
      return `<tr><td style="text-align:left">${esc(B.rules[rule])}</td>${cells.map(c => c ? `<td class="${c.final === best ? 'best' : ''}">${pct(c.final)} <small class="muted">±${Math.round(c.final_sd * 100)}</small></td>` : '<td>–</td>').join('')}${t === 'continual' ? `<td>${cells[0] && cells[0].forgetting != null ? pct(cells[0].forgetting) : '–'}</td>` : ''}</tr>`; }).join('')}</table></div>`).join('');
  $('rules').innerHTML = table('central_complex', Object.keys(B.rules)) + `<p class="hint">${Object.entries(B.backbones).map(([k, b]) => `${esc(b.label)}: ${b.connections.toLocaleString()} learned connections, ${b.inputs} input cells, ${b.outputs} output cells`).join(' · ')}. Bold: the best wiring in each row. ± is the spread over seeds.</p>`;
  $('motor').innerHTML = `<p class="note">Inputs at the descending and sensory neurons, outputs read from the motor neurons.</p>` + table('motor', ['readout', 'teachers_wiring', 'backprop']);
  const bios = Object.keys(B.biology).filter(b => b !== 'plain');
  $('biology').innerHTML = tasks.map(t => { const plain = { teachers_wiring: row({ task: t, backbone: 'central_complex', wiring: 'measured', rule: 'teachers_wiring', biology: 'plain' }), backprop: row({ task: t, backbone: 'central_complex', wiring: 'measured', rule: 'backprop', biology: 'plain' }) };
    const cell = (b, rule, w) => { const r = row({ task: t, backbone: 'central_complex', wiring: w, rule, biology: b }); if (!r) return '<td>–</td>'; const d = r.final - plain[rule].final;
      return w === 'measured' ? `<td class="${d > 0.03 ? 'up' : d < -0.03 ? 'down' : ''}">${pct(r.final)} (${signed(d)})</td>` : `<td>${pct(r.final)}</td>`; };
    return `<h3 style="font-size:13px;margin:14px 0 6px">${esc(B.tasks[t])}</h3><div class="tw"><table class="bench"><tr><th style="text-align:left">Unit</th><th>Teachers along the wiring</th><th><small>same, shuffled wiring</small></th><th>Backprop</th><th><small>same, shuffled wiring</small></th></tr>
      <tr><td style="text-align:left">${esc(B.biology.plain)}</td><td>${pct(plain.teachers_wiring.final)}</td><td>${pct((row({ task: t, backbone: 'central_complex', wiring: 'shuffled', rule: 'teachers_wiring', biology: 'plain' }) || {}).final)}</td><td>${pct(plain.backprop.final)}</td><td>${pct((row({ task: t, backbone: 'central_complex', wiring: 'shuffled', rule: 'backprop', biology: 'plain' }) || {}).final)}</td></tr>
      ${bios.map(b => `<tr><td style="text-align:left">${esc(B.biology[b])}</td>${cell(b, 'teachers_wiring', 'measured')}${cell(b, 'teachers_wiring', 'shuffled')}${cell(b, 'backprop', 'measured')}${cell(b, 'backprop', 'shuffled')}</tr>`).join('')}</table></div>`; }).join('')
    + '<p class="hint">Green and red mark changes of more than 3 points from the plain unit.</p>';
}

function dopamine() {
  if (!D) { $('dopamine').innerHTML = '<p class="hint">No results yet: python3 dopamine_loops.py</p>'; $('dbiology').innerHTML = ''; return; }
  const rules = Object.keys(D.rules), wir = Object.keys(D.wirings), tasks = Object.keys(D.tasks);
  lines('dcurve', D.checkpoints, rules.map(r => ({ name: D.rules[r], values: D.by_rule.aim[r].measured.curve.map(c => c.accuracy) })).concat([{ name: 'Standard dense network, told the answer', values: D.standard_network.aim.curve.map(c => c.accuracy) }]), 'trials (aiming task, measured wiring)', 'dlegend');
  const ls = D.loop_structure.measured;
  $('dopamine').innerHTML = tasks.map(t => `<h3 style="font-size:13px;margin:14px 0 6px">${esc(D.tasks[t])}: accuracy after ${D.by_rule[t].readout.measured.curve.slice(-1)[0].trials.toLocaleString()} trials (chance ${pct(D.by_rule[t].readout.measured.chance)})</h3><div class="tw"><table class="bench"><tr><th style="text-align:left">How dopamine is delivered</th>${wir.map(w => `<th>${esc(D.wirings[w])}</th>`).join('')}${t === 'continual' ? '<th>Forgetting (measured)</th>' : ''}</tr>
    ${rules.map(r => `<tr><td style="text-align:left">${esc(D.rules[r])}</td>${wir.map(w => `<td>${pct(D.by_rule[t][r][w].final)} <small class="muted">±${Math.round(D.by_rule[t][r][w].final_sd * 100)}</small></td>`).join('')}${t === 'continual' ? `<td>${pct(D.by_rule[t][r].measured.forgetting)}</td>` : ''}</tr>`).join('')}
    <tr><td style="text-align:left">Standard dense network, backpropagation, told the answer</td><td colspan="${wir.length}">${pct(D.standard_network[t].final)}</td>${t === 'continual' ? `<td>${pct(D.standard_network[t].forgetting)}</td>` : ''}</tr></table></div>`).join('')
    + `<p class="hint">Every rule but backpropagation and the standard network learns from reward and punishment alone. In the measured wiring, ${pct(ls.loops.from_own_compartment)} of the dopamine reaching an output neuron comes from dopamine neurons that listen to output neurons of its own compartment (${pct(D.loop_structure.measured.loops_shuffled.from_own_compartment)} with the targets shuffled), and ${pct(ls.loops.from_itself)} from ones that listen to that very neuron. The loops are real but loose. ${D.learned_synapses.toLocaleString()} learned synapses; rates: one per rule, chosen on a separate seed.</p>`;
  const bio = Object.keys(D.biology_text);
  const lc = D.loops_with_compartment_votes;
  $('dbiology').innerHTML = `<p class="note">On the rule that learns from reward alone (dopamine per output neuron), measured wiring. One change at a time; the change from the reference in brackets.${lc ? ` Dopamine through the measured loops, with votes organised by compartment: ${tasks.map(t => `${esc(D.tasks[t].split(' (')[0].toLowerCase())} ${pct(lc[t].loops.final)} (shuffled loops ${pct(lc[t].loops_shuffled.final)})`).join('; ')}.` : ''}</p><div class="tw"><table class="bench"><tr><th style="text-align:left">Model</th>${tasks.map(t => `<th>${esc(D.tasks[t].split(' (')[0])}</th>`).join('')}<th>Forgetting</th></tr>
    ${bio.map(b => `<tr><td style="text-align:left">${esc(D.biology_text[b])}</td>${tasks.map(t => { const v = D.biology[t][b].final, d = v - D.biology[t].plain.final; return `<td class="${b !== 'plain' && d > 0.03 ? 'up' : b !== 'plain' && d < -0.03 ? 'down' : ''}">${pct(v)}${b === 'plain' ? '' : ` (${signed(d)})`}</td>`; }).join('')}<td>${pct(D.biology.continual[b].forgetting)}</td></tr>`).join('')}</table></div>`;
}

function summary() {
  if (!B && !D) { $('summary').innerHTML = '<p class="hint">No results yet: python3 backbone.py and python3 dopamine_loops.py</p>'; return; }
  let h = '<h2>What came out</h2><ul>';
  if (B) {
    const tasks = Object.keys(B.tasks), f = (t, rule, w) => (row({ task: t, backbone: 'central_complex', wiring: w, rule, biology: 'plain' }) || {}).final;
    const gain = rule => mean(tasks.map(t => f(t, rule, 'measured') - f(t, 'readout', 'measured')));
    const wiringGap = rule => mean(tasks.map(t => f(t, rule, 'measured') - f(t, rule, 'shuffled')));
    h += `<li><b>Learning inside the wiring beats a frozen backbone.</b> Averaged over the three tasks, against readout-only: teachers along the wiring ${signed(gain('teachers_wiring'))} points, random teacher pathways ${signed(gain('teachers_random'))}, scalar reward ${signed(gain('reward'))}, backpropagation ${signed(gain('backprop'))}.</li>`;
    h += `<li><b>Does the measured wiring help once it learns?</b> Measured minus shuffled, averaged over tasks: readout ${signed(wiringGap('readout'))} points, teachers along the wiring ${signed(wiringGap('teachers_wiring'))}, backpropagation ${signed(wiringGap('backprop'))}. Positive means the measured wiring is better.</li>`;
    const bios = Object.keys(B.biology).filter(b => b !== 'plain');
    const eff = b => mean(tasks.map(t => ((row({ task: t, backbone: 'central_complex', wiring: 'measured', rule: 'teachers_wiring', biology: b }) || {}).final ?? NaN) - f(t, 'teachers_wiring', 'measured')));
    const ranked = bios.map(b => [b, eff(b)]).sort((a, b) => b[1] - a[1]);
    h += `<li><b>Which biology matters</b> (teachers along the wiring, change from the plain unit, averaged over tasks): ${ranked.map(([b, v]) => `${esc(B.biology[b].toLowerCase())} ${signed(v)}`).join('; ')}.</li>`;
  }
  if (D) {
    const tasks = ['aim', 'digits'], f = (r, w = 'measured') => mean(tasks.map(t => D.by_rule[t][r][w].final));
    h += `<li><b>Reward and punishment alone</b> (mushroom body, aiming and digits averaged): frozen backbone with a learned readout ${pct(f('readout'))}; ideal dopamine routing ${pct(f('per_cell'))}; dopamine through the measured loops ${pct(f('loops'))}, through shuffled loops ${pct(f('loops_shuffled'))}; one global dopamine signal ${pct(f('global'))}; backpropagation told the answer ${pct(f('backprop'))}.</li>`;
    h += `<li><b>Not forgetting</b> (digits two classes at a time): ideal dopamine routing forgets ${pct(D.by_rule.continual.per_cell.measured.forgetting)} of the first task, backpropagation on the same synapses ${pct(D.by_rule.continual.backprop.measured.forgetting)}, a standard dense network ${pct(D.standard_network.continual.forgetting)}.</li>`;
  }
  $('summary').innerHTML = h + '</ul>';
}

let TC = null;
async function trained() {
  try { const r = await fetch('/api/compass-trained'); if (!r.ok) throw 0; TC = await r.json(); } catch (e) { $('trained').innerHTML = '<p class="hint">No results yet: python3 compass_trained.py</p>'; return; }
  const t = TC, deg = v => v == null ? '–' : v.toFixed(0) + '°', c = t.circuit;
  const label = r => r.wiring === 'dense' ? 'Standard recurrent network, nothing fixed' : `${t.wirings[r.wiring]}, ${t.sharing[r.sharing].toLowerCase()}`;
  const get = (w, s) => t.rows.find(r => r.wiring === w && r.sharing === s);
  const m = get('measured', 'types'), sh = get('shuffled', 'types'), rn = get('random', 'types'), ms = get('measured', 'synapses'), ss = get('shuffled', 'synapses'), dn = get('dense', 'everything');
  $('trained').innerHTML = `<p class="note">The network: ${c.cells} cells of ${c.types.length} types (${Object.entries(c.cells_per_type).map(([k, v]) => `${v} ${esc(k)}`).join(', ')}) and their ${c.connections.toLocaleString()} measured connections, each cell's sign from its transmitter. A heading cue is shown for ${t.task.cue_s} s; then the network runs in the dark while the fly turns. The heading is read from the ${c.heading_cells_read} EPG cells, each cell's preferred direction fixed by its bridge glomerulus. Error is the angle between the network's heading and the true one (guessing: 90°). ${t.seeds.length} starting points each; the median is shown.</p>
    <div class="tw"><table class="bench"><tr><th style="text-align:left">Network</th><th>Numbers learned</th><th>Before training</th><th>After 4 s in the dark<br><small>(as trained)</small></th><th>After 20 s<br><small>(5× longer)</small></th><th>Standing still 10 s</th><th>Follows a turn by</th><th>Works<br><small>(under 30° at 20 s)</small></th></tr>
    ${t.rows.map(r => `<tr><td style="text-align:left">${esc(label(r))}</td><td>${r.numbers_learned.toLocaleString()}</td><td>${deg(r.error_deg_untrained)}</td><td><b>${deg(r.error_deg_4s)}</b></td><td>${deg(r.error_deg_20s)}</td><td>${deg(r.drift_deg_10s_still)}</td><td>${r.turn_gain == null ? '–' : Math.round(r.turn_gain * 100) + '%'}</td><td>${esc(r.works)}</td></tr>`).join('')}</table></div>`;
  const rows = [m, sh, rn, ms, ss, dn].filter(Boolean);
  const steps = m.curve.map(p => p[0]);
  lines('tcurve', steps.map((s, i) => i % 5 === 0 || i === steps.length - 1 ? s : ''), rows.map(r => ({ name: label(r), values: r.curve.map(p => Math.min(1, p[1])) })), 'training steps (curve: the training loss, 0 = perfect, 100% = guessing)', 'tlegend');
  const loc = t.location, at = k => loc && loc[k] ? pct(loc[k].three_modules[3]) : '–';
  const p = (t.pathways || []).slice(0, 8);
  $('trained2').innerHTML = `<p class="note"><b>With its measured wiring, ${m.numbers_learned} learned numbers are enough.</b> The compass goes from guessing (${deg(m.error_deg_untrained)}) to ${deg(m.error_deg_4s)} after 4 s in the dark and ${deg(m.error_deg_20s)} after 20 s, following ${Math.round(m.turn_gain * 100)}% of each turn, in ${esc(m.works)} starting points. <b>With the same cells and the same synapses sent by every cell, but the receiving partners swapped within each cell type, the same training fails</b>: ${deg(sh.error_deg_4s)} (${esc(sh.works)}); random partners ${deg(rn.error_deg_4s)}. What was lost is only who connects to whom inside a type: the ring.</p>
    <p class="note"><b>Can learning every synapse rescue the shuffled wiring?</b> With ${ss ? ss.numbers_learned.toLocaleString() : '?'} free strengths, shuffled reaches ${ss ? deg(ss.error_deg_4s) : '–'} at 4 s and ${ss ? deg(ss.error_deg_20s) : '–'} at 20 s (${ss ? esc(ss.works) : ''}); measured with free synapses ${ms ? deg(ms.error_deg_4s) : '–'} and ${ms ? deg(ms.error_deg_20s) : '–'} (${ms ? esc(ms.works) : ''}). A standard recurrent network of the same size with ${dn ? dn.numbers_learned.toLocaleString() : '?'} free numbers: ${dn ? deg(dn.error_deg_4s) : '–'} and ${dn ? deg(dn.error_deg_20s) : '–'} (${dn ? esc(dn.works) : ''}).</p>
    ${m.bump_profile ? `<p class="note"><b>Inside the trained network.</b> Activity in the heading cells forms one bump that moves with the turn and stays when the fly stands still (it drifts ${deg(m.drift_deg_10s_still)} in 10 s). Nothing in the training asked for a bump; it is how this wiring solves the task. The bump is broad, about ${m.bump_width_deg.toFixed(0)}° at half height. Published imaging of these cells also shows a single bump that tracks heading in the dark, but a narrower one, so the trained model is not yet a match for the real cells, and this page does not compare cell by cell with recordings.</p>` : ''}
    ${p.length ? `<h3 style="font-size:13px;margin:12px 0 6px">The strongest pathways after training (measured wiring)</h3><div class="tw"><table class="bench" style="width:auto"><tr><th style="text-align:left">From</th><th style="text-align:left">To</th><th>Sign</th><th>Learned multiplier</th><th>Input per receiving cell</th></tr>${p.map(x => `<tr><td style="text-align:left">${esc(x.from)}</td><td style="text-align:left">${esc(x.to)}</td><td>${x.sign > 0 ? 'excites' : 'inhibits'}</td><td>×${x.multiplier.toFixed(2)}</td><td>${x.input_per_cell.toFixed(2)}</td></tr>`).join('')}</table></div>` : ''}
    ${loc ? `<h3 style="font-size:13px;margin:14px 0 6px">The trained compass as the location signal</h3><p class="note">On the Scaffold page, objects are recognised by where their features are, and a hand-set compass gain had to be right within about 5%. Here the location comes from the trained circuit instead, with no gain set by hand. Recognition with three modules at four glimpses: true location ${at('true_location')}; published ring ${at('compass_idealised')}; <b>trained, measured wiring ${at('compass_trained_measured')}</b> (location error ${loc.compass_trained_measured.location_error_spacings.toFixed(2)} feature spacings); trained, shuffled wiring ${at('compass_trained_shuffled')}; no locations ${at('bag_of_features')}.</p>` : ''}
    <p class="hint">Measured: cells, connections, synapse counts, transmitters, bridge glomeruli. Assumed: the unit, that turning drives the rotation cells oppositely on the two sides, the left bridge numbered as the mirror of the right, the task, and backpropagation through time as the training method (it stands in for evolution and development; the fly does not use it). The shuffle keeps every cell's outgoing synapses and each type pair's total; only which cell of a type receives them changes.</p>`;
}
load(); trained();
addEventListener('resize', () => { clearTimeout(window._rs); window._rs = setTimeout(() => { if (D) dopamine(); if (TC) trained(); }, 200); });
