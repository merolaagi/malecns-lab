'use strict';
const $ = id => document.getElementById(id);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = v => v == null ? '–' : Math.round(v * 100) + '%';
const C = ['#2f5fd0', '#d0524a', '#4aa7ee', '#e9a23b', '#3f9a70', '#8150e6', '#2aa0a0', '#8a8a8a'];
const DASH = [[], [7, 4], [2, 4], [9, 4, 2, 4], [4, 3], [1, 3], [6, 2], [3, 6]];

function ctx(id) { const cv = $(id), w = cv.clientWidth, h = cv.clientHeight, d = devicePixelRatio || 1; cv.width = w * d; cv.height = h * d; const g = cv.getContext('2d'); g.scale(d, d); g.font = '12px Inter, sans-serif'; return [g, w, h]; }
function lines(id, xs, series, xlabel, legendId) {
  const [g, w, h] = ctx(id); const L = 46, R = 14, T = 12, B = 34;
  const X = i => L + i * (w - L - R) / Math.max(1, xs.length - 1), Y = v => T + (1 - v) * (h - T - B);
  g.strokeStyle = '#e6e6e3'; g.strokeRect(L, T, w - L - R, h - T - B);
  g.fillStyle = '#6e6e6e';
  for (const v of [0, 0.25, 0.5, 0.75, 1]) { g.strokeStyle = '#efefed'; g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(w - R, Y(v)); g.stroke(); g.fillText(Math.round(v * 100) + '%', 8, Y(v) + 4); }
  xs.forEach((x, i) => g.fillText(String(x), X(i) - 4, h - 16));
  g.fillText(xlabel, w / 2 - 40, h - 2);
  series.forEach((s, k) => {
    g.strokeStyle = C[k % C.length]; g.setLineDash(DASH[k % DASH.length]); g.lineWidth = k === 0 ? 2.6 : 1.6; g.beginPath();
    s.values.forEach((v, i) => i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))); g.stroke(); g.setLineDash([]);
    s.values.forEach((v, i) => { g.beginPath(); g.arc(X(i), Y(v), k === 0 ? 3.5 : 3, 0, 7); if (k === 0) { g.fillStyle = C[0]; g.fill(); g.strokeStyle = '#fff'; g.lineWidth = 1.2; g.stroke(); } else { g.fillStyle = '#fff'; g.fill(); g.strokeStyle = C[k % C.length]; g.lineWidth = 1.4; g.stroke(); } });
  });
  $(legendId).innerHTML = series.map((s, k) => `<span><i style="background:${C[k % C.length]}"></i>${esc(s.name)}</span>`).join('');
}

async function go() {
  let b;
  try { const r = await fetch('/api/scaffold-benchmark'); if (!r.ok) throw new Error(); b = await r.json(); }
  catch (e) { $('summary').innerHTML = '<p class="hint">No saved benchmark yet: python3 scaffold.py</p>'; return; }
  const f = b.few_shot, last = f[f.length - 1], first = f[0];
  const o = b.objects, a = b.arithmetic;
  $('summary').innerHTML = `<h2>What the scaffold does, and doesn't, buy</h2><table class="bench"><tr><th>Test</th><th>Scaffold</th><th>Best standard learner</th><th>Verdict</th></tr>
    <tr><td>Digits, 1 example per class</td><td>${pct(first.fly_measured)}</td><td>${pct(Math.max(first.pixels_nearest, first.pixels_softmax, first.mlp_backprop))}</td><td>no advantage</td></tr>
    <tr><td>Digits, 50 per class</td><td>${pct(last.fly_measured)}</td><td>${pct(Math.max(last.pixels_nearest, last.pixels_softmax, last.mlp_backprop))}</td><td>behind</td></tr>
    <tr><td>Objects by feature location, one exploration, 4 glimpses</td><td>${pct(o.voting_3_modules[3])} (3 modules voting)</td><td>${pct(o.bag_of_features[3])} (bag of features)</td><td>large advantage</td></tr>
    <tr><td>Addition on unseen pairs, after 19 successor pairs</td><td>${pct(a.addition)}</td><td>${pct(Math.max(...a.mlp_addition_curve.map(c => c.held_out_accuracy)))} (backprop, up to ${a.mlp_addition_curve[a.mlp_addition_curve.length - 1].train_examples} sums)</td><td>large advantage, with the frame given</td></tr>
    <tr><td>Long-context sequence, 1 exposure</td><td>${pct(b.sequences['fly_measured|1.0'][0])}</td><td>${pct(b.sequences.markov_order_1)} (first-order), ${pct(b.sequences.markov_order_5)} (fifth-order)</td><td>matches a model told the order</td></tr></table>
    <p class="hint">Measured, shuffled and random wiring perform alike everywhere (digits: ${pct(last.fly_measured)}, ${pct(last.fly_shuffled)}, ${pct(last.random_sparse)} at 50 per class): the gains come from the design, not the exact wiring.</p>`;
  lines('fewshot', f.map(r => r.shots), [
    { name: 'Scaffold, measured wiring', values: f.map(r => r.fly_measured) },
    { name: 'Scaffold, random wiring', values: f.map(r => r.random_sparse) },
    { name: 'Nearest neighbour on pixels', values: f.map(r => r.pixels_nearest) },
    { name: 'Softmax regression', values: f.map(r => r.pixels_softmax) },
    { name: 'Backprop MLP', values: f.map(r => r.mlp_backprop) },
    { name: 'Scaffold, 3 glimpses, prediction-assisted', values: f.map(r => r.glimpses_assisted) },
    { name: 'Scaffold, 3 glimpses, plain', values: f.map(r => r.glimpses_plain) },
    { name: 'Scaffold on shifted digits', values: f.map(r => r.fly_measured_shifted) }], 'examples per class', 'fewlegend');
  $('fewnote').textContent = `On plain digits the scaffold is no more data-efficient than nearest neighbour, softmax regression or a backprop MLP, and it falls behind as examples grow. Shifting test digits by one pixel drops every learner (scaffold ${pct(last.fly_measured_shifted)}, MLP ${pct(last.mlp_backprop_shifted)} at 50 per class). Prediction-assisted pooling helps when evidence is summed over glimpses (${pct(last.glimpses_assisted)} against ${pct(last.glimpses_plain)}), but the extra glimpses are shifted views, so both stay below a single centred look.`;
  lines('objects', o.glimpses, [
    { name: 'Reference frame, 3 modules voting', values: o.voting_3_modules },
    { name: 'Reference frame, 1 module', values: o.reference_frame },
    { name: 'Bag of features (no locations)', values: o.bag_of_features }], 'glimpses', 'objlegend');
  $('objnote').textContent = `${o.objects} objects built from only ${o.features} features, so they differ by where features sit. Learned from one exploration each; observations noisy. Chance ${pct(o.chance)}.`;
  const curve = a.mlp_addition_curve.map(c => `<tr><td>Backprop MLP, ${c.train_examples} addition examples</td><td></td><td>${pct(c.held_out_accuracy)}</td></tr>`).join('');
  $('arith').innerHTML = `<p class="note">Twenty symbols (random sparse patterns) are learned only from the 19 pairs "a, then a + 1", each seen once, in random order. Each symbol is placed by adding up steps in a one-dimensional frame. Then the model adds, subtracts and compares by moving.</p>
    <table class="bench"><tr><th>Learner</th><th></th><th>Accuracy on unseen problems</th></tr>
    <tr><td>Reference frame: addition</td><td></td><td>${pct(a.addition)}</td></tr><tr><td>Reference frame: subtraction</td><td></td><td>${pct(a.subtraction)}</td></tr>
    <tr><td>Reference frame: comparison</td><td></td><td>${pct(a.comparison)}</td></tr>${curve}</table>
    <p class="hint">The frame's line geometry is given; the network is not given it. That difference is the whole result: with the right reference frame, 19 relations are enough, and without it, 150 examples are not.</p>`;
  const s = b.sequences;
  $('seq').innerHTML = `<p class="note">Five sequences share their middle (four symbols); the last symbol depends on the first. Each code is biased by a fading trace of the previous codes, so the same symbol gets different codes in different contexts.</p>
    <div class="tw"><table class="bench"><tr><th>Model</th>${s.exposures.map(e => `<th>${e}×</th>`).join('')}</tr>
    ${s.betas.map(beta => `<tr><td>Head start ${beta}${beta === 0 ? ' (no context)' : ''}</td>${s[`fly_measured|${beta.toFixed(1)}`].map(v => `<td>${pct(v)}</td>`).join('')}</tr>`).join('')}
    <tr><td>Markov, first order</td><td colspan="${s.exposures.length}">${pct(s.markov_order_1)}</td></tr>
    <tr><td>Markov, fifth order (told how far back to look)</td><td colspan="${s.exposures.length}">${pct(s.markov_order_5)}</td></tr></table></div>
    <p class="hint">Columns: number of exposures to the sequences. Chance ${pct(s.chance)}. Any head start above zero solves it from one exposure, with every wiring. It is a small test; a counting model solves it too if told the right order.</p>`;
  const c = b.continual;
  $('cont').innerHTML = c ? `<p class="note">Digits arrive as five tasks of two classes; tested on all classes seen so far (from the sparse-expansion benchmark).</p>
    <table class="bench"><tr><th>Learner</th><th></th><th>Final accuracy</th><th>Forgetting</th></tr>${Object.entries(c).map(([k, v]) => `<tr><td>${esc({ fly_measured: 'Scaffold, measured wiring', fly_shuffled: 'Scaffold, shuffled', random_sparse: 'Scaffold, random sparse', mlp_backprop: 'Backprop MLP' }[k] || k)}</td><td></td><td>${pct(v.final_accuracy.mean)}</td><td>${pct(v.forgetting.mean)}</td></tr>`).join('')}</table>`
    : '<p class="hint">Run expansion.py --benchmark for this panel.</p>';
}

const FRAME_NAME = { frame: 'Discovered frame', chain_lookup: 'Lookup table, chained', backprop_net: 'Backprop network' };
async function frames() {
  let f; try { const r = await fetch('/api/frames-benchmark'); if (!r.ok) throw 0; f = await r.json(); } catch (e) { return; }
  const cell = (w, k, m) => w[k] ? pct(w[k][m]) : '–';
  $('frames').innerHTML = `<p class="note">The arithmetic above was handed a number line. Here the frame is worked out from relations alone. Each relation type is treated as one kind of movement; walking the relations gives every symbol a position; every relation that closes a loop says which combinations of movements bring you back. Diagonalising those constraints gives the frame: free directions are axes, directions that repeat are loops, and directions that are combinations of others vanish. Then held-out relations and multi-step questions are answered by moving.</p>
    <div class="tw"><table class="bench"><tr><th>World</th><th>Relations given</th><th>Frame found</th><th>Held-out relations<br><small>frame / lookup / network</small></th><th>Multi-step questions<br><small>frame / lookup / network</small></th></tr>
    ${f.worlds.map(w => `<tr><td>${esc(w.label)}</td><td>${w.relations_given} of ${w.relations_total}</td><td>${esc(w.shapes_found.join(', '))}${w.correct_shape === 1 ? '' : ` <small>(${pct(w.correct_shape)} of seeds as expected: ${esc(w.expected)})</small>`}</td>
      <td>${cell(w, 'frame', 'held_out_relations')} / ${cell(w, 'chain_lookup', 'held_out_relations')} / ${cell(w, 'backprop_net', 'held_out_relations')}</td>
      <td>${cell(w, 'frame', 'multi_step')} / ${cell(w, 'chain_lookup', 'multi_step')} / ${cell(w, 'backprop_net', 'multi_step')}</td></tr>`).join('')}</table></div>
    <p class="hint">Where every relation is given (numbers, clock), a lookup table that chains known relations does as well: the frame adds nothing there except knowing the shape. The frame earns its keep when relations are missing: it fills them in from geometry (a lookup table cannot, by construction) and answers multi-step questions that cross the gaps. The family tree has no frame of this kind, and the method says so.</p>`;
  const room = f.sweep.room, xs = room.map(r => r.relations_given + '/' + r.relations_total);
  lines('framesweep', xs, [
    { name: 'Room: frame, multi-step', values: room.map(r => r.frame.multi_step) },
    { name: 'Room: lookup table, multi-step', values: room.map(r => r.chain_lookup.multi_step) },
    { name: 'Room: right frame found', values: room.map(r => r.correct_shape) },
    { name: 'Rota: frame, multi-step', values: f.sweep.week_by_shift.map(r => r.frame.multi_step) },
    { name: 'Rota: lookup table, multi-step', values: f.sweep.week_by_shift.map(r => r.chain_lookup.multi_step) }], 'relations given (room; the rota has the same fractions)', 'framelegend');
  const n = f.one_wrong_relation;
  $('framenote').textContent = `How few relations are enough: below about three quarters of them, the relations no longer connect every symbol and the frame comes out in pieces. Limits: movements must commute, and every relation is trusted. With one wrong relation the method reports ${n.map(w => `"${w.shapes_found.join('" or "')}" for ${w.world}`).join(' and ')} instead of the right frame (multi-step accuracy ${n.map(w => pct(w.frame.multi_step)).join(' and ')}): it detects the damage but does not repair it.`;
}
async function compass() {
  let c; try { const r = await fetch('/api/compass-benchmark'); if (!r.ok) throw 0; c = await r.json(); } catch (e) { $('compassintro').innerHTML = '<p class="hint">Run compass.py for this panel.</p>'; return; }
  const m = c.main, order = Object.keys(c.labels).filter(k => m[k]);
  $('compassintro').innerHTML = `<p class="note">In the object test above, the location of each glimpse was given. Here a walker moves over the object (turn, then a ${c.setup.step_mm} mm step) and its location is whatever the central-complex model integrates from its own turning and speed: a heading bump on a ring, moved by rotation cells, with vector cells adding up the path. Features are stored where the compass says they are, in one walk, and recognised along a different walk. Three modules voting; ${c.seeds.length} seeds.</p>`;
  lines('compass', m[order[0]].glimpses, order.map(k => ({ name: c.labels[k], values: m[k].three_modules })), 'glimpses', 'compasslegend');
  const ms = c.measured;
  $('compasstable').innerHTML = `<div class="tw"><table class="bench"><tr><th>Location signal</th><th>1 module, 4 glimpses</th><th>3 modules, 4 glimpses</th><th>Location error<br><small>feature spacings</small></th></tr>
    ${order.map(k => `<tr><td>${esc(c.labels[k])}</td><td>${pct(m[k].one_module[3])}</td><td>${pct(m[k].three_modules[3])}</td><td>${k === 'bag_of_features' ? '–' : m[k].location_error_spacings.toFixed(2)}</td></tr>`).join('')}</table></div>
    ${explain(c)}`;
  if (c.gain_sweep) {
    $('compasstable').insertAdjacentHTML('beforeend', `<h3 style="font-size:13px;margin:14px 0 6px">How exactly must the compass follow a turn?</h3><div class="tw"><table class="bench"><tr><th>Bump moves this many turns per turn</th>${c.gain_sweep.map(g => `<th>${g.gain}</th>`).join('')}</tr>
      <tr><td>Recognition, 3 modules, 4 glimpses</td>${c.gain_sweep.map(g => `<td>${pct(g.three_modules[3])}</td>`).join('')}</tr>
      <tr><td>Location error (feature spacings)</td>${c.gain_sweep.map(g => `<td>${g.location_error_spacings.toFixed(2)}</td>`).join('')}</tr></table></div>`);
  }
}
function explain(c) {
  const ms = c.measured, m = c.main; if (!ms) return '<p class="hint">With the published ring offsets the fly\'s compass is as good a location signal as being told the location. Build data/central-complex.json to test the measured offsets.</p>';
  const b = ms.bridge, at4 = k => m[k] ? pct(m[k].three_modules[3]) : '–';
  const ok = c.noise_sweep.compass_idealised.filter(r => r.three_modules[3] > 0.9).map(r => r.compass_noise).pop();
  return `<p class="hint"><b>Published ring:</b> the compass is as good a location signal as being told the location (${at4('compass_idealised')} against ${at4('true_location')}); noise in the turn signal up to ${ok} rad/s still leaves recognition above 90%.</p>
    <p class="hint"><b>Measured wiring, read correctly:</b> ${b ? `within each side of the bridge, rotation cells reach heading cells ${Object.values(b.step_in_glomeruli_by_type).join(', ')} glomeruli away, the same step on both sides. Because the two sides are numbered as mirror images, that is a shift in opposite directions around the ellipsoid body: the mechanism the published account describes, found in this fly's wiring (opposite by side: ${b.opposite_by_side ? 'yes' : 'no'}). ` : ''}The wiring fixes which way each side pushes the bump. How far it moves per unit of turning is not anatomy; flies calibrate it against what they see. With the measured direction and a calibrated gain: ${at4('compass_measured_calibrated')}. Taking the measured step literally as the gain (${b ? b.rotation_gain.toFixed(1) : '?'} turns per turn) over-rotates and fails: ${at4('compass_measured_bridge')}. The table below shows how tight that calibration has to be.</p>
    <p class="hint"><b>The lab's earlier reading</b> treated the 16 bridge glomeruli as one ring; read that way the two sides seem to shift the same way and the compass follows ${Math.round(ms.rotation_gain_columns * 100)}% of a turn (${at4('compass_measured')}). That was the column assignment, not the fly; the Navigation page now offers both readings.</p>`;
}
async function all() { await go(); frames(); compass(); }
all();
addEventListener('resize', () => { clearTimeout(window._rs); window._rs = setTimeout(all, 200); });
