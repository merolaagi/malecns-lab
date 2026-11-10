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
go();
addEventListener('resize', () => { clearTimeout(window._rs); window._rs = setTimeout(go, 200); });
