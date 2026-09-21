// Run: node --test neuron-core.test.js
const test = require('node:test');
const assert = require('node:assert');
const N = require('./neuron-core.js');

// Kenyon cell 57729's measured PN synapse counts (MaleCNS v1.0).
const KC = [38, 25, 22, 20, 17, 13, 1];
const chans = (active, sign = 1) => N.weights(KC).map((w, i) => ({ weight: w, sign, active: active.includes(i) }));

test('weights follow log(1 + synapses), strongest = 1', () => {
  const w = N.weights(KC);
  assert.strictEqual(w[0], 1);
  assert.ok(w.every((x, i) => i === 0 || x < w[i - 1] + 1e-12));
});

test('no input: rests near -65 mV and never spikes', () => {
  const r = N.simulate({ channels: chans([]) });
  assert.strictEqual(r.spikes.length, 0);
  assert.ok(Math.abs(r.trace.v.at(-1) + 65) < 1.5);
  assert.strictEqual(r.events.length, 0);
});

test('deterministic for a fixed seed', () => {
  const a = N.simulate({ channels: chans([0, 1, 2]), seed: 3 }), b = N.simulate({ channels: chans([0, 1, 2]), seed: 3 });
  assert.deepStrictEqual(a.spikes, b.spikes);
});

test('coincidence: single claws rarely fire, all seven usually do (default strength)', () => {
  const rate = active => { let c = 0; for (let s = 1; s <= 20; s++) c += N.simulate({ channels: chans(active), seed: s }).spikes.length > 0; return c / 20; };
  for (let i = 0; i < 7; i++) assert.ok(rate([i]) <= 0.1, 'claw ' + i);
  assert.ok(rate([0, 1, 2, 3, 4, 5, 6]) >= 0.75);
  assert.ok(rate([0, 1, 2]) > rate([0]));
});

test('inhibition alone hyperpolarizes and cannot spike', () => {
  const r = N.simulate({ channels: chans([0, 1, 2, 3, 4, 5, 6], -1) });
  assert.strictEqual(r.spikes.length, 0);
  assert.ok(Math.min(...r.trace.vd) < -66);
});

test('zero-sign inputs have no effect', () => {
  const r = N.simulate({ channels: chans([0, 1, 2, 3, 4, 5, 6], 0) });
  assert.strictEqual(r.spikes.length, 0);
  assert.strictEqual(r.events.length, 0);
});

test('fire probability is 0 without inputs and within [0,1]', () => {
  assert.strictEqual(N.fireProbability([]), 0);
  const p = N.fireProbability(chans([]));
  assert.ok(p >= 0 && p <= 1);
});
