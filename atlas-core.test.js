// Run: node --test atlas-core.test.js
const test = require('node:test');
const assert = require('node:assert');
const fs = require('fs');
const path = require('path');
const core = require('./atlas-core.js');

const circuit = JSON.parse(fs.readFileSync(path.join(__dirname, 'data/circuit.json'), 'utf8'));
const A = core.build(circuit);

test('every cell and edge is represented once', () => {
  assert.strictEqual(A.nodes.length, circuit.nodes.length);
  assert.strictEqual(A.E, circuit.edges.length);
  const total = Object.values(A.layerEdges).reduce((a, b) => a + b.edges, 0);
  assert.strictEqual(total, circuit.edges.length);
  const syn = circuit.edges.reduce((a, e) => a + e[2], 0);
  assert.strictEqual(Object.values(A.layerEdges).reduce((a, b) => a + b.synapses, 0), syn);
});

test('layer and group partitions cover every cell exactly once', () => {
  const seen = new Set();
  for (const L of core.LAYERS) for (const g of A.groups[L.key]) for (const i of g.members) {
    assert.ok(!seen.has(i)); seen.add(i);
    assert.strictEqual(A.nodes[i].layer, L.key);
  }
  assert.strictEqual(seen.size, A.nodes.length);
  const size = k => A.groups[k].reduce((a, g) => a + g.members.length, 0);
  assert.deepStrictEqual(['dn', 'in', 'mn', 'sn'].map(size), [6, 398, 312, 91]);
});

test('leg assignment matches the simulator rule', () => {
  const mn = A.nodes.filter(n => n.layer === 'mn');
  assert.ok(mn.every(n => n.leg !== null));
  const sn = A.nodes.filter(n => n.layer === 'sn');
  assert.ok(sn.every(n => n.leg && n.leg[1] === { ProLN: 'F', MesoLN: 'M', MetaLN: 'H' }[n.raw.entryNerve]));
});

test('cells without soma are placed and flagged, real somata are untouched', () => {
  for (const n of A.nodes) {
    const s = n.raw.somaLocation;
    if (s) { assert.strictEqual(n.x, s[0]); assert.strictEqual(n.y, s[2]); assert.strictEqual(n.placed, false); }
    else if (n.x !== null) {
      assert.strictEqual(n.placed, true);
      assert.ok(core.treatment(A, n.i).flags.some(f => f.includes('centroid')));
    }
  }
  assert.ok(A.nodes.filter(n => n.layer === 'sn').every(n => n.placed));
});

test('whole-circuit summary has no external edges', () => {
  const s = core.summarize(A, A.nodes.map(n => n.i));
  assert.strictEqual(s.internal.edges, A.E);
  assert.strictEqual(s.input.edges, 0);
  assert.strictEqual(s.output.edges, 0);
});

test('region summary conserves edges', () => {
  const idx = A.groups.in[0].members;
  const s = core.summarize(A, idx);
  const outSum = idx.reduce((a, i) => a + A.outs[i].length, 0);
  const inSum = idx.reduce((a, i) => a + A.ins[i].length, 0);
  assert.strictEqual(s.internal.edges + s.output.edges, outSum);
  assert.strictEqual(s.internal.edges + s.input.edges, inSum);
});

test('partners are sorted and complete', () => {
  const dn = A.nodes.find(n => n.layer === 'dn').i;
  const p = core.partners(A, dn, 'out', 5);
  assert.strictEqual(p.total, A.outs[dn].length);
  for (let k = 1; k < p.top.length; k++) assert.ok(p.top[k - 1].synapses >= p.top[k].synapses);
});

test('known modeling problems are flagged', () => {
  const nonLeg = A.nodes.filter(n => n.layer === 'mn' && !['ProLN', 'MesoLN', 'MetaLN'].includes(n.raw.exitNerve));
  assert.strictEqual(nonLeg.length, 59);
  assert.ok(nonLeg.every(n => core.treatment(A, n.i).flags.some(f => f.includes('rather than a leg nerve') || f.includes('unnamed'))));
  const named = nonLeg.filter(n => /Tr |Sterno|Sternal|Tergo|Pleural|Fe /.test(n.raw.type || ''));
  assert.ok(named.length > 40);   // most are leg-muscle motor neurons taking an accessory nerve
  assert.ok(named.every(n => core.treatment(A, n.i).flags.some(f => f.includes('is a leg muscle'))));
  const silent = A.nodes.filter(n => n.sign === 0);
  assert.ok(silent.every(n => core.treatment(A, n.i).flags.some(f => f.includes('no effect'))));
});
