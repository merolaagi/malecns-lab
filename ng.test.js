// Run: node --test ng.test.js
const test = require('node:test');
const assert = require('node:assert');
const NG = require('./ng.js');

function fragment(verts, idx) {
  const b = Buffer.alloc(4 + verts.length * 4 + idx.length * 4);
  b.writeUInt32LE(verts.length / 3, 0);
  verts.forEach((v, i) => b.writeFloatLE(v, 4 + i * 4));
  idx.forEach((v, i) => b.writeUInt32LE(v, 4 + verts.length * 4 + i * 4));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
}

test('decodes a legacy mesh fragment', () => {
  const m = NG.decodeLegacy(fragment([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 2]));
  assert.deepStrictEqual(Array.from(m.vertices), [0, 0, 0, 1, 0, 0, 0, 1, 0]);
  assert.deepStrictEqual(Array.from(m.indices), [0, 1, 2]);
});

test('rejects malformed fragments', () => {
  assert.throws(() => NG.decodeLegacy(new ArrayBuffer(2)));
  assert.throws(() => NG.decodeLegacy(fragment([0, 0, 0], [0, 1, 2])));        // index out of range
  const f = fragment([0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2]);
  assert.throws(() => NG.decodeLegacy(f.slice(0, f.byteLength - 2)));         // truncated
});

test('converts nm and voxel coordinates to micrometres', () => {
  assert.strictEqual(NG.toMicrometres(new Float32Array([400000, 0, 0])).unit, 'nm');
  assert.ok(Math.abs(NG.toMicrometres(new Float32Array([400000, 0, 0])).vertices[0] - 400) < 1e-3);
  const v = NG.toMicrometres(new Float32Array([50000, 0, 0]));
  assert.strictEqual(v.unit, 'voxel'); assert.ok(Math.abs(v.vertices[0] - 400) < 1e-3);
});

test('maps segment property labels to ids', () => {
  const m = NG.labelsToIds({ inline: { ids: ['12', '40'], properties: [{ id: 'label', type: 'label', values: ['LAL(R)', 'VES(R)'] }] } });
  assert.deepStrictEqual(m, { 'LAL(R)': '12', 'VES(R)': '40' });
  assert.deepStrictEqual(NG.labelsToIds(null), {});
});

test('neuroglancer link round-trips and selects the cells', () => {
  const u = NG.url({ cells: [10360, 523769], position: [1, 2, 3] });
  assert.ok(u.startsWith(NG.VIEWER + '#!'));
  const s = JSON.parse(decodeURIComponent(u.split('#!')[1]));
  const seg = s.layers.find(l => l.name === 'cns-seg');
  assert.deepStrictEqual(seg.segments, ['10360', '523769']);
  assert.deepStrictEqual(s.position, [1, 2, 3]);
  assert.ok(s.layers.some(l => l.name === 'vnc-neuropil-shell'));
});

test('regions use ids when known, otherwise a name filter', () => {
  const withIds = NG.state({ regions: ['LAL(R)'], regionIds: { 'LAL(R)': 12 } }).layers.find(l => l.name === 'regions');
  assert.deepStrictEqual(withIds.segments, ['12']); assert.strictEqual(withIds.segmentQuery, undefined);
  const noIds = NG.state({ regions: ['LAL(R)', 'VES(R)'] }).layers.find(l => l.name === 'regions');
  assert.deepStrictEqual(noIds.segments, []); assert.strictEqual(noIds.segmentQuery, 'LAL(R) VES(R)');
});
