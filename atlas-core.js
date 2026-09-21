/* Pure atlas logic: no DOM. Loaded by atlas.html and by atlas-core.test.js under node. */
(function (root) {
  const LAYERS = [
    { key: 'dn', name: 'Descending neurons', superclass: 'descending_neuron' },
    { key: 'in', name: 'VNC interneurons', superclass: 'vnc_intrinsic' },
    { key: 'mn', name: 'Motor neurons', superclass: 'vnc_motor' },
    { key: 'sn', name: 'Sensory neurons', superclass: 'vnc_sensory' },
  ];
  const LAYER_BY_SUPER = Object.fromEntries(LAYERS.map(l => [l.superclass, l.key]));
  const LAYER_NAME = Object.fromEntries(LAYERS.map(l => [l.key, l.name]));
  const LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH'];
  const LEG_NAME = { F: 'front', M: 'middle', H: 'hind' };
  const LEG_NERVES = { ProLN: 'F', MesoLN: 'M', MetaLN: 'H' };
  // Same sign rule as model.py: ACh +1, GABA and glutamate -1, anything else 0.
  const SIGN = { acetylcholine: 1, gaba: -1, glutamate: -1 };

  function groupOf(n, layer) {
    if (layer === 'in') {
      const m = /^IN(\d\d[A-Z]|XXX)/.exec(n.type || '');
      if (!m) return 'Untyped';
      return m[1] === 'XXX' ? 'Unassigned hemilineage' : 'Hemilineage ' + m[1];
    }
    return n.type || 'Untyped';
  }

  // Mirrors model.leg_for exactly, so the atlas reports the pool the simulator actually uses.
  function legFor(n) {
    const side = n.rootSide || n.somaSide;
    let part = { fl: 'F', ml: 'M', hl: 'H' }[n.subclass];
    if (n.superclass === 'vnc_sensory') part = LEG_NERVES[n.entryNerve];
    const label = String(side) + String(part);
    return LEGS.includes(label) ? label : null;
  }

  function legText(label) {
    return label ? (label[0] === 'L' ? 'left ' : 'right ') + LEG_NAME[label[1]] + ' leg' : 'no leg';
  }

  function build(circuit) {
    const nodes = circuit.nodes.map((n, i) => {
      const layer = LAYER_BY_SUPER[n.superclass];
      if (!layer) throw new Error('Unknown superclass ' + n.superclass);
      return {
        i, raw: n, layer, group: groupOf(n, layer), leg: legFor(n),
        sign: SIGN[n.nt] ?? 0, x: null, y: null, placed: false,
      };
    });
    const index = new Map(nodes.map(n => [n.raw.bodyId, n.i]));
    const E = circuit.edges.length;
    const pre = new Int32Array(E), post = new Int32Array(E), w = new Float64Array(E);
    const outs = nodes.map(() => []), ins = nodes.map(() => []);
    circuit.edges.forEach(([a, b, c], e) => {
      if (!index.has(a) || !index.has(b)) throw new Error('Edge references unknown bodyId');
      pre[e] = index.get(a); post[e] = index.get(b); w[e] = c;
      outs[pre[e]].push(e); ins[post[e]].push(e);
    });

    // Top (dorsal) view: x is lateral, z is anterior-posterior in MaleCNS voxel space.
    for (const n of nodes) {
      const s = n.raw.somaLocation;
      if (s) { n.x = s[0]; n.y = s[2]; }
    }
    // Cells without a soma in the annotations (leg sensory neurons, a few others) are placed
    // at the synapse-weighted centroid of partners that do have one. Repeat for chains.
    for (let pass = 0; pass < 3; pass++) {
      for (const n of nodes) {
        if (n.x !== null) continue;
        let sx = 0, sy = 0, sw = 0;
        for (const e of outs[n.i]) { const p = nodes[post[e]]; if (p.x !== null) { sx += p.x * w[e]; sy += p.y * w[e]; sw += w[e]; } }
        for (const e of ins[n.i]) { const p = nodes[pre[e]]; if (p.x !== null) { sx += p.x * w[e]; sy += p.y * w[e]; sw += w[e]; } }
        if (sw > 0) { n.x = sx / sw; n.y = sy / sw; n.placed = true; }
      }
    }

    const groups = {};
    for (const L of LAYERS) {
      const m = new Map();
      for (const n of nodes) if (n.layer === L.key) {
        if (!m.has(n.group)) m.set(n.group, []);
        m.get(n.group).push(n.i);
      }
      groups[L.key] = [...m.entries()].map(([key, members]) => ({ key, layer: L.key, members }))
        .sort((a, b) => b.members.length - a.members.length || a.key.localeCompare(b.key));
    }

    const layerEdges = {};
    for (let e = 0; e < E; e++) {
      const k = nodes[pre[e]].layer + '>' + nodes[post[e]].layer;
      const v = layerEdges[k] || (layerEdges[k] = { edges: 0, synapses: 0 });
      v.edges++; v.synapses += w[e];
    }
    return { nodes, pre, post, w, outs, ins, groups, layerEdges, E, source: circuit };
  }

  function count(arr, f) {
    const m = new Map();
    for (const x of arr) { const k = f(x); m.set(k, (m.get(k) || 0) + 1); }
    return [...m.entries()].sort((a, b) => b[1] - a[1]);
  }

  function summarize(A, indices) {
    const set = new Set(indices), cells = indices.map(i => A.nodes[i]);
    const s = {
      cells: cells.length,
      layers: count(cells, n => n.layer),
      groups: count(cells, n => LAYER_NAME[n.layer] + ': ' + n.group),
      nt: count(cells, n => n.raw.nt || 'unknown'),
      sides: count(cells, n => n.raw.rootSide || n.raw.somaSide || 'unknown'),
      internal: { edges: 0, synapses: 0 }, input: { edges: 0, synapses: 0 }, output: { edges: 0, synapses: 0 },
      topInputs: [], topOutputs: [],
    };
    const inG = new Map(), outG = new Map();
    const label = n => LAYER_NAME[n.layer] + ': ' + n.group;
    for (const i of set) {
      for (const e of A.outs[i]) {
        const t = A.post[e];
        if (set.has(t)) { s.internal.edges++; s.internal.synapses += A.w[e]; }
        else { s.output.edges++; s.output.synapses += A.w[e]; const k = label(A.nodes[t]); outG.set(k, (outG.get(k) || 0) + A.w[e]); }
      }
      for (const e of A.ins[i]) {
        const f = A.pre[e];
        if (!set.has(f)) { s.input.edges++; s.input.synapses += A.w[e]; const k = label(A.nodes[f]); inG.set(k, (inG.get(k) || 0) + A.w[e]); }
      }
    }
    s.topInputs = [...inG.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
    s.topOutputs = [...outG.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
    return s;
  }

  function partners(A, i, dir, limit = 10) {
    const list = (dir === 'in' ? A.ins[i] : A.outs[i])
      .map(e => ({ cell: dir === 'in' ? A.pre[e] : A.post[e], synapses: A.w[e] }))
      .sort((a, b) => b.synapses - a.synapses);
    return { total: list.length, synapses: list.reduce((a, b) => a + b.synapses, 0), top: list.slice(0, limit) };
  }

  // Statements about how the simulator treats a cell. Flags mark known modeling problems.
  function treatment(A, i) {
    const n = A.nodes[i], r = n.raw, lines = [], flags = [];
    const signText = n.sign > 0 ? 'excitatory (+1)' : n.sign < 0 ? 'inhibitory (−1)' : 'zero weight';
    lines.push('Outgoing edges are ' + signText + ', scaled by log(1 + synapses) and normalized per target.');
    if (n.layer === 'dn') lines.push(r.seed ? 'Receives the descending stimulus current after 0.5 s.' : 'Not a stimulated seed.');
    if (n.layer === 'in') lines.push('Silenced in the "Silence VNC interneurons" condition.');
    if (n.layer === 'mn') lines.push(n.leg ? 'Firing rate is averaged into the ' + legText(n.leg) + ' pool that sets stride.' : 'Not assigned to any leg pool.');
    if (n.layer === 'sn') lines.push(n.leg ? 'Receives the engineered stance × stride current for the ' + legText(n.leg) + '.' : 'Receives no feedback current.');
    if (n.layer === 'mn' && r.exitNerve && !LEG_NERVES[r.exitNerve] && n.leg)
      flags.push('Leaves through ' + r.exitNerve + ', not a leg nerve, yet the model pools it into ' + legText(n.leg) + ' stride.');
    if (n.sign === 0) flags.push('Transmitter is "' + (r.nt || 'unknown') + '", so this cell has no effect on its targets in the model.');
    if (r.nt === 'glutamate') flags.push('Glutamate is treated as inhibitory on every central target; receptor identity is unknown.');
    if (n.placed) flags.push('No soma position in the annotations. Drawn at the synapse-weighted centroid of its partners.');
    if (n.x === null) flags.push('No position available. Not drawn in the anatomy view.');
    return { lines, flags };
  }

  const api = { LAYERS, LAYER_NAME, LEGS, build, summarize, partners, treatment, groupOf, legFor, legText };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.AtlasCore = api;
})(typeof window !== 'undefined' ? window : globalThis);
