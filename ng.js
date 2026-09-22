/* Neuroglancer links and precomputed-mesh decoding. Pure; used by the lab pages and by node tests. */
(function (root) {
  const VIEWER = 'https://neuroglancer-demo.appspot.com/';
  const B = 'precomputed://gs://flyem-male-cns/';
  const DEFAULT_POSITION = [48686.5, 27515.5, 24720.5];   // centre of the MaleCNS volume, 8 nm voxels

  function shell(name, url) {
    return { type: 'segmentation', source: { url: B + url, subsources: { default: true, properties: true, mesh: true }, enableDefaultSubsources: false },
             pick: false, tab: 'rendering', selectedAlpha: 0, saturation: 0, meshSilhouetteRendering: 7, segments: ['1'], segmentDefaultColor: '#ffffff', name };
  }
  // State for the official viewer: selected neurons, optional regions, brain and VNC outlines.
  function state({ cells = [], regions = [], regionIds = {}, position = null, title = 'MaleCNS lab' } = {}) {
    const layers = [
      { type: 'image', source: { url: B + 'em/em-clahe-jpeg', subsources: { default: true }, enableDefaultSubsources: false }, tab: 'rendering', name: 'em-clahe', visible: false },
      { type: 'segmentation', name: 'cns-seg', tab: 'segments', segments: cells.map(String),
        source: [{ url: B + 'v1.0/segmentation', subsources: { default: true, mesh: true }, enableDefaultSubsources: false },
                 { url: B + 'v1.0/segmentation/type_property', subsources: { default: true }, enableDefaultSubsources: false },
                 { url: B + 'v1.0/segmentation/instance_property', enableDefaultSubsources: false },
                 { url: B + 'v1.0/segmentation/meshes-malecns/single-res-meshes', enableDefaultSubsources: false }] },
      shell('brain-neuropil-shell', 'rois/fullbrain-major-shells'),
      shell('vnc-neuropil-shell', 'rois/vnc-neuropil-shell-v2'),
    ];
    if (regions.length) {
      const known = regions.filter(r => regionIds[r] !== undefined);
      layers.push({ type: 'segmentation', name: 'regions', tab: 'segments', selectedAlpha: 0, objectAlpha: 0.35,
        source: { url: B + 'rois/fullbrain-roi-v5', subsources: { default: true, properties: true, mesh: true }, enableDefaultSubsources: false },
        segments: known.map(r => String(regionIds[r])),
        // Without ids (probe not run yet) the segment list is filtered by name, so one click selects it.
        ...(known.length ? {} : { segmentQuery: regions.join(' ') }) });
    }
    return { title, dimensions: { x: [8e-9, 'm'], y: [8e-9, 'm'], z: [8e-9, 'm'] }, position: position || DEFAULT_POSITION,
             projectionScale: position ? 60000 : 134522, layers, showSlices: false, layout: '3d', selectedLayer: { visible: true, layer: 'cns-seg' } };
  }
  function url(opts) { return VIEWER + '#!' + encodeURIComponent(JSON.stringify(state(opts))); }

  // Neuroglancer legacy precomputed mesh fragment: uint32 n, float32[3n] vertices, uint32[] triangle indices.
  function decodeLegacy(buffer) {
    const dv = new DataView(buffer);
    if (buffer.byteLength < 4) throw new Error('fragment too short');
    const n = dv.getUint32(0, true), vEnd = 4 + 12 * n;
    if (vEnd > buffer.byteLength || (buffer.byteLength - vEnd) % 12 !== 0) throw new Error('not a legacy mesh fragment');
    const vertices = new Float32Array(buffer.slice(4, vEnd)), indices = new Uint32Array(buffer.slice(vEnd));
    for (let i = 0; i < indices.length; i++) if (indices[i] >= n) throw new Error('index out of range');
    return { vertices, indices };
  }
  // Vertex units differ between layers (nm or 8 nm voxels); convert to micrometres by magnitude.
  function toMicrometres(vertices) {
    let max = 0; for (let i = 0; i < vertices.length; i++) { const a = Math.abs(vertices[i]); if (a > max) max = a; }
    const f = max > 2e5 ? 1e-3 : 0.008, out = new Float32Array(vertices.length);
    for (let i = 0; i < vertices.length; i++) out[i] = vertices[i] * f;
    return { vertices: out, unit: f === 1e-3 ? 'nm' : 'voxel' };
  }
  // Segment properties (inline format) -> {label: id}.
  function labelsToIds(props) {
    const inl = props && props.inline; if (!inl) return {};
    const p = (inl.properties || []).find(x => x.type === 'label') || (inl.properties || [])[0];
    const out = {}; if (!p) return out;
    inl.ids.forEach((id, i) => { out[p.values[i]] = id; });
    return out;
  }
  const api = { state, url, decodeLegacy, toMicrometres, labelsToIds, VIEWER };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.NG = api;
})(typeof window !== 'undefined' ? window : globalThis);
