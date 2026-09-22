"""Server-side mesh assembly and simplification for the 3D viewer.

Full-resolution MaleCNS neuron meshes can exceed 25 MB (DNa02's is one such file), too large to send to a
browser. For each requested segment this fetches every fragment of the legacy precomputed mesh once,
merges them, simplifies by vertex clustering until the triangle count is under a per-layer target, and
caches only the simplified mesh (git-ignored). Clustering keeps the overall shape and branch layout;
fine surface detail below the cluster size is lost, and very thin branches can thin out further.
"""
import json
import struct
import threading
from pathlib import Path

import numpy as np

import gcs

LAYERS = {  # key -> (mesh directory in the bucket, target triangles)
    'neuron': ('v1.0/segmentation/meshes-malecns/single-res-meshes', 250_000),
    'roi': ('rois/fullbrain-roi-v5/mesh', 60_000),
    'vnc-roi': ('rois/malecns-vnc-neuropil-roi-v0/mesh', 60_000),
    'brain-shell': ('rois/fullbrain-major-shells/mesh', 80_000),
    'vnc-shell': ('rois/vnc-neuropil-shell-v2/mesh', 80_000),
}
RAW_MAX = 400 * 1024 * 1024
_locks, _guard = {}, threading.Lock()


def decode(buf):
    n = struct.unpack_from('<I', buf, 0)[0]; end = 4 + 12 * n
    if end > len(buf) or (len(buf) - end) % 12: raise ValueError('not a legacy mesh fragment')
    v = np.frombuffer(buf, '<f4', 3 * n, 4).reshape(-1, 3)
    t = np.frombuffer(buf, '<u4', offset=end).reshape(-1, 3)
    if t.size and t.max() >= n: raise ValueError('triangle index out of range')
    return v, t


def encode(v, t):
    v = np.ascontiguousarray(v, '<f4'); t = np.ascontiguousarray(t, '<u4')
    return struct.pack('<I', len(v)) + v.tobytes() + t.tobytes()


def simplify(v, t, target):
    """Vertex clustering: snap vertices to a grid, average each cell, drop collapsed and duplicate triangles.
    The grid doubles until the mesh is under `target` triangles."""
    if len(t) <= target: return v, t
    lo, hi = v.min(0), v.max(0)
    cell = float(np.linalg.norm(hi - lo)) / 2000 or 1.0
    while True:
        key = np.floor((v - lo) / cell).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
        k = inv.max() + 1
        nv = np.zeros((k, 3)); np.add.at(nv, inv, v); nv /= np.bincount(inv, minlength=k)[:, None]
        nt = inv[t]
        nt = nt[(nt[:, 0] != nt[:, 1]) & (nt[:, 1] != nt[:, 2]) & (nt[:, 0] != nt[:, 2])]
        if len(nt): nt = nt[np.unique(np.sort(nt, axis=1), axis=0, return_index=True)[1]]   # drop duplicate triangles
        if len(nt) <= target or cell > np.linalg.norm(hi - lo):
            used = np.unique(nt); remap = np.full(k, -1, np.int64); remap[used] = np.arange(len(used))
            return nv[used].astype('<f4'), remap[nt].astype('<u4')
        cell *= 2


def get(layer, seg_id):
    """Return (legacy mesh bytes, info dict) for a whitelisted layer and integer segment id."""
    if layer not in LAYERS: raise gcs.GCSError('Unknown mesh layer', 400)
    try: seg_id = int(seg_id)
    except (TypeError, ValueError): raise gcs.GCSError('Segment id must be an integer', 400)
    if seg_id < 0: raise gcs.GCSError('Segment id must be an integer', 400)
    directory, target = LAYERS[layer]
    out = gcs.CACHE / 'simplified' / layer / f'{seg_id}.bin'
    meta = out.with_suffix('.json')
    with _guard: lock = _locks.setdefault((layer, seg_id), threading.Lock())
    with lock:                      # one build per mesh even if several requests arrive together
        if out.is_file() and meta.is_file(): return out.read_bytes(), json.loads(meta.read_text())
        manifest = json.loads(gcs.fetch(f'{directory}/{seg_id}:0'))
        frags = manifest.get('fragments', [])
        if not isinstance(frags, list) or not all(isinstance(f, str) for f in frags): raise gcs.GCSError('Malformed mesh manifest', 502)
        # Fragment names are relative to the manifest; a name may itself contain subdirectories.
        parts = [decode(gcs.fetch(f'{directory}/{f}', max_bytes=RAW_MAX, cache=False)) for f in frags]
        if not parts: raise gcs.GCSError('Mesh has no fragments', 404)
        offs = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
        v = np.concatenate([p[0] for p in parts]); t = np.concatenate([p[1] + o for p, o in zip(parts, offs)])
        sv, st = simplify(v, t, target)
        body = encode(sv, st)
        info = {'layer': layer, 'id': seg_id, 'triangles_original': int(len(t)), 'triangles': int(len(st)), 'vertices': int(len(sv))}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(body); meta.write_text(json.dumps(info))
        return body, info
