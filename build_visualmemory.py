"""The fly's measured route from seeing to remembering: visual inputs onto Kenyon cells.

About 7% of Kenyon cells get little or no input from olfactory projection neurons. In data/mushroom-body.json,
252 of 4,064 get none and 51 more get fewer than ten synapses, almost all of type KCg-d and KCab-p. Those types
take visual input instead, from optic-lobe projection neurons that reach the mushroom body's accessory calyx
(Vogt et al. 2016; Li et al. 2020). This builder extracts those inputs.

Selected, by anatomy only:
  * every Traced Kenyon cell of a visual type (VISUAL_KC_TYPES)
  * every Traced cell outside the mushroom body that makes at least MIN_SYNAPSES synapses onto them in total
  * every edge from those inputs onto any Kenyon cell (visual or not), with its synapse count
  * per input: type, class, superclass, soma side, and assigned optic-lobe hex coordinates when annotated

Run on a machine with the raw tables:
    .venv/bin/python build_visualmemory.py raw-data --probe     # who feeds visual vs olfactory Kenyon cells
    .venv/bin/python build_visualmemory.py raw-data             # writes data/visual-memory.json, then --upstream
    .venv/bin/python build_visualmemory.py raw-data --upstream  # only add upstream measures to an existing file

Upstream measures, per input (one and two hops up):
  visual_share   fraction of its annotated input synapses that come from optic-lobe cells
  channel_mix    which kinds of optic-lobe cell feed it: ON, OFF, colour/luminance, or form
  rf             where in the eye: synapse-weighted position of columnar cells with optic-lobe hex coordinates
"""
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from model import BASE

VISUAL_KC_TYPES = ('KCg-d', 'KCab-p')
MB_PREFIXES = ('KC', 'MBON', 'PAM', 'PPL1', 'PPL2', 'DPM', 'APL')
MIN_SYNAPSES = 5


def annotations(raw):
    return feather.read_table(Path(raw) / 'annotations.feather').to_pylist()


def is_mb(type_name):
    return str(type_name or '').startswith(MB_PREFIXES)


def batches(raw):
    with pa.memory_map(str(Path(raw) / 'weights.feather'), 'r') as handle:
        reader = pa.ipc.open_file(handle)
        for k in range(reader.num_record_batches):
            batch = reader.get_batch(k)
            yield k, batch.column('body_pre').to_numpy(), batch.column('body_post').to_numpy(), batch.column('weight').to_numpy()


def hex_of(r):
    a, b = r.get('assignedOlHex1'), r.get('assignedOlHex2')
    try:
        return [int(a), int(b)] if a is not None and b is not None else None
    except (TypeError, ValueError):
        return None


def kenyon_cells(rows):
    kcs = {int(r['bodyId']): r for r in rows if str(r.get('type') or '').startswith('KC') and r.get('status') == 'Traced'}
    visual = {b for b, r in kcs.items() if str(r.get('type')).startswith(VISUAL_KC_TYPES)}
    return kcs, visual


def inputs_onto(raw, posts):
    """Total synapses from every presynaptic cell onto a set of cells, and the per-edge counts."""
    posts = np.array(sorted(posts))
    total, edges = Counter(), []
    for k, a, b, w in batches(raw):
        mask = np.isin(b, posts) & (w > 0)
        for x, y, z in zip(a[mask], b[mask], w[mask]):
            total[int(x)] += int(z); edges.append((int(x), int(y), int(z)))
        if k % 600 == 0: print('batch', k, 'edges', len(edges), flush=True)
    return total, edges


def probe(raw):
    rows = annotations(raw)
    info = {int(r['bodyId']): r for r in rows}
    kcs, visual = kenyon_cells(rows)
    print('Kenyon cells by type (Traced):', Counter(str(r.get('type')) for r in kcs.values()).most_common(14))
    print('visual Kenyon cells:', len(visual))
    total, edges = inputs_onto(raw, set(kcs))
    onto_visual, onto_olf = Counter(), Counter()
    for x, y, z in edges:
        if is_mb(info.get(x, {}).get('type')): continue
        (onto_visual if y in visual else onto_olf)[x] += z
    for name, counter in [('visual', onto_visual), ('olfactory', onto_olf)]:
        by_super = Counter(); by_type = Counter()
        for x, z in counter.items():
            r = info.get(x, {})
            by_super[(r.get('superclass'), r.get('class'))] += z; by_type[r.get('type')] += z
        print(f'\ninputs from outside the mushroom body onto {name} Kenyon cells: {len(counter)} cells, {sum(counter.values())} synapses')
        print('  by superclass/class:', by_super.most_common(8))
        print('  by type:', by_type.most_common(16))
    strong = [x for x, z in onto_visual.items() if z >= MIN_SYNAPSES]
    with_hex = sum(1 for x in strong if hex_of(info.get(x, {})))
    print(f'\ninputs with >= {MIN_SYNAPSES} synapses onto visual KCs: {len(strong)}; with optic-lobe hex coordinates: {with_hex}')
    sides = Counter(info.get(x, {}).get('somaSide') for x in strong)
    print('soma sides:', dict(sides))

    # The other visual route: what reaches the ellipsoid-body ring neurons (ER), which bind scenes to heading.
    ring = {int(r['bodyId']) for r in rows if str(r.get('type') or '').startswith('ER') and r.get('status') == 'Traced'}
    if ring:
        _, ring_edges = inputs_onto(raw, ring)
        by_type = Counter()
        for x, _, z in ring_edges: by_type[info.get(x, {}).get('type')] += z
        print(f'\ninputs onto {len(ring)} ring neurons by type:', by_type.most_common(12))


# --- one and two hops upstream: does an input carry vision, which kind, and from where in the eye -----------------
OPTIC = {'visual_projection', 'ol_intrinsic'}       # superclasses whose output is optic-lobe processed vision
FAMILY = re.compile(r'^([A-Za-z]+?)(\d+)')
# Optic-lobe cell types with a well-established contrast polarity. Everything else from the optic lobe is "form":
# lobula and medulla projection cells, Li, LC, TmY and the like, which respond to shapes, edges and objects.
CHANNEL_OF_TYPE = {
    'ON':  ['L1', 'Mi1', 'Mi4', 'Tm3', 'T4', 'C3'],
    'OFF': ['L2', 'L3', 'Tm1', 'Tm2', 'Tm4', 'Tm9', 'Mi9', 'T5', 'T3'],
    'luminance': ['R7', 'R8', 'Dm8', 'Dm9', 'Tm5', 'Tm20'],          # colour/UV pathway; monochrome here
}
TYPE_CHANNEL = {t: c for c, ts in CHANNEL_OF_TYPE.items() for t in ts}
MIX_KEYS = ('ON', 'OFF', 'luminance', 'form')
MIN_UPSTREAM = 3


def channel_of(type_name, superclass):
    m = FAMILY.match(str(type_name or ''))
    if m and f'{m.group(1)}{m.group(2)}' in TYPE_CHANNEL: return TYPE_CHANNEL[f'{m.group(1)}{m.group(2)}']
    return 'form' if superclass in OPTIC else None


def hex_xy(h):
    """Axial optic-lobe hex coordinates (neighbours differ by (1,0), (0,1) or (1,1)) to plane coordinates."""
    return (h[0] - 0.5 * h[1], h[1] * math.sqrt(3) / 2)


def weighted_position(points):
    """points: [(x, y, spread, weight)] -> (x, y, spread, total weight) or None."""
    w = sum(p[3] for p in points)
    if not w: return None
    x = sum(p[0] * p[3] for p in points) / w; y = sum(p[1] * p[3] for p in points) / w
    var = sum(p[3] * ((p[0] - x) ** 2 + (p[1] - y) ** 2 + p[2] ** 2) for p in points) / w
    return x, y, math.sqrt(var / 2), w


def upstream(raw, rows, sources):
    """For each source cell: visual share, channel mix and receptive field, from one and two hops upstream."""
    info = {int(r['bodyId']): r for r in rows}
    sup = lambda b: info.get(b, {}).get('superclass')
    hexes = {b: hex_of(r) for b, r in info.items() if hex_of(r)}
    pts = np.array([hex_xy(h) for h in hexes.values()]) if hexes else np.zeros((1, 2))
    centre, scale = pts.mean(0), max(float(np.abs(pts - pts.mean(0)).max()), 1e-9)
    norm = lambda h: tuple((np.array(hex_xy(h)) - centre) / scale)

    _, hop1 = inputs_onto(raw, sources)
    into = defaultdict(Counter)
    for x, y, z in hop1: into[y][x] += z
    # Optic-lobe cells one hop up without coordinates (LC, LoVP, MeVP...): their own inputs give position and kind.
    second = {x for c in into.values() for x, z in c.items() if z >= MIN_UPSTREAM and sup(x) in OPTIC and x not in hexes}
    _, hop2 = inputs_onto(raw, second) if second else (None, [])
    into2 = defaultdict(Counter)
    for x, y, z in hop2: into2[y][x] += z

    def describe(counter):
        """Channel mix and position of the optic-lobe input in one cell's input counter (single hop)."""
        mix, points = Counter(), []
        for pre, z in counter.items():
            channel = channel_of(info.get(pre, {}).get('type'), sup(pre))
            if channel: mix[channel] += z
            if pre in hexes:
                x, y = norm(hexes[pre]); points.append((x, y, 0.0, z, info[pre].get('somaSide')))
        return mix, points

    derived = {}
    for cell in second:
        mix, points = describe(into2[cell])
        pos = weighted_position([p[:4] for p in points])
        side = Counter()
        for p in points: side[p[4]] += p[3]
        derived[cell] = (mix, pos, side)

    out = {}
    for cell in sources:
        counter = into[cell]
        annotated = sum(z for x, z in counter.items() if sup(x))
        optic = sum(z for x, z in counter.items() if sup(x) in OPTIC)
        mix, points = describe(counter)
        side = Counter()
        for p in points: side[p[4]] += p[3]
        # Optic-lobe inputs without coordinates contribute their own derived mix and position, by synapse weight.
        for pre, z in counter.items():
            if pre in derived and z >= MIN_UPSTREAM:
                dmix, dpos, dside = derived[pre]
                dtotal = sum(dmix.values())
                if dtotal:
                    for k, v in dmix.items(): mix[k] += 0.5 * z * v / dtotal       # half own label, half derived
                    mix['form'] -= 0.5 * z if channel_of(info.get(pre, {}).get('type'), sup(pre)) == 'form' else 0
                if dpos:
                    points.append((dpos[0], dpos[1], dpos[2], z, None))
                    for s, v in dside.items(): side[s] += z * v / max(sum(dside.values()), 1)
        total_mix = sum(max(v, 0) for v in mix.values())
        pos = weighted_position([p[:4] for p in points])
        out[cell] = {
            'visual_share': round(optic / annotated, 4) if annotated else 0.0,
            'channel_mix': {k: round(max(mix.get(k, 0), 0) / total_mix, 4) for k in MIX_KEYS} if total_mix else None,
            'rf': {'x': round(pos[0], 4), 'y': round(pos[1], 4), 'spread': round(pos[2], 4),
                   'side': side.most_common(1)[0][0] if side else None, 'weight': int(pos[3])} if pos else None,
            'upstream_types': Counter({str(info.get(x, {}).get('type')): z for x, z in counter.items()}).most_common(8),
        }
    return out


def add_upstream(raw):
    """Add visual share, channel mix and receptive fields to an existing data/visual-memory.json."""
    path = BASE / 'data/visual-memory.json'
    d = json.loads(path.read_text())
    rows = annotations(raw)
    found = upstream(raw, rows, {n['bodyId'] for n in d['nodes']})
    for n in d['nodes']: n.update(found.get(n['bodyId'], {}))
    d['upstream'] = {'optic_superclasses': sorted(OPTIC), 'channels': CHANNEL_OF_TYPE, 'min_synapses': MIN_UPSTREAM,
                     'method': 'visual_share: fraction of annotated input synapses from optic-lobe superclasses. channel_mix: '
                               'input synapses by optic-lobe cell type (known ON, OFF and colour types; other optic-lobe '
                               'cells count as form), with coordinate-less optic-lobe inputs contributing half their own '
                               'derived mix. rf: synapse-weighted position of columnar inputs with optic-lobe hex '
                               'coordinates, one or two hops up, normalised to the whole lattice.'}
    path.write_text(json.dumps(d, separators=(',', ':')))
    shares = [n['visual_share'] for n in d['nodes']]
    print('UPSTREAM', json.dumps({
        'inputs': len(shares), 'mostly_visual(>0.5)': sum(s > 0.5 for s in shares), 'some_visual(0.1-0.5)': sum(0.1 < s <= 0.5 for s in shares),
        'not_visual(<=0.1)': sum(s <= 0.1 for s in shares), 'with_rf': sum(1 for n in d['nodes'] if n.get('rf')),
        'synapse_weighted_visual_share': round(sum(n['visual_share'] * n['synapses_to_visual_kcs'] for n in d['nodes'])
                                               / max(sum(n['synapses_to_visual_kcs'] for n in d['nodes']), 1), 3),
        'mean_mix': {k: round(float(np.mean([n['channel_mix'][k] for n in d['nodes'] if n.get('channel_mix')] or [0])), 3) for k in MIX_KEYS},
        'most_visual_types': Counter({n['type']: n['visual_share'] for n in d['nodes']}).most_common(8)}), flush=True)


def build(raw):
    rows = annotations(raw)
    info = {int(r['bodyId']): r for r in rows}
    kcs, visual = kenyon_cells(rows)
    # One pass over every edge onto any Kenyon cell, so controls can compare where else the inputs reach.
    _, all_edges = inputs_onto(raw, set(kcs))
    total = Counter()
    for x, y, z in all_edges:
        if y in visual: total[x] += z
    sources = {x for x, z in total.items() if z >= MIN_SYNAPSES and not is_mb(info.get(x, {}).get('type'))
               and info.get(x, {}).get('status') == 'Traced'}
    kept = [[x, y, z] for x, y, z in all_edges if x in sources]
    nodes = []
    for x in sorted(sources):
        r = info[x]
        nodes.append({'bodyId': x, 'type': r.get('type'), 'instance': r.get('instance'), 'class': r.get('class'),
                      'superclass': r.get('superclass'), 'somaSide': r.get('somaSide'), 'hex': hex_of(r),
                      'role': 'VPN', 'synapses_to_visual_kcs': int(total[x])})
    out = {'dataset': 'male-cns:v1.0', 'license': 'CC-BY-4.0', 'nodes': nodes, 'edges': kept,
           'visual_kc_types': list(VISUAL_KC_TYPES), 'visual_kcs': sorted(visual), 'min_synapses': MIN_SYNAPSES,
           'scope': f'Traced cells outside the mushroom body with at least {MIN_SYNAPSES} synapses onto Kenyon cells of '
                    f'types {", ".join(VISUAL_KC_TYPES)}, and all their edges onto any Traced Kenyon cell.'}
    (BASE / 'data/visual-memory.json').write_text(json.dumps(out, separators=(',', ':')))
    print('DONE', json.dumps({'inputs': len(nodes), 'edges': len(kept), 'visual_kcs': len(visual),
                              'with_hex': sum(1 for n in nodes if n['hex']),
                              'types': Counter(n['type'] for n in nodes).most_common(10)}), flush=True)
    add_upstream(raw)


if __name__ == '__main__':
    if len(sys.argv) < 2: sys.exit(__doc__)
    if '--probe' in sys.argv: probe(sys.argv[1])
    elif '--upstream' in sys.argv: add_upstream(sys.argv[1])
    else: build(sys.argv[1])
