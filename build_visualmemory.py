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
    .venv/bin/python build_visualmemory.py raw-data             # writes data/visual-memory.json
"""
import json
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


if __name__ == '__main__':
    if len(sys.argv) < 2: sys.exit(__doc__)
    probe(sys.argv[1]) if '--probe' in sys.argv else build(sys.argv[1])
