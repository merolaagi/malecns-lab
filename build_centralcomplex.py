"""Central-complex navigation subset: the heading, vector and steering cells.

The fly's navigation machinery is here: EPG cells hold heading as a bump around the ellipsoid body,
PEN/PEG cells rotate that bump with turning, Delta7 cells enforce a single bump, PFN cells combine
heading with movement, FC2/hDelta cells hold a goal, and PFL cells compare goal against heading and
drive the descending neurons, PFL3 onto DNa02 in particular, which this lab's vision bridge already
targets.

Run on a machine with the raw tables:
    .venv/bin/python build_centralcomplex.py raw-data --probe     # what the annotations actually hold
    .venv/bin/python build_centralcomplex.py raw-data             # writes data/central-complex.json

The probe prints every annotation column, the counts of each central-complex type, sample instance
strings and which columns are populated for those cells. Column identity (which glomerulus or wedge a
cell belongs to) is what the navigation model needs, and it is not guaranteed to be stored the same way
as the optic-lobe hex coordinates, so the probe is worth reading before trusting the build.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from model import BASE

# Functional groups, by type prefix. Each entry: (group, what it is taken to do).
GROUPS = {
    'EPG': ('heading', 'Heading bump around the ellipsoid body'),
    'PEN': ('rotate', 'Shifts the bump with turning'),
    'PEG': ('sustain', 'Feeds the bump back to itself'),
    'Delta7': ('inhibit', 'Long-range inhibition that keeps one bump'),
    'PFN': ('vector', 'Combines heading with movement direction'),
    'FC2': ('goal', 'Holds a goal direction'),
    'hDelta': ('goal', 'Holds or accumulates a vector'),
    'PFL': ('steer', 'Compares goal against heading and drives descending neurons'),
    'ER': ('cue', 'Ring neurons carrying sensory cues to the compass'),
    'ExR': ('cue', 'Extrinsic ring neurons'),
}
DN_TARGETS = ['DNa02', 'DNa01', 'DNg13']
# Column labels live in the instance suffix, as the probe on the real annotations showed:
#   EPG(PB08)_R2          protocerebral-bridge glomerulus R2 (the parenthetical is a type code)
#   PEN_a(PB06a)_L3       glomerulus L3
#   PFNv(PB05)_L4_C3      glomerulus L4 and fan-shaped-body column C3
#   PFL2(PB12b)_R4_C2     glomerulus R4 and column C2
#   Delta7(PB15)_L3R6_R   spans glomeruli L3 and R6; the trailing _R is the soma side
#   FC2A_C2_L             column C2, soma side L
#   hDeltaB_08_C7         fan-shaped-body index 08 and column C7
#   ER5_L, ExR1_R         no column at all
PAREN = re.compile(r'\([^)]*\)')
# A glomerulus token is a whole underscore-separated field made only of side+number pairs, so that
# 'ER5' and 'ExR1' (type names) do not masquerade as glomerulus R5 and R1.
GLOMERULUS_TOKEN = re.compile(r'^(?:[LR]\d+)+$')
GLOMERULUS = re.compile(r'([LR])(\d+)')
FB_COLUMN = re.compile(r'_C(\d+)')
FB_INDEX = re.compile(r'_(\d{2})(?:_|$)')
RING = 16                      # 8 glomeruli per side of the bridge


def group_of(type_name):
    if not type_name: return None
    for prefix, (group, _) in GROUPS.items():
        if str(type_name).startswith(prefix): return prefix, group
    return None


def column_of(instance):
    """Ring column 0-15 from an instance string, with the rule used. (None, None) when unparseable.

    Bridge glomeruli map directly: R1-R8 to 0-7 and L1-L8 to 8-15. Cells labelled only by a
    fan-shaped-body column are spread over the same ring, two ring columns per body column. Cells that
    span several glomeruli, like Delta7, take the circular mean of the ones they list."""
    if not instance: return None, None
    text = PAREN.sub('', str(instance))
    glomeruli = []
    for token in text.split('_'):
        if GLOMERULUS_TOKEN.match(token):
            glomeruli += [(side, int(number)) for side, number in GLOMERULUS.findall(token) if 1 <= int(number) <= 9]
    if glomeruli:
        positions = [((number - 1) % 8) + (8 if side == 'L' else 0) for side, number in glomeruli]
        if len(positions) == 1: return positions[0], 'glomerulus'
        vectors = np.exp(2j * np.pi * np.array(positions) / RING).sum()
        return int(round(np.angle(vectors) / (2 * np.pi) * RING)) % RING, 'glomerulus_mean'
    column = FB_COLUMN.search(str(instance))
    if column: return ((int(column.group(1)) - 1) * 2) % RING, 'fb_column'
    index = FB_INDEX.search(str(instance))
    if index: return ((int(index.group(1)) - 1) * 2) % RING, 'fb_index'
    return None, None


def probe(raw):
    table = feather.read_table(raw / 'annotations.feather')
    rows = table.to_pylist()
    print('annotation columns:', sorted(table.column_names))
    counts, samples, populated = Counter(), {}, Counter()
    for r in rows:
        hit = group_of(r.get('type'))
        if not hit: continue
        prefix, group = hit
        counts[(prefix, group, r.get('status'))] += 1
        samples.setdefault(prefix, []).append({'bodyId': r.get('bodyId'), 'type': r.get('type'), 'instance': r.get('instance'),
                                               'somaSide': r.get('somaSide'), 'column_guess': column_of(r.get('instance'))})
        for key, value in r.items():
            if value not in (None, '', float('nan')) and value == value: populated[key] += 1
    print('\ncentral-complex cells by type prefix, group and status:')
    for (prefix, group, status), n in sorted(counts.items()): print(f'  {prefix:8} {group:8} {status!s:12} {n}')
    print('\nsample instances (3 per type):')
    for prefix, rows_ in samples.items():
        for r in rows_[:3]: print(f'  {prefix:8} {r["bodyId"]} {r["type"]!s:14} {r["instance"]!s:38} side={r["somaSide"]} column_guess={r["column_guess"]}')
    parsed = sum(1 for rows_ in samples.values() for r in rows_ if r['column_guess'][0] is not None)
    total = sum(len(rows_) for rows_ in samples.values())
    print(f'\ncolumn parsed from instance for {parsed} of {total} central-complex cells')
    print('populated fields across these cells:', dict(populated.most_common(25)))


def build(raw):
    rows = feather.read_table(raw / 'annotations.feather').to_pylist()
    dn_ids, nodes = [], []
    for r in rows:
        if r.get('status') != 'Traced': continue
        name = str(r.get('type') or '')
        if any(name.startswith(t) for t in DN_TARGETS): dn_ids.append(int(r['bodyId']))
        hit = group_of(name)
        if not hit: continue
        prefix, group = hit
        column, rule = column_of(r.get('instance'))
        nodes.append({'bodyId': int(r['bodyId']), 'type': r.get('type'), 'instance': r.get('instance'),
                      'somaSide': r.get('somaSide'), 'prefix': prefix, 'group': group,
                      'column': column, 'column_rule': rule})
    ids = np.array(sorted({n['bodyId'] for n in nodes} | set(dn_ids)))
    edges = []
    with pa.memory_map(str(raw / 'weights.feather'), 'r') as file:
        reader = pa.ipc.open_file(file)
        for k in range(reader.num_record_batches):
            batch = reader.get_batch(k)
            a, b, w = batch.column('body_pre').to_numpy(), batch.column('body_post').to_numpy(), batch.column('weight').to_numpy()
            mask = np.isin(a, ids) & np.isin(b, ids) & (w > 0)
            edges.extend([[int(x), int(y), int(z)] for x, y, z in zip(a[mask], b[mask], w[mask])])
            if k % 600 == 0: print('batch', k, 'edges', len(edges), flush=True)
    transmitters = {}
    path = raw / 'neurotransmitters.feather'
    if path.exists():
        for r in feather.read_table(path).to_pylist():
            if int(r['bodyId']) in set(ids): transmitters[int(r['bodyId'])] = r.get('consensus_nt') or r.get('predicted_nt')
    for n in nodes: n['nt'] = transmitters.get(n['bodyId'])
    provenance = json.loads((BASE / 'data/circuit.json').read_text())
    out = {'dataset': 'male-cns:v1.0', 'license': 'CC-BY-4.0', 'source': provenance['source'], 'files': provenance['files'],
           'groups': {k: {'group': v[0], 'role': v[1]} for k, v in GROUPS.items()},
           'nodes': nodes, 'descending_targets': sorted(dn_ids), 'edges': edges,
           'columns_parsed': sum(1 for n in nodes if n['column'] is not None),
           'column_rules': dict(Counter(n['column_rule'] for n in nodes)),
           'scope': 'Traced central-complex navigation types plus the lab\'s descending targets, with the measured edges among '
                    'them. Functional group labels are assignments by type prefix from published roles, not measurements. Column '
                    'identity is parsed from instance strings and may be incomplete; check with --probe.'}
    (BASE / 'data/central-complex.json').write_text(json.dumps(out, separators=(',', ':')))
    by_group = Counter(n['group'] for n in nodes)
    print('DONE nodes', len(nodes), 'edges', len(edges), 'columns parsed', out['columns_parsed'], dict(by_group), flush=True)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    raw = Path(args[0] if args else 'raw-data')
    (probe if '--probe' in sys.argv else build)(raw)
