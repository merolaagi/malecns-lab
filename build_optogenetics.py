"""Circuit for virtual optogenetics: descending neurons whose activation has a known effect in real flies.

Each target type is a descending neuron (or pair) that experimenters have switched on with light in
genetically targeted flies, with a reported behaviour:

  MDN     moonwalker descending neurons      backward walking            Bidaye et al. 2014
  DNp09   P9                                 forward walking; turning when one side is driven   Bidaye et al. 2020
  DNa02                                      ipsilateral turning         Rayshubskiy et al.
  DNa01                                      ipsilateral steering, smaller and slower   Rayshubskiy et al.
  DNp01   giant fiber                        escape take-off via the jump muscle (TTM)   classic giant-fiber work
  DNg13                                      in the lab's original circuit; no single published effect, run as exploratory

Selection follows build_dataset.py, by anatomy only: every Traced cell of each target type; its 80 strongest
VNC targets (>= 5 synapses); every leg motor target of the selected population (>= 5); the jump-muscle (TTM)
motor neurons; and up to 20 strongly connected leg proprioceptive cells per leg and side. All positive edges
among the selected cells are kept.

Not in this data: electrical synapses. The giant fiber reaches the jump-muscle motor neuron mainly through
gap junctions, which electron-microscopy connectomes do not record, so the model is expected to miss part
of that effect. That is a test of what the chemical connectome alone can explain.

    .venv/bin/python build_optogenetics.py raw-data --probe   # which target types and TTM motor neurons exist
    .venv/bin/python build_optogenetics.py raw-data           # writes data/optogenetics-circuit.json
"""
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

BASE = Path(__file__).resolve().parent
TARGET_TYPES = ['MDN', 'DNp09', 'DNa02', 'DNa01', 'DNp01', 'DNg13']
SEARCH = re.compile(r'MDN|moonwalk|giant|\bGF\b|P9|DNp0?9|DNp0?1\b|TTM', re.I)
TTM = re.compile(r'TTM', re.I)
PER_SEED = 80
MIN_SYNAPSES = 5


def annotations(raw):
    return feather.read_table(Path(raw) / 'annotations.feather').to_pylist()


def probe(raw):
    rows = annotations(raw)
    exact = Counter((r.get('type'), r.get('status'), r.get('rootSide') or r.get('somaSide'))
                    for r in rows if r.get('type') in TARGET_TYPES)
    print('target types (type, status, side): count')
    for key, n in sorted(exact.items(), key=str): print('  ', key, n)
    missing = [t for t in TARGET_TYPES if not any(k[0] == t for k in exact)]
    print('missing target types:', missing or 'none')
    hits = Counter()
    for r in rows:
        text = ' '.join(str(r.get(k) or '') for k in ('type', 'instance', 'synonyms', 'group', 'mancType'))
        if SEARCH.search(text): hits[(r.get('type'), r.get('superclass'), r.get('status'))] += 1
    print('\nannotations mentioning MDN / giant fiber / P9 / TTM (type, superclass, status): count')
    for key, n in hits.most_common(40): print('  ', key, n)
    ttm = [r for r in rows if TTM.search(str(r.get('type') or '') + str(r.get('instance') or '')) and r.get('superclass') == 'vnc_motor']
    print('\nTTM motor neurons:', [(r['bodyId'], r.get('type'), r.get('subclass'), r.get('rootSide') or r.get('somaSide')) for r in ttm][:10])


def build(raw):
    raw = Path(raw)
    rows = annotations(raw)
    ann = {r['bodyId']: r for r in rows if r['status'] == 'Traced'}
    table = feather.read_table(raw / 'weights.feather', memory_map=True)
    pre, post, weight = (table[k].to_numpy() for k in ['body_pre', 'body_post', 'weight'])
    nt = {r['body']: r for r in feather.read_table(raw / 'neurotransmitters.feather').to_pylist()}
    seeds = {i for i, r in ann.items() if r['type'] in TARGET_TYPES}
    intrinsic = {i for i, r in ann.items() if r['superclass'] == 'vnc_intrinsic'}
    ttm = {i for i, r in ann.items() if r['superclass'] == 'vnc_motor'
           and TTM.search(str(r.get('type') or '') + str(r.get('instance') or ''))}
    motors = {i for i, r in ann.items() if r['superclass'] == 'vnc_motor' and r['subclass'] in ['fl', 'ml', 'hl']} | ttm
    eligible = np.isin(post, list(intrinsic | motors)) & (weight >= MIN_SYNAPSES)
    selected = set(seeds) | ttm
    for seed in sorted(seeds):
        idx = np.flatnonzero(eligible & (pre == seed))
        idx = sorted(idx, key=lambda j: (-int(weight[j]), int(post[j])))[:PER_SEED]
        selected.update(int(post[j]) for j in idx)
    mask = np.isin(pre, list(selected)) & np.isin(post, list(motors)) & (weight >= MIN_SYNAPSES)
    selected.update(int(i) for i in post[mask])
    sensory = {i for i, r in ann.items() if r['superclass'] == 'vnc_sensory'
               and r['subclass'] in ['chordotonal organ', 'campaniform sensilla']
               and r['entryNerve'] in ['ProLN', 'MesoLN', 'MetaLN']}
    mask = np.isin(pre, list(sensory)) & np.isin(post, list(selected)) & (weight >= MIN_SYNAPSES)
    scores = Counter()
    for a, w in zip(pre[mask], weight[mask]): scores[int(a)] += int(w)
    for nerve in ['ProLN', 'MesoLN', 'MetaLN']:
        for side in ['L', 'R']:
            candidates = [i for i in scores if ann[i]['entryNerve'] == nerve and (ann[i]['rootSide'] or ann[i]['somaSide']) == side]
            selected.update(sorted(candidates, key=lambda i: (-scores[i], i))[:20])
    ids = sorted(selected)
    nodes = []
    for i in ids:
        r, transmitter = ann[i], nt.get(i, {})
        nodes.append({k: r.get(k) for k in ['bodyId', 'type', 'instance', 'superclass', 'subclass', 'somaSide',
                                          'rootSide', 'entryNerve', 'exitNerve', 'status', 'somaLocation']}
                     | {'nt': transmitter.get('consensus_nt'), 'nt_confidence': transmitter.get('predicted_nt_confidence'),
                        'seed': i in seeds, 'ttm': i in ttm})
    mask = np.isin(pre, ids) & np.isin(post, ids) & (weight > 0)
    edges = [[int(a), int(b), int(w)] for a, b, w in zip(pre[mask], post[mask], weight[mask])]
    found = Counter(ann[i]['type'] for i in seeds)
    out = {'dataset': 'male-cns:v1.0', 'license': 'CC-BY-4.0', 'built_at': datetime.now(timezone.utc).isoformat(),
           'source': 'https://male-cns.janelia.org/download/', 'target_types': TARGET_TYPES, 'found': dict(found),
           'selection': f'Traced cells of {", ".join(TARGET_TYPES)}; top {PER_SEED} VNC targets per seed (>= {MIN_SYNAPSES} synapses); '
                        f'their leg motor targets; TTM motor neurons; top 20 connected leg proprioceptive cells per nerve and side. '
                        'All positive edges among selected cells. Chemical synapses only.',
           'nodes': nodes, 'edges': edges}
    (BASE / 'data/optogenetics-circuit.json').write_text(json.dumps(out, separators=(',', ':')))
    print('DONE', json.dumps({'neurons': len(nodes), 'edges': len(edges), 'synapses': int(sum(e[2] for e in edges)),
                              'targets': dict(found), 'ttm_motor_neurons': len(ttm),
                              'missing': [t for t in TARGET_TYPES if t not in found]}), flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2: sys.exit(__doc__)
    probe(sys.argv[1]) if '--probe' in sys.argv else build(sys.argv[1])
