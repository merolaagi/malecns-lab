"""Add the cells that coordinate the legs, and P9's route to the front legs, to the optogenetics circuit.

The diagnosis (diagnose.py) found two gaps in data/optogenetics-circuit.json:
  * P9 reaches no front-leg hip motor neuron, at any light strength;
  * nothing in the circuit links the legs, so each leg steps at its own speed and the body barely walks.
Real flies coordinate their legs through ventral-nerve-cord interneurons that connect the leg neuropils, plus
leg sense organs. This builder adds, by anatomy only:

  premotor     for each leg (side x segment), the interneurons with the most synapses onto that leg's motor
               neurons (at least PREMOTOR_MIN, top PREMOTOR_PER_LEG)
  linking      interneurons whose outputs onto premotor cells and motor neurons reach at least two leg
               segments with at least LINK_MIN synapses each (top LINK_N, ranked by their second-largest segment)
  P9 route     cells receiving at least ROUTE_MIN synapses from P9 (DNp09) that make at least ROUTE_MIN onto
               front-leg premotor cells or front-leg motor neurons, and those premotor cells
  mirror       every added cell type's cells on the other side, as in build_optogenetics.py

It also measures how strongly each leg's premotor pool reaches every other leg's, directly or through one
interneuron, split into excitatory (acetylcholine) and inhibitory (GABA, glutamate) paths: a 6 x 6 leg-link matrix
that bodysim.py can use for the coupling between leg rhythms instead of a uniform constant.

    .venv/bin/python build_coordination.py raw-data --probe   # sizes only
    .venv/bin/python build_coordination.py raw-data           # writes data/coordination-circuit.json
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from scipy.sparse import csr_matrix

import build_optogenetics as bo

BASE = Path(__file__).resolve().parent
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
SEGMENT = {'fl': 'F', 'ml': 'M', 'hl': 'H'}
PREMOTOR_MIN, PREMOTOR_PER_LEG = 10, 60
LINK_MIN, LINK_N = 10, 200
ROUTE_MIN = 5
SIGN = {'acetylcholine': 1.0, 'gaba': -1.0, 'glutamate': -1.0}


def leg_of(r):
    side = r.get('rootSide') or r.get('somaSide')
    seg = SEGMENT.get(r.get('subclass'))
    label = f'{side}{seg}'
    return label if label in LEGS else None


def load(raw):
    raw = Path(raw)
    rows = bo.annotations(raw)
    ann = {r['bodyId']: r for r in rows if r['status'] == 'Traced'}
    table = feather.read_table(raw / 'weights.feather', memory_map=True)
    pre, post, weight = (table[k].to_numpy() for k in ['body_pre', 'body_post', 'weight'])
    nt = {r['body']: r for r in feather.read_table(raw / 'neurotransmitters.feather').to_pylist()}
    return rows, ann, pre, post, weight, nt


def sums_by(src, w):
    """Total weight per source id: (ids, totals)."""
    if not len(src): return np.zeros(0, np.int64), np.zeros(0)
    ids, inv = np.unique(src, return_inverse=True)
    return ids, np.bincount(inv, weights=w.astype(float))


def select(ann, pre, post, weight, base_ids):
    """The cells to add, and why each was added."""
    mn_leg = {i: leg_of(r) for i, r in ann.items() if r['superclass'] == 'vnc_motor' and leg_of(r)}
    inter = np.array(sorted(i for i, r in ann.items() if r['superclass'] == 'vnc_intrinsic'), dtype=pre.dtype)
    from_inter = np.isin(pre, inter) & (weight > 0)

    # premotor pools
    premotor = {}
    for leg in LEGS:
        mns = np.array([i for i, l in mn_leg.items() if l == leg], dtype=post.dtype)
        m = from_inter & np.isin(post, mns)
        ids, tot = sums_by(pre[m], weight[m])
        order = np.argsort(-tot)[:PREMOTOR_PER_LEG]
        premotor[leg] = {int(ids[k]) for k in order if tot[k] >= PREMOTOR_MIN}

    # linking interneurons: outputs onto premotor cells or motor neurons of >= 2 segments
    per_segment = {}
    for seg in 'FMH':
        targets = {i for i, l in mn_leg.items() if l[1] == seg}
        for leg in LEGS:
            if leg[1] == seg: targets |= premotor[leg]
        m = from_inter & np.isin(post, np.array(sorted(targets), dtype=post.dtype))
        ids, tot = sums_by(pre[m], weight[m])
        per_segment[seg] = dict(zip(ids.tolist(), tot.tolist()))
    scored = []
    for i in set().union(*per_segment.values()):
        top = sorted((per_segment[s].get(i, 0.0) for s in 'FMH'), reverse=True)
        if top[1] >= LINK_MIN: scored.append((top[1], i))
    linking = {i for _, i in sorted(scored, reverse=True)[:LINK_N]}

    # P9's route to the front legs
    p9 = np.array([i for i, r in ann.items() if r.get('type') == 'DNp09'], dtype=pre.dtype)
    front_targets = premotor['LF'] | premotor['RF'] | {i for i, l in mn_leg.items() if l[1] == 'F'}
    first = np.unique(post[np.isin(pre, p9) & (weight >= ROUTE_MIN)])
    m = np.isin(pre, first) & np.isin(post, np.array(sorted(front_targets), dtype=post.dtype)) & (weight >= ROUTE_MIN)
    inter_set = set(inter.tolist())
    route = {int(x) for x in pre[m]}
    route_pm = {int(y) for y in post[m] if int(y) in inter_set}

    added = {'premotor': set().union(*premotor.values()) - set(base_ids), 'linking': linking - set(base_ids),
             'p9_route': (route | route_pm) - set(base_ids)}
    # mirror completion of everything added
    pools = defaultdict(set)
    for i, r in ann.items():
        if r['superclass'] in ('vnc_intrinsic', 'vnc_motor') and r.get('type'): pools[(r['type'], r.get('subclass'))].add(i)
    chosen = set().union(*added.values())
    mirrored = set()
    for i in chosen:
        r = ann.get(i)
        if r and r.get('type'): mirrored |= pools[(r['type'], r.get('subclass'))]
    added['mirror'] = mirrored - chosen - set(base_ids)
    added = {k: {i for i in v if i in ann} for k, v in added.items()}
    return added, premotor, mn_leg


def leg_links(ann, pre, post, weight, nt, premotor):
    """6 x 6 influence of each leg's premotor pool on each other leg's: direct synapses plus two-step paths through
    any one interneuron, separately for excitatory and inhibitory first steps. Rows: from; columns: to."""
    inter = np.array(sorted(i for i, r in ann.items() if r['superclass'] == 'vnc_intrinsic'), dtype=pre.dtype)
    index = {int(b): k for k, b in enumerate(inter)}
    m = np.isin(pre, inter) & np.isin(post, inter) & (weight > 0)
    rows, cols = np.searchsorted(inter, post[m]), np.searchsorted(inter, pre[m])
    W = csr_matrix((weight[m].astype(float), (rows, cols)), shape=(len(inter), len(inter)))   # post x pre
    sign = np.array([SIGN.get(str(nt.get(int(b), {}).get('consensus_nt')), 0.0) for b in inter])
    pools = {leg: np.array(sorted(index[i] for i in premotor[leg] if i in index), int) for leg in LEGS}
    out = {}
    for kind, s in (('excitatory', 1.0), ('inhibitory', -1.0)):
        L = np.zeros((6, 6))
        for i, a in enumerate(LEGS):
            src = np.zeros(len(inter)); src[pools[a]] = (sign[pools[a]] == s)
            one = W @ src                                   # direct reach of leg a's (sign s) premotor cells
            two = W @ (one * (sign != 0))                   # through one more interneuron of any sign
            for j, b in enumerate(LEGS):
                if i != j and len(pools[b]): L[i, j] = float(one[pools[b]].sum() + 0.1 * two[pools[b]].sum())
        out[kind] = np.round(L, 1).tolist()
    out['legs'] = LEGS
    out['method'] = ('Synapses from leg i\'s premotor cells (of the given sign) onto leg j\'s premotor cells, plus 0.1 x '
                     'the two-step paths through any interneuron. Premotor pools as in this builder.')
    return out


def probe(raw):
    rows, ann, pre, post, weight, nt = load(raw)
    base = {n['bodyId'] for n in json.loads((BASE / 'data/optogenetics-circuit.json').read_text())['nodes']}
    added, premotor, mn_leg = select(ann, pre, post, weight, base)
    print('premotor cells per leg:', {leg: len(c) for leg, c in premotor.items()})
    print('to add:', {k: len(v) for k, v in added.items()})
    print('front-leg motor neurons:', sum(1 for l in mn_leg.values() if l[1] == 'F'))


def build(raw):
    rows, ann, pre, post, weight, nt = load(raw)
    base_data = json.loads((BASE / 'data/optogenetics-circuit.json').read_text())
    base = {n['bodyId'] for n in base_data['nodes']}
    added, premotor, mn_leg = select(ann, pre, post, weight, base)
    why = {}
    for k, cells in added.items():
        for i in cells: why.setdefault(i, k)
    ids = sorted(base | set(why))
    seeds = {n['bodyId'] for n in base_data['nodes'] if n.get('seed')}
    ttm = {n['bodyId'] for n in base_data['nodes'] if n.get('ttm')}
    nodes = []
    for i in ids:
        r, transmitter = ann[i], nt.get(i, {})
        nodes.append({k: r.get(k) for k in ['bodyId', 'type', 'instance', 'superclass', 'subclass', 'somaSide',
                                          'rootSide', 'entryNerve', 'exitNerve', 'status', 'somaLocation']}
                     | {'nt': transmitter.get('consensus_nt'), 'nt_confidence': transmitter.get('predicted_nt_confidence'),
                        'seed': i in seeds, 'ttm': i in ttm, 'added_as': why.get(i),
                        'premotor_of': sorted(l for l, c in premotor.items() if i in c)})
    m = np.isin(pre, ids) & np.isin(post, ids) & (weight > 0)
    edges = [[int(a), int(b), int(w)] for a, b, w in zip(pre[m], post[m], weight[m])]
    out = {'dataset': 'male-cns:v1.0', 'license': 'CC-BY-4.0', 'built_at': datetime.now(timezone.utc).isoformat(),
           'source': 'https://male-cns.janelia.org/download/', 'base': 'data/optogenetics-circuit.json',
           'target_types': base_data.get('target_types'), 'found': base_data.get('found'),
           'added': {k: len(v) for k, v in added.items()},
           'leg_links': leg_links(ann, pre, post, weight, nt, premotor),
           'selection': (f'The optogenetics circuit plus, per leg, the top {PREMOTOR_PER_LEG} premotor interneurons '
                         f'(>= {PREMOTOR_MIN} synapses onto its motor neurons); the top {LINK_N} interneurons reaching '
                         f'premotor cells or motor neurons of two or more leg segments (>= {LINK_MIN} synapses each); '
                         f'P9\'s two-step route to front-leg premotor cells and motor neurons (>= {ROUTE_MIN}); mirror '
                         'completion. All positive edges among selected cells. Chemical synapses only.'),
           'nodes': nodes, 'edges': edges}
    (BASE / 'data/coordination-circuit.json').write_text(json.dumps(out, separators=(',', ':')))
    print('DONE', json.dumps({'neurons': len(nodes), 'edges': len(edges), 'added': out['added'],
                              'premotor_per_leg': {l: len(c) for l, c in premotor.items()}}), flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2: sys.exit(__doc__)
    probe(sys.argv[1]) if '--probe' in sys.argv else build(sys.argv[1])
