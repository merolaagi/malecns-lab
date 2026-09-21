"""Vision-to-steering bridge: how strongly can each flyvis output cell type reach the lab's
descending neurons (DNa01, DNa02, DNg13) in the MaleCNS connectome?

Runs on the official raw tables (same files as build_dataset.py). Works at the level of
cell-type groups split by side, because flyvis models cell types, not individual cells.

Measured: synapse counts between typed MaleCNS bodies, predicted transmitters.
Derived (not measured): input fractions and multi-hop path influence. Influence of A on B along
a path is the product of input fractions (synapses from the previous group / all synapses onto the
next group, typed or not). It ranks anatomical routes; it is not a prediction of activity.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import scipy.sparse as sp

BASE = Path(__file__).resolve().parent
SOURCE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = {'annotations': 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
         'weights': 'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
         'neurotransmitters': 'body-neurotransmitters-male-cns-v1.0.feather'}
# Decoded (output) cell types of the flyvis connectome (flyvis 1.2.0, fib25-fib19_v2.2.json).
FLYVIS_OUTPUTS = ['T1', 'T2', 'T2a', 'T3', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d',
                  'Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm5Y', 'Tm5a', 'Tm5b', 'Tm5c', 'Tm9', 'Tm16', 'Tm20',
                  'Tm28', 'Tm30', 'TmY3', 'TmY4', 'TmY5a', 'TmY9', 'TmY10', 'TmY13', 'TmY14', 'TmY15', 'TmY18']
DN_TYPES = ['DNa01', 'DNa02', 'DNg13']
SIGN = {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1}


def side_of(r):
    return r.get('rootSide') or r.get('somaSide') or '?'


def file_hashes(directory):
    out = {}
    for key, name in FILES.items():
        h = hashlib.sha256()
        with (directory / (key + '.feather')).open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
        out[key] = {'url': SOURCE + name, 'sha256': h.hexdigest()}
    return out


def build(directory, out_path, keep=60):
    ann_rows = feather.read_table(directory / 'annotations.feather').to_pylist()
    typed = [r for r in ann_rows if r.get('type')]
    status = Counter(r.get('status') for r in typed)
    nt = {r['body']: r.get('consensus_nt') for r in feather.read_table(directory / 'neurotransmitters.feather').to_pylist()}

    keys = sorted({(r['type'], side_of(r)) for r in typed})
    gindex = {k: i for i, k in enumerate(keys)}
    body_ids = np.array([r['bodyId'] for r in typed], dtype=np.int64)
    body_group = np.array([gindex[(r['type'], side_of(r))] for r in typed], dtype=np.int64)
    order = np.argsort(body_ids); body_ids, body_group = body_ids[order], body_group[order]

    def groups_of(ids):
        pos = np.searchsorted(body_ids, ids)
        pos = np.clip(pos, 0, len(body_ids) - 1)
        return np.where(body_ids[pos] == ids, body_group[pos], -1)

    t = feather.read_table(directory / 'weights.feather', memory_map=True)
    pre, post, w = (t[k].to_numpy() for k in ['body_pre', 'body_post', 'weight'])
    gpre, gpost = groups_of(pre.astype(np.int64)), groups_of(post.astype(np.int64))
    G = len(keys)
    # Denominator: every synapse onto the group, including from untyped bodies.
    total_in = np.bincount(gpost[gpost >= 0], weights=w[gpost >= 0], minlength=G)
    m = (gpre >= 0) & (gpost >= 0)
    S = sp.coo_matrix((w[m].astype(np.float64), (gpre[m], gpost[m])), shape=(G, G)).tocsr()
    S.sum_duplicates()
    F = sp.csr_matrix(S.multiply(1 / np.maximum(total_in, 1)[None, :]))

    O = [gindex[k] for k in keys if k[0] in FLYVIS_OUTPUTS]
    D = [gindex[k] for k in keys if k[0] in DN_TYPES]
    if not D: raise SystemExit('No DNa01/DNa02/DNg13 bodies found in annotations')
    if not O: raise SystemExit('No flyvis output cell types found in annotations')
    FO, FD = F[O, :], F[:, D]
    E1 = FO[:, D].toarray()
    FOF = (FO @ F)
    E2 = (FO @ FD).toarray()
    E3 = (FOF @ FD).toarray()

    # How much of the depth-2 and depth-3 influence passes through each intermediate group.
    to_D = np.asarray(FD.sum(axis=1)).ravel()                 # X -> any DN
    from_O = np.asarray(FO.sum(axis=0)).ravel()               # any output -> X
    via1 = from_O * np.asarray((F @ FD).sum(axis=1)).ravel()  # X as first hop of a 3-hop path
    via2 = np.asarray(FOF.sum(axis=0)).ravel() * to_D         # X as second hop of a 3-hop path
    through = from_O * to_D + via1 + via2
    through[O] = 0; through[D] = 0
    mids = [int(i) for i in np.argsort(-through)[:keep] if through[i] > 0]

    members = {}
    for r in typed: members.setdefault(gindex[(r['type'], side_of(r))], []).append(r)

    def describe(g, role):
        rs = members[g]
        nts = Counter(nt.get(r['bodyId']) for r in rs)
        top_nt, n_top = nts.most_common(1)[0]
        return {'id': g, 'type': keys[g][0], 'side': keys[g][1], 'role': role, 'cells': len(rs),
                'superclass': Counter(r.get('superclass') for r in rs).most_common(1)[0][0],
                'nt': top_nt, 'nt_agreement': round(n_top / len(rs), 3), 'sign': SIGN.get(top_nt, 0),
                'total_input_synapses': int(total_in[g]), 'through_influence': float(through[g]) if role == 'intermediate' else None}
    keep_ids = O + mids + D
    groups = [describe(g, 'flyvis_output') for g in O] + [describe(g, 'intermediate') for g in mids] + [describe(g, 'descending') for g in D]
    sub = S[keep_ids, :][:, keep_ids].tocoo()
    edges = [{'pre': keep_ids[i], 'post': keep_ids[j], 'synapses': int(v), 'input_fraction': float(F[keep_ids[i], keep_ids[j]])}
             for i, j, v in zip(sub.row, sub.col, sub.data) if v > 0]
    influence = [{'output': O[a], 'dn': D[b], 'direct': float(E1[a, b]), 'two_hop': float(E2[a, b]), 'three_hop': float(E3[a, b]),
                  'total': float(E1[a, b] + E2[a, b] + E3[a, b])} for a in range(len(O)) for b in range(len(D))]
    result = {
        'dataset': 'male-cns:v1.0', 'license': 'CC-BY-4.0', 'built_at': datetime.now(timezone.utc).isoformat(),
        'source': 'https://male-cns.janelia.org/download/', 'files': file_hashes(directory),
        'method': 'Typed bodies grouped by (type, side). Input fraction = synapses from group A onto group B / all synapses onto B '
                  '(typed or untyped). Path influence = product of input fractions, summed over all paths of 1, 2 or 3 hops. '
                  f'Intermediates: the {keep} groups carrying the most depth-2 and depth-3 influence from flyvis outputs to the DNs.',
        'flyvis_outputs': FLYVIS_OUTPUTS, 'dn_types': DN_TYPES, 'typed_status_counts': dict(status),
        'groups': groups, 'edges': edges, 'influence': influence}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, separators=(',', ':')))
    summary = {'groups': len(groups), 'intermediates': len(mids), 'edges': len(edges),
               'strongest_routes': sorted(({'from': f"{keys[i['output']][0]} {keys[i['output']][1]}", 'to': f"{keys[i['dn']][0]} {keys[i['dn']][1]}",
                                            'total': round(i['total'], 6)} for i in influence), key=lambda x: -x['total'])[:10]}
    print(json.dumps(summary, indent=2))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path, help='Folder with annotations.feather, weights.feather, neurotransmitters.feather')
    ap.add_argument('--out', type=Path, default=BASE / 'data/vision-bridge.json')
    ap.add_argument('--keep', type=int, default=60, help='Number of intermediate groups to retain')
    a = ap.parse_args()
    build(a.directory, a.out, a.keep)
