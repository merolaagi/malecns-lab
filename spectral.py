"""Spectral structure of the measured graphs, against randomisations that preserve what we choose.

The question: is there anything about the measured wiring that random wiring with the same statistics
does not reproduce? Task performance said no (shuffled learned as well as measured in odour learning,
numerosity and vision). This asks a structural question instead, about the eigenvalues of the weight
matrix, which is what recurrent-network theory cares about.

Read the normalisation before reading the numbers. The lab's locomotion matrix normalises incoming
weights so each cell's absolute inputs sum to one, which bounds the spectral radius at 1 by
construction. A measured radius near 1 therefore means the matrix nearly attains its bound, not that the
fly is poised at the edge of chaos. Two other normalisations are offered so that the comparison does not
rest on that choice.

Randomisations, each preserving something different:
  degree_shuffle  - permute destinations across edges (the lab's existing shuffled condition)
  weight_shuffle  - keep topology and signs, permute weight magnitudes among edges
  rewire_targets  - keep each cell's out-degree and its weights, choose new targets
  sign_permute    - keep topology and weights, permute which cells are inhibitory
  full_random     - keep only the number of edges and the multiset of weights

Everything here is a property of the matrices the lab builds, not a claim about the fly: the subset
selection, the log compression of synapse counts and the sign assignment are all lab choices.
"""
import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph
from scipy.sparse.linalg import eigs

from model import BASE, Circuit

NORMALISATIONS = {
    'incoming_l1': 'Incoming absolute weights sum to 1 (the lab default; radius is bounded by 1)',
    'log_raw': 'log(1 + synapses) with sign, no normalisation',
    'global_scale': 'log(1 + synapses) with sign, scaled so the largest singular value is 1',
}
RANDOMISATIONS = {
    'measured': 'The measured wiring',
    'degree_shuffle': 'Destinations permuted across edges',
    'weight_shuffle': 'Weights permuted among existing edges',
    'rewire_targets': 'Targets redrawn, out-degree and weights kept',
    'sign_permute': 'Inhibitory cells permuted',
    'full_random': 'Edge count and weight multiset only',
    'reciprocal_random': 'Random wiring with the measured fraction of reciprocal pairs',
}
DENSE_LIMIT = 1600           # above this, only sparse leading eigenvalues are computed


def graphs():
    """Every measured graph the lab has, as (name, pre, post, count, sign-per-cell, size)."""
    available = {}
    circuit = Circuit()
    signs = np.array([{'acetylcholine': 1., 'gaba': -1., 'glutamate': -1.}.get(n.get('nt'), 0.) for n in circuit.nodes])
    available['locomotion'] = (circuit.pre, circuit.post, circuit.count, signs, circuit.n, 'Locomotion subset')
    for name, path, label in [('learning', 'data/learning-circuit.json', 'Mushroom-body subset'),
                              ('vision', 'data/vision-circuit.json', 'Optic-lobe patch'),
                              ('central_complex', 'data/central-complex.json', 'Central-complex navigation subset')]:
        file = BASE / path
        if not file.exists(): continue
        data = json.loads(file.read_text())
        index = {n['bodyId']: i for i, n in enumerate(data['nodes'])}
        edges = [(index[a], index[b], w) for a, b, w in data['edges'] if a in index and b in index]
        if not edges: continue
        pre = np.array([e[0] for e in edges]); post = np.array([e[1] for e in edges]); count = np.array([e[2] for e in edges], float)
        sign = np.array([{'acetylcholine': 1., 'gaba': -1., 'glutamate': -1.}.get(n.get('nt'), 0.) for n in data['nodes']])
        if not np.any(sign): sign = np.ones(len(data['nodes']))        # no transmitter calls in this file
        available[name] = (pre, post, count, sign, len(data['nodes']), label)
    return available


def matrix(name, randomisation='measured', normalisation='incoming_l1', seed=7, data=None):
    pre, post, count, sign, n, _ = (data or graphs())[name]
    rng = np.random.default_rng(seed + 1000)
    pre, post, count, sign = pre.copy(), post.copy(), count.copy(), sign.copy()
    if randomisation == 'degree_shuffle':
        post = rng.permutation(post)
    elif randomisation == 'weight_shuffle':
        count = rng.permutation(count)
    elif randomisation == 'rewire_targets':
        post = rng.integers(0, n, len(post))
    elif randomisation == 'sign_permute':
        sign = sign[rng.permutation(n)]
    elif randomisation == 'full_random':
        pre = rng.integers(0, n, len(pre)); post = rng.integers(0, n, len(post)); count = rng.permutation(count)
    elif randomisation == 'reciprocal_random':
        # Random wiring built to carry the same share of reciprocal pairs as the measured graph, which
        # is the statistic that plausibly explains the measured spectrum.
        share = _reciprocal_share(pre, post)
        total = len(pre); pairs = int(share * total / 2)
        a = rng.integers(0, n, pairs); b = rng.integers(0, n, pairs)
        rest = total - 2 * pairs
        pre = np.concatenate([a, b, rng.integers(0, n, rest)])
        post = np.concatenate([b, a, rng.integers(0, n, rest)])
        count = rng.permutation(count)
    elif randomisation != 'measured':
        raise ValueError('Unknown randomisation')
    values = np.log1p(count) * sign[pre]
    if normalisation == 'incoming_l1':
        denominator = np.bincount(post, weights=np.abs(values), minlength=n)
        values = values / np.maximum(denominator[post], 1e-9)
    elif normalisation not in NORMALISATIONS:
        raise ValueError('Unknown normalisation')
    W = sp.csr_matrix((values, (post, pre)), shape=(n, n))
    if normalisation == 'global_scale':
        largest = sp.linalg.norm(W) if n > DENSE_LIMIT else float(np.linalg.norm(W.toarray(), 2))
        W = W / max(largest, 1e-9)
    return W


def _reciprocal_share(pre, post):
    edges = set(zip(pre.tolist(), post.tolist()))
    return sum(1 for a, b in edges if (b, a) in edges) / max(len(edges), 1)


def spectrum(W, dense_limit=DENSE_LIMIT):
    """Radius and, when the matrix is small enough to diagonalise, its shape measures."""
    n = W.shape[0]
    if n > dense_limit:
        values = eigs(W.astype(float), k=min(24, n - 2), return_eigenvectors=False, maxiter=5000, tol=1e-6)
        return {'radius': float(np.max(np.abs(values))), 'eigenvalues': [[float(v.real), float(v.imag)] for v in values],
                'henrici': None, 'transient_gain': None, 'full_spectrum': False}
    M = W.toarray()
    values, vectors = np.linalg.eig(M)
    frobenius = float(np.linalg.norm(M, 'fro'))
    henrici = float(math.sqrt(max(frobenius ** 2 - float(np.sum(np.abs(values) ** 2)), 0.0)) / max(frobenius, 1e-9))
    # Transient growth of dx/dt = (-I + W)x at a few horizons. A normal matrix cannot exceed 1 here;
    # anything above it comes from non-orthogonal eigenvectors.
    from scipy.linalg import expm
    A = -np.eye(n) + M
    gain = max(float(np.linalg.norm(expm(A * t), 2)) for t in (0.1, 0.5, 1.0, 2.0, 5.0))
    order = np.argsort(-np.abs(values))[:400]
    leading = np.abs(vectors[:, int(np.argmax(np.abs(values)))])
    leading = leading / max(leading.sum(), 1e-12)
    participation = float(1.0 / max(np.sum(leading ** 2) * n, 1e-12))     # 1 = spread over all cells
    return {'radius': float(np.max(np.abs(values))), 'henrici': round(henrici, 4),
            'transient_gain': round(gain, 3), 'full_spectrum': True,
            'participation_ratio': round(participation, 4),
            'leading_cells': [[int(i), round(float(leading[i]), 4)] for i in np.argsort(-leading)[:12]],
            'eigenvalues': [[round(float(values[i].real), 4), round(float(values[i].imag), 4)] for i in order]}


def closed_blocks(name, data=None, coverage=None):
    """Cells whose entire measured input comes from inside this subset, and the blocks they form.

    Under incoming-L1 normalisation such a group is row-stochastic, so it carries an eigenvalue of 1 by
    construction. That is a property of how the subset was cut, not of the fly: a cell with 4% input
    coverage can look self-contained here simply because the rest of its inputs were never included."""
    pre, post, _, _, n, _ = (data or graphs())[name]
    incoming = {}
    for a, b in zip(pre.tolist(), post.tolist()): incoming.setdefault(b, set()).add(a)
    inside = [cell for cell, sources in incoming.items() if sources]
    graph = sp.csr_matrix((np.ones(len(pre)), (post, pre)), shape=(n, n))
    count, labels = sp.csgraph.connected_components(graph, directed=True, connection='strong')
    sizes = np.bincount(labels, minlength=count)
    # A block is closed when no edge enters it from outside.
    entering = np.zeros(count, dtype=bool)
    for a, b in zip(pre.tolist(), post.tolist()):
        if labels[a] != labels[b]: entering[labels[b]] = True
    closed = [int(i) for i in range(count) if not entering[i] and sizes[i] >= 2]
    members = {int(i): [int(c) for c in np.flatnonzero(labels == i)] for i in closed}
    # The unit that matters for the spectrum is a cell whose every input comes from its own strongly
    # connected component: with incoming-L1 normalisation such rows form a stochastic sub-block.
    self_contained = [cell for cell, sources in incoming.items() if sources and all(labels[a] == labels[cell] for a in sources)]
    result = {'cells_with_input': len(inside), 'strong_components': int(count),
              'closed_blocks': len(closed), 'closed_cells': int(sum(len(v) for v in members.values())),
              'largest_closed_block': max((len(v) for v in members.values()), default=0), 'members': members,
              'self_contained_cells': len(self_contained),
              'self_contained_components': sorted({int(labels[c]) for c in self_contained})[:20]}
    if coverage:
        result['closed_block_mean_coverage'] = round(float(np.mean([coverage.get(c, 0.0) for v in members.values() for c in v])) if members else 0.0, 4)
    return result


def coverage_map():
    """Per-cell input coverage from data/quality.json, keyed by index into the locomotion circuit."""
    path = BASE / 'data/quality.json'
    if not path.exists(): return {}
    cells = json.loads(path.read_text())['cells']
    return {i: (cells.get(str(n['bodyId'])) or {}).get('coverage_in') or 0.0 for i, n in enumerate(Circuit().nodes)}


def coverage_control(name='locomotion', normalisation='incoming_l1', seed=7,
                     minimums=(0.0, 0.02, 0.05, 0.1, 0.2)):
    """Radius when only reasonably observed cells are kept.

    Cells with almost no input coverage are the ones that can look self-contained inside a subset. If the
    measured-versus-random gap survives dropping them, it is not merely a coverage artefact."""
    data = graphs(); coverage = coverage_map()
    if name != 'locomotion' or not coverage: raise ValueError('Coverage control needs the locomotion subset and data/quality.json')
    pre, post, count, sign, n, label = data[name]
    rows = []
    for minimum in minimums:
        keep = np.array([coverage.get(i, 0.0) >= minimum for i in range(n)])
        mask = keep[pre] & keep[post]
        if mask.sum() < 100: continue
        subset = {name: (pre[mask], post[mask], count[mask], sign, n, label)}
        measured = spectrum(matrix(name, 'measured', normalisation, seed, subset))['radius']
        shuffled = spectrum(matrix(name, 'degree_shuffle', normalisation, seed, subset))['radius']
        blocks = closed_blocks(name, subset)
        rows.append({'min_coverage': minimum, 'cells_kept': int(keep.sum()), 'edges': int(mask.sum()),
                     'measured_radius': round(measured, 4), 'shuffled_radius': round(shuffled, 4),
                     'closed_blocks': blocks['closed_blocks'], 'closed_cells': blocks['closed_cells']})
    return {'graph': name, 'normalisation': normalisation, 'rows': rows,
            'note': 'Closed blocks are groups whose entire input lies inside the subset; under incoming-L1 they carry an '
                    'eigenvalue of 1 by construction, so the radius tracks them rather than anything about the fly.'}


def reciprocity(name, data=None):
    pre, post, _, _, n, _ = (data or graphs())[name]
    edges = set(zip(pre.tolist(), post.tolist()))
    both = sum(1 for a, b in edges if (b, a) in edges)
    return round(both / max(len(edges), 1), 4)


def compare(name='locomotion', normalisation='incoming_l1', seed=7, data=None):
    data = data or graphs()
    if name not in data: raise ValueError(f'No graph named {name}')
    out = {'graph': name, 'label': data[name][5], 'cells': int(data[name][4]), 'edges': int(len(data[name][0])),
           'normalisation': normalisation, 'normalisation_text': NORMALISATIONS[normalisation],
           'reciprocity': reciprocity(name, data), 'closed': closed_blocks(name, data), 'conditions': {}}
    for randomisation in RANDOMISATIONS:
        pre, post, count, sign, n, _ = data[name]
        W = matrix(name, randomisation, normalisation, seed, data)
        shuffled_edges = matrix(name, randomisation, 'log_raw', seed, data).tocoo()
        out['conditions'][randomisation] = spectrum(W) | {
            'text': RANDOMISATIONS[randomisation],
            'reciprocity': round(_reciprocal_share(shuffled_edges.col, shuffled_edges.row), 4)}
    # Name the cells that carry the leading eigenvector, when the graph has names for them.
    cells = None
    if name == 'locomotion':
        cells = [(n.get('instance') or n.get('type') or str(n['bodyId']), n.get('superclass')) for n in Circuit().nodes]
    for condition in out['conditions'].values():
        if cells and condition.get('leading_cells'):
            condition['leading_named'] = [{'cell': cells[i][0], 'superclass': cells[i][1], 'share': share}
                                          for i, share in condition['leading_cells'] if i < len(cells)]
    measured = out['conditions']['measured']['radius']
    rewired = [v['radius'] for k, v in out['conditions'].items() if k in ('degree_shuffle', 'rewire_targets', 'full_random', 'reciprocal_random')]
    out['radius_ratio_vs_rewired'] = round(measured / max(max(rewired, default=1e-9), 1e-9), 3)
    out['radius_ratio_vs_weight_shuffle'] = round(measured / max(out['conditions']['weight_shuffle']['radius'], 1e-9), 3)
    out['note'] = ('Under incoming_l1 the radius cannot exceed 1, so a measured value near 1 means the matrix nearly attains '
                   'its bound.' if normalisation == 'incoming_l1' else 'No normalisation caps the radius here.')
    return out


def threshold_control(name='locomotion', normalisation='incoming_l1', thresholds=(1, 3, 5, 10, 20, 40), seed=7):
    """Does the measured-versus-random gap survive changing which edges are included?

    The lab's subsets were selected with a synapse-count threshold, so the spectral signature could be an
    artefact of that choice. This rebuilds the matrix at several thresholds and compares each against its
    own matched randomisations."""
    data = graphs()
    if name not in data: raise ValueError(f'No graph named {name}')
    pre, post, count, sign, n, label = data[name]
    rows = []
    for threshold in thresholds:
        keep = count >= threshold
        if keep.sum() < 50: continue
        subset = {name: (pre[keep], post[keep], count[keep], sign, n, label)}
        measured = spectrum(matrix(name, 'measured', normalisation, seed, subset))['radius']
        randomised = {r: spectrum(matrix(name, r, normalisation, seed, subset))['radius']
                      for r in ['degree_shuffle', 'weight_shuffle', 'rewire_targets']}
        # The ratio compares against randomisations that move edges. weight_shuffle keeps the measured
        # topology, so it is reported but excluded: it is a check that weights are not what matters.
        rewired = max(randomised['degree_shuffle'], randomised['rewire_targets'])
        rows.append({'threshold': int(threshold), 'edges': int(keep.sum()), 'measured_radius': round(measured, 4),
                     'randomised_radius': {k: round(v, 4) for k, v in randomised.items()},
                     'ratio_vs_rewired': round(measured / max(rewired, 1e-9), 3),
                     'ratio_vs_weight_shuffle': round(measured / max(randomised['weight_shuffle'], 1e-9), 3)})
    return {'graph': name, 'normalisation': normalisation, 'rows': rows,
            'note': 'If the ratio stays above one across thresholds, the gap is not an artefact of where the subset was cut.'}


def suite(seed=7):
    data = graphs()
    out = {'graphs': {k: {'label': v[5], 'cells': int(v[4]), 'edges': int(len(v[0]))} for k, v in data.items()},
           'normalisations': NORMALISATIONS, 'randomisations': RANDOMISATIONS, 'comparisons': {}, 'thresholds': {}}
    for name in data:
        for normalisation in NORMALISATIONS:
            if name in ('learning',) and normalisation == 'global_scale': continue      # too large for a dense norm
            out['comparisons'][f'{name}|{normalisation}'] = compare(name, normalisation, seed, data)
    out['thresholds']['locomotion'] = threshold_control('locomotion', 'incoming_l1', seed=seed)
    out['thresholds']['locomotion_log_raw'] = threshold_control('locomotion', 'log_raw', seed=seed)
    try: out['coverage'] = coverage_control('locomotion', 'incoming_l1', seed=seed)
    except ValueError as error: out['coverage'] = {'error': str(error)}
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--benchmark', action='store_true'); ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    if a.benchmark:
        result = suite(a.seed)
        (BASE / 'data/spectral-benchmark.json').write_text(json.dumps(result, separators=(',', ':')))
        print('Wrote data/spectral-benchmark.json')
        for key, value in result['comparisons'].items():
            print(f"{key:34} measured {value['conditions']['measured']['radius']:.3f} "
                  + ' '.join(f"{k[:6]} {v['radius']:.3f}" for k, v in value['conditions'].items() if k != 'measured'))
    else:
        print(json.dumps(compare(), indent=1)[:2000])
