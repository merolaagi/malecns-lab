"""Where inhibition sits, tested against what it does.

The spectral work ended with one effect that survived both controls. On the coverage-selected subset,
under a normalisation that does not bound the radius, the measured network is less excitable than
matched random wiring, and permuting which cells are inhibitory raises the spectral radius by about
half, even though only about 6% of cells are inhibitory and they are not the highest-degree ones.

This module asks whether that placement does anything computational, using the measured graph as a
fixed recurrent network and training only a linear readout, the standard reservoir test. Conditions
keep the number of inhibitory cells identical and move only which cells they are:

  measured        the transmitter predictions as they are
  random          the same number of inhibitory cells, chosen at random
  degree_matched  random, but drawn to match the measured inhibitory out-degree distribution
  hubs            the cells with the most outgoing synapses
  antihubs        the cells with the fewest
  none            no inhibition at all

Two tasks, both standard: memory capacity (how much of a random input stream can be reconstructed at
increasing delays) and a delayed-integration task. Measured: the graph and the transmitter calls.
Assumed: the reservoir dynamics, the scaling, the readout, the tasks, and that a transmitter prediction
fixes a cell's sign. A negative result here is as informative as a positive one.
"""
import json
import math
from pathlib import Path

import numpy as np

from model import BASE

PLACEMENTS = {
    'measured': 'Inhibitory cells as the transmitter predictions call them',
    'random': 'Same number of inhibitory cells, chosen at random',
    'degree_matched': 'Random, matched to the measured inhibitory out-degree distribution',
    'hubs': 'The cells with the most outgoing synapses',
    'antihubs': 'The cells with the fewest outgoing synapses',
    'none': 'No inhibition',
}
DEFAULT = dict(seed=7, cells=700, radius=0.95, scaling='shared', steps=1200, washout=150, delays=12, leak=0.3, noise=0.01)
SCALINGS = {
    'per_condition': 'Each placement rescaled to the same spectral radius: differences are structural only',
    'shared': 'One scale factor for every placement, set by the measured network: gain differences survive',
    'natural': 'No rescaling at all',
}


def load_graph(path=None):
    path = Path(path or BASE / 'data/coverage-subset.json')
    if not path.exists():
        raise ValueError('data/coverage-subset.json is missing. Build it with build_coverage_subset.py.')
    data = json.loads(path.read_text())
    index = {n['bodyId']: i for i, n in enumerate(data['nodes'])}
    edges = np.array([[index[a], index[b], w] for a, b, w in data['edges'] if a in index and b in index], dtype=float)
    inhibitory = np.array([str(n.get('nt')) in ('gaba', 'glutamate') for n in data['nodes']])
    return {'n': len(data['nodes']), 'edges': edges, 'inhibitory': inhibitory,
            'names': [n.get('instance') or n.get('type') or str(n['bodyId']) for n in data['nodes']]}


def subsample(graph, cells, rng):
    """An induced subgraph, so the measured topology is kept while the matrix stays small enough to train."""
    keep = rng.choice(graph['n'], min(cells, graph['n']), replace=False)
    index = -np.ones(graph['n'], int); index[keep] = np.arange(len(keep))
    edges = graph['edges']
    mask = (index[edges[:, 0].astype(int)] >= 0) & (index[edges[:, 1].astype(int)] >= 0)
    sub = edges[mask]
    return {'n': len(keep), 'pre': index[sub[:, 0].astype(int)], 'post': index[sub[:, 1].astype(int)],
            'weight': sub[:, 2], 'inhibitory': graph['inhibitory'][keep]}


def place(sub, placement, rng):
    """Which cells are inhibitory under each condition, keeping the count fixed."""
    n, measured = sub['n'], sub['inhibitory']
    count = int(measured.sum())
    out_synapses = np.bincount(sub['pre'].astype(int), weights=sub['weight'], minlength=n)
    out_degree = np.bincount(sub['pre'].astype(int), minlength=n)
    flags = np.zeros(n, bool)
    if placement == 'measured': flags = measured.copy()
    elif placement == 'random': flags[rng.choice(n, count, replace=False)] = True
    elif placement == 'hubs': flags[np.argsort(-out_synapses)[:count]] = True
    elif placement == 'antihubs': flags[np.argsort(out_synapses)[:count]] = True
    elif placement == 'none': pass
    elif placement == 'degree_matched':
        # Draw cells whose out-degree resembles the measured inhibitory population.
        targets = np.sort(out_degree[measured]) if count else np.array([])
        available = list(np.argsort(out_degree))
        for target in targets:
            position = int(np.searchsorted([out_degree[i] for i in available], target))
            position = min(max(position, 0), len(available) - 1)
            flags[available.pop(position)] = True
    else: raise ValueError('Unknown placement')
    return flags


def matrix(sub, flags, radius, scaling, reference=None):
    """Weight matrix for one placement. `reference` is the measured network's natural radius, used by
    the shared scaling so that every placement is divided by the same number."""
    sign = np.where(flags, -1.0, 1.0)
    values = np.log1p(sub['weight']) * sign[sub['pre'].astype(int)]
    W = np.zeros((sub['n'], sub['n']))
    np.add.at(W, (sub['post'].astype(int), sub['pre'].astype(int)), values)
    scale = float(np.max(np.abs(np.linalg.eigvals(W)))) if W.any() else 0.0
    if scaling == 'per_condition' and scale > 1e-9: W = W * (radius / scale)
    elif scaling == 'shared' and (reference or 0) > 1e-9: W = W * (radius / reference)
    return W, scale


def reservoir(W, inputs, leak, noise, rng):
    """Leaky tanh dynamics driven by a scalar input stream."""
    n = W.shape[0]
    weights_in = rng.normal(0, 1, n)
    state = np.zeros(n)
    states = np.zeros((len(inputs), n))
    for t, value in enumerate(inputs):
        drive = W @ state + weights_in * value + rng.normal(0, noise, n)
        state = (1 - leak) * state + leak * np.tanh(drive)
        states[t] = state
    return states


def ridge(states, targets, penalty=1e-4):
    design = np.hstack([states, np.ones((len(states), 1))])
    gram = design.T @ design + penalty * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ targets)
    predicted = design @ weights
    residual = targets - predicted
    variance = targets.var(axis=0)
    return 1 - residual.var(axis=0) / np.maximum(variance, 1e-12)          # explained variance per column


def run(placement='measured', graph=None, **options):
    p = DEFAULT | options
    if placement not in PLACEMENTS: raise ValueError('Unknown placement')
    for key, lo, hi in [('cells', 100, 3000), ('steps', 300, 5000), ('washout', 50, 1000), ('delays', 2, 30)]:
        if not float(p[key]).is_integer() or not lo <= p[key] <= hi: raise ValueError(f'{key} must be an integer in [{lo},{hi}]')
        p[key] = int(p[key])
    for key, lo, hi in [('radius', 0.05, 2.0), ('leak', 0.01, 1.0), ('noise', 0.0, 0.5)]:
        p[key] = float(p[key])
        if not math.isfinite(p[key]) or not lo <= p[key] <= hi: raise ValueError(f'{key} must be between {lo} and {hi}')
    if p['scaling'] not in SCALINGS: raise ValueError('Unknown scaling')
    rng = np.random.default_rng(int(p['seed']))
    graph = graph or load_graph()
    sub = subsample(graph, p['cells'], rng)
    flags = place(sub, placement, rng)
    reference = None
    if p['scaling'] == 'shared':
        reference = matrix(sub, place(sub, 'measured', rng), p['radius'], 'natural')[1]
    W, natural_radius = matrix(sub, flags, p['radius'], p['scaling'], reference)

    inputs = rng.uniform(-1, 1, p['steps'])
    states = reservoir(W, inputs, p['leak'], p['noise'], rng)
    used = states[p['washout']:]
    # Memory: reconstruct the input at increasing delays. Integration: reconstruct a leaky running sum.
    delays = np.arange(1, p['delays'] + 1)
    memory_targets = np.stack([np.concatenate([np.zeros(d), inputs[:-d]])[p['washout']:] for d in delays], axis=1)
    integral = np.zeros(len(inputs)); accumulator = 0.0
    for t, value in enumerate(inputs):
        accumulator = 0.9 * accumulator + value; integral[t] = accumulator
    scores = ridge(used, memory_targets)
    integration = float(ridge(used, integral[p['washout']:, None])[0])
    activity = float(np.abs(states).mean())
    saturated = float((np.abs(states) > 0.99).mean())
    return {
        'placement': placement, 'placement_text': PLACEMENTS[placement], 'parameters': p,
        'cells': int(sub['n']), 'edges': int(len(sub['weight'])),
        'inhibitory_cells': int(flags.sum()), 'inhibitory_share': round(float(flags.mean()), 4),
        'natural_radius': round(natural_radius, 4),
        'effective_radius': round(float(np.max(np.abs(np.linalg.eigvals(W)))), 4),
        'memory_capacity': round(float(np.clip(scores, 0, 1).sum()), 3),
        'memory_by_delay': [round(float(max(s, 0)), 4) for s in scores],
        'integration_score': round(max(integration, 0.0), 4),
        'mean_activity': round(activity, 4), 'saturated_fraction': round(saturated, 4),
        'interpretation': 'Fixed recurrent weights from the measured graph with only the readout trained. Dynamics, scaling, '
                          'readout and tasks are assumptions; the graph and the transmitter calls are measured.'}


def paired(reference='measured', against=None, seeds=15, graph=None, **options):
    """Per-seed differences against one placement. Seeds fix the subsample, so runs are paired and the
    difference is the right statistic; overlapping error bars across conditions are not."""
    graph = graph or load_graph()
    against = against or [p for p in PLACEMENTS if p != reference]
    base = {seed: run(reference, graph=graph, seed=seed, **options)['memory_capacity'] for seed in range(7, 7 + seeds)}
    rows = {}
    for placement in against:
        differences = np.array([base[seed] - run(placement, graph=graph, seed=seed, **options)['memory_capacity']
                                for seed in range(7, 7 + seeds)])
        spread = float(differences.std(ddof=1)) if seeds > 1 else 0.0
        rows[placement] = {'mean_difference': round(float(differences.mean()), 4), 'sd': round(spread, 4),
                           'favours_reference': int((differences > 0).sum()), 'seeds': seeds,
                           't': round(float(differences.mean() / (spread / math.sqrt(seeds))), 2) if spread > 1e-9 else None}
    return {'reference': reference, 'options': options, 'rows': rows,
            'note': 'Positive means the reference placement stored more of the input stream. Paired by seed, since the seed '
                    'fixes which cells the subsample contains.'}


def suite(seeds=3, scaling='shared', **options):
    graph = load_graph()
    rows = {}
    for placement in PLACEMENTS:
        runs = [run(placement, graph=graph, seed=seed, scaling=scaling, **options) for seed in range(7, 7 + seeds)]
        rows[placement] = {'text': PLACEMENTS[placement],
                           'inhibitory_cells': runs[0]['inhibitory_cells'],
                           'natural_radius': round(float(np.mean([r['natural_radius'] for r in runs])), 4),
                           'effective_radius': round(float(np.mean([r['effective_radius'] for r in runs])), 4),
                           'memory_capacity': round(float(np.mean([r['memory_capacity'] for r in runs])), 3),
                           'memory_sd': round(float(np.std([r['memory_capacity'] for r in runs])), 3),
                           'integration_score': round(float(np.mean([r['integration_score'] for r in runs])), 4),
                           'mean_activity': round(float(np.mean([r['mean_activity'] for r in runs])), 4),
                           'saturated_fraction': round(float(np.mean([r['saturated_fraction'] for r in runs])), 4)}
    return {'seeds': seeds, 'scaling': scaling, 'scaling_text': SCALINGS[scaling], 'placements': PLACEMENTS, 'summary': rows}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--benchmark', action='store_true'); ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    if a.benchmark:
        result = {name: suite(a.seeds, scaling=name) for name in SCALINGS}
        (BASE / 'data/inhibition-benchmark.json').write_text(json.dumps(result, indent=1))
        print('Wrote data/inhibition-benchmark.json')
        for mode, block in result.items():
            print(mode)
            for name, row in block['summary'].items():
                print(f"  {name:15} radius {row['natural_radius']:7.3f}  memory {row['memory_capacity']:6.2f}  "
                      f"integration {row['integration_score']:.3f}  activity {row['mean_activity']:.3f}")
    else:
        for name, value in run().items():
            if name not in ('memory_by_delay', 'parameters', 'interpretation'): print(f'{name:22} {value}')
