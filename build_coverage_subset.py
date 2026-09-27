"""A subset chosen so its cells are actually observed, for the spectral question.

Every subset in this lab so far was cut by synapse count and cell type, which leaves cells whose entire
measured input comes from inside the subset only because the rest of their inputs were never included.
Under the lab's incoming-L1 normalisation those cells form stochastic blocks with an eigenvalue of one,
so `/spectrum` was measuring the cut, not the fly.

This builds the opposite kind of subset. Coverage is circular, since a cell's input coverage depends on
which other cells are present, so it proceeds outward:

  1. Seed with the lab's existing subsets.
  2. Read every edge whose target is in the current set, and compute each cell's coverage: the share of
     its true input synapses (from body-stats) that comes from inside the set.
  3. Add the strongest missing upstream partners of the worst-covered cells, up to a cell budget.
  4. Read the edges among the grown set, and keep only cells whose coverage now reaches the target.

The result is smaller and better observed: cells that stay are ones whose inputs are largely accounted
for, so their rows are not artefacts of where the subset was cut.

Run on a machine with the raw tables (two passes over weights.feather, a few minutes each):
    .venv/bin/python build_coverage_subset.py raw-data --target 0.5 --budget 3000
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.feather as feather

from model import BASE

SEEDS = ['data/circuit.json', 'data/learning-circuit.json', 'data/central-complex.json', 'data/vision-circuit.json']


def pick(schema, *names):
    for name in names:
        if name in schema: return name
    raise SystemExit(f'None of {names} in {schema}')


def seed_ids(paths=None):
    ids = set()
    for relative in (paths or SEEDS):
        path = BASE / relative
        if path.exists():
            ids |= {int(n['bodyId']) for n in json.loads(path.read_text())['nodes']}
    if not ids: raise SystemExit('No seed subsets found; build data/circuit.json first.')
    return ids


def input_totals(raw, ids):
    """True input synapse count per cell, from body-stats."""
    path = raw / 'body-stats.feather'
    schema = ds.dataset(path, format='ipc').schema.names
    body, post = pick(schema, 'body', 'bodyId'), pick(schema, 'post', 'PostSyn')
    table = ds.dataset(path, format='ipc').to_table(columns=[body, post],
                                                    filter=ds.field(body).isin(list(ids))).to_pydict()
    return {int(b): int(p) for b, p in zip(table[body], table[post]) if p}


def incoming(raw, targets, progress='pass'):
    """Every measured edge whose target is in `targets`, as {target: {source: synapses}}."""
    found = defaultdict(dict)
    wanted = np.array(sorted(targets))
    with pa.memory_map(str(raw / 'weights.feather'), 'r') as handle:
        reader = pa.ipc.open_file(handle)
        for k in range(reader.num_record_batches):
            batch = reader.get_batch(k)
            a = batch.column('body_pre').to_numpy(); b = batch.column('body_post').to_numpy(); w = batch.column('weight').to_numpy()
            mask = np.isin(b, wanted) & (w > 0)
            for source, target, weight in zip(a[mask], b[mask], w[mask]):
                found[int(target)][int(source)] = int(weight)
            if k % 600 == 0: print(f'  {progress}: batch {k}, targets covered {len(found)}', flush=True)
    return found


def coverage_of(found, totals, members):
    """Share of each cell's true input that comes from sources inside the subset.

    Only sources that are themselves included count: the matrix contains no other edges, so input from
    outside is input the model does not have."""
    members = set(members)
    return {cell: min(sum(w for source, w in sources.items() if source in members) / totals[cell], 1.0)
            for cell, sources in found.items() if totals.get(cell)}


def build(raw, target=0.5, budget=3000, out_path=None):
    ids = seed_ids()
    print(f'seed cells: {len(ids)}', flush=True)
    totals = input_totals(raw, ids)
    first = incoming(raw, ids, 'pass 1')
    coverage = coverage_of(first, totals, ids)
    print(f'seed coverage: median {np.median(list(coverage.values())):.3f}, '
          f'cells at or above target {sum(v >= target for v in coverage.values())}', flush=True)

    # Grow: add the strongest missing upstream partners of the worst-covered cells.
    wanted = set(ids)
    candidates = defaultdict(int)
    for cell, sources in first.items():
        if coverage.get(cell, 0.0) >= target: continue
        for source, weight in sources.items():
            if source not in wanted: candidates[source] += weight
    for source, _ in sorted(candidates.items(), key=lambda kv: -kv[1]):
        if len(wanted) >= budget: break
        wanted.add(source)
    print(f'grown to {len(wanted)} cells (added {len(wanted) - len(ids)})', flush=True)

    totals = input_totals(raw, wanted)
    second = incoming(raw, wanted, 'pass 2')
    coverage = coverage_of(second, totals, wanted)
    keep = sorted(cell for cell, value in coverage.items() if value >= target)
    # Dropping a cell removes its edges, which lowers its neighbours' coverage: iterate until stable.
    for _ in range(20):
        recomputed = coverage_of(second, totals, keep)
        shrunk = sorted(cell for cell in keep if recomputed.get(cell, 0.0) >= target)
        if len(shrunk) == len(keep): break
        keep = shrunk
    coverage = coverage_of(second, totals, keep)
    print(f'cells reaching {target:.0%} coverage: {len(keep)}', flush=True)
    if len(keep) < 20: raise SystemExit(f'Only {len(keep)} cells reach {target:.0%} coverage; lower --target or raise --budget.')

    keep_set = set(keep)
    edges = [[source, target_cell, weight]
             for target_cell, sources in second.items() if target_cell in keep_set
             for source, weight in sources.items() if source in keep_set]

    annotations = feather.read_table(raw / 'annotations.feather').to_pylist()
    info = {int(r['bodyId']): r for r in annotations if int(r['bodyId']) in keep_set}
    transmitters = {}
    nt_path = raw / 'neurotransmitters.feather'
    if nt_path.exists():
        rows = feather.read_table(nt_path).to_pylist()
        id_key = next((k for k in ('body', 'bodyId') if rows and k in rows[0]), None)
        nt_keys = [k for k in ('consensus_nt', 'predicted_nt', 'nt') if rows and k in rows[0]]
        if id_key and nt_keys:
            for r in rows:
                body = int(r[id_key])
                if body in keep_set: transmitters[body] = next((r[k] for k in nt_keys if r.get(k)), None)

    nodes = [{'bodyId': cell, 'type': (info.get(cell) or {}).get('type'),
              'instance': (info.get(cell) or {}).get('instance'),
              'superclass': (info.get(cell) or {}).get('superclass'),
              'somaSide': (info.get(cell) or {}).get('somaSide'),
              'nt': transmitters.get(cell),
              'coverage_in': round(coverage[cell], 4),
              'input_synapses': totals.get(cell, 0)} for cell in keep]
    covered = [n['coverage_in'] for n in nodes]
    out = {'dataset': 'male-cns:v1.0', 'license': 'CC-BY-4.0', 'target_coverage': target, 'budget': budget,
           'nodes': nodes, 'edges': edges,
           'summary': {'cells': len(nodes), 'edges': len(edges),
                       'median_coverage': round(float(np.median(covered)), 4),
                       'min_coverage': round(float(np.min(covered)), 4),
                       'mean_input_synapses': round(float(np.mean([n['input_synapses'] for n in nodes])), 1)},
           'scope': 'Cells whose measured input inside this subset reaches the target share of their true input synapse '
                    'count. Selected for observability, not by synapse threshold or cell type, so the spectral question '
                    'is not decided by where the subset was cut. Transmitter calls and the sign convention are unchanged.'}
    path = Path(out_path or BASE / 'data/coverage-subset.json')
    path.write_text(json.dumps(out, separators=(',', ':')))
    print('DONE', json.dumps(out['summary']), flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('directory', type=Path, nargs='?', default=Path('raw-data'))
    ap.add_argument('--target', type=float, default=0.5, help='required share of each cell\'s true input')
    ap.add_argument('--budget', type=int, default=3000, help='maximum cells to grow to')
    a = ap.parse_args()
    build(a.directory, a.target, a.budget)
