"""Brain-region (neuropil) connectivity for MaleCNS v1.0 from neuPrint.

Needs your neuPrint token in the environment (never commit it; the repo is public):
    export NEUPRINT_APPLICATION_CREDENTIALS='<token from your neuPrint account page>'
    .venv/bin/python build_regions.py

Measured (from neuPrint's per-neuron roiInfo): for every traced neuron, how many of its input
connections (post) and output connections (downstream) fall in each primary region.

Derived, not measured: region-to-region flow. A neuron that receives a fraction f of its input in
region A and makes D output connections in region B contributes f*D to flow A -> B. Summed over all
traced neurons, this says how much output in B is driven by neurons that listen in A. It routes
through neurons; it is not a count of synapses between regions (synapses sit inside one region).
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
SERVER, DATASET = 'neuprint.janelia.org', 'male-cns:v1.0'
PAGE = 10000


def neuprint_backend(server=SERVER, dataset=DATASET):
    from neuprint import Client, fetch_custom, fetch_primary_rois, fetch_roi_hierarchy
    client = Client(server, dataset=dataset)   # reads NEUPRINT_APPLICATION_CREDENTIALS
    return {'query': lambda cypher: fetch_custom(cypher, client=client),
            'primary_rois': lambda: list(fetch_primary_rois(client=client)),
            'hierarchy': lambda: fetch_roi_hierarchy(include_subprimary=False, mark_primary=True, format='dict', client=client)}


def traced_roi_info(query, page=PAGE):
    """Yield (bodyId, roiInfo dict) for every traced neuron, keyset-paginated by bodyId."""
    last = -1
    while True:
        df = query(f"MATCH (n:Neuron) WHERE n.status = 'Traced' AND n.bodyId > {int(last)} "
                   f"RETURN n.bodyId AS bodyId, n.roiInfo AS roiInfo ORDER BY n.bodyId LIMIT {int(page)}")
        ids, infos = list(df['bodyId']), list(df['roiInfo'])
        if not ids: return
        for b, info in zip(ids, infos):
            yield int(b), (json.loads(info) if isinstance(info, str) else (info or {}))
        last = int(ids[-1])
        print(f'  read through body {last}', flush=True)
        if len(ids) < page: return


def lab_cells(paths):
    cells = {}
    for key, p in paths.items():
        for n in json.loads(Path(p).read_text())['nodes']:
            cells[int(n['bodyId'])] = key
    return cells


def build(backend, out_path, circuits=None):
    circuits = circuits or {'locomotion': BASE / 'data/circuit.json', 'learning': BASE / 'data/learning-circuit.json'}
    wanted = lab_cells(circuits)
    rois = backend['primary_rois']()
    idx = {r: i for i, r in enumerate(rois)}
    R = len(rois)
    flow = np.zeros((R, R))
    post_tot, down_tot, neurons_in = np.zeros(R), np.zeros(R), np.zeros(R, int)
    profiles, n_traced = {}, 0
    for body, info in traced_roi_info(backend['query']):
        n_traced += 1
        post = np.zeros(R); down = np.zeros(R)
        for roi, c in info.items():
            i = idx.get(roi)
            if i is None: continue                    # non-primary (super- or sub-) regions would double count
            post[i] = c.get('post', 0) or 0
            down[i] = c.get('downstream', 0) or 0
        post_tot += post; down_tot += down; neurons_in += post > 0
        if post.sum() > 0 and down.sum() > 0:
            flow += np.outer(post / post.sum(), down)
        if body in wanted:
            def top(v):
                s = v.sum()
                return [{'roi': rois[i], 'count': int(v[i]), 'fraction': round(float(v[i] / s), 4)}
                        for i in np.argsort(-v)[:8] if v[i] > 0] if s else []
            profiles[str(body)] = {'circuit': wanted[body], 'input': top(post), 'output': top(down),
                                   'input_total': int(post.sum()), 'output_total': int(down.sum())}
    keep = [i for i in range(R) if post_tot[i] > 0 or down_tot[i] > 0]
    sub = flow[np.ix_(keep, keep)]
    result = {
        'dataset': DATASET, 'server': SERVER, 'license': 'CC-BY-4.0', 'built_at': datetime.now(timezone.utc).isoformat(),
        'method': __doc__.strip(), 'traced_neurons': n_traced,
        'rois': [rois[i] for i in keep], 'hierarchy': backend['hierarchy'](),
        'roi_totals': [{'roi': rois[i], 'input_connections': int(post_tot[i]), 'output_connections': int(down_tot[i]),
                        'neurons_with_input': int(neurons_in[i])} for i in keep],
        'flow': np.round(sub, 2).tolist(),
        'lab_cells_found': len(profiles), 'lab_cells_total': len(wanted), 'profiles': profiles}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, separators=(',', ':')))
    strongest = sorted(((sub[a, b], result['rois'][a], result['rois'][b]) for a in range(len(keep)) for b in range(len(keep)) if a != b), reverse=True)[:10]
    print(json.dumps({'traced_neurons': n_traced, 'regions': len(keep), 'lab_cells_found': f"{len(profiles)}/{len(wanted)}",
                      'strongest_flows': [f'{a} -> {b}: {v:,.0f}' for v, a, b in strongest]}, indent=1))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=BASE / 'data/regions.json')
    a = ap.parse_args()
    build(neuprint_backend(), a.out)
