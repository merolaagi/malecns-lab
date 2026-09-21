"""Build a reproducible, bounded MaleCNS steering/VNC circuit from official files."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pyarrow.feather as feather

BASE = Path(__file__).resolve().parent
SOURCE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = {'annotations': 'body-annotations-male-cns-v1.0-minconf-0.5.feather',
         'weights': 'connectome-weights-male-cns-v1.0-minconf-0.5.feather',
         'neurotransmitters': 'body-neurotransmitters-male-cns-v1.0.feather'}

def build(directory):
    ann = feather.read_table(directory / 'annotations.feather').to_pylist()
    ann = {r['bodyId']: r for r in ann if r['status'] == 'Traced'}
    table = feather.read_table(directory / 'weights.feather', memory_map=True)
    pre, post, weight = (table[k].to_numpy() for k in ['body_pre', 'body_post', 'weight'])
    nt = {r['body']: r for r in feather.read_table(directory / 'neurotransmitters.feather').to_pylist()}
    seeds = {i for i, r in ann.items() if r['type'] in ['DNa01', 'DNa02', 'DNg13']}
    intrinsic = {i for i, r in ann.items() if r['superclass'] == 'vnc_intrinsic'}
    motors = {i for i, r in ann.items() if r['superclass'] == 'vnc_motor' and r['subclass'] in ['fl', 'ml', 'hl']}
    eligible = np.isin(post, list(intrinsic | motors)) & (weight >= 5)
    selected = set(seeds)
    # Strongest 80 VNC targets per seed; selection uses anatomy, never simulated outcome.
    for seed in sorted(seeds):
        idx = np.flatnonzero(eligible & (pre == seed))
        idx = sorted(idx, key=lambda j: (-int(weight[j]), int(post[j])))[:80]
        selected.update(int(post[j]) for j in idx)
    # Include every leg motor target of the selected VNC/descending population at >=5 synapses.
    mask = np.isin(pre, list(selected)) & np.isin(post, list(motors)) & (weight >= 5)
    selected.update(int(i) for i in post[mask])
    sensory = {i for i, r in ann.items() if r['superclass'] == 'vnc_sensory'
               and r['subclass'] in ['chordotonal organ', 'campaniform sensilla']
               and r['entryNerve'] in ['ProLN', 'MesoLN', 'MetaLN']}
    mask = np.isin(pre, list(sensory)) & np.isin(post, list(selected)) & (weight >= 5)
    scores = {}
    for a, w in zip(pre[mask], weight[mask]):
        scores[int(a)] = scores.get(int(a), 0) + int(w)
    # Up to 20 sensory cells per leg; anatomical side and nerve select the feedback channel.
    for nerve in ['ProLN', 'MesoLN', 'MetaLN']:
        for side in ['L', 'R']:
            candidates = [i for i in scores if ann[i]['entryNerve'] == nerve and
                          (ann[i]['rootSide'] or ann[i]['somaSide']) == side]
            selected.update(sorted(candidates, key=lambda i: (-scores[i], i))[:20])
    ids = sorted(selected)
    nodes = []
    for i in ids:
        r, transmitter = ann[i], nt.get(i, {})
        nodes.append({k: r.get(k) for k in ['bodyId', 'type', 'instance', 'superclass', 'subclass',
                                          'somaSide', 'rootSide', 'entryNerve', 'exitNerve', 'status', 'somaLocation']}
                     | {'nt': transmitter.get('consensus_nt'), 'nt_confidence': transmitter.get('predicted_nt_confidence'),
                        'seed': i in seeds})
    # Retain ALL positive induced edges, including edges below the selection threshold.
    mask = np.isin(pre, ids) & np.isin(post, ids) & (weight > 0)
    edges = [[int(a), int(b), int(w)] for a, b, w in zip(pre[mask], post[mask], weight[mask])]
    hashes = {}
    for key, filename in FILES.items():
        h = hashlib.sha256()
        with (directory / (key + '.feather')).open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
        hashes[key] = {'url': SOURCE + filename, 'sha256': h.hexdigest()}
    result = {'dataset': 'male-cns:v1.0', 'license': 'CC-BY-4.0', 'built_at': datetime.now(timezone.utc).isoformat(),
              'source': 'https://male-cns.janelia.org/download/', 'files': hashes,
              'selection': 'Traced DNa01/DNa02/DNg13; top 80 VNC targets per seed (>=5 synapses); their leg motor targets (>=5); top 20 connected leg proprioceptive sensory cells per nerve/side. All positive induced edges retained.',
              'nodes': nodes, 'edges': edges}
    (BASE / 'data').mkdir(exist_ok=True)
    (BASE / 'data/circuit.json').write_text(json.dumps(result, separators=(',', ':')))
    print(json.dumps({'neurons': len(nodes), 'edges': len(edges), 'synapses': sum(e[2] for e in edges),
                      'seeds': sorted(seeds)}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    build(parser.parse_args().directory)
