"""Data-quality layer for every cell in the lab's two circuits.

Inputs (official MaleCNS v1.0 tables, see README):
  body-stats.feather                synapse totals for every body: pre, post, downstream
  tbar-neurotransmitters.feather    transmitter probabilities for every pre-synapse (T-bar)

For each cell in data/circuit.json and data/learning-circuit.json this computes:
  coverage   fraction of the cell's input (and output) connections that the lab's subset contains
  nt         per-synapse transmitter evidence: mean probability per transmitter, the share of the
             cell's synapses whose most likely transmitter is each one, and a sign probability

Sign probability uses the same rule as the models (acetylcholine +, GABA and glutamate -, others 0):
p_positive = mean P(acetylcholine); p_negative = mean P(GABA) + mean P(glutamate). Transmitter
predictions come from EM appearance; receptor identity, and therefore glutamate's real sign, is not
measured. Nothing here changes the models; it reports how much each result can trust its inputs.
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

BASE = Path(__file__).resolve().parent
SOURCE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'
FILES = {'body-stats': 'body-stats-male-cns-v1.0-minconf-0.5.feather',
         'tbar-neurotransmitters': 'tbar-neurotransmitters-male-cns-v1.0.feather'}
POSITIVE, NEGATIVE = {'acetylcholine'}, {'gaba', 'glutamate'}


def pick(columns, *names):
    for n in names:
        if n in columns: return n
    raise SystemExit(f'None of {names} found; columns are {columns}')


def read_filtered(path, ids, columns, body_col):
    """Stream only the rows for our bodies from a large Arrow/Feather file."""
    d = ds.dataset(path, format='ipc')
    typ = d.schema.field(body_col).type
    wanted = pa.array(sorted(ids), type=typ)
    return d.to_table(columns=columns, filter=pc.field(body_col).isin(wanted))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def captured(circuit):
    ins, outs = defaultdict(int), defaultdict(int)
    for a, b, w in circuit['edges']:
        outs[a] += w; ins[b] += w
    return ins, outs


def build(raw, out_path, circuits=None):
    circuits = circuits or {'locomotion': BASE / 'data/circuit.json', 'learning': BASE / 'data/learning-circuit.json'}
    data = {k: json.loads(Path(v).read_text()) for k, v in circuits.items()}
    ids = sorted({n['bodyId'] for c in data.values() for n in c['nodes']})

    stats_path, tbar_path = raw / 'body-stats.feather', raw / 'tbar-neurotransmitters.feather'
    sschema = ds.dataset(stats_path, format='ipc').schema.names
    sbody, spost = pick(sschema, 'body', 'bodyId'), pick(sschema, 'post', 'PostSyn')
    spre, sdown = pick(sschema, 'pre', 'PreSyn'), pick(sschema, 'downstream', 'OutputPartners')
    st = read_filtered(stats_path, ids, [sbody, spre, spost, sdown], sbody).to_pydict()
    stats = {int(b): (int(pr), int(po), int(dn)) for b, pr, po, dn in zip(st[sbody], st[spre], st[spost], st[sdown])}

    tschema = ds.dataset(tbar_path, format='ipc').schema.names
    tbody = pick(tschema, 'body', 'bodyId')
    ntcols = [c for c in tschema if re.fullmatch(r'nt_(.+)_prob', c)]
    if not ntcols: raise SystemExit(f'No nt_*_prob columns in {tbar_path}; columns are {tschema}')
    names = [re.fullmatch(r'nt_(.+)_prob', c).group(1) for c in ntcols]
    tb = read_filtered(tbar_path, ids, [tbody, *ntcols], tbody)
    body = tb.column(tbody).to_numpy()
    probs = np.stack([np.nan_to_num(tb.column(c).to_numpy(zero_copy_only=False).astype(float)) for c in ntcols], axis=1)
    order = np.argsort(body, kind='stable'); body, probs = body[order], probs[order]
    starts = np.flatnonzero(np.r_[True, body[1:] != body[:-1]]) if len(body) else np.array([], int)
    ends = np.r_[starts[1:], len(body)] if len(body) else np.array([], int)
    pos_idx = [i for i, n in enumerate(names) if n in POSITIVE]
    neg_idx = [i for i, n in enumerate(names) if n in NEGATIVE]
    nt = {}
    for s, e in zip(starts, ends):
        p = probs[s:e]
        mean = p.mean(axis=0)
        top = np.bincount(p.argmax(axis=1), minlength=len(names)) / len(p)
        nt[int(body[s])] = {'tbars': int(e - s),
                            'mean_prob': {n: round(float(v), 4) for n, v in zip(names, mean)},
                            'top_share': {n: round(float(v), 4) for n, v in zip(names, top) if v > 0},
                            'dominant': names[int(top.argmax())], 'dominant_share': round(float(top.max()), 4),
                            'p_positive': round(float(mean[pos_idx].sum()), 4), 'p_negative': round(float(mean[neg_idx].sum()), 4)}

    cells, summary = {}, {}
    for key, c in data.items():
        ins, outs = captured(c)
        layer_of = (lambda n: n.get('superclass')) if key == 'locomotion' else (lambda n: n.get('role'))
        by_layer = defaultdict(list)
        for n in c['nodes']:
            b = n['bodyId']
            pre, post, down = stats.get(b, (None, None, None))
            cov_in = ins[b] / post if post else None
            cov_out = outs[b] / down if down else None
            rec = {'circuit': key, 'layer': layer_of(n), 'post_total': post, 'pre_total': pre, 'downstream_total': down,
                   'captured_in': ins[b], 'captured_out': outs[b],
                   'coverage_in': None if cov_in is None else round(min(cov_in, 1.0), 4),
                   'coverage_out': None if cov_out is None else round(min(cov_out, 1.0), 4),
                   'aggregate_nt': n.get('nt'), 'nt': nt.get(b)}
            cells[str(b)] = rec
            by_layer[rec['layer']].append(rec)
        summary[key] = {}
        for layer, recs in by_layer.items():
            cov = [r['coverage_in'] for r in recs if r['coverage_in'] is not None]
            unclear = [r for r in recs if r['aggregate_nt'] in (None, 'unclear', 'unknown')]
            summary[key][layer] = {
                'cells': len(recs), 'median_coverage_in': None if not cov else round(float(np.median(cov)), 4),
                'cells_with_synapse_nt': sum(r['nt'] is not None for r in recs),
                'aggregate_unclear': len(unclear),
                'unclear_with_dominant_share_over_0.6': sum(1 for r in unclear if r['nt'] and r['nt']['dominant_share'] > .6),
                'mixed_cells': sum(1 for r in recs if r['nt'] and r['nt']['dominant_share'] < .6)}
    result = {'dataset': 'male-cns:v1.0', 'license': 'CC-BY-4.0', 'built_at': datetime.now(timezone.utc).isoformat(),
              'files': {k: {'url': SOURCE + v, 'sha256': sha256(raw / (k + '.feather'))} for k, v in FILES.items()},
              'transmitters': names, 'method': __doc__.strip(), 'summary': summary, 'cells': cells}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, separators=(',', ':')))
    print(json.dumps(summary, indent=1))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('directory', type=Path, help='Folder containing body-stats.feather and tbar-neurotransmitters.feather')
    ap.add_argument('--out', type=Path, default=BASE / 'data/quality.json')
    a = ap.parse_args()
    build(a.directory, a.out)
