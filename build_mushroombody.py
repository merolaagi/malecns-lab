"""The mushroom body as a memory circuit: every cell type the biology needs, with compartments.

The lab's learning subset has projection neurons, Kenyon cells, PAM dopamine neurons and output
neurons, but only three kinds of edge (PN to KC, PAM to KC, KC to MBON). Memory needs more:

  PPL1 dopamine neurons   the aversive teaching signal; without them punishment cannot be learned
  APL                     the inhibitory neuron that keeps Kenyon-cell codes sparse
  DPM                     implicated in consolidating long-term memory
  MBON to DAN loops       extinction and second-order conditioning run through these
  MBON to MBON            output neurons that inhibit each other
  MBON output targets     where memory leaves the mushroom body on its way to behaviour

Compartment identity is parsed from instance names. From the lab's own data:
  MBON01(y5B'2a)_R        dendrites in gamma5 and beta'2a
  MBON11(y1pedc>a/B)_L    dendrites in gamma1-pedunculus, axon to the alpha/beta lobes
  PAM07(y4<y1y2)_L        teaches in gamma4, receives input in gamma1-gamma2

Run on a machine with the raw tables:
    .venv/bin/python build_mushroombody.py raw-data --probe     # what the annotations hold
    .venv/bin/python build_mushroombody.py raw-data             # writes data/mushroom-body.json
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

from model import BASE

# PPL2 cells are dopaminergic but innervate the calyx and lateral horn, not the lobe compartments where
# associative memory is written, so they get their own role and never count as compartment teachers.
ROLES = [('KC', 'KC'), ('MBON', 'MBON'), ('PAM', 'DAN'), ('PPL1', 'DAN'), ('PPL2', 'DAN_calyx'), ('DPM', 'DPM'), ('APL', 'APL')]
# Instance names write beta both as 'B' and 'b' (MBON26(b'2d)), and the pedunculus as 'pedc' or 'ped'.
LOBE = {'y': 'γ', "a'": "α'", "B'": "β'", "b'": "β'", 'a': 'α', 'B': 'β', 'b': 'β'}
# A trailing letter is a sub-compartment ("B'2a") only when no digit follows it; otherwise it starts the
# next compartment ("y1y2" is gamma1 and gamma2, not "gamma1y").
COMPARTMENT = re.compile(r"(a'|B'|b'|a|B|b|y)(\d)(pedc|ped|[a-z](?!\d)(?!'))?")
LOBE_ONLY = re.compile(r"(?<![\w'])(a'|B'|b'|a|B|b|y)(?![\d'\w])")
MBON_TARGETS = 200          # strongest downstream partners of the MBONs, where memory leaves the MB


def role_of(type_name):
    if not type_name: return None
    name = str(type_name)
    for prefix, role in ROLES:
        if name.startswith(prefix): return role
    return None


def kc_class(type_name):
    name = str(type_name or '')
    if name.startswith("KCa'b'"): return "α'/β'"
    if name.startswith('KCab'): return 'α/β'
    if name.startswith('KCg'): return 'γ'
    return None


def compartments(text):
    """Normalised compartments in one parenthetical field, e.g. "y5B'2a" -> ['γ5', "β'2a"]."""
    found = []
    for lobe, number, suffix in COMPARTMENT.findall(text or ''):
        suffix = 'pedc' if suffix == 'ped' else suffix
        found.append(f'{LOBE[lobe]}{number}{suffix or ""}')
    if not found:
        found = [f'{LOBE[lobe]} lobe' for lobe in LOBE_ONLY.findall(text or '')]
    return found


def parse_instance(instance, role):
    """(input compartments, output compartments) from an instance name.

    MBONs write dendrites first and axons after '>'. Dopamine neurons write their teaching (axonal)
    compartment first and where they receive input after '<'. Cells without a parenthetical get none."""
    match = re.search(r'\(([^)]*)\)', str(instance or ''))
    if not match or role not in ('MBON', 'DAN'): return [], []
    field = match.group(1)
    if role == 'MBON':
        dendrites, _, axon = field.partition('>')
        return compartments(dendrites), compartments(axon)
    teaching, _, receiving = field.partition('<')
    return compartments(receiving), compartments(teaching)


def annotations(raw):
    return feather.read_table(raw / 'annotations.feather').to_pylist()


def probe(raw):
    rows = annotations(raw)
    counts, samples, parsed = Counter(), defaultdict(list), Counter()
    for r in rows:
        role = role_of(r.get('type'))
        if not role: continue
        counts[(role, str(r.get('type'))[:6], r.get('status'))] += 1
        samples[role].append((r.get('bodyId'), r.get('type'), r.get('instance')))
        inputs, outputs = parse_instance(r.get('instance'), role)
        if role in ('MBON', 'DAN'): parsed[(role, bool(inputs or outputs))] += 1
    by_role = Counter()
    for (role, _, status), n in counts.items():
        if status == 'Traced': by_role[role] += n
    print('traced cells by role:', dict(by_role))
    print('\nby type prefix and status:')
    for (role, prefix, status), n in sorted(counts.items()):
        print(f'  {role:5} {prefix:7} {status!s:12} {n}')
    print('\nsample instances with parsed compartments (input -> output):')
    for role, rows_ in samples.items():
        seen = set()
        for body, type_name, instance in rows_:
            if type_name in seen: continue
            seen.add(type_name)
            inputs, outputs = parse_instance(instance, role)
            print(f'  {role:5} {body} {str(type_name):12} {str(instance):32} {inputs} -> {outputs}')
            if len(seen) >= (10 if role in ('MBON', 'DAN') else 3): break
    for role in ('MBON', 'DAN'):
        ok, total = parsed[(role, True)], parsed[(role, True)] + parsed[(role, False)]
        print(f'{role}: compartment parsed for {ok} of {total}')


def build(raw):
    rows = annotations(raw)
    pn_seed = set()
    learning = BASE / 'data/learning-circuit.json'
    if learning.exists():
        pn_seed = {int(n['bodyId']) for n in json.loads(learning.read_text())['nodes'] if n.get('role') == 'PN'}
    nodes = {}
    for r in rows:
        if r.get('status') != 'Traced': continue
        body = int(r['bodyId'])
        role = role_of(r.get('type')) or ('PN' if body in pn_seed else None)
        if not role: continue
        inputs, outputs = parse_instance(r.get('instance'), role)
        nodes[body] = {'bodyId': body, 'type': r.get('type'), 'instance': r.get('instance'), 'somaSide': r.get('somaSide'),
                       'role': role, 'kc_class': kc_class(r.get('type')) if role == 'KC' else None,
                       'compartments_in': inputs, 'compartments_out': outputs}
    ids = np.array(sorted(nodes))
    mbons = np.array(sorted(b for b, n in nodes.items() if n['role'] == 'MBON'))

    edges, kc_kc, downstream = [], 0, Counter()
    with pa.memory_map(str(raw / 'weights.feather'), 'r') as handle:
        reader = pa.ipc.open_file(handle)
        for k in range(reader.num_record_batches):
            batch = reader.get_batch(k)
            a = batch.column('body_pre').to_numpy(); b = batch.column('body_post').to_numpy(); w = batch.column('weight').to_numpy()
            inside = np.isin(a, ids) & np.isin(b, ids) & (w > 0)
            for x, y, z in zip(a[inside], b[inside], w[inside]):
                if nodes[int(x)]['role'] == 'KC' and nodes[int(y)]['role'] == 'KC':
                    kc_kc += int(z); continue                   # summarised, not stored: there are very many
                edges.append([int(x), int(y), int(z)])
            leaving = np.isin(a, mbons) & ~np.isin(b, ids) & (w > 0)
            for y, z in zip(b[leaving], w[leaving]): downstream[int(y)] += int(z)
            if k % 600 == 0: print('batch', k, 'edges', len(edges), flush=True)

    # Where memory leaves the mushroom body: the strongest downstream partners of the output neurons.
    targets = [body for body, _ in downstream.most_common(MBON_TARGETS)]
    info = {int(r['bodyId']): r for r in rows if int(r['bodyId']) in set(targets)}
    for body in targets:
        r = info.get(body, {})
        nodes[body] = {'bodyId': body, 'type': r.get('type'), 'instance': r.get('instance'), 'somaSide': r.get('somaSide'),
                       'role': 'MBON_target', 'kc_class': None, 'compartments_in': [], 'compartments_out': []}
    target_set = set(targets)
    with pa.memory_map(str(raw / 'weights.feather'), 'r') as handle:
        reader = pa.ipc.open_file(handle)
        for k in range(reader.num_record_batches):
            batch = reader.get_batch(k)
            a = batch.column('body_pre').to_numpy(); b = batch.column('body_post').to_numpy(); w = batch.column('weight').to_numpy()
            mask = np.isin(a, mbons) & np.isin(b, list(target_set)) & (w > 0)
            edges.extend([[int(x), int(y), int(z)] for x, y, z in zip(a[mask], b[mask], w[mask])])

    transmitters = {}
    try:
        path = raw / 'neurotransmitters.feather'
        if path.exists():
            table = feather.read_table(path).to_pylist()
            id_key = next((k for k in ('body', 'bodyId') if table and k in table[0]), None)
            nt_keys = [k for k in ('consensus_nt', 'predicted_nt', 'nt') if table and k in table[0]]
            if id_key and nt_keys:
                for r in table:
                    body = int(r[id_key])
                    if body in nodes: transmitters[body] = next((r[k] for k in nt_keys if r.get(k)), None)
    except Exception as error:
        print('transmitters skipped:', error, flush=True)
    for body, n in nodes.items(): n['nt'] = transmitters.get(body)

    infer_compartments(nodes, edges)
    role_of_body = {b: n['role'] for b, n in nodes.items()}
    kinds = Counter(f'{role_of_body[a]}->{role_of_body[b]}' for a, b, _ in edges)
    out = {'dataset': 'male-cns:v1.0', 'license': 'CC-BY-4.0', 'nodes': list(nodes.values()), 'edges': edges,
           'kc_to_kc_synapses': kc_kc, 'edge_kinds': dict(kinds),
           'roles': dict(Counter(n['role'] for n in nodes.values())),
           'compartments': compartment_table(nodes),
           'scope': 'Traced mushroom-body cells by type prefix (KC, MBON, PAM, PPL1, PPL2, DPM, APL), the projection neurons '
                    'from the learning subset, and the strongest downstream partners of the output neurons. Compartments are '
                    'parsed from instance names. KC-to-KC edges are summarised as a total rather than stored.'}
    (BASE / 'data/mushroom-body.json').write_text(json.dumps(out, separators=(',', ':')))
    print('DONE', json.dumps({'roles': out['roles'], 'edges': len(edges), 'edge_kinds': dict(kinds.most_common(12)),
                              'compartments': len(out['compartments'])}), flush=True)


def infer_compartments(nodes, edges, minimum_overlap=0.05):
    """Place dopamine and output neurons whose names carry no compartment, from their Kenyon-cell partners.

    Within a compartment, the dopamine neurons and the output neurons contact the same stretch of Kenyon
    cell axons, so the Kenyon cells a dopamine neuron targets overlap with those that feed that compartment's
    output neurons. Each unnamed cell is assigned the compartment with the largest overlap (Jaccard), and the
    assignment is marked as inferred so it is never mistaken for an annotation."""
    kc_out, kc_in = defaultdict(set), defaultdict(set)
    for a, b, _ in edges:
        if nodes[a]['role'] == 'KC' and nodes[b]['role'] == 'MBON': kc_in[b].add(a)
        if nodes[a]['role'] == 'DAN' and nodes[b]['role'] == 'KC': kc_out[a].add(b)
    by_compartment = defaultdict(set)
    for body, n in nodes.items():
        if n['role'] == 'MBON':
            for c in n['compartments_in']: by_compartment[c] |= kc_in[body]
        if n['role'] == 'DAN':
            for c in n['compartments_out']: by_compartment[c] |= kc_out[body]
    def best(kcs):
        scores = {c: len(kcs & members) / max(len(kcs | members), 1) for c, members in by_compartment.items() if members}
        if not scores: return None, 0.0
        compartment = max(scores, key=scores.get)
        return compartment, scores[compartment]
    for body, n in nodes.items():
        n['compartment_source'] = 'name' if (n['compartments_in'] or n['compartments_out']) else None
        if n['role'] == 'DAN' and not n['compartments_out'] and kc_out[body]:
            compartment, score = best(kc_out[body])
            if compartment and score >= minimum_overlap:
                n['compartments_out'] = [compartment]; n['compartment_source'] = f'inferred (overlap {score:.2f})'
        if n['role'] == 'MBON' and not n['compartments_in'] and kc_in[body]:
            compartment, score = best(kc_in[body])
            if compartment and score >= minimum_overlap:
                n['compartments_in'] = [compartment]; n['compartment_source'] = f'inferred (overlap {score:.2f})'


def compartment_table(nodes):
    """For each compartment: which dopamine neurons teach there and which output neurons read it."""
    table = defaultdict(lambda: {'teachers': Counter(), 'readers': Counter()})
    for n in nodes.values():
        if n['role'] == 'DAN':
            for c in n['compartments_out']: table[c]['teachers'][n['type']] += 1
        if n['role'] == 'MBON':
            for c in n['compartments_in']: table[c]['readers'][n['type']] += 1
    return {c: {'teachers': dict(v['teachers']), 'readers': dict(v['readers'])} for c, v in sorted(table.items())}


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    raw = Path(args[0] if args else 'raw-data')
    (probe if '--probe' in sys.argv else build)(raw)
