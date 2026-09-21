"""Measured synaptic inputs for one cell, for the single-neuron workbench.

Returns only anatomy plus the sign rule the network models already use. The workbench's
biophysics (Hodgkin-Huxley kinetics, two compartments, input firing) is assumed and lives in
neuron-core.js.
"""
import json
from collections import defaultdict
from model import BASE
from model import Circuit
from learning import LearningCircuit
from numerosity import DEFAULT as NUM_DEFAULT, Numerosity

SIGN = {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1}


class NeuronIndex:
    def __init__(self, locomotion=None, learning=None, quality_path=None):
        qp = quality_path or BASE / 'data/quality.json'
        self.quality = json.loads(qp.read_text())['cells'] if qp.exists() else {}
        self.loco = locomotion or Circuit()
        self.learn = learning or LearningCircuit()
        self.nodes = {'locomotion': {n['bodyId']: n for n in self.loco.nodes},
                      'learning': {n['bodyId']: n for n in self.learn.data['nodes']}}
        self.inputs = {'locomotion': defaultdict(list), 'learning': defaultdict(list)}
        self.outputs = {'locomotion': defaultdict(int), 'learning': defaultdict(int)}
        for key, edges in [('locomotion', self.loco.data['edges']), ('learning', self.learn.data['edges'])]:
            for a, b, w in edges:
                self.inputs[key][b].append((a, w))
                self.outputs[key][a] += 1
        self.kcs = sorted(n['bodyId'] for n in self.learn.data['nodes'] if n['role'] == 'KC')
        self.pn_index = self.learn.maps['PN']

    def describe(self, circuit, body_id):
        if circuit not in self.nodes: raise ValueError('circuit must be locomotion or learning')
        try: body_id = int(body_id)
        except (TypeError, ValueError): raise ValueError('id must be an integer body ID')
        node = self.nodes[circuit].get(body_id)
        if node is None: raise ValueError(f'No cell {body_id} in the {circuit} circuit')
        nodes = self.nodes[circuit]
        channels, modulatory = [], []
        for pre, w in sorted(self.inputs[circuit][body_id], key=lambda x: -x[1]):
            p = nodes[pre]
            if circuit == 'learning':
                if p['role'] == 'PAM':
                    modulatory.append({'bodyId': pre, 'name': p['instance'], 'synapses': w}); continue
                qual = self.quality.get(str(pre)) or {}
                if qual.get('consensus_nt') not in (None, 'unclear', 'unknown'):   # per-body consensus from build_quality.py
                    nt = qual['consensus_nt']; sign = SIGN.get(nt, 0)
                    note = f'Sign from the MaleCNS consensus transmitter ({nt}).'
                else:   # the learning dataset has no transmitter field
                    sign, nt, note = 1, None, 'Transmitter not in the learning dataset; assumed excitatory.'
            else:
                nt = p.get('nt'); sign = SIGN.get(nt, 0)
                note = None if sign else 'Unclear transmitter: carries zero weight in the network model.'
            qual = self.quality.get(str(pre)) or {}
            channels.append({'synapse_nt': qual.get('nt'), 'coverage_in': qual.get('coverage_in'), 'bodyId': pre, 'name': p.get('instance') or p.get('type'), 'type': p.get('type'),
                             'class': p.get('superclass') or p.get('role'), 'synapses': w, 'sign': sign, 'nt': nt, 'note': note})
        out = {'circuit': circuit, 'cell': node, 'inputs': channels, 'modulatory': modulatory,
               'output_partners': self.outputs[circuit][body_id], 'quality': self.quality.get(str(body_id))}
        if circuit == 'learning' and node.get('role') == 'KC':
            out['number_codes'] = self.number_codes(channels)
        return out

    def number_codes(self, channels):
        """Which of this KC's PN inputs each number activates under the default scalar and one-hot codes."""
        exp = Numerosity(self.learn)
        result = {}
        for cond in ['scalar', 'onehot']:
            p = exp.validate({'condition': cond, 'seed': NUM_DEFAULT['seed']})
            codes = [set(c.tolist()) for c in exp.codes(p)]
            result[cond] = [[i for i, ch in enumerate(channels) if self.pn_index.get(ch['bodyId']) in code] for code in codes]
        return {'seed': NUM_DEFAULT['seed'], 'codes': result}

    def random_kc(self, seed):
        return self.kcs[int(seed) % len(self.kcs)]
