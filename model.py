"""Experimental LIF circuit coupled to an explicitly engineered 2D body proxy.

Anatomical edges are measured. Every dynamic parameter and body mapping is a
model assumption. This is not a validated emulation, muscle model, or gait model.
"""
import json
import math
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

BASE = Path(__file__).resolve().parent
DT = .001
LEGS = ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']
DEFAULT = dict(duration=8, drive=2.0, bias=0., gain=10., feedback=2.,
               condition='intact', seed=7, mode='tonic')
CONDITIONS = ['intact', 'silence_vnc', 'silence_left_dn', 'no_feedback', 'no_stimulus', 'shuffled']

def leg_for(n):
    side = n.get('rootSide') or n.get('somaSide')
    part = {'fl': 'F', 'ml': 'M', 'hl': 'H'}.get(n.get('subclass'))
    if n['superclass'] == 'vnc_sensory':
        part = {'ProLN': 'F', 'MesoLN': 'M', 'MetaLN': 'H'}.get(n.get('entryNerve'))
    label = str(side) + str(part)
    return LEGS.index(label) if label in LEGS else -1

class Circuit:
    def __init__(self, path=None):
        self.data = json.loads(Path(path or BASE / 'data/circuit.json').read_text())
        self.nodes = self.data['nodes']
        self.n = len(self.nodes)
        index = {n['bodyId']: i for i, n in enumerate(self.nodes)}
        edges = self.data['edges']
        self.pre = np.array([index[e[0]] for e in edges])
        self.post = np.array([index[e[1]] for e in edges])
        self.count = np.array([e[2] for e in edges], dtype=float)
        # Glutamate is assumed inhibitory inside this CNS circuit; receptor-specific exceptions unknown.
        self.sign = np.array([{'acetylcholine': 1., 'gaba': -1., 'glutamate': -1.}.get(n['nt'], 0.) for n in self.nodes])
        self.dn = np.array([n['seed'] for n in self.nodes])
        self.vnc = np.array([n['superclass'] == 'vnc_intrinsic' for n in self.nodes])
        self.side = np.array([-1 if (n['rootSide'] or n['somaSide']) == 'L' else
                              1 if (n['rootSide'] or n['somaSide']) == 'R' else 0 for n in self.nodes])
        self.leg = np.array([leg_for(n) for n in self.nodes])
        self.motor = [np.array([i for i, n in enumerate(self.nodes) if n['superclass'] == 'vnc_motor' and self.leg[i] == k], dtype=int) for k in range(6)]
        self.sensory = [np.array([i for i, n in enumerate(self.nodes) if n['superclass'] == 'vnc_sensory' and self.leg[i] == k], dtype=int) for k in range(6)]

    def matrix(self, condition, seed):
        post = self.post.copy()
        if condition == 'shuffled':
            # Permute destinations across existing edges: preserve source out-degree,
            # destination in-degree multiplicity and source-associated weights/signs.
            post = np.random.default_rng(seed + 10000).permutation(post)
        values = np.log1p(self.count) * self.sign[self.pre]
        # Incoming L1 normalization prevents synapse count from being treated as conductance.
        denom = np.bincount(post, weights=np.abs(values), minlength=self.n)
        return csr_matrix((values / np.maximum(denom[post], 1), (post, self.pre)), shape=(self.n, self.n))

    def simulate(self, **options):
        p = DEFAULT | options
        if p['condition'] not in CONDITIONS: raise ValueError('Unknown condition')
        if p['mode'] not in ['tonic', 'pulse', 'target']: raise ValueError('Unknown stimulus mode')
        bounds = {'duration': (.5, 20), 'drive': (0, 5), 'bias': (-1, 1), 'gain': (0, 20), 'feedback': (0, 2)}
        for key, (lo, hi) in bounds.items():
            p[key] = float(p[key])
            if not math.isfinite(p[key]) or not lo <= p[key] <= hi: raise ValueError(f'{key} must be between {lo} and {hi}')
        p['seed'] = int(p['seed'])
        rng = np.random.default_rng(p['seed'])
        w = self.matrix(p['condition'], p['seed'])
        v = rng.uniform(0, .3, self.n)
        syn = np.zeros(self.n)
        rates = np.zeros(self.n)
        refractory = np.zeros(self.n, dtype=int)
        total = np.zeros(self.n, dtype=int)
        silenced = np.zeros(self.n, dtype=bool)
        if p['condition'] == 'silence_vnc': silenced = self.vnc.copy()
        if p['condition'] == 'silence_left_dn': silenced = self.dn & (self.side == -1)
        phases = np.array([0, math.pi, 0, math.pi, 0, math.pi], dtype=float)
        x = y = heading = path_length = 0.
        target = np.array([12., 6.])
        trace, raster = [], []
        contacts = np.zeros(6)
        strides = np.zeros(6)
        saturation = 0
        target_drive = p['drive']
        for step in range(round(p['duration'] / DT)):
            t = step * DT
            external = np.zeros(self.n)
            bias = p['bias']
            stimulus = t >= .5 and (p['mode'] != 'pulse' or t < 1.5)
            if p['condition'] == 'no_stimulus': stimulus = False
            if p['mode'] == 'target':
                bearing = math.atan2(target[1] - y, target[0] - x) - heading
                bias = float(np.clip(math.sin(bearing), -1, 1))
                target_drive = p['drive'] * min(1., float(np.linalg.norm(target - [x, y])) / 2)
            if stimulus:
                external[self.dn] = target_drive * (1 + bias * self.side[self.dn])
            if p['condition'] != 'no_feedback':
                for k, idx in enumerate(self.sensory):
                    external[idx] += p['feedback'] * contacts[k] * strides[k]
            # No tonic motor drive. Gaussian current noise is an assumption; seed is fixed.
            current = external + p['gain'] * w.dot(syn) + rng.normal(0, .015, self.n)
            refractory = np.maximum(refractory - 1, 0)
            active = (refractory == 0) & ~silenced
            v[active] += DT / .020 * (-v[active] + current[active])
            v[silenced] = 0
            spikes = (v >= 1) & active
            v[spikes] = 0
            refractory[spikes] = 2
            # 10 ms exponentially decaying synaptic trace, unit increment per spike.
            syn *= math.exp(-DT / .010)
            syn[spikes] += 1
            syn[silenced] = 0
            rates *= math.exp(-DT / .050)
            rates[spikes] += 1 / .050
            total += spikes
            if len(raster) < 14000 and step % 2 == 0:
                raster.extend([round(t, 3), int(i)] for i in np.flatnonzero(spikes)[:40])
            # Kinematic proxy only: pooled motor firing modulates engineered leg oscillators.
            if step % 10 == 0:
                motor_hz = np.array([float(rates[idx].mean()) if len(idx) else 0 for idx in self.motor])
                strides = np.clip(motor_hz / 50., 0, 1)
                saturation += int(np.any(motor_hz >= 50))
                phases += .010 * 2 * math.pi * 5 * strides
                contacts = (np.sin(phases) <= 0).astype(float)
                left, right = strides[:3].mean(), strides[3:].mean()
                speed = 6 * (left + right) / 2
                omega = 3 * (right - left)
                heading += omega * .010
                x += math.cos(heading) * speed * .010
                y += math.sin(heading) * speed * .010
                path_length += speed * .010
            if step % 20 == 0:
                trace.append({'t': round(t, 3), 'x': x, 'y': y, 'heading': heading,
                              'motor': motor_hz.tolist(), 'strides': strides.tolist(), 'phases': phases.tolist(),
                              'dn_hz': float(rates[self.dn].mean()), 'vnc_hz': float(rates[self.vnc].mean()),
                              'motor_hz': float(motor_hz.mean()), 'speed': float(speed)})
        duration = round(p['duration'] / DT) * DT
        return {'parameters': p, 'trace': trace, 'raster': raster,
                'neuron_hz': (total / duration).tolist(),
                'metrics': {'path_mm': path_length, 'displacement_mm': math.hypot(x, y),
                            'turn_deg': math.degrees(heading), 'spikes': int(total.sum()),
                            'active_neurons': int((total > 0).sum()),
                            'motor_mean_hz': float(np.mean([np.mean(total[idx] / duration) if len(idx) else 0 for idx in self.motor])),
                            'clipped_motor_fraction': saturation / math.ceil(round(duration / DT) / 10)},
                'interpretation': 'Model output, not biological validation. Motor pools drive an engineered 2D gait proxy. Target mode injects a hand-designed bearing signal into descending neurons.'}
