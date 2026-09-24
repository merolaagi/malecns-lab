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
               condition='intact', seed=7, mode='tonic', neuron='lif')
NEURON_MODELS = {
    'lif': 'Leaky integrate-and-fire, current-based synapses (the lab default)',
    'adex': 'Adaptive exponential integrate-and-fire, conductance-based synapses',
}
CONDITIONS = ['intact', 'silence_vnc', 'silence_left_dn', 'no_feedback', 'no_stimulus', 'shuffled']

# Calibration of the engineered gait readout against reported Drosophila walking:
# forward speed roughly 5-25 mm/s, step frequency about 5-15 Hz, stride length about 1-2 mm.
# A pooled motor rate of MOTOR_REF_HZ counts as a full stride; step frequency scales with it (so a
# resting fly does not drift) and stride length grows with it. These are calibration choices, not
# measurements: there is no muscle model, load or inertia anywhere in this readout.
MOTOR_REF_HZ = 50.0
STEP_HZ_MAX = 12.0
STRIDE_MM = (1.1, 2.0)
TRACK_MM = 1.0          # distance between the left and right legs, used for differential turning
YAW_MAX = 10.0          # rad/s, about 570 deg/s


def turn_rate(left, right):
    """Yaw from the difference between the two sides' leg speeds over the track width (rad/s)."""
    return float(np.clip((gait(right)[2] - gait(left)[2]) / TRACK_MM, -YAW_MAX, YAW_MAX))


def gait(stride):
    """(step frequency Hz, stride length mm, forward speed mm/s) for a pooled stride in [0, 1]."""
    stride = float(min(max(stride, 0.0), 1.0))
    if stride < 0.02: return 0.0, 0.0, 0.0
    frequency = STEP_HZ_MAX * stride
    length = STRIDE_MM[0] + (STRIDE_MM[1] - STRIDE_MM[0]) * stride
    return frequency, length, frequency * length


# Adaptive exponential integrate-and-fire with conductance-based synapses (Brette & Gerstner, 2005).
# Adds three things the leaky integrate-and-fire model lacks: a spike-initiation nonlinearity, spike
# frequency adaptation, and synapses as conductances, so inhibition shunts rather than subtracting a
# fixed current. Parameters are standard cortical values, not Drosophila measurements: no cell-type
# fitting exists for this subset. Reversal potentials follow the transmitter signs already in use.
ADEX = dict(C=200.0, g_L=10.0, E_L=-60.0, V_T=-50.0, delta_T=2.0, V_peak=0.0, V_reset=-58.0,
            tau_w=100.0, a=2.0, b=30.0, E_exc=0.0, E_inh=-75.0)
EXT_TO_PA = 250.0       # dimensionless drive -> injected current (pA); calibrated to match the LIF rates
SYN_TO_NS = 8.0         # synaptic trace * gain -> conductance (nS); calibrated the same way
SUBSTEPS = 10           # the exponential term needs a finer step than the 1 ms network step


def steer(bearing):
    """Descending bias that turns the body readout toward `bearing`.

    The readout turns with omega = 3*(right stride - left stride), and a positive bias drives the
    left-side descending neurons harder, which turns the fly clockwise. Steering toward a bearing
    therefore needs the opposite sign; before this was fixed, target mode steered away from its target.
    """
    return -math.sin(bearing)


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

    def conductance_matrices(self, condition, seed):
        """Excitatory and inhibitory weight matrices (both non-negative) for conductance synapses."""
        w = self.matrix(condition, seed)
        return w.maximum(0), (-w).maximum(0)

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
        if p['neuron'] not in NEURON_MODELS: raise ValueError('Unknown neuron model')
        bounds = {'duration': (.5, 20), 'drive': (0, 5), 'bias': (-1, 1), 'gain': (0, 20), 'feedback': (0, 2)}
        for key, (lo, hi) in bounds.items():
            p[key] = float(p[key])
            if not math.isfinite(p[key]) or not lo <= p[key] <= hi: raise ValueError(f'{key} must be between {lo} and {hi}')
        p['seed'] = int(p['seed'])
        rng = np.random.default_rng(p['seed'])
        w = self.matrix(p['condition'], p['seed'])
        w_exc, w_inh = self.conductance_matrices(p['condition'], p['seed']) if p['neuron'] == 'adex' else (None, None)
        v = (ADEX['E_L'] + rng.uniform(0, 3, self.n)) if p['neuron'] == 'adex' else rng.uniform(0, .3, self.n)
        wad = np.zeros(self.n)                      # AdEx adaptation current (pA)
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
                bias = float(np.clip(steer(bearing), -1, 1))
                target_drive = p['drive'] * min(1., float(np.linalg.norm(target - [x, y])) / 2)
            if stimulus:
                external[self.dn] = target_drive * (1 + bias * self.side[self.dn])
            if p['condition'] != 'no_feedback':
                for k, idx in enumerate(self.sensory):
                    external[idx] += p['feedback'] * contacts[k] * strides[k]
            # No tonic motor drive. Gaussian current noise is an assumption; seed is fixed.
            refractory = np.maximum(refractory - 1, 0)
            active = (refractory == 0) & ~silenced
            if p['neuron'] == 'adex':
                A = ADEX
                ge = SYN_TO_NS * p['gain'] * w_exc.dot(syn)
                gi = SYN_TO_NS * p['gain'] * w_inh.dot(syn)
                I = EXT_TO_PA * (external + rng.normal(0, .015, self.n))
                spikes = np.zeros(self.n, dtype=bool)
                sub = DT * 1000.0 / SUBSTEPS                      # ms
                for _ in range(SUBSTEPS):
                    drift = (-A['g_L'] * (v - A['E_L'])
                             + A['g_L'] * A['delta_T'] * np.exp(np.clip((v - A['V_T']) / A['delta_T'], -30, 20))
                             + ge * (A['E_exc'] - v) + gi * (A['E_inh'] - v) + I - wad)
                    v[active] += sub / A['C'] * drift[active]
                    wad += sub / A['tau_w'] * (A['a'] * (v - A['E_L']) - wad)
                    fired = (v >= A['V_peak']) & active
                    v[fired] = A['V_reset']; wad[fired] += A['b']; spikes |= fired
                v[silenced] = A['E_L']; wad[silenced] = 0
            else:
                current = external + p['gain'] * w.dot(syn) + rng.normal(0, .015, self.n)
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
                strides = np.clip(motor_hz / MOTOR_REF_HZ, 0, 1)
                saturation += int(np.any(motor_hz >= MOTOR_REF_HZ))
                step_hz = np.array([gait(s)[0] for s in strides])
                phases += .010 * 2 * math.pi * step_hz
                contacts = (np.sin(phases) <= 0).astype(float)
                left, right = strides[:3].mean(), strides[3:].mean()
                speed = gait((left + right) / 2)[2]
                omega = turn_rate(left, right)
                heading += omega * .010
                x += math.cos(heading) * speed * .010
                y += math.sin(heading) * speed * .010
                path_length += speed * .010
            if step % 20 == 0:
                trace.append({'t': round(t, 3), 'x': x, 'y': y, 'heading': heading,
                              'motor': motor_hz.tolist(), 'strides': strides.tolist(), 'phases': phases.tolist(),
                              'dn_hz': float(rates[self.dn].mean()), 'vnc_hz': float(rates[self.vnc].mean()),
                              'motor_hz': float(motor_hz.mean()), 'speed': float(speed),
                              'step_hz': float(np.mean([gait(s)[0] for s in strides])),
                              'stride_mm': float(np.mean([gait(s)[1] for s in strides]))})
        duration = round(p['duration'] / DT) * DT
        return {'parameters': p, 'trace': trace, 'raster': raster,
                'neuron_hz': (total / duration).tolist(),
                'metrics': {'path_mm': path_length, 'displacement_mm': math.hypot(x, y),
                            'turn_deg': math.degrees(heading), 'spikes': int(total.sum()),
                            'active_neurons': int((total > 0).sum()),
                            'motor_mean_hz': float(np.mean([np.mean(total[idx] / duration) if len(idx) else 0 for idx in self.motor])),
                            'clipped_motor_fraction': saturation / math.ceil(round(duration / DT) / 10),
                            'mean_speed_mm_s': float(np.mean([t['speed'] for t in trace])) if trace else 0.0,
                            'mean_step_hz': float(np.mean([t['step_hz'] for t in trace])) if trace else 0.0,
                            'mean_stride_mm': float(np.mean([t['stride_mm'] for t in trace])) if trace else 0.0,
                            'reported_ranges': {'speed_mm_s': [5, 25], 'step_hz': [5, 15], 'stride_mm': [1, 2]}},
                'interpretation': 'Model output, not biological validation. Motor pools drive an engineered 2D gait proxy. Target mode injects a hand-designed bearing signal into descending neurons.'}
