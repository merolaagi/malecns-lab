"""What arrives at one neuron: measured synapses turned into conductances.

The locomotion network is leaky integrate-and-fire with current-based synapses, and the workbench runs
Hodgkin-Huxley on a single cell with generic conductances. Neither connects the measured synapse counts
of a real cell to receptor conductances. This module does.

For a chosen cell, every measured presynaptic partner opens a conductance with transmitter-specific
kinetics: acetylcholine acts on a nicotinic cation channel (reversal 0 mV), GABA on a GABA-A chloride
channel (-70 mV), glutamate on a glutamate-gated chloride channel (-70 mV, the insect inhibitory
receptor), histamine on a histamine-gated chloride channel (-80 mV). Conductances sum and drive one
passive compartment with an optional spike threshold.

Two ways of using the transmitter predictions:
  'dominant'    - each partner uses its consensus transmitter, the same call the network model makes.
  'probability' - each partner's conductance is split across receptors in proportion to the measured
                  per-transmitter probabilities in data/quality.json, so prediction uncertainty reaches
                  the membrane instead of being hidden by a hard label.

Measured: which cells connect, how many synapses they make, and the transmitter predictions with their
probabilities. Assumed: every kinetic constant, the conductance per synapse, membrane capacitance and
leak, the threshold, and the presynaptic firing rates. No fly recording sets these; there is no
morphology, no cable structure, no receptor subtypes and no desensitisation. A transmitter prediction
is not a receptor measurement, and these traces are illustrative dynamics, not measurements of the cell.
"""
import json
import math
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent

# reversal potential (mV), rise and decay time constants (ms)
RECEPTORS = {
    'acetylcholine': {'name': 'Nicotinic acetylcholine receptor', 'reversal': 0.0, 'rise': 0.3, 'decay': 5.0, 'kind': 'excitatory'},
    'gaba': {'name': 'GABA-A receptor', 'reversal': -70.0, 'rise': 0.5, 'decay': 8.0, 'kind': 'inhibitory'},
    'glutamate': {'name': 'Glutamate-gated chloride channel', 'reversal': -70.0, 'rise': 0.5, 'decay': 10.0, 'kind': 'inhibitory'},
    'histamine': {'name': 'Histamine-gated chloride channel', 'reversal': -80.0, 'rise': 0.3, 'decay': 6.0, 'kind': 'inhibitory'},
}
MEMBRANE = {'capacitance_pf': 20.0, 'leak_ns': 1.0, 'rest_mv': -60.0, 'threshold_mv': -45.0, 'reset_mv': -60.0, 'refractory_ms': 2.0}
UNITARY_NS = 0.03        # assumed peak conductance per anatomical synapse
DT = 0.05                # ms
MODES = {'dominant': 'Consensus transmitter per cell', 'probability': 'Conductance split by measured transmitter probabilities'}


class Synapses:
    def __init__(self, circuit=None, quality_path=None):
        from model import Circuit
        self.c = circuit or Circuit()
        self.index = {n['bodyId']: i for i, n in enumerate(self.c.nodes)}
        path = Path(quality_path or BASE / 'data/quality.json')
        self.quality = json.loads(path.read_text())['cells'] if path.exists() else {}
        self.incoming = {}
        for pre, post, count in self.c.data['edges']:
            self.incoming.setdefault(post, []).append((pre, int(count)))

    def transmitter(self, body_id):
        """Consensus transmitter and the measured per-transmitter probabilities, when available."""
        cell = self.quality.get(str(body_id), {})
        node = self.c.nodes[self.index[body_id]] if body_id in self.index else {}
        name = cell.get('consensus_nt') or node.get('nt')
        probabilities = (cell.get('nt') or {}).get('mean_prob') or ({name: 1.0} if name else {})
        return name, {k: float(v) for k, v in probabilities.items() if k in RECEPTORS}

    def partners(self, body_id):
        body_id = int(body_id)
        if body_id not in self.index: raise ValueError(f'No neuron {body_id} in the locomotion circuit')
        rows = []
        for pre, count in self.incoming.get(body_id, []):
            node = self.c.nodes[self.index[pre]]
            name, probabilities = self.transmitter(pre)
            rows.append({'bodyId': int(pre), 'name': node.get('instance') or node.get('type') or str(pre),
                         'superclass': node.get('superclass'), 'synapses': count, 'transmitter': name,
                         'receptor': RECEPTORS.get(name, {}).get('name', 'no receptor assigned'),
                         'probabilities': {k: round(v, 3) for k, v in sorted(probabilities.items(), key=lambda kv: -kv[1])}})
        rows.sort(key=lambda r: -r['synapses'])
        return rows

    def weights(self, body_id, mode, excitation_scale, inhibition_scale):
        """Total conductance weight per receptor, in nS, plus how many cells contribute to each."""
        groups = {}
        for row in self.partners(body_id):
            if mode == 'probability':
                shares = row['probabilities'] or ({row['transmitter']: 1.0} if row['transmitter'] in RECEPTORS else {})
                total = sum(shares.values()) or 1.0
                shares = {k: v / total for k, v in shares.items()}
            else:
                shares = {row['transmitter']: 1.0} if row['transmitter'] in RECEPTORS else {}
            for transmitter, share in shares.items():
                receptor = RECEPTORS[transmitter]
                scale = excitation_scale if receptor['kind'] == 'excitatory' else inhibition_scale
                group = groups.setdefault(transmitter, {'receptor': receptor, 'weight': 0.0, 'sources': 0.0})
                group['weight'] += UNITARY_NS * row['synapses'] * share * scale
                group['sources'] += share
        return groups

    def run(self, body_id=10360, duration_ms=300.0, rate_hz=40.0, excitation_scale=1.0, inhibition_scale=1.0,
            mode='dominant', spiking=True, seed=7):
        for name, value, lo, hi in [('duration_ms', duration_ms, 20, 1000), ('rate_hz', rate_hz, 0, 200),
                                    ('excitation_scale', excitation_scale, 0, 5), ('inhibition_scale', inhibition_scale, 0, 5)]:
            value = float(value)
            if not math.isfinite(value) or not lo <= value <= hi: raise ValueError(f'{name} must be between {lo} and {hi}')
        if mode not in MODES: raise ValueError('mode must be dominant or probability')
        duration_ms, rate_hz = float(duration_ms), float(rate_hz)
        excitation_scale, inhibition_scale, seed = float(excitation_scale), float(inhibition_scale), int(seed)
        rows = self.partners(body_id)
        groups = self.weights(body_id, mode, excitation_scale, inhibition_scale)
        rng = np.random.default_rng(seed)
        steps = int(round(duration_ms / DT))
        state = {k: [0.0, 0.0] for k in groups}                 # rising and decaying components
        voltage, refractory = MEMBRANE['rest_mv'], 0.0
        times, trace_v, spikes = [], [], []
        trace_g = {k: [] for k in groups}
        for step in range(steps):
            current = MEMBRANE['leak_ns'] * (MEMBRANE['rest_mv'] - voltage)
            for transmitter, group in groups.items():
                arrivals = rng.poisson(max(group['sources'], 0.0) * rate_hz * DT / 1000.0)
                unit = group['weight'] / max(group['sources'], 1e-9)
                rise, decay = group['receptor']['rise'], group['receptor']['decay']
                state[transmitter][0] = state[transmitter][0] * math.exp(-DT / rise) + arrivals * unit
                state[transmitter][1] += (state[transmitter][0] - state[transmitter][1]) * (1 - math.exp(-DT / decay))
                g = state[transmitter][1]
                current += g * (group['receptor']['reversal'] - voltage)
                trace_g[transmitter].append(round(g, 5))
            if refractory > 0: refractory -= DT
            else: voltage += DT * current / MEMBRANE['capacitance_pf']
            if spiking and voltage >= MEMBRANE['threshold_mv']:
                spikes.append(round(step * DT, 3)); voltage = MEMBRANE['reset_mv']; refractory = MEMBRANE['refractory_ms']
            times.append(round(step * DT, 3)); trace_v.append(round(voltage, 4))
        settled = [v for t, v in zip(times, trace_v) if t > duration_ms * 0.25] or [MEMBRANE['rest_mv']]
        node = self.c.nodes[self.index[int(body_id)]]
        return {
            'bodyId': int(body_id), 'name': node.get('instance') or node.get('type') or str(body_id),
            'superclass': node.get('superclass'),
            'parameters': {'duration_ms': duration_ms, 'rate_hz': rate_hz, 'excitation_scale': excitation_scale,
                           'inhibition_scale': inhibition_scale, 'mode': mode, 'spiking': bool(spiking), 'seed': seed,
                           'unitary_ns': UNITARY_NS, 'dt_ms': DT},
            'modes': MODES, 'membrane': MEMBRANE, 'receptors': RECEPTORS,
            'partners': rows[:40], 'partner_count': len(rows),
            'synapses_total': int(sum(r['synapses'] for r in rows)),
            'groups': [{'transmitter': k, 'receptor': v['receptor']['name'], 'kind': v['receptor']['kind'],
                        'sources': round(v['sources'], 2), 'weight_ns': round(v['weight'], 4),
                        'peak_conductance_ns': round(max(trace_g[k], default=0.0), 4)} for k, v in groups.items()],
            'time_ms': times, 'voltage_mv': trace_v, 'conductance_ns': trace_g,
            'spikes_ms': spikes, 'spike_rate_hz': round(len(spikes) / (duration_ms / 1000.0), 2),
            'mean_voltage_mv': round(float(np.mean(settled)), 3),
            'interpretation': 'Measured synapse counts and transmitter predictions with assumed receptor kinetics, unitary '
                              'conductance, membrane constants and presynaptic rates. Not a measurement of this cell and not a '
                              'morphology-based cable model; a transmitter prediction is not a receptor identification.'}

    def shunting(self, body_id=10360, duration_ms=300.0, rate_hz=40.0, excitation_scale=1.0, mode='dominant', seed=7):
        """Same excitatory drive with and without the inhibitory conductances."""
        common = dict(body_id=body_id, duration_ms=duration_ms, rate_hz=rate_hz, excitation_scale=excitation_scale,
                      mode=mode, spiking=False, seed=seed)
        alone = self.run(inhibition_scale=0.0, **common)
        both = self.run(inhibition_scale=1.0, **common)
        rest = MEMBRANE['rest_mv']
        depolarisation = alone['mean_voltage_mv'] - rest
        return {'bodyId': int(body_id), 'parameters': common,
                'excitation_only_mv': alone['mean_voltage_mv'], 'with_inhibition_mv': both['mean_voltage_mv'],
                'depolarisation_alone_mv': round(depolarisation, 3),
                'depolarisation_with_inhibition_mv': round(both['mean_voltage_mv'] - rest, 3),
                'ratio': round((both['mean_voltage_mv'] - rest) / depolarisation, 3) if abs(depolarisation) > 1e-9 else None,
                'note': 'A ratio below one means the inhibitory conductance shrank the same excitatory drive. A current-based '
                        'synapse cannot do that: it subtracts a fixed current regardless of membrane potential.'}
