"""Olfactory memory in the mushroom body: written, kept, retrieved and extinguished.

Built on data/mushroom-body.json (build_mushroombody.py). The measured wiring decides where memory can
be written; the model decides how.

Mechanism, following the established account:
  * An odour activates projection neurons, which drive Kenyon cells through the measured PN-to-KC wiring.
    APL keeps the code sparse; here that is a top-k threshold (about 5% of Kenyon cells).
  * Kenyon cells excite output neurons (MBONs) through the measured KC-to-MBON synapses.
  * A memory is written where Kenyon-cell activity coincides with dopamine in the same compartment: the
    active Kenyon cells' synapses onto that compartment's output neurons are depressed.
  * Which compartments a reinforcer reaches is measured: reward activates the PAM cells, punishment the
    PPL1 cells, and each dopamine cell teaches only in the compartments it innervates.
  * An output neuron's valence is derived from its compartment's teachers: a compartment taught by
    punishment holds an approach-promoting output neuron (depressing it tips behaviour toward avoidance),
    and one taught by reward holds an avoidance-promoting one.
  * Two traces per synapse: a labile trace that forms on every pairing and fades within hours, and a
    consolidated trace that forms only with spaced training and fades over days.
  * Retrieval without reinforcement lets output neurons drive dopamine neurons through the measured
    MBON-to-DAN loops. If those loops reach compartments of the opposite valence, they write an opposing
    trace, which is extinction as a new memory rather than erasure.

Measured: every cell, compartment, and edge used. Assumed: the reward and punishment split by dopamine
family (PAM rewarding, PPL1 punishing, with known exceptions in the real fly), the valence rule for output
neurons, learning rates, time constants, the consolidation rule, the sparseness level, odour codes as random
projection-neuron patterns, and the behavioural readout. Results are model behaviour, not fly measurements.
"""
import json
import math
from pathlib import Path

import numpy as np

from model import BASE

DEFAULT = dict(seed=7, sparsity=0.05, fast_rate=0.35, slow_rate=0.06, fast_hours=3.0, slow_hours=96.0,
               spacing_hours=0.25, loop_gain=1.0, odour_fraction=0.1)
CONDITIONS = {
    'intact': 'Measured circuit',
    'no_loops': 'MBON-to-DAN loops cut',
    'shuffled': 'KC-to-MBON destinations permuted, which scrambles compartment membership',
    'no_consolidation': 'No consolidated trace',
}


def load(path=None):
    path = Path(path or BASE / 'data/mushroom-body.json')
    if not path.exists():
        raise ValueError('data/mushroom-body.json is missing. Build it with build_mushroombody.py.')
    return json.loads(path.read_text())


class MushroomBody:
    def __init__(self, data, condition='intact', seed=7, sparsity=0.05, odour_fraction=0.1):
        if condition not in CONDITIONS: raise ValueError('Unknown condition')
        self.condition, self.rng = condition, np.random.default_rng(seed)
        nodes = data['nodes']
        self.role = {n['bodyId']: n['role'] for n in nodes}
        by_role = {role: [n for n in nodes if n['role'] == role] for role in ('PN', 'KC', 'MBON', 'DAN')}
        self.pn = {n['bodyId']: i for i, n in enumerate(by_role['PN'])}
        self.kc = {n['bodyId']: i for i, n in enumerate(by_role['KC'])}
        # Output neurons without a compartment cannot be taught or read by valence, so they are left out.
        mbons = [n for n in by_role['MBON'] if n['compartments_in']]
        self.mbon = {n['bodyId']: i for i, n in enumerate(mbons)}
        self.mbon_nodes = mbons
        dans = [n for n in by_role['DAN'] if n['compartments_out']]
        self.dan = {n['bodyId']: i for i, n in enumerate(dans)}
        self.dan_nodes = dans
        self.compartments = sorted({c for n in mbons for c in n['compartments_in']} | {c for n in dans for c in n['compartments_out']})
        self.comp_index = {c: i for i, c in enumerate(self.compartments)}

        P, K, M = len(self.pn), len(self.kc), len(self.mbon)
        self.pn_kc = np.zeros((K, P)); self.kc_mbon = np.zeros((M, K)); self.mbon_dan = np.zeros((len(dans), M))
        edges = data['edges']
        kc_mbon_pairs = []
        for pre, post, w in edges:
            if pre in self.pn and post in self.kc: self.pn_kc[self.kc[post], self.pn[pre]] += w
            elif pre in self.kc and post in self.mbon: kc_mbon_pairs.append((self.kc[pre], self.mbon[post], w))
            elif pre in self.mbon and post in self.dan: self.mbon_dan[self.dan[post], self.mbon[pre]] += w
        if condition == 'shuffled' and kc_mbon_pairs:
            targets = self.rng.permutation([m for _, m, _ in kc_mbon_pairs])
            kc_mbon_pairs = [(k, int(t), w) for (k, _, w), t in zip(kc_mbon_pairs, targets)]
        for k, m, w in kc_mbon_pairs: self.kc_mbon[m, k] += w
        # Log-compress synapse counts and normalise each target's input, as elsewhere in the lab.
        self.pn_kc = np.log1p(self.pn_kc); self.pn_kc /= np.maximum(self.pn_kc.sum(1, keepdims=True), 1e-9)
        self.kc_mbon = np.log1p(self.kc_mbon); self.kc_mbon /= np.maximum(self.kc_mbon.sum(1, keepdims=True), 1e-9)
        signs = np.array([-1.0 if str(n.get('nt')) in ('gaba', 'glutamate') else 1.0 for n in mbons])
        self.mbon_dan = np.log1p(self.mbon_dan) * signs[None, :]
        if condition == 'no_loops': self.mbon_dan[:] = 0

        # Teaching: which reinforcer reaches which compartment, from the measured dopamine cells.
        self.family = np.array(['reward' if str(n['type']).startswith('PAM') else 'punishment' for n in dans])
        self.dan_comp = np.zeros((len(dans), len(self.compartments)))
        for i, n in enumerate(dans):
            for c in n['compartments_out']: self.dan_comp[i, self.comp_index[c]] = 1.0
        # Each output neuron's teachers and the valence that follows from them.
        self.mbon_comp = np.zeros((M, len(self.compartments)))
        for i, n in enumerate(mbons):
            for c in n['compartments_in']: self.mbon_comp[i, self.comp_index[c]] = 1.0 / len(n['compartments_in'])
        punishment = self.dan_comp[self.family == 'punishment'].sum(0); reward = self.dan_comp[self.family == 'reward'].sum(0)
        teacher = self.mbon_comp @ np.sign(punishment - reward)
        self.valence = np.where(teacher > 0, 1.0, np.where(teacher < 0, -1.0, 0.0))   # +1 approach, -1 avoid

        self.sparsity, self.odour_fraction = sparsity, odour_fraction
        self.fast = np.zeros_like(self.kc_mbon); self.slow = np.zeros_like(self.kc_mbon)
        self.odours = {}

    # --- representation -------------------------------------------------------------------------------
    def odour(self, name):
        if name not in self.odours:
            pattern = np.zeros(len(self.pn))
            active = self.rng.choice(len(self.pn), max(1, int(self.odour_fraction * len(self.pn))), replace=False)
            pattern[active] = 1.0
            self.odours[name] = pattern
        return self.odours[name]

    def kenyon(self, name):
        drive = self.pn_kc @ self.odour(name)
        k = max(1, int(round(self.sparsity * len(drive))))
        code = np.zeros_like(drive); code[np.argsort(-drive)[:k]] = 1.0
        return code

    def outputs(self, name):
        strength = np.clip(1 - self.fast - self.slow, 0, 1)
        return (self.kc_mbon * strength) @ self.kenyon(name)

    def value(self, name):
        return float(self.valence @ self.outputs(name))

    # --- learning ---------------------------------------------------------------------------------------
    def teach(self, name, reinforcer, p):
        """One pairing: dopamine from the reinforcer plus, through the loops, from the output neurons."""
        code = self.kenyon(name)
        dopamine = np.zeros(len(self.dan_nodes))
        if reinforcer in ('reward', 'punishment'): dopamine[self.family == reinforcer] = 1.0
        loop = self.mbon_dan @ self.outputs(name)
        dopamine = np.clip(dopamine + p['loop_gain'] * np.maximum(loop, 0) / max(np.abs(loop).max(), 1e-9) * 0.5, 0, 1)
        compartment_signal = dopamine @ self.dan_comp                       # dopamine reaching each compartment
        gate = self.mbon_comp @ compartment_signal                          # reaching each output neuron's synapses
        room = np.clip(1 - self.fast - self.slow, 0, 1)
        self.fast += p['fast_rate'] * gate[:, None] * code[None, :] * room
        return float(gate.max())

    def consolidate(self, p):
        if self.condition == 'no_consolidation': return
        moved = p['slow_rate'] * self.fast
        self.slow += moved

    def wait(self, hours, p):
        self.fast *= math.exp(-hours / p['fast_hours'])
        self.slow *= math.exp(-hours / p['slow_hours'])


def snapshot(mb, odours=('A', 'B')):
    """Output-neuron activity before training, the baseline memory is measured against."""
    return {o: mb.outputs(o).copy() for o in odours}


def preference(mb, naive, trained='A', control='B'):
    """Change in approach drive toward the trained odour, relative to the control and to the naive state.

    Positive: the fly now approaches the trained odour more than before; negative: it avoids it. Scaled
    by the trained odour's naive output so that values are comparable across circuits."""
    change_trained = mb.valence @ (mb.outputs(trained) - naive[trained])
    change_control = mb.valence @ (mb.outputs(control) - naive[control])
    scale = float(np.abs(naive[trained]).sum()) + 1e-9
    return float((change_trained - change_control) / scale)


def validate(options):
    p = DEFAULT | options
    for key, lo, hi in [('sparsity', 0.01, 0.5), ('fast_rate', 0, 1), ('slow_rate', 0, 1), ('fast_hours', 0.1, 100),
                        ('slow_hours', 1, 1000), ('spacing_hours', 0, 24), ('loop_gain', 0, 5), ('odour_fraction', 0.01, 0.5)]:
        p[key] = float(p[key])
        if not math.isfinite(p[key]) or not lo <= p[key] <= hi: raise ValueError(f'{key} must be between {lo} and {hi}')
    p['seed'] = int(p['seed'])
    return p


def train(mb, p, reinforcer, trials=5, spaced=False):
    for trial in range(trials):
        mb.teach('A', reinforcer, p)
        if spaced:
            mb.consolidate(p); mb.wait(p['spacing_hours'], p)
        else:
            mb.wait(0.02, p)                                  # massed training: no consolidation


def forgetting(data=None, reinforcer='punishment', spaced=False, condition='intact', hours=(0, 1, 3, 6, 24, 48, 96), **options):
    p = validate(options); data = data or load()
    mb = MushroomBody(data, condition, p['seed'], p['sparsity'], p['odour_fraction'])
    naive = snapshot(mb)
    train(mb, p, reinforcer, spaced=spaced)
    curve, elapsed = [], 0.0
    for h in hours:
        mb.wait(h - elapsed, p); elapsed = h
        curve.append({'hours': h, 'preference': round(preference(mb, naive), 4)})
    return {'reinforcer': reinforcer, 'spaced': spaced, 'condition': condition, 'curve': curve}


def extinction(data=None, condition='intact', exposures=6, recovery_hours=24, **options):
    """Aversive training, then the odour without punishment, then a delay.

    Each extinguished fly is compared with the same fly trained identically but never re-exposed, at the
    same time points, so ordinary forgetting cancels out. The extinction effect is the difference between
    the two; spontaneous recovery is that effect shrinking over the delay, the signature of extinction as a
    separate, faster-fading memory rather than erasure of the original."""
    p = validate(options); data = data or load()
    def trained():
        mb = MushroomBody(data, condition, p['seed'], p['sparsity'], p['odour_fraction'])
        naive = snapshot(mb); train(mb, p, 'punishment', spaced=True)
        return mb, naive
    extinguished, naive = trained()
    control, control_naive = trained()
    after_training = preference(extinguished, naive)
    trace = [{'step': 'trained', 'preference': round(after_training, 4), 'control': round(after_training, 4)}]
    for i in range(exposures):
        extinguished.teach('A', None, p); extinguished.wait(0.02, p); control.wait(0.02, p)
        trace.append({'step': f'exposure {i + 1}', 'preference': round(preference(extinguished, naive), 4),
                      'control': round(preference(control, control_naive), 4)})
    effect_now = preference(extinguished, naive) - preference(control, control_naive)
    extinguished.wait(recovery_hours, p); control.wait(recovery_hours, p)
    after_delay, control_delay = preference(extinguished, naive), preference(control, control_naive)
    effect_later = after_delay - control_delay
    trace.append({'step': f'{recovery_hours} h later', 'preference': round(after_delay, 4), 'control': round(control_delay, 4)})
    sign = -1.0 if after_training < 0 else 1.0                 # extinction moves memory back toward zero
    return {'condition': condition, 'trace': trace, 'after_training': round(after_training, 4),
            'extinction_effect': round(effect_now, 4), 'extinction_effect_after_delay': round(effect_later, 4),
            'extinction_fraction': round(-sign * effect_now / max(abs(after_training), 1e-9), 3),
            'extinguished': bool(-sign * effect_now > 0.05 * abs(after_training)),
            'spontaneous_recovery': bool(abs(effect_later) < 0.5 * abs(effect_now)) if abs(effect_now) > 1e-6 else False}


def ablation(data=None, reinforcer='punishment', **options):
    """Block plasticity in one compartment at a time: which memories need which compartment?"""
    p = validate(options); data = data or load()
    base = MushroomBody(data, 'intact', p['seed'], p['sparsity'], p['odour_fraction'])
    rows = []
    for compartment in [None] + base.compartments:
        mb = MushroomBody(data, 'intact', p['seed'], p['sparsity'], p['odour_fraction'])
        if compartment is not None:
            mb.dan_comp[:, mb.comp_index[compartment]] = 0.0
        naive = snapshot(mb)
        train(mb, p, reinforcer, spaced=True)
        rows.append({'blocked': compartment or 'none', 'memory': round(preference(mb, naive), 4),
                     'teachers': sorted({n['type'] for n in mb.dan_nodes if compartment in n['compartments_out']}) if compartment else [],
                     'readers': sorted({n['type'] for n in mb.mbon_nodes if compartment in n['compartments_in']}) if compartment else []})
    full = rows[0]['memory']
    for row in rows[1:]: row['share_lost'] = round(1 - row['memory'] / full, 3) if abs(full) > 1e-9 else None
    return {'reinforcer': reinforcer, 'rows': rows}


def report(seed=7, **options):
    """Everything the page shows, in one call."""
    data = load()
    common = dict(options, seed=seed)
    return {
        'summary': summary(data),
        'forgetting': {f'{r}|{"spaced" if s else "massed"}': forgetting(data, r, s, **common)
                       for r in ('punishment', 'reward') for s in (False, True)},
        'controls': {c: forgetting(data, 'punishment', True, condition=c, **common)['curve'][0]['preference']
                     for c in CONDITIONS},
        'extinction': {c: extinction(data, condition=c, **common) for c in ('intact', 'no_loops')},
        'ablation': {r: ablation(data, r, **common) for r in ('punishment', 'reward')},
        'conditions': CONDITIONS,
        'interpretation': 'Measured cells, compartments and edges; assumed learning rule, rates, time constants, valence '
                          'rule, reward/punishment split by dopamine family, odour codes and readout.'}


def summary(data=None):
    data = data or load()
    mb = MushroomBody(data)
    return {'cells': {role: sum(1 for n in data['nodes'] if n['role'] == role) for role in ('PN', 'KC', 'MBON', 'DAN', 'APL', 'DPM', 'MBON_target')},
            'compartments': mb.compartments, 'output_neurons_used': len(mb.mbon), 'dopamine_neurons_used': len(mb.dan),
            'approach_outputs': int((mb.valence > 0).sum()), 'avoid_outputs': int((mb.valence < 0).sum()),
            'unassigned_outputs': int((mb.valence == 0).sum()),
            'loop_synapses': int(sum(w for a, b, w in data['edges'] if data and mb.role.get(a) == 'MBON' and mb.role.get(b) == 'DAN')),
            'edge_kinds': data.get('edge_kinds', {})}


if __name__ == '__main__':
    data = load()
    print(json.dumps(summary(data), indent=1)[:1200])
    for reinforcer in ('punishment', 'reward'):
        for spaced in (False, True):
            r = forgetting(data, reinforcer, spaced)
            print(reinforcer, 'spaced' if spaced else 'massed', [c['preference'] for c in r['curve']])
    print('extinction', extinction(data)['trace'])
