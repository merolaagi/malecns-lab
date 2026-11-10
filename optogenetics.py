"""Virtual optogenetics: switch on one descending-neuron type with "light" and compare the model with real flies.

In real flies a light-gated channel is expressed in one genetically targeted cell type and switched on with
light. Several such experiments have well-known outcomes (backward walking, forward walking, turning,
escape). Here the same cell type is driven in the lab's spiking model, alone, on both sides or on one side,
and the model's legs are scored against the published outcome. A mismatch says which physiology the model
lacks; a match is necessary, not sufficient, before trusting the model to plan experiments on real flies.

Measured: the cells, their wiring and transmitter predictions (data/optogenetics-circuit.json from
build_optogenetics.py; until it exists, the lab's original circuit with DNa01, DNa02 and DNg13 only).
Assumed: everything model.py assumes (uniform leaky integrate-and-fire cells, synapse weights from counts,
glutamate as inhibitory, an engineered leg readout), light as a constant current into every targeted cell,
and the thresholds that turn model output into a verdict. The leg readout cannot walk backwards, so
backward walking is scored by a proxy and labelled as such. There are no electrical synapses.
"""
import json
import math
from pathlib import Path

import numpy as np

import bodyplan
import model
from model import BASE

EXPERIMENTS = {
    'DNa02': dict(types=['DNa02'], test='turn', expect='Turning toward the lit side (ipsilateral)',
                  reference='Rayshubskiy et al., DNa02 activity predicts and drives ipsilateral turning'),
    'DNa01': dict(types=['DNa01'], test='turn', expect='Ipsilateral steering, smaller and slower than DNa02',
                  reference='Rayshubskiy et al.'),
    'P9': dict(types=['DNp09'], test='forward_turn', expect='Forward walking; turning toward the lit side when one side is lit',
               reference='Bidaye et al. 2020, Neuron'),
    'MDN': dict(types=['MDN'], test='backward', expect='Backward walking ("moonwalking")',
                reference='Bidaye et al. 2014, Science; hindlegs lead backward stepping (Feng et al. 2020)'),
    'Giant fiber': dict(types=['DNp01'], test='escape', expect='Escape take-off: the jump-muscle (TTM) motor neuron fires within milliseconds',
                        reference='Giant-fiber escape pathway (e.g. von Reyn et al. 2014); mainly electrical synapses'),
    'DNg13': dict(types=['DNg13'], test=None, expect='No single published activation effect; exploratory', reference=None),
}
SIDES = {'both': 0.0, 'right': 1.0, 'left': -1.0}
CONDITIONS = {
    'intact': 'Measured wiring',
    'shuffled': 'Wiring shuffled (destinations permuted, source weights and in-degree kept)',
    'sham': 'Light on the same number of random VNC interneurons instead (same sides)',
}
DEFAULT = dict(experiment='DNa02', side='right', light=2.5, duration=3.0, seed=1, condition='intact', resting=0.0, gap_junctions=False)
# Physiology the connectome does not contain, added one piece at a time to see what each one fixes.
#   resting        constant current into every VNC interneuron, as a fraction of spike threshold: below 1 no cell
#                  fires on its own, but the network becomes excitable, as real VNC circuits with ongoing activity are
#   gap_junctions  known electrical synapses, placed on the measured chemical contacts between the same partners:
#                  giant fiber -> jump-muscle (TTM) motor neuron. A presynaptic spike kicks the partner's voltage.
GAPS = [('DNp01', 'TTMn')]
GAP_KICK = 1.2                  # in units of spike threshold: enough to make the partner follow one-to-one
PHYSIOLOGY = {
    'connectome': dict(resting=0.0, gap_junctions=False, text='Connectome only'),
    'electrical': dict(resting=0.0, gap_junctions=True, text='+ electrical synapses'),
    'resting': dict(resting=0.5, gap_junctions=False, text='+ resting excitability (0.5 of threshold)'),
    'both': dict(resting=0.5, gap_junctions=True, text='+ both'),
}
LIGHT_ON = 0.5                 # model.simulate switches the stimulus on at 0.5 s
TURN_DEG = 20.0                # verdict thresholds over the light period
FORWARD_MM_S = 2.0
STRAIGHT_DEG = 60.0
HIND_OVER_FRONT = 1.2
ESCAPE_S = 0.02


def side_of(n):
    return n.get('rootSide') or n.get('somaSide')


def symmetric(data):
    """Left-right matched copy of a circuit: for every cell type (and leg, for motor and sensory cells), keep the
    same number of cells on each side, dropping the cells with the fewest synapses inside the circuit. Cells without
    a type or a side are kept. Returns (data, report)."""
    nodes = data['nodes']
    degree = {}
    for a, b, w in data['edges']:
        degree[a] = degree.get(a, 0) + w; degree[b] = degree.get(b, 0) + w
    pools = {}
    for n in nodes:
        if n.get('type') and side_of(n) in ('L', 'R'):
            key = (n['type'], n.get('subclass'), n.get('entryNerve'))
            pools.setdefault(key, {'L': [], 'R': []})[side_of(n)].append(n)
    drop = set()
    for key, sides in pools.items():
        k = min(len(sides['L']), len(sides['R']))
        for s in ('L', 'R'):
            ranked = sorted(sides[s], key=lambda n: (-degree.get(n['bodyId'], 0), n['bodyId']))
            drop.update(n['bodyId'] for n in ranked[k:])
    # never drop the lit cell types themselves
    keep_types = {t for e in EXPERIMENTS.values() for t in e['types']}
    drop = {b for b in drop if next(n for n in nodes if n['bodyId'] == b)['type'] not in keep_types}
    out = dict(data, nodes=[n for n in nodes if n['bodyId'] not in drop],
               edges=[e for e in data['edges'] if e[0] not in drop and e[1] not in drop])
    unmatched = sum(1 for sides in pools.values() if len(sides['L']) != len(sides['R']))
    return out, {'dropped': len(drop), 'kept': len(out['nodes']), 'unmatched_pools_before': unmatched}


class Optogenetics:
    def __init__(self, path=None, matched=False):
        default = BASE / 'data/optogenetics-circuit.json'
        self.path = Path(path) if path else (default if default.exists() else BASE / 'data/circuit.json')
        self.source = {'optogenetics-circuit.json': 'optogenetics', 'coordination-circuit.json': 'coordination'}.get(self.path.name, 'original')
        data = json.loads(self.path.read_text())
        self.leg_links = data.get('leg_links')
        self.matched, self.matching = matched, None
        if matched: data, self.matching = symmetric(data)
        self.circuit = model.Circuit(data=data)
        nodes = self.circuit.nodes
        self.types = np.array([str(n.get('type')) for n in nodes])
        self.ttm = np.array([bool(n.get('ttm')) for n in nodes])
        self.interneurons = np.flatnonzero(self.circuit.vnc)
        self.available = {k: int(np.isin(self.types, e['types']).sum()) for k, e in EXPERIMENTS.items()}
        c = self.circuit
        self.gaps = [(int(a), int(b)) for a, b in sorted(set(zip(c.pre.tolist(), c.post.tolist())))
                     if any(self.types[a] == x and self.types[b] == y for x, y in GAPS)]
        # Motor neurons grouped by leg and muscle (bodyplan), to see what a lit cell type does to each joint.
        self.muscle_groups = {}
        for i, n in enumerate(nodes):
            if n.get('superclass') != 'vnc_motor': continue
            leg, group = bodyplan.leg_of(n), bodyplan.muscle_of(n.get('type'))
            if leg and group: self.muscle_groups.setdefault((leg, group), []).append(i)

    def targets(self, experiment, condition, seed):
        mask = np.isin(self.types, EXPERIMENTS[experiment]['types'])
        if condition != 'sham': return mask
        rng = np.random.default_rng(seed + 777)
        sham = np.zeros_like(mask)
        for side in np.unique(self.circuit.side[mask]):
            need = int(np.sum(mask & (self.circuit.side == side)))
            pool = self.interneurons[self.circuit.side[self.interneurons] == side]
            if len(pool): sham[rng.choice(pool, min(need, len(pool)), replace=False)] = True
        return sham

    def simulate(self, p, **extra):
        """Run the spiking model with the lit cells driven. Returns (raw model result, number of lit cells)."""
        c = self.circuit
        saved = c.dn
        try:
            c.dn = self.targets(p['experiment'], p['condition'], p['seed'])
            resting = np.where(c.vnc, p['resting'], 0.0).tolist() if p['resting'] else None
            gaps = [(a, b, GAP_KICK) for a, b in self.gaps] if p['gap_junctions'] else []
            r = c.simulate(duration=p['duration'], drive=p['light'], bias=SIDES[p['side']], seed=p['seed'], resting=resting, gaps=gaps,
                           condition='shuffled' if p['condition'] == 'shuffled' else 'intact', mode='tonic', **extra)
            return r, int(c.dn.sum())
        finally:
            c.dn = saved

    def stimulate(self, **options):
        p = validate(options)
        if not self.available[p['experiment']]:
            raise ValueError(f"{p['experiment']} is not in this circuit; build data/optogenetics-circuit.json with build_optogenetics.py")
        r, lit = self.simulate(p)
        trace = r['trace']
        on = [t for t in trace if t['t'] >= LIGHT_ON]
        start = next((t['heading'] for t in trace if t['t'] >= LIGHT_ON), 0.0)
        legs = np.mean([t['motor'] for t in on], 0) if on else np.zeros(6)
        ttm_idx = np.flatnonzero(self.ttm)
        latency = None
        if len(ttm_idx):
            times = [t for t, i in r['raster'] if i in set(ttm_idx.tolist()) and t >= LIGHT_ON]
            latency = round(min(times) - LIGHT_ON, 4) if times else None
        hz = np.array(r['neuron_hz'])
        return {'parameters': p, 'source': self.source, 'lit_cells': lit,
                'metrics': {'turn_deg': math.degrees(trace[-1]['heading'] - start) if trace else 0.0,
                            'speed_mm_s': float(np.mean([t['speed'] for t in on])) if on else 0.0,
                            'leg_motor_hz': dict(zip(model.LEGS, np.round(legs, 2).tolist())),
                            'lit_side_share': (float(legs[3:].sum() / max(legs.sum(), 1e-9)) if p['side'] == 'right' else
                                               float(legs[:3].sum() / max(legs.sum(), 1e-9)) if p['side'] == 'left' else None),
                            'front_hz': float((legs[0] + legs[3]) / 2), 'middle_hz': float((legs[1] + legs[4]) / 2),
                            'hind_hz': float((legs[2] + legs[5]) / 2),
                            'target_hz': float(hz[self.targets(p['experiment'], p['condition'], p['seed'])].mean()),
                            'ttm_cells': int(len(ttm_idx)), 'ttm_hz': float(hz[ttm_idx].mean()) if len(ttm_idx) else None,
                            'ttm_latency_s': latency,
                            'muscles': {f'{leg} {group}': round(float(hz[idx].mean()), 2)
                                        for (leg, group), idx in sorted(self.muscle_groups.items()) if hz[idx].mean() > 0.5}},
                'trace': [{k: t[k] for k in ('t', 'x', 'y', 'heading', 'speed', 'motor')} for t in trace]}


def validate(options):
    p = DEFAULT | {k: v for k, v in options.items() if v is not None}
    if p['experiment'] not in EXPERIMENTS: raise ValueError('Unknown experiment')
    if p['side'] not in SIDES: raise ValueError('side must be both, right or left')
    if p['condition'] not in CONDITIONS: raise ValueError('Unknown condition')
    p['light'] = float(p['light']); p['duration'] = float(p['duration']); p['seed'] = int(p['seed'])
    if not 0 <= p['light'] <= 5: raise ValueError('light must be between 0 and 5')
    if not 1 <= p['duration'] <= 10: raise ValueError('duration must be between 1 and 10 s')
    p['resting'] = float(p['resting'])
    if not 0 <= p['resting'] <= 0.95: raise ValueError('resting must be between 0 and 0.95 of threshold')
    if not isinstance(p['gap_junctions'], bool): raise ValueError('gap_junctions must be true or false')
    return p


def verdict(experiment, runs):
    """Score runs ({side: [metrics per seed]}) against the published outcome."""
    e = EXPERIMENTS[experiment]
    mean = lambda side, key: float(np.mean([m[key] for m in runs[side]])) if runs.get(side) else None
    turn = {s: mean(s, 'turn_deg') for s in SIDES}
    out = {'turn_deg': turn, 'speed_mm_s': {s: mean(s, 'speed_mm_s') for s in SIDES}}

    shares = [m['lit_side_share'] for side in ('right', 'left') for m in runs.get(side, []) if m['lit_side_share'] is not None]
    out['lit_side_share'] = float(np.mean(shares)) if shares else None

    def turning():
        r, l = turn['right'], turn['left']
        if r is None or l is None: return 'not run', ''
        if r < -TURN_DEG and l > TURN_DEG: return 'match', f'right side lit turns right ({r:.0f}°), left side lit turns left ({l:+.0f}°)'
        if r > TURN_DEG and l < -TURN_DEG: return 'opposite', f'turns away from the lit side (right {r:+.0f}°, left {l:+.0f}°)'
        return 'no clear turn', f'right {r:+.0f}°, left {l:+.0f}° (threshold ±{TURN_DEG:.0f}°)'

    if e['test'] is None:
        out.update(result='exploratory', detail=f"both sides: {out['speed_mm_s']['both']:.1f} mm/s, {turn['both']:+.0f}°"); return out
    if e['test'] == 'turn':
        out['result'], out['detail'] = turning()
        if out['lit_side_share'] is not None:
            out['detail'] += f"; {out['lit_side_share'] * 100:.0f}% of leg motor activity on the lit side"
        return out
    if e['test'] == 'forward_turn':
        speed, straight = out['speed_mm_s']['both'], abs(turn['both'])
        forward = speed > FORWARD_MM_S and straight < STRAIGHT_DEG
        t_result, t_detail = turning()
        out['result'] = 'match' if forward and t_result == 'match' else 'partial' if forward or t_result == 'match' else 'mismatch'
        out['detail'] = f"both sides: {speed:.1f} mm/s, {turn['both']:+.0f}° ({'forward' if forward else 'not straight forward walking'}); one side: {t_detail}"
        return out
    if e['test'] == 'backward':
        front = float(np.mean([m['front_hz'] for m in runs['both']])); hind = float(np.mean([m['hind_hz'] for m in runs['both']]))
        out['front_hz'], out['hind_hz'] = front, hind
        if hind + front < 0.5: out['result'] = 'no leg activity'
        else: out['result'] = 'proxy match' if hind > HIND_OVER_FRONT * front else 'proxy mismatch'
        out['detail'] = (f'the leg readout cannot walk backwards; proxy is hindleg motor drive above foreleg '
                         f'({hind:.1f} vs {front:.1f} Hz)')
        return out
    if e['test'] == 'escape':
        ms = runs['both']
        if not ms or not ms[0]['ttm_cells']:
            out.update(result='not testable', detail='no jump-muscle (TTM) motor neuron in this circuit'); return out
        lat = [m['ttm_latency_s'] for m in ms if m['ttm_latency_s'] is not None]
        hz = float(np.mean([m['ttm_hz'] for m in ms]))
        out['ttm_hz'] = hz; out['ttm_latency_s'] = float(np.median(lat)) if lat else None
        if not lat: out['result'] = 'no TTM activity'
        elif np.median(lat) <= ESCAPE_S: out['result'] = 'match'
        else: out['result'] = 'too slow'
        out['detail'] = (f'TTM {hz:.1f} Hz, first spike {out["ttm_latency_s"] * 1000:.0f} ms after light' if lat else
                         'the jump-muscle motor neuron never fired. Expected without gap junctions: the real '
                         'giant-fiber synapse onto it is mainly electrical, absent from the connectome')
        return out
    raise ValueError('Unknown test')


def experiment(lab, name, light=2.5, seeds=(1, 2, 3, 4, 5), condition='intact', duration=3.0, physiology='connectome'):
    extra = {k: v for k, v in PHYSIOLOGY[physiology].items() if k != 'text'}
    runs = {side: [lab.stimulate(experiment=name, side=side, light=light, seed=s, condition=condition, duration=duration, **extra)['metrics']
                   for s in seeds] for side in SIDES}
    return verdict(name, runs) | {'runs': len(seeds)}


def scorecard(path=None, light=2.5, seeds=(1, 2, 3, 4, 5), out=None):
    lab = Optogenetics(path)
    rows = {}
    for name, e in EXPERIMENTS.items():
        if not lab.available[name]:
            rows[name] = {'available': 0}; continue
        rows[name] = {'available': lab.available[name], 'expect': e['expect'], 'reference': e['reference'],
                      'physiology': {v: {c: experiment(lab, name, light, seeds, c, physiology=v) for c in CONDITIONS}
                                     for v in PHYSIOLOGY}}
        rows[name]['conditions'] = rows[name]['physiology']['connectome']
    result = {'source': lab.source, 'circuit': {'neurons': lab.circuit.n, 'ttm_motor_neurons': int(lab.ttm.sum()),
                                                'electrical_synapses': len(lab.gaps)},
              'physiology_variants': {k: v['text'] for k, v in PHYSIOLOGY.items()},
              'light': light, 'seeds': list(seeds), 'conditions': CONDITIONS, 'experiments': rows,
              'thresholds': {'turn_deg': TURN_DEG, 'forward_mm_s': FORWARD_MM_S, 'straight_deg': STRAIGHT_DEG,
                             'hind_over_front': HIND_OVER_FRONT, 'escape_s': ESCAPE_S}}
    (out or BASE / 'data/optogenetics-benchmark.json').write_text(json.dumps(result, indent=1))
    return result


def summary(lab=None):
    lab = lab or Optogenetics()
    return {'source': lab.source, 'available': lab.available, 'conditions': CONDITIONS, 'electrical_synapses': len(lab.gaps),
            'physiology': {k: v['text'] for k, v in PHYSIOLOGY.items()},
            'experiments': {k: {kk: v[kk] for kk in ('expect', 'reference', 'test')} for k, v in EXPERIMENTS.items()}}


if __name__ == '__main__':
    r = scorecard()
    print('circuit:', r['source'], r['circuit'])
    for name, row in r['experiments'].items():
        if not row['available']: print(f'{name:12} not in circuit'); continue
        for variant, conds in row['physiology'].items():
            print(f'{name:12} {variant:10}', ' | '.join(f"{c}: {v['result']}" for c, v in conds.items()))
            print(' ' * 24, conds['intact']['detail'])
