"""Three flies in one arena: two labelled male, one labelled female.

What is measured: each agent runs the same MaleCNS locomotion circuit (807 cells, 21,161 edges) with the
dynamics of model.py, unchanged. Nothing else here is measured.

What is engineered, and why it has to be:
  * There is no female connectome in this lab. MaleCNS is one male fly, and the selected subset (leg
    circuits) holds none of the sexually dimorphic courtship circuitry. Every agent therefore runs the
    same male circuit; "female" is a label attached to engineered emissions and responses, not a
    different brain.
  * Sensing is hand-designed: odour concentration falls off with distance, each agent compares
    concentration at two antenna positions, and other flies are seen as a bearing plus an apparent size.
    These become the descending-neuron bias and drive that model.py already accepts.
  * Signalling is hand-designed from described behaviour, not from circuitry: a near male "sings", which
    slows the female and raises the rival male's drive. Real courtship song is produced by wing motor
    circuits that are not in the subset.

So this shows how three copies of a measured circuit behave when coupled by assumed senses. It is not a
model of fly social behaviour, and the male/female difference is a parameter choice.
"""
import argparse
import json
import math

import numpy as np

import bodyplan
from model import DT, Circuit, gait, steer, turn_rate

SCENARIOS = {
    'rivalry': 'Two males, one female. Both males approach her odour; each male raises his drive when he hears a rival sing.',
    'courtship': 'One male approaches the female; the other male starts far away and is not attracted.',
    'food': 'A food patch emits odour. All three approach it; the patch is small, so they crowd.',
    'threat': 'A looming threat appears above the arena centre at 4 s and all three flee it.',
}
CONDITIONS = {
    'intact': 'All senses and signals active',
    'no_vision': 'Agents cannot see each other',
    'no_odour': 'No odour sensing',
    'no_signals': 'No song signalling between agents',
    'isolated': 'No vision, odour or signals: the agents ignore each other',
    'shuffled': 'Shuffled connectome in every agent',
    'silence_vnc': 'VNC interneurons silenced in every agent',
}
DEFAULT = dict(seed=7, duration=16.0, scenario='rivalry', condition='intact', drive=2.4, gain=10.0,
               feedback=2.0, odour_gain=1.0, vision_gain=0.6, song_gain=0.5, body='pooled', steer_gain=0.6, steer_tau=0.12,
               drive_floor=2.1, approach_max=0.30)
# Measured operating band of this subset: below about 2.1 of descending drive the motor pools are silent,
# above about 3.0 they saturate. With only 10-15% of their real input the cells have a steep, narrow
# range, so a behavioural command in [0, 1] is mapped into that band instead of being fed in raw.
# Default drive 2.8: at 2.0 the agents crawl at under 1 mm/s, and above 3.0 the motor pools saturate.
BODY_MODES = {
    'pooled': 'Body speed and turning from pooled motor rates (the lab\'s original readout)',
    'kinematic': 'Body motion from the planted feet, whose positions come from joint angles',
}
AGENTS = [
    {'id': 'male_1', 'label': 'Male 1', 'sex': 'male', 'start': (-8.0, -4.0), 'heading': 0.5},
    {'id': 'male_2', 'label': 'Male 2', 'sex': 'male', 'start': (-8.0, 5.0), 'heading': -0.4},
    {'id': 'female', 'label': 'Female (same circuit, different label)', 'sex': 'female', 'start': (7.0, 1.0), 'heading': 2.6},
]
FOOD = np.array([0.0, 8.0])
THREAT = np.array([0.0, 0.0])
CONTACT_MM = 2.0
NEAR_MM = 5.0


def concentration(pos, sources):
    """Assumed odour field: each source falls off as 1/(1 + (d/2 mm)^2)."""
    total = 0.0
    for p, strength in sources:
        d = float(np.linalg.norm(pos - p))
        total += strength / (1 + (d / 2.0) ** 2)
    return total


class Agent:
    def __init__(self, spec, circuit, w, rng, p, legs):
        self.spec, self.c, self.w, self.rng, self.p, self.legs = spec, circuit, w, rng, p, legs
        n = circuit.n
        self.v = rng.uniform(0, .3, n); self.syn = np.zeros(n); self.rates = np.zeros(n)
        self.refractory = np.zeros(n, dtype=int); self.total = np.zeros(n, dtype=int)
        self.silenced = circuit.vnc.copy() if p['condition'] == 'silence_vnc' else np.zeros(n, dtype=bool)
        self.phases = np.array([0, math.pi, 0, math.pi, 0, math.pi], float)
        self.contacts = np.zeros(6); self.strides = np.zeros(6)
        self.x, self.y = spec['start']; self.heading = spec['heading']
        self.speed = 0.0; self.motor_hz = np.zeros(6); self.path = 0.0
        self.bias = self.drive = 0.0; self.bias_target = 0.0; self.singing = False; self.heard_song = 0.0
        self.joints = bodyplan.angles(legs, self.rates)
        # Two stance rules. Absolute: the trochanter is depressed (femur angle <= 0). Relative: the
        # leg is lower than its own recent average, which is what the drawing uses, because the
        # measured levator/depressor balance is mostly static and a leg would otherwise never lift.
        self.femur_ema = {leg: self.joints[leg]['femur'] for leg in bodyplan.LEGS}
        self.feet = {leg: bodyplan.foot(leg, self.joints[leg]) for leg in bodyplan.LEGS}
        self.planted_abs = {leg: self.feet[leg][2] for leg in bodyplan.LEGS}
        self.planted_rel = dict(self.planted_abs)

    pos = property(lambda self: np.array([self.x, self.y]))

    def antennae(self, sep=0.5):
        n = np.array([-math.sin(self.heading), math.cos(self.heading)]) * sep / 2
        return self.pos + n, self.pos - n

    def sense(self, world):
        """Turn the world into a descending-neuron bias in [-1, 1] and a drive scale in [0, 1].

        Each target contributes a steering term (toward it if attractive, away if not) weighted by how
        salient it is: odour by concentration difference between the antennae, another fly by apparent
        size. Drive is a baseline exploration level, raised when approaching something far away or
        escaping something close, and lowered on arrival."""
        p = self.p
        bias = 0.0
        drive = 0.15            # baseline exploration, as a fraction of the usable drive band
        alignment = 1.0         # how squarely the fly faces whatever it is heading for
        if p['condition'] not in ('no_odour', 'isolated') and world['odour_sources']:
            left, right = self.antennae()
            cl, cr = concentration(left, world['odour_sources']), concentration(right, world['odour_sources'])
            src = min(world['odour_sources'], key=lambda s: float(np.linalg.norm(s[0] - self.pos)))[0]
            nearest = float(np.linalg.norm(src - self.pos))
            # Left antenna stronger means the source is to the left; steering left needs a negative bias.
            bias -= p['odour_gain'] * float(np.clip(20 * (cl - cr) / (cl + cr + 1e-6), -1, 1))
            drive = max(drive, min(p['approach_max'], 0.2 + nearest / 20))
            alignment = min(alignment, math.cos(math.atan2(src[1] - self.y, src[0] - self.x) - self.heading))
        if p['condition'] not in ('no_vision', 'isolated'):
            for other in world['others']:
                d = float(np.linalg.norm(other.pos - self.pos))
                bearing = math.atan2(other.y - self.y, other.x - self.x) - self.heading
                apparent = min(1.0, 2.0 / max(d, 0.4))          # apparent size stands in for looming
                attract = world['attraction'](self, other)
                if attract > 0:
                    bias += p['vision_gain'] * attract * steer(bearing)
                    drive = max(drive, min(p['approach_max'], 0.2 + d / 20))
                    alignment = min(alignment, math.cos(bearing))
                elif attract < 0:
                    bias += p['vision_gain'] * abs(attract) * steer(bearing + math.pi) * apparent
                    if d < NEAR_MM: drive = max(drive, min(0.8, apparent))
        if world['threat'] is not None:
            bearing = math.atan2(world['threat'][1] - self.y, world['threat'][0] - self.x) - self.heading
            bias += 2.0 * steer(bearing + math.pi); drive = 1.0
        if p['condition'] not in ('no_signals', 'isolated'):
            if self.spec['sex'] == 'female': drive *= max(0.0, 1 - p['song_gain'] * self.heard_song)
            else: drive = min(1.0, drive * (1 + p['song_gain'] * self.heard_song))
        # Steering gain and smoothing: with calibrated speed and differential turning the loop is fast
        # enough to oscillate, so the bias is damped. Both are control parameters, not measurements.
        # Flies slow down while turning; without this the loop overshoots at calibrated speeds.
        drive *= 0.25 + 0.75 * max(0.0, alignment)
        self.bias_target = float(np.clip(p['steer_gain'] * bias, -1, 1))
        self.drive = float(np.clip(drive, 0, 1))
        return self.bias_target, self.drive

    def step(self, step_index):
        p, c = self.p, self.c
        external = np.zeros(c.n)
        alpha = min(1.0, DT / max(p['steer_tau'], DT))
        self.bias += (self.bias_target - self.bias) * alpha
        if step_index * DT >= 0.5:
            level = p['drive_floor'] + (p['drive'] - p['drive_floor']) * self.drive
            external[c.dn] = level * (1 + self.bias * c.side[c.dn])
        for k, idx in enumerate(c.sensory):
            external[idx] += p['feedback'] * self.contacts[k] * self.strides[k]
        current = external + p['gain'] * self.w.dot(self.syn) + self.rng.normal(0, .015, c.n)
        self.refractory = np.maximum(self.refractory - 1, 0)
        active = (self.refractory == 0) & ~self.silenced
        self.v[active] += DT / .020 * (-self.v[active] + current[active])
        self.v[self.silenced] = 0
        spikes = (self.v >= 1) & active
        self.v[spikes] = 0; self.refractory[spikes] = 2
        self.syn *= math.exp(-DT / .010); self.syn[spikes] += 1; self.syn[self.silenced] = 0
        self.rates *= math.exp(-DT / .050); self.rates[spikes] += 1 / .050
        self.total += spikes
        if step_index % 10 == 0:                                  # same 10 ms body update as model.py
            self.motor_hz = np.array([float(self.rates[idx].mean()) if len(idx) else 0 for idx in c.motor])
            self.strides = np.clip(self.motor_hz / 50., 0, 1)
            self.phases += .010 * 2 * math.pi * np.array([gait(v)[0] for v in self.strides])
            self.contacts = (np.sin(self.phases) <= 0).astype(float)
            # Every leg joint is driven by the motor neurons of its own muscles.
            self.joints = bodyplan.angles(self.legs, self.rates)
            previous, self.feet = self.feet, {leg: bodyplan.foot(leg, self.joints[leg]) for leg in bodyplan.LEGS}
            for leg in bodyplan.LEGS:
                f = self.joints[leg]['femur']
                self.femur_ema[leg] += (f - self.femur_ema[leg]) * .02      # ~0.5 s average
                self.planted_abs[leg] = self.feet[leg][2]
                self.planted_rel[leg] = f <= self.femur_ema[leg]
            if p['body'] == 'kinematic':
                # Planted feet do not slide: the body moves opposite to their motion in the body frame.
                dx = dy = dtheta = 0.0; planted = 0
                for leg, (fx, fy, down) in self.feet.items():
                    px, py, pdown = previous[leg]
                    if down and pdown:
                        dx += fx - px; dy += fy - py
                        r2 = px * px + py * py
                        if r2 > 1e-6: dtheta += (px * (fy - py) - py * (fx - px)) / r2
                        planted += 1
                if planted:
                    dx /= planted; dy /= planted; dtheta /= planted
                    self.heading -= dtheta
                    self.x -= math.cos(self.heading) * dx - math.sin(self.heading) * dy
                    self.y -= math.sin(self.heading) * dx + math.cos(self.heading) * dy
                    self.speed = math.hypot(dx, dy) / .010
                    self.path += math.hypot(dx, dy)
                else:
                    self.speed = 0.0
            else:
                left, right = self.strides[:3].mean(), self.strides[3:].mean()
                self.speed = gait((left + right) / 2)[2]      # calibrated speed, see model.gait
                self.heading += turn_rate(left, right) * .010
                self.x += math.cos(self.heading) * self.speed * .010
                self.y += math.sin(self.heading) * self.speed * .010
                self.path += self.speed * .010


class Arena:
    def __init__(self, circuit=None): self.c = circuit or Circuit()

    def validate(self, options):
        p = DEFAULT | options
        if p['scenario'] not in SCENARIOS: raise ValueError('Unknown scenario')
        if p['condition'] not in CONDITIONS: raise ValueError('Unknown condition')
        if p['body'] not in BODY_MODES: raise ValueError('Unknown body mode')
        if p['drive_floor'] > p['drive']: raise ValueError('drive_floor must not exceed drive')
        for k, lo, hi in [('duration', 2, 30), ('drive', 0, 5), ('gain', 0, 20), ('feedback', 0, 2),
                          ('odour_gain', 0, 3), ('vision_gain', 0, 3), ('song_gain', 0, 2),
                          ('steer_gain', 0, 2), ('steer_tau', 0.01, 1), ('drive_floor', 0, 5), ('approach_max', 0.05, 1)]:
            p[k] = float(p[k])
            if not math.isfinite(p[k]) or not lo <= p[k] <= hi: raise ValueError(f'{k} must be between {lo} and {hi}')
        s = p['seed']
        if isinstance(s, bool) or not isinstance(s, (int, float)) or s != int(s) or not 0 <= s <= 999999: raise ValueError('seed must be an integer in [0,999999]')
        p['seed'] = int(s)
        return p

    def attraction(self, p):
        """Engineered: who is drawn to whom, and how strongly. Positive approaches, negative avoids."""
        def f(self_agent, other):
            if p['scenario'] == 'threat': return 0.0
            if self_agent.spec['sex'] == 'male' and other.spec['sex'] == 'female':
                if p['scenario'] == 'courtship' and self_agent.spec['id'] == 'male_2': return 0.0
                return 1.0
            if self_agent.spec['sex'] == 'male' and other.spec['sex'] == 'male': return -0.3   # males avoid each other
            if self_agent.spec['sex'] == 'female': return -0.5                                  # she moves away from a close male
            return 0.0
        return f

    def run(self, **options):
        p = self.validate(options)
        rng = np.random.default_rng(p['seed'])
        w = self.c.matrix('shuffled' if p['condition'] == 'shuffled' else 'intact', p['seed'])
        legs = bodyplan.index(self.c.nodes)
        agents = [Agent(spec, self.c, w, np.random.default_rng(p['seed'] + 100 * i), p, legs) for i, spec in enumerate(AGENTS)]
        steps = round(p['duration'] / DT)
        trace, events = [], []
        stance = {spec['id']: {'rel': [], 'abs': []} for spec in AGENTS}
        pair_keys = [('male_1', 'male_2'), ('male_1', 'female'), ('male_2', 'female')]
        stats = {k: {'min_distance': math.inf, 'contact_s': 0.0, 'near_s': 0.0} for k in map('|'.join, pair_keys)}
        arrivals, song_s, both_near_s = {}, {a['id']: 0.0 for a in AGENTS}, 0.0
        for step in range(steps):
            t = step * DT
            by_id = {a.spec['id']: a for a in agents}
            female = by_id['female']
            # Engineered signalling: a male within 5 mm of the female and roughly facing her sings.
            heard = {a.spec['id']: 0.0 for a in agents}
            for a in agents:
                a.singing = False
                if p['condition'] in ('no_signals', 'isolated') or a.spec['sex'] != 'male': continue
                d = float(np.linalg.norm(female.pos - a.pos))
                facing = math.cos(math.atan2(female.y - a.y, female.x - a.x) - a.heading)
                if d < NEAR_MM and facing > 0.3 and p['scenario'] in ('rivalry', 'courtship'):
                    a.singing = True; song_s[a.spec['id']] += DT
                    loudness = 1 - d / NEAR_MM
                    for other in agents:
                        if other is not a: heard[other.spec['id']] = max(heard[other.spec['id']], loudness)
            for a in agents: a.heard_song = heard[a.spec['id']]
            if step % 10 == 0:                                   # sensing at the body update rate
                threat = THREAT if (p['scenario'] == 'threat' and t >= 4.0) else None
                for a in agents:
                    sources = []
                    if p['scenario'] in ('rivalry', 'courtship') and a is not female:
                        # In courtship only male 1 is drawn to her; in rivalry both males are.
                        if not (p['scenario'] == 'courtship' and a.spec['id'] == 'male_2'): sources.append((female.pos, 1.0))
                    if p['scenario'] == 'food': sources.append((FOOD, 1.5))
                    a.sense({'others': [o for o in agents if o is not a], 'odour_sources': sources,
                             'attraction': self.attraction(p), 'threat': threat})
            for a in agents: a.step(step)
            if step % 10 == 0:
                for a in agents:
                    stance[a.spec['id']]['rel'].append([a.planted_rel[leg] for leg in bodyplan.LEGS])
                    stance[a.spec['id']]['abs'].append([a.planted_abs[leg] for leg in bodyplan.LEGS])
                for (i, j) in pair_keys:
                    d = float(np.linalg.norm(by_id[i].pos - by_id[j].pos)); s = stats[f'{i}|{j}']
                    s['min_distance'] = min(s['min_distance'], d)
                    if d < CONTACT_MM: s['contact_s'] += .010
                    if d < NEAR_MM: s['near_s'] += .010
                if all(float(np.linalg.norm(by_id[m].pos - female.pos)) < NEAR_MM for m in ('male_1', 'male_2')): both_near_s += .010
                if p['scenario'] == 'food':
                    for a in agents:
                        if a.spec['id'] not in arrivals and float(np.linalg.norm(a.pos - FOOD)) < 3.0: arrivals[a.spec['id']] = round(t, 2)
            if step % 20 == 0:
                trace.append({'t': round(t, 3), 'agents': [{'id': a.spec['id'], 'x': round(a.x, 3), 'y': round(a.y, 3), 'heading': round(a.heading, 4),
                                                            'speed': round(a.speed, 3), 'bias': round(a.bias, 3), 'drive': round(a.drive, 3),
                                                            'singing': bool(a.singing), 'dn_hz': round(float(a.rates[self.c.dn].mean()), 2),
                                                            # Leg phases and strides drive the drawn legs, the same engineered gait as the motion lab.
                                                            'phases': [round(v, 3) for v in a.phases.tolist()],
                                                            'strides': [round(v, 3) for v in a.strides.tolist()],
                                                            # Joint angles and foot positions, per leg, from that leg's muscles.
                                                            'legs': {leg: {'coxa': round(a.joints[leg]['coxa'], 3), 'femur': round(a.joints[leg]['femur'], 3),
                                                                           'tibia': round(a.joints[leg]['tibia'], 3), 'tarsus': round(a.joints[leg]['tarsus'], 3),
                                                                           'foot': [round(a.feet[leg][0], 3), round(a.feet[leg][1], 3)],
                                                                           'planted': bool(a.planted_rel[leg]), 'planted_absolute': bool(a.planted_abs[leg])}
                                                                     for leg in bodyplan.LEGS}} for a in agents]})
        duration = steps * DT
        gait_stats = {}
        tripod_a, tripod_b = [0, 4, 2], [3, 1, 5]          # LF, RM, LH against RF, LM, RH
        for a in agents:
            rel = np.array(stance[a.spec['id']]['rel']); ab = np.array(stance[a.spec['id']]['abs'])
            steps_per_leg = [int(np.count_nonzero(np.diff(rel[:, k].astype(int)) == 1)) for k in range(6)]
            within = float(np.mean([np.mean(rel[:, i] == rel[:, j]) for group in (tripod_a, tripod_b) for i in group for j in group if i < j]))
            across = float(np.mean([np.mean(rel[:, i] == rel[:, j]) for i in tripod_a for j in tripod_b]))
            speeds = [t['agents'][[x['id'] for x in t['agents']].index(a.spec['id'])]['speed'] for t in trace]
            gait_stats[a.spec['id']] = {
                'duty_factor_relative': [round(float(v), 3) for v in rel.mean(0)],
                'duty_factor_absolute': [round(float(v), 3) for v in ab.mean(0)],
                'steps_per_second': [round(k / duration, 2) for k in steps_per_leg],
                'tripod_within': round(within, 3), 'tripod_across': round(across, 3),
                'tripod_index': round(within - across, 3),
                'mean_speed_mm_s': round(float(np.mean(speeds)), 2), 'peak_speed_mm_s': round(float(np.max(speeds)), 2)}
        for k in stats:
            if not math.isfinite(stats[k]['min_distance']): stats[k]['min_distance'] = None
        return {
            'parameters': p, 'scenario_text': SCENARIOS[p['scenario']], 'condition_text': CONDITIONS[p['condition']],
            'body_text': BODY_MODES[p['body']], 'leg_coverage': bodyplan.coverage(legs),
            'unmatched_motor_neurons': int(len(legs['unmatched'])),
            'agents': [{'id': a.spec['id'], 'label': a.spec['label'], 'sex': a.spec['sex'],
                        'path_mm': round(a.path, 2), 'mean_dn_hz': float(a.total[self.c.dn].sum() / (max(1, int(np.count_nonzero(self.c.dn))) * duration)),
                        'spikes': int(a.total.sum())} for a in agents],
            'pairs': stats, 'song_seconds': {k: round(v, 2) for k, v in song_s.items()},
            'gait': gait_stats, 'legs_order': bodyplan.LEGS,
            'reported_ranges': {'speed_mm_s': [5, 25], 'step_hz': [5, 15], 'duty_factor': [0.5, 0.7],
                                'note': 'Reported ranges for walking Drosophila, for comparison only.'},
            'both_males_near_female_s': round(both_near_s, 2),
            'food': {'position': FOOD.tolist(), 'arrivals': arrivals} if p['scenario'] == 'food' else None,
            'threat': {'position': THREAT.tolist(), 'time': 4.0} if p['scenario'] == 'threat' else None,
            'trace': trace, 'events': events,
            'interpretation': 'Three copies of one measured male circuit coupled by assumed senses and signals. The female label marks '
                              'engineered emissions and responses, not a different connectome. Not a model of fly social behaviour.'}

    def suite(self, seed=7):
        rows = []
        for scenario in SCENARIOS:
            for condition in ['intact', 'no_vision', 'no_odour', 'isolated']:
                for s in range(seed, seed + 2):
                    r = self.run(seed=s, scenario=scenario, condition=condition)
                    rows.append({'scenario': scenario, 'condition': condition, 'seed': s,
                                 'pairs': r['pairs'], 'song_seconds': r['song_seconds'],
                                 'both_males_near_female_s': r['both_males_near_female_s'],
                                 'paths': {a['id']: a['path_mm'] for a in r['agents']},
                                 'food_arrivals': (r['food'] or {}).get('arrivals', {})})
        summary = {}
        for scenario in SCENARIOS:
            summary[scenario] = {}
            for condition in ['intact', 'no_vision', 'no_odour', 'isolated']:
                sel = [r for r in rows if r['scenario'] == scenario and r['condition'] == condition]
                if not sel: continue
                summary[scenario][condition] = {
                    'male1_female_near_s': float(np.mean([r['pairs']['male_1|female']['near_s'] for r in sel])),
                    'male1_female_min_mm': float(np.mean([r['pairs']['male_1|female']['min_distance'] for r in sel])),
                    'both_males_near_female_s': float(np.mean([r['both_males_near_female_s'] for r in sel])),
                    'song_s': float(np.mean([sum(r['song_seconds'].values()) for r in sel])),
                    'mean_path_mm': float(np.mean([np.mean(list(r['paths'].values())) for r in sel])),
                    'food_arrived': float(np.mean([len(r['food_arrivals']) for r in sel])),
                    'food_first_arrival_s': (float(np.mean([min(r['food_arrivals'].values()) for r in sel]))
                                             if all(r['food_arrivals'] for r in sel) else None)}
        return {'defaults': DEFAULT, 'scenarios': SCENARIOS, 'conditions': CONDITIONS, 'seeds': 2, 'summary': summary,
                'interpretation': 'Three seeds per scenario and condition. Model experiments with engineered senses; the differences between '
                                  'conditions show what the assumed sensing does, not what flies do.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--benchmark', action='store_true'); ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--scenario', default='rivalry', choices=list(SCENARIOS))
    a = ap.parse_args(); arena = Arena()
    if a.benchmark:
        from model import BASE
        (BASE / 'data/arena-benchmark.json').write_text(json.dumps(arena.suite(a.seed), allow_nan=False, indent=1))
        print('Wrote data/arena-benchmark.json')
    else:
        r = arena.run(seed=a.seed, scenario=a.scenario)
        print(json.dumps({'pairs': r['pairs'], 'song_seconds': r['song_seconds'],
                          'both_males_near_female_s': r['both_males_near_female_s'],
                          'paths': {x['id']: x['path_mm'] for x in r['agents']}, 'food': r['food']}, indent=1))
