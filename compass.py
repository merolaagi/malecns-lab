"""The fly's compass as the scaffold's location signal.

In scaffold.py, objects are recognised by where their features are, and the location of each glimpse is handed to
the learner. Here nothing hands it over: a walker moves over the object (turn, then step), and its location is
whatever the central-complex model (navigation.Navigator) integrates from its own turning and speed: a heading
bump on a ring, moved by rotation cells, and vector cells that add up the path. Features are stored at the
locations the compass reports, in one exploration, and recognised later along a different path.

The question for the connectome: does the compass wiring matter? Conditions:
- true location (the upper bound, what scaffold.py assumed)
- compass with the published ring offsets
- compass with rotation offsets as measured in data/central-complex.json, read two ways: each side of the bridge
  as its own full circle (the anatomy), and the 16 glomeruli as one ring (the lab's earlier reading)
- compass with column identities shuffled, and with rotation cells removed (controls)
- no locations at all (bag of features)

Measured: the rotation offsets (when 'measured'). Assumed: everything else, including the ring dynamics, the
walker, a 4 mm step between features, and the noise in the turn signal reaching the ring.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

import navigation as nav
import scaffold as sc
from learning import LearningCircuit

BASE = Path(__file__).resolve().parent
STEP_MM, SPEED, TURN_RATE = 4.0, 8.0, 4.0          # distance between features, walking speed (mm/s), turning speed (rad/s)
SOURCES = {'true_location': 'True location (given)', 'compass_idealised': 'Compass, published ring',
           'compass_measured_bridge': 'Compass, measured offsets, each bridge side a full circle',
           'compass_measured_calibrated': 'Compass, measured shift direction, gain calibrated',
           'compass_measured': 'Compass, measured offsets, bridge read as one ring of 16', 'compass_shuffled': 'Compass, columns shuffled',
           'compass_no_rotation': 'Compass, rotation cells removed', 'bag_of_features': 'No locations (bag of features)'}
NAV = {'compass_idealised': ('idealised', 'intact'), 'compass_measured': ('measured', 'intact'), 'compass_measured_bridge': ('measured_bridge', 'intact'),
       'compass_measured_calibrated': ('measured_bridge', 'intact'), 'compass_gain': ('idealised', 'intact'),
       'compass_shuffled': ('idealised', 'shuffled_columns'), 'compass_no_rotation': ('idealised', 'no_rotation')}


class Walker:
    """Walks between neighbouring features and reports where the compass thinks it is, in feature spacings."""
    def __init__(self, source, rng, compass_noise, offsets=None, gain=None):
        self.rng, self.noise, self.source = rng, compass_noise, source
        self.heading = float(rng.uniform(-math.pi, math.pi))         # arbitrary starting direction
        self.cell = None
        if source in NAV:
            connectivity, condition = NAV[source]
            self.nav = nav.Navigator(connectivity, condition, offsets=offsets, rng=rng)
            # calibrated: keep what the wiring fixes (which way each side shifts the bump), set how far to one turn per turn,
            # as flies do by comparing the compass with what they see
            if source == 'compass_measured_calibrated': self.nav.rotation_gain = 1.0 if self.nav.rotation_gain > 0 else 0.0
            if gain is not None: self.nav.rotation_gain = gain
            # the ring starts aligned with the body (as if set by a visual landmark)
            bump = np.exp(np.cos(self.nav.theta[self.nav.map] - self.heading) * 2.0)
            self.nav.heading_cells = bump / bump.sum()
        else:
            self.nav = None

    def start(self, cell):
        self.cell = cell
        return self.where()

    def go(self, cell):
        dr, dc = cell[0] - self.cell[0], cell[1] - self.cell[1]
        target = math.atan2(dr, dc)
        turn = math.atan2(math.sin(target - self.heading), math.cos(target - self.heading))
        if self.nav:
            steps = max(1, round(abs(turn) / TURN_RATE / nav.DT)); omega = turn / (steps * nav.DT)
            for _ in range(steps): self.nav.step(omega + float(self.rng.normal(0, self.noise)), 0.0)
            for _ in range(round(STEP_MM / SPEED / nav.DT)): self.nav.step(float(self.rng.normal(0, self.noise)), SPEED)
        self.heading = target
        self.cell = cell
        return self.where()

    def where(self):
        if not self.nav: return (float(self.cell[0]), float(self.cell[1]))
        x, y = self.nav.home_vector()
        return (y / STEP_MM, x / STEP_MM)                             # (row, column) in feature spacings


def serpentine(size):
    return [(r, c if r % 2 == 0 else size - 1 - c) for r in range(size) for c in range(size)]


def random_path(start, size, length, rng):
    locs = [start]
    while len(locs) < length:
        r, c = locs[-1]
        moves = [(dr, dc) for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)) if 0 <= r + dr < size and 0 <= c + dc < size]
        dr, dc = moves[rng.integers(len(moves))]
        locs.append((r + dr, c + dc))
    return locs


def experiment(circuit, seed, source, compass_noise=0.15, n_objects=20, features=6, size=4, noise=0.1, max_glimpses=8, trials=100, offsets=None, gain=None):
    rng = np.random.default_rng(seed + 71)
    enc = sc.Encoder(circuit, 'fly_measured', seed)
    pats = (rng.random((features, 64)) < 0.25).astype(float)
    objs = rng.integers(features, size=(n_objects, size, size))
    clean = enc.code(pats)

    def sense(o, cell):
        p = pats[objs[o][cell]].copy(); flip = rng.random(64) < noise; p[flip] = 1 - p[flip]
        return enc.code(p[None])[0]

    # learning: one walk over every feature of every object; store each feature where the compass says it is
    maps, errors, lost = [], [], 0
    for o in range(n_objects):
        w = Walker(source, rng, compass_noise, offsets, gain); path = serpentine(size)
        origin = w.start(path[0]); stored = {}
        for i, cell in enumerate(path):
            est = origin if i == 0 else w.go(cell)
            rel = (est[0] - origin[0], est[1] - origin[1])
            errors.append(math.hypot(rel[0] - (cell[0] - path[0][0]), rel[1] - (cell[1] - path[0][1])))
            key = (round(rel[0]), round(rel[1]))
            if key in stored: lost += 1                               # two features landed on one location
            else: stored[key] = clean[objs[o][cell]]
        maps.append(stored)
    if source == 'bag_of_features':
        bag = np.array([sum(m.values()) for m in maps]); bag /= np.linalg.norm(bag, axis=1, keepdims=True)

    hits = {'one_module': np.zeros(max_glimpses), 'three_modules': np.zeros(max_glimpses)}
    for _ in range(trials):
        target = int(rng.integers(n_objects))
        per_module = []
        for m in range(3):
            path = random_path((int(rng.integers(size)), int(rng.integers(size))), size, max_glimpses, rng)
            w = Walker(source, rng, compass_noise, offsets, gain); origin = w.start(path[0])
            ev = [dict.fromkeys(mp, 0.0) for mp in maps]; glimpses = []; bag_ev = np.zeros(n_objects)
            for g, cell in enumerate(path):
                est = origin if g == 0 else w.go(cell)
                d = (round(est[0] - origin[0]), round(est[1] - origin[1]))
                obs = sense(target, cell)
                if source == 'bag_of_features':
                    bag_ev += bag @ obs; glimpses.append(bag_ev.copy()); continue
                best = np.full(n_objects, -1e9)
                for o, mp in enumerate(maps):                         # hypotheses: (object, where the walk started on its map)
                    for s in mp:
                        code = mp.get((s[0] + d[0], s[1] + d[1]))
                        ev[o][s] += (obs @ code) / enc.k if code is not None else -1.0
                        if ev[o][s] > best[o]: best[o] = ev[o][s]
                glimpses.append(best)
            per_module.append(glimpses)
        for g in range(max_glimpses):
            hits['one_module'][g] += int(per_module[0][g].argmax() == target)
            hits['three_modules'][g] += int(sum(pm[g] for pm in per_module).argmax() == target)
    return {'source': source, 'compass_noise': compass_noise, 'gain': gain, 'glimpses': list(range(1, max_glimpses + 1)),
            'one_module': (hits['one_module'] / trials).round(3).tolist(), 'three_modules': (hits['three_modules'] / trials).round(3).tolist(),
            'location_error_spacings': float(np.mean(errors)), 'features_lost_per_object': lost / n_objects,
            'chance': 1 / n_objects, 'objects': n_objects}


def average(runs):
    out = dict(runs[0])
    for k in ('one_module', 'three_modules'): out[k] = np.mean([r[k] for r in runs], 0).round(3).tolist()
    for k in ('location_error_spacings', 'features_lost_per_object'): out[k] = float(np.mean([r[k] for r in runs]))
    return out


def benchmark(seeds=(7, 8, 9), out=None):
    circuit = LearningCircuit()
    sources = [s for s in SOURCES if not s.startswith('compass_measured') or (BASE / 'data/central-complex.json').exists()]
    main = {s: average([experiment(circuit, seed, s) for seed in seeds]) for s in sources}
    sweep = {s: [average([experiment(circuit, seed, s, compass_noise=n, trials=60) for seed in seeds]) for n in (0.0, 0.15, 0.3, 0.6, 1.0)]
             for s in ('compass_idealised',)}
    gains = [average([experiment(circuit, seed, 'compass_gain', gain=g, trials=60) for seed in seeds]) for g in (0.25, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5)]
    measured = None
    if 'compass_measured' in sources:
        rep = nav.measured_report()
        measured = {'rotation_gain_columns': nav.Navigator('measured').rotation_gain, 'rotation_opposite_by_side': rep['rotation_opposite_by_side'],
                    'rotation_offsets_by_type': rep['rotation_offsets_by_type'], 'bridge': rep['bridge']}
    result = {'seeds': list(seeds), 'labels': SOURCES, 'main': main, 'noise_sweep': sweep, 'gain_sweep': gains, 'measured': measured,
              'setup': {'step_mm': STEP_MM, 'speed_mm_s': SPEED, 'turn_rate_rad_s': TURN_RATE, 'default_compass_noise_rad_s': 0.15}}
    (out or BASE / 'data/compass-benchmark.json').write_text(json.dumps(result, indent=1))
    return result


if __name__ == '__main__':
    r = benchmark()
    for s, v in r['main'].items():
        print(f"{SOURCES[s]:<38} 1 module {v['one_module'][3]:.2f} / 3 modules {v['three_modules'][3]:.2f} at 4 glimpses; "
              f"location error {v['location_error_spacings']:.2f} spacings; lost {v['features_lost_per_object']:.1f}", file=sys.stderr)
    print(r['measured'], file=sys.stderr)
