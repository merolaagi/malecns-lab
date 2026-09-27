"""Navigation: a heading compass, a home vector, and steering toward a goal.

The algorithm follows the published account of the fly's central complex. A ring of heading cells (EPG)
holds one activity bump; turning shifts it through the rotation cells (PEN), long-range inhibition
(Delta7) keeps a single bump; vector cells (PFN) multiply movement speed by heading-shifted tuning and
accumulate, which integrates a home vector; goal cells (FC2/hDelta) hold a desired direction; and
steering cells (PFL) compare goal against heading and push the descending neurons left or right.

Two sources of connectivity:
  'idealised' - a clean ring built from the published offsets. Always available.
  'measured'  - offsets and weights read from data/central-complex.json, built by
                build_centralcomplex.py from the raw tables. Requires that file and parsed columns.

Measured, when that file exists: which central-complex cells connect, with how many synapses, and each
cell's type. Assumed everywhere: that these types play the roles above, the ring dynamics, the gains and
time constants, the conversion from steering activity to turning, and the movement model itself. The
descending output is the lab's own body readout, not a measurement.
"""
import json
import math
from pathlib import Path

import numpy as np

from model import BASE

N = 16                       # heading columns around the ring
DEFAULT = dict(seed=7, outbound_s=12.0, homing_s=18.0, speed=8.0, connectivity='idealised',
               condition='intact', turn_noise=1.2, gain=1.0, compass_noise=0.15)
# compass_noise: error in the turn signal reaching the ring, in rad/s. With none, integration is exact
# and homing is trivially perfect; real compasses drift, so the default is deliberately not zero.
CONNECTIVITY = {'idealised': 'Published ring offsets', 'measured': 'Offsets from data/central-complex.json'}
CONDITIONS = {
    'intact': 'Full circuit',
    'no_rotation': 'Rotation cells removed: the bump cannot follow turning',
    'no_inhibition': 'Long-range inhibition removed: nothing enforces a single bump',
    'no_vector': 'Vector cells silenced: no home vector accumulates',
    'shuffled_columns': 'Column identities shuffled: the same cells, the wrong map',
    'random_goal': 'Goal direction randomised: steering works, the target does not',
}
DT = 0.02                    # s
RECURRENT_HZ = 15.0          # how fast the ring pulls its own shape back
SHARPEN = 1.05               # mild sharpening; stronger values pin the bump to one column
PFN_OFFSETS = (math.pi / 4, -math.pi / 4)      # left and right vector populations
PFL_OFFSET = math.pi / 2                       # steering populations compare at right angles


def angles(n=N):
    return np.arange(n) * 2 * math.pi / n


def ring_weights(n=N, width=1.2, inhibition=0.6):
    """Local excitation, global inhibition: the standard ring attractor."""
    theta = angles(n)
    difference = theta[:, None] - theta[None, :]
    excitation = np.exp(np.cos(difference) / width ** 2)
    excitation /= excitation.sum(1, keepdims=True)
    return excitation - inhibition / n


def measured_offsets(path=None):
    """Column offsets between groups, averaged over the measured edges."""
    path = Path(path or BASE / 'data/central-complex.json')
    if not path.exists():
        raise ValueError('data/central-complex.json is missing. Build it with build_centralcomplex.py on a machine with the raw tables.')
    data = json.loads(path.read_text())
    nodes = {n['bodyId']: n for n in data['nodes']}
    parsed = [n for n in data['nodes'] if n.get('column') is not None]
    if len(parsed) < 20:
        raise ValueError(f'Only {len(parsed)} central-complex cells have a parsed column; run build_centralcomplex.py --probe and check the instance format.')
    offsets, weights = {}, {}
    for pre, post, count in data['edges']:
        a, b = nodes.get(pre), nodes.get(post)
        if not a or not b or a.get('column') is None or b.get('column') is None: continue
        key = (a['group'], b['group'])
        shift = (abs(b['column']) - abs(a['column'])) % N
        offsets.setdefault(key, np.zeros(N))[shift] += count
        weights[key] = weights.get(key, 0) + count
    return {'offsets': {k: (v / v.sum()).tolist() for k, v in offsets.items()}, 'totals': weights,
            'cells': len(data['nodes']), 'columns_parsed': len(parsed)}


class Navigator:
    """Heading ring plus home-vector accumulation, stepped in place."""

    def __init__(self, connectivity='idealised', condition='intact', offsets=None, rng=None):
        self.condition, self.connectivity = condition, connectivity
        self.rng = rng or np.random.default_rng(0)
        inhibition = 0.0 if condition == 'no_inhibition' else 0.6
        self.ring = ring_weights(inhibition=inhibition)
        self.map = np.arange(N)
        if condition == 'shuffled_columns': self.map = self.rng.permutation(N)
        self.shift_profile = None
        if connectivity == 'measured':
            data = offsets if offsets is not None else measured_offsets()
            profile = data['offsets'].get(('rotate', 'heading')) or data['offsets'].get(('heading', 'rotate'))
            self.shift_profile = np.array(profile) if profile else None
            self.measured_summary = {'cells': data['cells'], 'columns_parsed': data['columns_parsed'],
                                     'group_pairs': len(data['offsets'])}
        bump = np.exp(np.cos(angles() - 0.0) * 2.0)
        self.heading_cells = bump / bump.sum()
        self.vector_cells = np.zeros(len(PFN_OFFSETS))       # accumulated home vector, per population
        self.theta = angles()

    def decode_heading(self):
        z = (self.heading_cells * np.exp(1j * self.theta[self.map])).sum()
        return float(np.angle(z)), float(abs(z) / max(self.heading_cells.sum(), 1e-9))

    def bumps(self, threshold=0.6):
        x = self.heading_cells / max(self.heading_cells.max(), 1e-9)
        above = x > threshold
        return int(sum(1 for i in range(N) if above[i] and not above[i - 1]))

    def step(self, omega, speed, dt=DT):
        """One step. omega is angular velocity in rad/s, speed in mm/s.

        The bump is transported by -omega * d(activity)/d(angle), the standard way a rotation cell pair
        shifts a ring attractor, then sharpened and renormalised so one bump survives."""
        recurrent = self.ring @ self.heading_cells
        b = self.heading_cells + RECURRENT_HZ * dt * (recurrent - self.heading_cells)
        if self.condition != 'no_rotation':
            # Rotation cells shift the bump by the turn. Implemented as an exact circular rotation
            # (a Fourier shift), because the discretised gradient version is pinned by the attractor at
            # 16 columns; the mechanism is the same, the numerics are not.
            frequencies = np.fft.fftfreq(N, d=1.0 / N)
            b = np.real(np.fft.ifft(np.fft.fft(b) * np.exp(-1j * frequencies * omega * dt)))
            if self.shift_profile is not None:
                # Measured offsets sharpen or blur that shift, depending on how the rotation cells land.
                b = np.real(np.fft.ifft(np.fft.fft(b) * np.fft.fft(self.shift_profile)))
        b = np.maximum(b, 0)
        if self.condition != 'no_inhibition': b = b ** SHARPEN        # keeps one bump without pinning it
        total = b.sum()
        self.heading_cells = b / total if total > 1e-9 else np.full(N, 1 / N)
        estimate, _ = self.decode_heading()
        if self.condition != 'no_vector':
            for i, offset in enumerate(PFN_OFFSETS):
                self.vector_cells[i] += dt * speed * math.cos(estimate + offset)
        return estimate

    def home_vector(self):
        """Decode the accumulated populations back into a travelled vector."""
        # v_i = integral of speed*cos(heading + offset_i) dt = cos(offset)*dx - sin(offset)*dy
        basis = np.array([[math.cos(o), -math.sin(o)] for o in PFN_OFFSETS])
        solution, *_ = np.linalg.lstsq(basis, self.vector_cells, rcond=None)
        return float(solution[0]), float(solution[1])

    def steer(self, goal, estimate):
        """PFL-style comparison: two populations offset by 90 degrees, their difference turns the fly."""
        left = math.cos(goal - estimate + PFL_OFFSET)
        right = math.cos(goal - estimate - PFL_OFFSET)
        return float(right - left) / 2


def run(**options):
    p = DEFAULT | options
    if p['connectivity'] not in CONNECTIVITY: raise ValueError('Unknown connectivity')
    if p['condition'] not in CONDITIONS: raise ValueError('Unknown condition')
    for key, lo, hi in [('outbound_s', 2, 60), ('homing_s', 2, 60), ('speed', 1, 30), ('turn_noise', 0, 5),
                        ('gain', 0, 5), ('compass_noise', 0, 2)]:
        p[key] = float(p[key])
        if not math.isfinite(p[key]) or not lo <= p[key] <= hi: raise ValueError(f'{key} must be between {lo} and {hi}')
    seed = p['seed']
    if isinstance(seed, bool) or not isinstance(seed, (int, float)) or seed != int(seed) or not 0 <= seed <= 999999:
        raise ValueError('seed must be an integer in [0,999999]')
    p['seed'] = seed = int(seed)

    rng = np.random.default_rng(seed)
    navigator = Navigator(p['connectivity'], p['condition'], rng=rng)
    x = y = heading = 0.0
    trace, heading_error = [], []
    steps_out, steps_home = int(p['outbound_s'] / DT), int(p['homing_s'] / DT)
    goal = None
    for step in range(steps_out + steps_home):
        homing = step >= steps_out
        # omega is an angular velocity in rad/s; the heading and the ring both advance by omega * DT.
        speed_scale = 1.0
        if not homing:
            omega = float(rng.normal(0, p['turn_noise']))
        else:
            hx, hy = navigator.home_vector()
            remaining = math.hypot(hx, hy)                        # how far home is, by the fly's own estimate
            if remaining < 1e-6:
                goal = None                                      # nothing accumulated: nothing to steer towards
            elif p['condition'] != 'random_goal':
                goal = math.atan2(-hy, -hx)                       # home lies opposite the travelled vector
            if p['condition'] == 'random_goal' and goal is not None and step == steps_out:
                goal = float(rng.uniform(-math.pi, math.pi))
            estimate, _ = navigator.decode_heading()
            omega = (float(rng.normal(0, p['turn_noise'])) if goal is None
                     else float(np.clip(p['gain'] * 4.0 * navigator.steer(goal, estimate), -6, 6)))
            if remaining < 3.0: speed_scale = 0.15               # local search once it believes it is home
            else: speed_scale = 1.0
        heading += omega * DT
        speed = p['speed'] * (speed_scale if homing else 1.0)
        x += math.cos(heading) * speed * DT
        y += math.sin(heading) * speed * DT
        # The ring sees the turn through a noisy efference copy, so the heading estimate drifts.
        estimate = navigator.step(omega + float(rng.normal(0, p['compass_noise'])), speed)
        error = math.atan2(math.sin(estimate - heading), math.cos(estimate - heading))
        heading_error.append(abs(error))
        if step % 5 == 0:
            hx, hy = navigator.home_vector()
            trace.append({'t': round(step * DT, 2), 'x': round(x, 3), 'y': round(y, 3),
                          'heading': round(heading, 4), 'estimate': round(estimate, 4),
                          'home_x': round(-hx, 3), 'home_y': round(-hy, 3),
                          'bump': [round(v, 4) for v in navigator.heading_cells],
                          'homing': homing})
    distance_home = math.hypot(x, y)
    furthest = max(math.hypot(f['x'], f['y']) for f in trace)
    closest = min((math.hypot(f['x'], f['y']) for f in trace if f['homing']), default=distance_home)
    estimated = navigator.home_vector()
    vector_error = math.hypot(estimated[0] - x, estimated[1] - y)   # accumulated against travelled
    return {
        'parameters': p, 'connectivity_text': CONNECTIVITY[p['connectivity']], 'condition_text': CONDITIONS[p['condition']],
        'trace': trace, 'goal': None if goal is None else round(goal, 4),
        'metrics': {
            'final_distance_from_home_mm': round(distance_home, 2),
            'furthest_distance_mm': round(furthest, 2),
            'closest_during_homing_mm': round(closest, 2),
            'homing_improvement': round(1 - closest / max(furthest, 1e-9), 3),
            'mean_heading_error_deg': round(float(np.degrees(np.mean(heading_error))), 2),
            'final_heading_error_deg': round(float(np.degrees(heading_error[-1])), 2),
            'home_vector_error_mm': round(vector_error, 2),
            'bumps_at_end': navigator.bumps(),
            'homed': bool(closest < 10.0),
        },
        'interpretation': 'Published central-complex algorithm run as a model. Whether the fly computes it this way is not tested '
                          'here; whether the measured wiring supports it is only tested with connectivity="measured".'}


def suite(seed=7, connectivity='idealised'):
    rows = {}
    for condition in CONDITIONS:
        runs = [run(seed=s, condition=condition, connectivity=connectivity) for s in range(seed, seed + 5)]
        rows[condition] = {key: round(float(np.mean([r['metrics'][key] for r in runs])), 3)
                           for key in runs[0]['metrics']}
    return {'seeds': 5, 'connectivity': connectivity, 'conditions': CONDITIONS, 'summary': rows,
            'interpretation': 'Five seeds per condition. Controls remove one mechanism at a time: the heading bump\'s rotation, '
                              'its single-bump inhibition, the vector accumulation, the column map, or the goal itself.'}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--benchmark', action='store_true'); ap.add_argument('--connectivity', default='idealised')
    ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    if a.benchmark:
        (BASE / 'data/navigation-benchmark.json').write_text(json.dumps(suite(a.seed, a.connectivity), indent=1))
        print('Wrote data/navigation-benchmark.json')
    else:
        r = run(seed=a.seed, connectivity=a.connectivity)
        for k, v in r['metrics'].items(): print(f'{k:32} {v}')
