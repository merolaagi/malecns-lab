"""Why the physics fly does not walk, and what coordination buys.

Three questions, each answered in the body (NeuroMechFly) under the walking commands (P9 forward, MDN backward,
both sides lit, light 5.0 as chosen by diagnose.py), three seeds:

  A. What stops it? The full model, then with the jump muscle removed, against the reference of a supplied rhythm.
  B. Which coordination between legs works? No coupling; the original tripod coupling (uniform, or scaled by the
     measured leg links); tripod plus two rules (feet on the ground share one speed; a leg with drive but no
     direction of its own takes it from linked legs); and the same two rules with the phase relations taken from
     the measured links instead of assuming a tripod (net excitatory pairs step together, net inhibitory apart),
     against that rule with the links shuffled between leg pairs.
  C. Is it the wiring? The best rule again with the circuit's wiring shuffled, and with the light on the same number
     of random interneurons instead (sham).

Also a circuit-only check: does the measured balance of excitation and inhibition between the legs' premotor
pools match the tripod pattern?

Nothing is tuned per experiment. Requires flygym:
    .venv/bin/python walking.py            # writes data/walking.json
"""
import json
import math
import os
import sys

import numpy as np

import bodysim as bs
import optogenetics as og
from model import BASE

LIGHT = 5.0
COMMANDS = {'P9': ('forward', 1), 'MDN': ('backward', -1)}
LIGHTS = {'P9': [5.0], 'MDN': [5.0, 2.5]}     # 5.0: chosen by diagnose.py for P9; 2.5: the scorecard's standard light
MODES = ['independent', 'uniform', 'measured', 'tripod_sync', 'signed', 'signed_shuffled']
NO_JUMP = {'muscle': False}


def measure(traj, legs):
    on = [s for s in traj if s['t'] >= og.LIGHT_ON]
    a, b = on[0], on[-1]
    h = a['yaw']; dx, dy = b['x'] - a['x'], b['y'] - a['y']
    yaws = np.unwrap([s['yaw'] for s in on])
    lg = [l for l in legs if l['t'] >= og.LIGHT_ON + 0.2]
    freq = np.mean([l['freq'] for l in lg], 0); amp = np.mean([l['amp'] for l in lg], 0); direction = np.mean([l['dir'] for l in lg], 0)
    stepping = amp > 0.1
    return {'forward_mm': dx * math.cos(h) + dy * math.sin(h), 'sideways_mm': -dx * math.sin(h) + dy * math.cos(h),
            'yaw_deg': math.degrees(yaws[-1] - yaws[0]),
            'path_mm': float(sum(math.hypot(q['x'] - p['x'], q['y'] - p['y']) for p, q in zip(on, on[1:]))),
            'legs_stepping': int(stepping.sum()), 'own_direction': np.round(direction, 2).tolist(),
            'frequency_spread': float(np.std(freq[stepping]) / np.mean(freq[stepping])) if stepping.sum() >= 2 and np.mean(freq[stepping]) > 0 else None}


def job(args):
    command, mode, muscle, condition, seed, rhythm, light = args
    cfg = {'coupling': mode, 'muscle': muscle, 'rhythm': rhythm}
    traj, legs = bs.run_once(command, 'both', seed, condition, cfg, light=light)
    return measure(traj, legs)


def verdict(command, rows):
    sign = COMMANDS[command][1]
    f = float(np.mean([r['forward_mm'] for r in rows])) * sign
    yaw = float(np.mean([abs(r['yaw_deg']) for r in rows]))
    walks = [r['forward_mm'] * sign > bs.WALK_MM and abs(r['yaw_deg']) < bs.STRAIGHT_DEG for r in rows]
    return {'travel_mm': round(f * sign, 2), 'travel_in_commanded_direction_mm': round(f, 2), 'turn_deg': round(yaw, 1),
            'path_mm': round(float(np.mean([r['path_mm'] for r in rows])), 1),
            'walks': f"{sum(walks)} of {len(walks)} seeds", 'walks_all': bool(all(walks)),
            'per_seed_mm': [round(r['forward_mm'], 2) for r in rows],
            'frequency_spread': (lambda v: round(float(np.mean(v)), 3) if v else None)([r['frequency_spread'] for r in rows if r['frequency_spread'] is not None]),
            'own_direction': np.mean([r['own_direction'] for r in rows], 0).round(2).tolist()}


def link_balance():
    """Circuit only: does the measured excitation-inhibition balance between leg premotor pools match the tripod?"""
    lab = bs.lab_for(False)
    links = lab.leg_links or bs.leg_links_from_circuit(lab)
    E, I = np.array(links['excitatory']), np.array(links['inhibitory'])
    net = (E - I) / np.maximum(E + I, 1e-9)
    tripod = np.array([0, 1, 0, 1, 0, 1]); same = tripod[:, None] == tripod[None, :]
    off = ~np.eye(6, dtype=bool)
    agree = ((net > 0) == same)[off]
    return {'legs': links['legs'], 'net_balance_from_row_to_column': np.round(net, 3).tolist(),
            'pairs_matching_tripod': int(agree.sum()), 'pairs': int(agree.size), 'chance': 0.5,
            'total_link_into_leg': dict(zip(links['legs'], np.round((E + I).sum(0) / (E + I).sum(0).mean(), 2).tolist()))}


def main(out=None, seeds=(1, 2, 3), workers=None):
    if not bs.available(): raise SystemExit('flygym is not installed (pip install -r requirements-body.txt)')
    jobs, keys = [], []

    def add(part, command, mode, muscle, condition, rhythm='circuit', light=LIGHT):
        for seed in seeds:
            jobs.append((command, mode, muscle, condition, seed, rhythm, light)); keys.append((part, command, mode, muscle, condition, rhythm, light))
    for command in COMMANDS:
        for light in LIGHTS[command]:
            add('A', command, 'uniform', True, 'intact', light=light); add('A', command, 'uniform', False, 'intact', light=light)
            add('A', command, 'uniform', False, 'intact', 'supplied', light=light)
            for mode in (MODES if light == LIGHT else ('uniform', 'tripod_sync')): add('B', command, mode, False, 'intact', light=light)
            for mode in (('tripod_sync', 'signed') if light == LIGHT else ('tripod_sync',)):
                for condition in ('shuffled', 'sham'): add('C', command, mode, False, condition, light=light)
            add('C', command, 'tripod_sync', True, 'intact', light=light)
    unique = sorted(set(jobs)); workers = workers or max(1, min(4, (os.cpu_count() or 2)))
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(workers) as pool: results = dict(zip(unique, pool.map(job, unique, chunksize=1)))
    else:
        results = {j: job(j) for j in unique}
    (BASE / 'data/walking-runs.partial').write_text(json.dumps([[list(k), v] for k, v in results.items()]))   # raw runs, in case of a crash
    rows = {}
    for j, k in zip(jobs, keys): rows.setdefault(k, []).append(results[j])
    table = [{'part': p, 'command': c, 'coupling': m, 'coupling_text': bs.COORDINATION_MODES[m], 'jump_muscle': mu, 'condition': cond,
              'rhythm': rh, 'light': li, **verdict(c, v)} for (p, c, m, mu, cond, rh, li), v in rows.items()]
    result = {'light': LIGHT, 'seeds': list(seeds), 'commands': {k: v[0] for k, v in COMMANDS.items()}, 'modes': bs.COORDINATION_MODES,
              'thresholds': {'walk_mm': bs.WALK_MM, 'straight_deg': bs.STRAIGHT_DEG}, 'link_balance': link_balance(), 'rows': table,
              'jump_neurons_hz': jump_rates()}
    (out or BASE / 'data/walking.json').write_text(json.dumps(result, indent=1))
    return result


def jump_rates(ms=1000):
    """Circuit only: how hard the jump-muscle motor neurons fire under each command (they should fire for the giant fiber only)."""
    import model
    lab = bs.lab_for(False); c = lab.circuit
    ttm = np.flatnonzero(lab.ttm); phys = og.PHYSIOLOGY['both']
    out = {}
    for name, light in (('P9', LIGHT), ('MDN', LIGHT), ('Giant fiber', 2.5)):
        if not lab.available.get(name): continue
        lit, rest = bs.drive_vector(lab, name, 'both', light, 'intact', 1, phys['resting'])
        net = model.Stepper(c, rest, seed=1, gaps=[(a, b, og.GAP_KICK) for a, b in lab.gaps]); n = 0
        for step in range(500 + ms):
            net.drive = rest + (lit if step >= 500 else 0); sp = net.step(None)
            if step >= 500: n += sp[ttm].sum()
        out[name] = round(float(n) / max(len(ttm), 1) / (ms / 1000), 1)
    return out


if __name__ == '__main__':
    r = main()
    for row in r['rows']:
        print(f"{row['part']} {row['command']:4} {row['light']} {row['coupling']:16} jump {str(row['jump_muscle']):5} {row['condition']:8} {row['rhythm']:8} "
              f"travel {row['travel_mm']:+6.2f} mm  turn {row['turn_deg']:5.1f}°  walks {row['walks']}  spread {row['frequency_spread']}", file=sys.stderr)
    print('links matching tripod:', r['link_balance']['pairs_matching_tripod'], 'of', r['link_balance']['pairs'], '| jump neurons Hz:', r['jump_neurons_hz'], file=sys.stderr)
