"""Do the leg-linking interneurons let the circuit coordinate the legs?

Compares the optogenetics circuit with the coordination circuit (build_coordination.py: plus each leg's premotor
interneurons, interneurons reaching two or more leg segments, and P9's route to the front legs), and uniform
coupling between leg rhythms with coupling measured from the connectome (the leg-link matrix).

  Stage A, circuit alone: under P9 (both sides, one side) and MDN, at several light strengths, does the drive
           reach every leg (the front legs especially), and are the legs driven evenly (similar step frequency)?
  Stage B, physics body: does bilateral P9 now walk the fly forward, MDN backward, and DNa02 turn it, against
           shuffled and sham controls? Three seeds.

Nothing is tuned: same rules and constants as bodysim.py; the only changes are the circuit and the coupling source.

    .venv/bin/python coordination.py       # writes data/coordination.json (stage B needs flygym)
"""
import json
import os
from functools import lru_cache

import numpy as np

import bodysim as bs
import diagnose as dg
import model
import optogenetics as og
from model import BASE

CIRCUITS = {'optogenetics': 'data/optogenetics-circuit.json', 'coordination': 'data/coordination-circuit.json'}
COUPLINGS = {'uniform': 'Same coupling between every pair of legs', 'measured': 'Coupling scaled by the measured leg links'}
LIGHTS = [2.5, 3.5, 5.0]
CASES = [('P9', 'both'), ('P9', 'right'), ('MDN', 'both')]
TESTS = [('P9', 'forward_turn', None), ('MDN', 'backward', None), ('DNa02', 'turn', 'P9')]


def available_circuits():
    return {k: v for k, v in CIRCUITS.items() if (BASE / v).exists()}


@lru_cache(maxsize=4)
def lab(circuit):
    return og.Optogenetics(BASE / CIRCUITS[circuit])


def evenness(legs):
    f = [legs[g]['freq_hz'] for g in model.LEGS if legs[g]['amp'] > 0.1]
    return round(float(np.std(f) / np.mean(f)), 3) if len(f) >= 2 and np.mean(f) > 0 else None


def stage_a(seeds=(1, 2)):
    rows = []
    for circuit in available_circuits():
        l = lab(circuit)
        for experiment, side in CASES:
            for light in LIGHTS:
                per = [dg.leg_state(l, experiment, side, light, s) for s in seeds]
                legs = {g: {k: float(np.mean([p[g][k] for p in per])) for k in per[0][g]} for g in model.LEGS}
                rows.append({'circuit': circuit, 'experiment': experiment, 'side': side, 'light': light, 'legs': legs,
                             'front_amp': round(float(np.mean([legs[g]['amp'] for g in ('LF', 'RF')])), 3),
                             'rear_amp': round(float(np.mean([legs[g]['amp'] for g in ('LM', 'LH', 'RM', 'RH')])), 3),
                             'front_dir': round(float(np.mean([legs[g]['dir'] for g in ('LF', 'RF')])), 3),
                             'rear_dir': round(float(np.mean([legs[g]['dir'] for g in ('LM', 'LH', 'RM', 'RH')])), 3),
                             'frequency_spread': evenness(legs)})
    return rows


def chosen_light(rows, circuit):
    ok = [r['light'] for r in rows if r['circuit'] == circuit and r['experiment'] == 'P9' and r['side'] == 'both'
          and r['rear_amp'] >= 0.3 and abs(r['rear_dir']) > 0.2]
    return min(ok, default=max(LIGHTS))


def _job(args):
    circuit, coupling, light, experiment, side, seed, condition, background = args
    path = BASE / CIRCUITS[circuit]
    cfg = {'coupling': coupling}
    traj, _ = bs.run_once(experiment, side, seed, condition, cfg, background=background, light=light, path=path)
    return bs.summarise(traj, _comparison(circuit, coupling, light, seed, condition, background))


@lru_cache(maxsize=128)
def _comparison(circuit, coupling, light, seed, condition, background):
    traj, _ = bs.run_once(None, 'both', seed, condition, {'coupling': coupling}, background=background, light=light,
                          path=BASE / CIRCUITS[circuit])
    return traj


def stage_b(lights, seeds=(1, 2, 3), workers=None):
    jobs = []
    for circuit, light in lights.items():
        for coupling in COUPLINGS:
            for experiment, test, background in TESTS:
                for condition in og.CONDITIONS:
                    for side in og.SIDES:
                        for seed in seeds:
                            jobs.append((circuit, coupling, light, experiment, side, seed, condition, background))
    workers = workers or max(1, min(8, os.cpu_count() or 2))
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(workers) as pool:
            results = pool.map(_job, jobs, chunksize=1)
    else:
        results = [_job(j) for j in jobs]
    grouped = {}
    for job, summary in zip(jobs, results):
        circuit, coupling, light, experiment, side, seed, condition, background = job
        grouped.setdefault((circuit, coupling, experiment, condition), {}).setdefault(side, []).append(summary)
    test_of = {e: t for e, t, _ in TESTS}
    out = []
    for (circuit, coupling, experiment, condition), runs in grouped.items():
        v = bs.verdict(test_of[experiment], runs)
        out.append({'circuit': circuit, 'coupling': coupling, 'light': lights[circuit], 'experiment': experiment,
                    'condition': condition, 'result': v['result'], 'detail': v['detail'],
                    'forward_mm_lit': v['forward_mm_lit'], 'yaw_deg': v['yaw_deg'], 'agreement': v.get('agreement')})
    return out


def main(out=None):
    rows = stage_a()
    lights = {c: chosen_light(rows, c) for c in available_circuits()}
    circuits = {}
    for c in available_circuits():
        d = json.loads((BASE / CIRCUITS[c]).read_text())
        circuits[c] = {'neurons': len(d['nodes']), 'added': d.get('added'), 'leg_links_measured_in_build': bool(d.get('leg_links'))}
    result = {'circuits': circuits, 'couplings': COUPLINGS, 'lights': LIGHTS, 'stage_a': rows, 'chosen_light': lights,
              'coupling_matrices': {c: np.round(bs.coupling_matrix(lab(c), 'measured'), 2).tolist() for c in available_circuits()},
              'stage_b': stage_b(lights) if bs.available() else None}
    (out or BASE / 'data/coordination.json').write_text(json.dumps(result, indent=1))
    return result


if __name__ == '__main__':
    import time
    t0 = time.time()
    r = main()
    print('circuits:', r['circuits'])
    print('stage A: front amp / rear amp / front dir / rear dir / frequency spread')
    for x in r['stage_a']:
        print(f"  {x['circuit']:13} {x['experiment']:4} {x['side']:5} light {x['light']}: {x['front_amp']:.2f} / {x['rear_amp']:.2f} / "
              f"{x['front_dir']:+.2f} / {x['rear_dir']:+.2f} / {x['frequency_spread']}")
    print('chosen light:', r['chosen_light'])
    for x in r['stage_b'] or []:
        if x['condition'] == 'intact' or x['result'] not in ('mismatch', 'no clear turn'):
            print(f"  {x['circuit']:13} {x['coupling']:8} {x['experiment']:6} {x['condition']:8} {x['result']:22} {x['detail'][:110]}")
    print(f'{time.time() - t0:.0f} s')
