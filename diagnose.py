"""Why doesn't the fly walk? Three assumptions tested against each other, not tuned.

The closed-loop body (bodysim.py) failed in three ways on the left-right complete circuit: lighting P9 on both
sides silenced every leg, MDN reversed only the front legs, and the front legs got almost no hip-muscle drive
under P9. Three assumptions could cause that:

  light       One light strength (2.5) for every cell type. Real experiments set intensity per fly line, and
              too much drive can push a recurrent network into shutdown. Swept from 1 to 5.
  glutamate   Glutamate is treated as inhibitory everywhere (as through GluCl channels). In flies it also
              excites, through other receptors. Many cells that cross the midline release glutamate, so the
              sign decides whether the two sides cancel. Tested both ways: inhibitory (current) and excitatory.
  front legs  The step direction rule (coxa retractors over protractors = forward) may not suit front legs,
              whose coxa moves differently. Reported per leg so it can be checked; a second rule, where the
              front legs follow the direction set by the middle and hind legs, is tested in the body.

Stage 1 runs the circuit alone (fast) over every combination and reports, per leg, how strongly it would step
and which way. Stage 2 runs the physics body for the combinations stage 1 picks out. Every stage-2 result has
its shuffled and sham controls. Nothing here changes the defaults; the point is to find which assumption is
blocking walking.

    .venv/bin/python diagnose.py            # writes data/diagnosis.json (stage 2 needs flygym)
"""
import json
import math
import os
from functools import lru_cache

import numpy as np

import bodysim as bs
import model
import optogenetics as og
from model import BASE

LIGHTS = [1.0, 1.5, 2.0, 2.5, 3.5, 5.0]
GLUTAMATE = {'inhibitory': 'Glutamate inhibitory everywhere (current model)',
             'excitatory': 'Glutamate excitatory everywhere'}
FRONT = {'own': 'Each leg takes its direction from its own coxa muscles (current model)',
         'follow': 'Front legs take the mean direction of the middle and hind legs'}
CASES = [('P9', 'both'), ('P9', 'right'), ('P9', 'left'), ('MDN', 'both'), ('DNa02', 'right'), ('DNa02', 'left')]
SEEDS = (1, 2)
WINDOW = (1.0, 2.0)              # s; the second half of the light


@lru_cache(maxsize=4)
def lab(glutamate='inhibitory', matched=False):
    """The optogenetics circuit with glutamate's sign set. The sign multiplies every outgoing synapse of a cell."""
    l = og.Optogenetics(matched=matched)
    if glutamate == 'excitatory':
        glu = np.array([str(n.get('nt')) == 'glutamate' for n in l.circuit.nodes])
        l.circuit.sign = np.where(glu, 1.0, l.circuit.sign)
    return l


def leg_state(lab_, experiment, side, light, seed, physiology='both'):
    """Circuit alone: per-leg step amplitude, frequency and direction as bodysim's rhythm would read them."""
    p = og.validate(dict(experiment=experiment, side=side, light=light, seed=seed, duration=WINDOW[1],
                         **{k: v for k, v in og.PHYSIOLOGY[physiology].items() if k != 'text'}))
    g, _ = bs.groups(lab_, muscle=True)
    rec = {f'{leg}|{grp}': idx for (leg, grp), idx in g.items() if idx}
    for k, idx in enumerate(lab_.circuit.motor): rec[f'{model.LEGS[k]}|pool'] = list(map(int, idx))
    r, _ = lab_.simulate(p, record_groups=rec, feedback=0.0)
    win = [t['groups'] for t in r['trace'] if WINDOW[0] <= t['t'] <= WINDOW[1]]
    m = {k: float(np.mean([w[k] for w in win])) for k in rec}
    legs = {}
    for leg in model.LEGS:
        drive = m.get(f'{leg}|pool', 0.0)
        ret, pro = m.get(f'{leg}|coxa_retract', 0.0), m.get(f'{leg}|coxa_protract', 0.0)
        legs[leg] = {'drive_hz': round(drive, 2), 'amp': round(math.tanh(drive / bs.A_SCALE), 3),
                     'freq_hz': round(bs.F_MAX * math.tanh(drive / bs.F_SCALE), 2),
                     'dir': round((ret - pro) / (ret + pro + 1.0), 3), 'retract_hz': round(ret, 2), 'protract_hz': round(pro, 2)}
    return legs


def stage1(seeds=SEEDS):
    rows = []
    for glu in GLUTAMATE:
        l = lab(glu)
        for experiment, side in CASES:
            if not l.available[experiment]: continue
            for light in LIGHTS:
                per_seed = [leg_state(l, experiment, side, light, s) for s in seeds]
                legs = {leg: {k: float(np.mean([ps[leg][k] for ps in per_seed])) for k in per_seed[0][leg]} for leg in model.LEGS}
                amp = np.mean([legs[g]['amp'] for g in model.LEGS])
                rear = [legs[g]['dir'] for g in ('LM', 'LH', 'RM', 'RH')]
                front = [legs[g]['dir'] for g in ('LF', 'RF')]
                rows.append({'glutamate': glu, 'experiment': experiment, 'side': side, 'light': light, 'legs': legs,
                             'mean_amp': round(float(amp), 3), 'rear_dir': round(float(np.mean(rear)), 3),
                             'front_dir': round(float(np.mean(front)), 3),
                             'left_right_drive': [round(sum(legs[g]['drive_hz'] for g in model.LEGS[:3]), 1),
                                                  round(sum(legs[g]['drive_hz'] for g in model.LEGS[3:]), 1)]})
    return rows


def pick(rows):
    """Stage-2 settings: for each glutamate sign, the lowest light at which bilateral P9 makes the legs step
    (mean amplitude at least 0.3) with a step direction (mean rear-leg direction above 0.2 in size). A sign whose
    legs only saturate without direction is not carried to the body."""
    chosen = {}
    for glu in GLUTAMATE:
        if not any(r['glutamate'] == glu and r['experiment'] == 'P9' and r['side'] == 'both' and r['mean_amp'] >= 0.3
                   and abs(r['rear_dir']) > 0.2 for r in rows):
            continue
        ok = [r for r in rows if r['glutamate'] == glu and r['experiment'] == 'P9' and r['side'] == 'both' and r['mean_amp'] >= 0.3]
        chosen[glu] = min((r['light'] for r in ok), default=2.5)
    return chosen


# --- stage 2: the body ------------------------------------------------------------------------------------------------
def _run_body(args):
    glu, front, light, experiment, side, seed, condition, background = args
    traj, legs = run_with(lab(glu), front, experiment, side, seed, condition, light, background)
    return bs.summarise(traj, _comparison(glu, front, seed, condition, light, background))


@lru_cache(maxsize=128)
def _comparison(glu, front, seed, condition, light, background):
    return run_with(lab(glu), front, None, 'both', seed, condition, light, background)[0]


def run_with(l, front, experiment, side, seed, condition, light, background):
    """bodysim.run_once on a given circuit object, with the chosen front-leg direction rule."""
    original_lab, original_front = bs.lab_for, bs.FRONT_RULE
    try:
        bs.lab_for = lambda matched, path=None: l
        bs.FRONT_RULE = front
        return bs.run_once(experiment, side, seed, condition, {}, background=background, light=light)
    finally:
        bs.lab_for, bs.FRONT_RULE = original_lab, original_front


def stage2(chosen, seeds=(1, 2, 3), workers=None):
    jobs = []
    tests = [('P9', 'forward_turn', None), ('MDN', 'backward', None), ('DNa02', 'turn', 'P9')]
    for glu, light in chosen.items():
        for front in FRONT:
            for experiment, test, background in tests:
                for condition in og.CONDITIONS:
                    for side in og.SIDES:
                        for seed in seeds:
                            jobs.append((glu, front, light, experiment, side, seed, condition, background))
    workers = workers or max(1, min(8, os.cpu_count() or 2))
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(workers) as pool:
            results = pool.map(_run_body, jobs, chunksize=1)
    else:
        results = [_run_body(j) for j in jobs]
    grouped = {}
    for job, summary in zip(jobs, results):
        glu, front, light, experiment, side, seed, condition, background = job
        grouped.setdefault((glu, front, experiment, condition), {}).setdefault(side, []).append(summary)
    test_of = {'P9': 'forward_turn', 'MDN': 'backward', 'DNa02': 'turn'}
    out = []
    for (glu, front, experiment, condition), runs in grouped.items():
        v = bs.verdict(test_of[experiment], runs)
        out.append({'glutamate': glu, 'front': front, 'light': chosen[glu], 'experiment': experiment, 'condition': condition,
                    'result': v['result'], 'detail': v['detail'], 'forward_mm_lit': v['forward_mm_lit'], 'yaw_deg': v['yaw_deg']})
    return out


COORDINATION = {
    'circuit': ('Circuit rhythm as in the model', {}),
    'no_posture': ('Circuit rhythm, posture offsets removed', {'OFFSET_SCALE': 0.0}),
    'strong_coupling': ('Circuit rhythm, leg coupling 6× stronger', {'COUPLING': 60.0}),
    'shared_frequency': ('Circuit rhythm, all legs at one shared frequency', {'SHARED_FREQ': True}),
    'supplied': ('Reference: supplied rhythm, every leg at full amplitude and 12 Hz', None),
}


def stage3(light, seeds=(1, 2)):
    """Coordination: with bilateral P9 making the legs step forward (front legs following), how far does the body
    walk under each rhythm variant? The supplied rhythm is the reference for what this body can do."""
    out = []
    for key, (text, settings) in COORDINATION.items():
        travel, yaw = [], []
        for seed in seeds:
            saved = {k: getattr(bs, k) for k in ('OFFSET_SCALE', 'COUPLING', 'SHARED_FREQ')}
            try:
                for k, v in (settings or {}).items(): setattr(bs, k, v)
                if settings is None:
                    tr, _ = bs.run_once(None, 'both', seed, 'intact', {'rhythm': 'supplied'})
                    s = bs.summarise(tr, [dict(p, x=0.0, y=0.0, yaw=0.0) for p in tr])
                else:
                    tr, _ = run_with(lab('inhibitory'), 'follow', 'P9', 'both', seed, 'intact', light, None)
                    comp, _ = run_with(lab('inhibitory'), 'follow', None, 'both', seed, 'intact', light, None)
                    s = bs.summarise(tr, comp)
            finally:
                for k, v in saved.items(): setattr(bs, k, v)
            travel.append(s['forward_mm_lit']); yaw.append(s['yaw_deg'])
        out.append({'variant': key, 'text': text, 'forward_mm': round(float(np.mean(travel)), 2),
                    'yaw_deg': round(float(np.mean(yaw)), 1), 'per_seed_forward_mm': [round(t, 2) for t in travel]})
    return out


def main(out=None):
    rows = stage1()
    chosen = pick(rows)
    result = {'glutamate': GLUTAMATE, 'front': FRONT, 'lights': LIGHTS, 'cases': CASES, 'stage1': rows, 'chosen_light': chosen,
              'stage2': stage2(chosen) if bs.available() else None,
              'stage3': stage3(chosen.get('inhibitory', 5.0)) if bs.available() else None}
    (out or BASE / 'data/diagnosis.json').write_text(json.dumps(result, indent=1))
    return result


if __name__ == '__main__':
    import time
    t0 = time.time()
    r = main()
    print('stage 1: mean step amplitude / rear direction / front direction, by light')
    for glu in GLUTAMATE:
        for experiment, side in CASES:
            line = [x for x in r['stage1'] if x['glutamate'] == glu and x['experiment'] == experiment and x['side'] == side]
            print(f'  {glu:10} {experiment:6} {side:5}', '  '.join(f"{x['light']}:{x['mean_amp']:.2f}/{x['rear_dir']:+.2f}/{x['front_dir']:+.2f}" for x in line))
    print('chosen light:', r['chosen_light'])
    for x in r['stage2'] or []:
        if x['condition'] == 'intact' or x['result'] != 'mismatch':
            print(f"  {x['glutamate']:10} front={x['front']:6} {x['experiment']:6} {x['condition']:8} {x['result']:22} {x['detail'][:120]}")
    print(f'{time.time() - t0:.0f} s')
