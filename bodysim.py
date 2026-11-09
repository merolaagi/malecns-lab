"""The lab's motor neurons driving a physics body: NeuroMechFly v2 (flygym) in MuJoCo.

Replaces the engineered leg readout that failed the optogenetics scorecard (it turned a stronger side into
faster stepping, so DNa02 turned the model the wrong way, and it could not walk backwards). Here the spiking
model's motor neurons set joint angles on a physical fly with legs, ground contact and foot adhesion, and the
physics decides which way the body goes.

Pipeline for one experiment:
  1. Light one cell type in the spiking model (optogenetics.py), recording every leg muscle group's motor
     neuron rate every 20 ms (muscle groups from bodyplan.py: coxa protract/retract, femur levate/depress,
     tibia extend/flex, tarsus levate/depress, per leg; the jump-muscle TTM motor neurons count as middle-leg
     femur depressors).
  2. Each joint's angle offset = span * tanh((agonist rate - antagonist rate) / 12 Hz), on the matching
     NeuroMechFly degree of freedom. Which direction of each DoF is "protract", "levate" or "extend" is
     measured on the model at start-up (move the joint, see where the foot goes), not assumed.
  3. Two body modes:
       walking   a standard tripod stepping pattern (recorded fly steps from flygym) runs at a fixed rate, and
                 the motor-neuron offsets are added to it. The lab's circuit has no rhythm generator, so the
                 rhythm is supplied; what is tested is how the connectome-driven motor output bends it.
       standing  the neutral pose plus the offsets, feet adhered.
  4. Compare with the same body without light (physics is deterministic), so only the light's effect counts.

Measured: the circuit, the muscle identity of each motor neuron. Assumed: everything in optogenetics.py and
model.py, the rate-to-angle mapping (one span per joint, never tuned per experiment), the supplied rhythm, the
mapping of bodyplan joints onto NeuroMechFly DoFs, and verdict thresholds. Open loop: the body's contacts do not
yet feed back into the circuit's sensory neurons.

Requires flygym (pip install flygym; Python 3.12):
    .venv/bin/python bodysim.py            # scorecard -> data/body-benchmark.json
"""
import json
import math
import os
from functools import lru_cache

import numpy as np

import bodyplan
import optogenetics as og
from model import BASE

DURATION = 2.0                   # s; light on from 0.5 s, as in optogenetics.py
SAMPLE = 0.02                    # s between recorded motor-neuron rates
STEP_HZ = 12.0                   # supplied stepping rhythm (walking mode)
TRIPOD = np.array([0, np.pi, 0, np.pi, 0, np.pi])   # lf, lm, lh, rf, rm, rh
SPAN = {'coxa': 0.35, 'femur': 0.35, 'tibia': 0.35, 'tarsus': 0.25}   # rad at saturation
RATE_SCALE = bodyplan.RATE_SCALE
DOF = {   # bodyplan joint -> (NeuroMechFly DoF name template, agonist group, antagonist group, test of the agonist)
    'coxa': ('c_thorax-{leg}_coxa-pitch', 'coxa_protract', 'coxa_retract', 'forward'),
    'femur': ('{leg}_coxa-{leg}_trochanterfemur-pitch', 'femur_levate', 'femur_depress', 'up'),
    'tibia': ('{leg}_trochanterfemur-{leg}_tibia-pitch', 'tibia_extend', 'tibia_flex', 'reach'),
    'tarsus': ('{leg}_tibia-{leg}_tarsus1-pitch', 'tarsus_levate', 'tarsus_depress', 'up'),
}
LEGS = ['lf', 'lm', 'lh', 'rf', 'rm', 'rh']
LAB_LEG = dict(zip(LEGS, ['LF', 'LM', 'LH', 'RF', 'RM', 'RH']))
MODES = {
    'walking': 'Supplied tripod stepping at 12 Hz, bent by the motor-neuron offsets',
    'standing': 'Neutral pose plus the motor-neuron offsets, feet adhered',
}
TEST_MODE = {'turn': 'walking', 'forward_turn': 'walking', 'backward': 'walking', 'escape': 'standing', None: 'walking'}
TURN_DEG = 10.0                  # verdict thresholds, relative to the unlit body
SLOWER = 0.3
RISE_MM = 0.05


def available():
    try:
        import flygym  # noqa: F401
        return True
    except Exception:
        return False


# --- the physics body -------------------------------------------------------------------------------------------
class Body:
    def __init__(self):
        from flygym import Simulation
        from flygym.anatomy import BodySegment
        from flygym.compose import ActuatorType, FlatGroundWorld
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain import PreprogrammedSteps, make_locomotion_fly
        self.POS = ActuatorType.POSITION
        self.fly = make_locomotion_fly(name='nmf', add_adhesion=True)
        world = FlatGroundWorld()
        world.add_fly(self.fly, [0, 0, 0.5], Rotation3D('quat', [1, 0, 0, 0]))
        self.sim = Simulation(world)
        self.steps = PreprogrammedSteps()
        self.order = self.fly.get_actuated_jointdofs_order(self.POS)
        names = [str(d.name) for d in self.order]
        self.slot = {(leg, joint): names.index(tmpl.format(leg=leg)) for leg in LEGS for joint, (tmpl, *_) in DOF.items()}
        segs = self.fly.get_bodysegs_order()
        self.thorax = segs.index(BodySegment('c_thorax'))
        self.signs = self.calibrate()

    def calibrate(self):
        """Which direction of each DoF moves the foot forward, up, or away from the body (measured, not assumed)."""
        import mujoco as mj
        m, d = self.sim.mj_model, self.sim.mj_data
        self.sim.reset()
        pose = self.steps.default_pose_by_dof_order(self.order)
        for dof, angle in zip(self.order, pose):
            d.qpos[m.joint(f'nmf/{dof.name}').qposadr[0]] = angle
        mj.mj_forward(m, d)
        signs = {}
        for leg in LEGS:
            tip, base = m.body(f'nmf/{leg}_tarsus5').id, m.body(f'nmf/{leg}_coxa').id
            for joint, (tmpl, _, _, test) in DOF.items():
                adr = m.joint(f'nmf/{tmpl.format(leg=leg)}').qposadr[0]
                saved = d.qpos[adr]
                def measure():
                    mj.mj_forward(m, d)
                    th = m.body('nmf/c_thorax').id
                    local = d.xmat[th].reshape(3, 3).T @ (d.xpos[tip] - d.xpos[th])
                    reach = float(np.linalg.norm(d.xpos[tip] - d.xpos[base]))
                    return {'forward': local[0], 'up': local[2], 'reach': reach}[test]
                before = measure(); d.qpos[adr] = saved + 0.2; after = measure(); d.qpos[adr] = saved
                signs[(leg, joint)] = 1.0 if after > before else -1.0
        mj.mj_forward(m, d)
        return signs

    def run(self, rates, mode, duration=DURATION):
        """rates: list (one per SAMPLE) of {(leg, muscle group): Hz}. Returns the thorax trajectory."""
        sim, POS = self.sim, self.POS
        dt = sim.timestep
        sim.reset()
        default = self.steps.default_pose_by_dof_order(self.order)
        sim.set_actuator_inputs('nmf', POS, default)
        sim.set_leg_adhesion_states('nmf', np.ones(6, bool))
        sim.warmup()
        offsets_cache = {}
        out = []
        for i in range(int(round(duration / dt))):
            t = i * dt
            k = min(int(t / SAMPLE), len(rates) - 1) if rates else -1
            if k not in offsets_cache: offsets_cache[k] = self.offsets(rates[k] if k >= 0 else {})
            if mode == 'walking':
                phases = TRIPOD + 2 * np.pi * STEP_HZ * t
                target = self.steps.get_joint_angles_by_dof_order(phases, np.ones(6), self.order)
                sim.set_leg_adhesion_states('nmf', self.steps.get_adhesion_onoff_by_phase(phases))
            else:
                target = default
            sim.set_actuator_inputs('nmf', POS, target + offsets_cache[k])
            sim.step()
            if i % 100 == 0:
                pos = sim.get_body_positions('nmf')[self.thorax]; q = sim.get_body_rotations('nmf')[self.thorax]
                yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                out.append({'t': round(t, 3), 'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]), 'yaw': yaw})
        return out

    def offsets(self, rates):
        off = np.zeros(len(self.order))
        for leg in LEGS:
            for joint, (_, ag, an, _) in DOF.items():
                diff = rates.get((LAB_LEG[leg], ag), 0.0) - rates.get((LAB_LEG[leg], an), 0.0)
                off[self.slot[(leg, joint)]] = self.signs[(leg, joint)] * SPAN[joint] * math.tanh(diff / RATE_SCALE)
        return off


@lru_cache(maxsize=1)
def body():
    return Body()


# --- motor neurons by leg and muscle group ------------------------------------------------------------------------
def groups(lab):
    g = {f'{leg}|{grp}': idx for (leg, grp), idx in lab.muscle_groups.items()}
    # Jump-muscle (TTM) motor neurons depress the middle-leg femur (they drive the escape jump).
    for i in np.flatnonzero(lab.ttm):
        leg = ('L' if lab.circuit.side[i] < 0 else 'R') + 'M'
        g.setdefault(f'{leg}|femur_depress', []).append(int(i))
    return {k: list(map(int, v)) for k, v in g.items()}


def motor_rates(lab, p):
    r, lit = lab.simulate(p, record_groups=groups(lab), feedback=0.0)
    series = [{tuple(k.split('|')): v for k, v in t['groups'].items()} for t in r['trace']]
    # trace samples are every 20 ms, matching SAMPLE
    return series, lit


def summarise(traj, baseline):
    """Effect of the light: yaw change, forward and lateral travel and height, relative to the unlit body."""
    def window(tr):
        on = [s for s in tr if s['t'] >= og.LIGHT_ON]
        a, b = on[0], on[-1]
        yaw = math.degrees(np.unwrap([s['yaw'] for s in on])[-1] - np.unwrap([s['yaw'] for s in on])[0])
        h = a['yaw']; dx, dy = b['x'] - a['x'], b['y'] - a['y']
        forward = dx * math.cos(h) + dy * math.sin(h); lateral = -dx * math.sin(h) + dy * math.cos(h)
        pre = [s['z'] for s in tr if s['t'] < og.LIGHT_ON] or [a['z']]
        return {'yaw_deg': yaw, 'forward_mm': forward, 'lateral_mm': lateral, 'rise_mm': max(s['z'] for s in on) - float(np.mean(pre))}
    lit, base = window(traj), window(baseline)
    return {k + '_lit': v for k, v in lit.items()} | {k: lit[k] - base[k] for k in lit} | {'baseline_' + k: v for k, v in base.items()}


def experiment_run(lab, experiment, side, seed, condition, physiology, mode):
    extra = {k: v for k, v in og.PHYSIOLOGY[physiology].items() if k != 'text'}
    p = og.validate(dict(experiment=experiment, side=side, seed=seed, condition=condition, duration=DURATION, **extra))
    rates, lit = motor_rates(lab, p)
    traj = body().run(rates, mode)
    return traj, lit


@lru_cache(maxsize=4)
def baseline(mode):
    return tuple(body().run([], mode))


def verdict(test, runs):
    """runs: {side: [summary per seed]}. Effects are relative to the unlit body."""
    mean = lambda side, key: float(np.mean([r[key] for r in runs[side]])) if runs.get(side) else None
    out = {k: {s: mean(s, k) for s in og.SIDES} for k in ('yaw_deg', 'forward_mm', 'lateral_mm', 'rise_mm')}
    yaw = out['yaw_deg']

    def turning():
        r, l = yaw['right'], yaw['left']
        if r < -TURN_DEG and l > TURN_DEG: return 'match', f'right side lit turns right ({r:+.0f}°), left side lit turns left ({l:+.0f}°)'
        if r > TURN_DEG and l < -TURN_DEG: return 'opposite', f'turns away from the lit side (right {r:+.0f}°, left {l:+.0f}°)'
        if r < 0 < l: return 'weak, right direction', f'turns toward the lit side, below ±{TURN_DEG:.0f}° (right {r:+.0f}°, left {l:+.0f}°)'
        if r > 0 > l: return 'weak, wrong direction', f'turns away from the lit side, below ±{TURN_DEG:.0f}° (right {r:+.0f}°, left {l:+.0f}°)'
        return 'no clear turn', f'right {r:+.0f}°, left {l:+.0f}° relative to the unlit body (threshold ±{TURN_DEG:.0f}°)'

    base_fwd = float(np.mean([r['baseline_forward_mm'] for rs in runs.values() for r in rs]))
    if test is None:
        out['result'], out['detail'] = 'exploratory', f"both sides: {yaw['both']:+.0f}°, forward {out['forward_mm']['both']:+.2f} mm vs unlit"
    elif test == 'turn':
        out['result'], out['detail'] = turning()
    elif test == 'forward_turn':
        t_result, t_detail = turning()
        out['result'] = 'partial' if t_result in ('match', 'weak, right direction') else 'not testable'
        out['detail'] = ('walking is supplied, so starting it cannot be tested; one side: ' + t_detail +
                         f"; both sides change forward travel by {out['forward_mm']['both']:+.2f} mm")
    elif test == 'backward':
        lit_fwd = base_fwd + out['forward_mm']['both']
        out['result'] = 'match' if lit_fwd < 0 else 'partial' if lit_fwd < (1 - SLOWER) * base_fwd else 'mismatch'
        out['detail'] = f'forward travel during light {lit_fwd:+.2f} mm against {base_fwd:+.2f} mm unlit (backward walking would be negative)'
    elif test == 'escape':
        rise = out['rise_mm']['both']
        out['result'] = 'match' if rise > RISE_MM else 'mismatch'
        out['detail'] = f'body rises {rise:+.3f} mm more than unlit (threshold {RISE_MM} mm)'
    return out


def scorecard(path=None, seeds=(1, 2), out=None, workers=None):
    lab = og.Optogenetics(path)
    jobs = []
    for name, e in og.EXPERIMENTS.items():
        if not lab.available[name]: continue
        for physiology in ('connectome', 'both'):
            for condition in og.CONDITIONS:
                for side in og.SIDES:
                    for seed in seeds:
                        jobs.append((name, side, seed, condition, physiology, TEST_MODE[e['test']]))
    workers = workers or max(1, min(8, (os.cpu_count() or 2)))
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(workers, initializer=_init, initargs=(str(lab.path),)) as pool:
            results = pool.map(_job, jobs)
    else:
        _init(str(lab.path)); results = [_job(j) for j in jobs]
    rows, traces = {}, {}
    for job, (summary, traj) in zip(jobs, results):
        name, side, seed, condition, physiology, mode = job
        rows.setdefault(name, {}).setdefault(physiology, {}).setdefault(condition, {}).setdefault(side, []).append(summary)
        if seed == seeds[0] and condition == 'intact':
            traces[f'{name}|{physiology}|{side}'] = [{k: round(s[k], 4) for k in ('t', 'x', 'y', 'yaw')} for s in traj]
    result = {'source': lab.source, 'modes': MODES, 'seeds': list(seeds), 'conditions': og.CONDITIONS,
              'physiology': {'connectome': 'Connectome only', 'both': '+ electrical synapses and resting excitability'},
              'calibration': {f'{leg} {joint}': s for (leg, joint), s in body().signs.items()},
              'baseline': {m: [{k: round(s[k], 4) for k in ('t', 'x', 'y', 'yaw')} for s in baseline(m)] for m in MODES},
              'parameters': {'span_rad': SPAN, 'rate_scale_hz': RATE_SCALE, 'step_hz': STEP_HZ, 'duration_s': DURATION,
                             'turn_deg': TURN_DEG, 'slower': SLOWER, 'rise_mm': RISE_MM},
              'experiments': {}, 'traces': traces}
    for name, by_phys in rows.items():
        e = og.EXPERIMENTS[name]
        result['experiments'][name] = {'expect': e['expect'], 'mode': TEST_MODE[e['test']],
                                       'physiology': {ph: {c: verdict(e['test'], sides) for c, sides in conds.items()}
                                                      for ph, conds in by_phys.items()}}
    (out or BASE / 'data/body-benchmark.json').write_text(json.dumps(result, indent=1))
    return result


_LAB = None
def _init(path):
    global _LAB
    _LAB = og.Optogenetics(path)


def _job(job):
    name, side, seed, condition, physiology, mode = job
    traj, _ = experiment_run(_LAB, name, side, seed, condition, physiology, mode)
    return summarise(traj, list(baseline(mode))), traj


def run(**options):
    """One experiment for the page: neural run, physics body, effect against the unlit body."""
    if not available(): raise ValueError('flygym is not installed in this environment: .venv/bin/pip install flygym')
    allowed = {'experiment', 'side', 'seed', 'condition', 'physiology', 'mode'}
    extra = set(options) - allowed
    if extra: raise ValueError('Unknown option: ' + ', '.join(sorted(extra)))
    name = options.get('experiment', 'DNa02')
    if name not in og.EXPERIMENTS: raise ValueError('Unknown experiment')
    physiology = options.get('physiology', 'both')
    if physiology not in og.PHYSIOLOGY: raise ValueError('Unknown physiology')
    mode = options.get('mode') or TEST_MODE[og.EXPERIMENTS[name]['test']]
    if mode not in MODES: raise ValueError('Unknown mode')
    lab = _lab()
    traj, lit = experiment_run(lab, name, options.get('side', 'right'), int(options.get('seed', 1)),
                               options.get('condition', 'intact'), physiology, mode)
    base = list(baseline(mode))
    return {'experiment': name, 'mode': mode, 'lit_cells': lit, 'effect': summarise(traj, base),
            'trace': traj, 'baseline': base}


@lru_cache(maxsize=1)
def _lab():
    return og.Optogenetics()


if __name__ == '__main__':
    import sys, time
    t0 = time.time()
    r = scorecard()
    print('calibration (+1: increasing the DoF angle does the agonist action):', r['calibration'])
    for name, row in r['experiments'].items():
        for ph, conds in row['physiology'].items():
            print(f'{name:12} {ph:10} {row["mode"]:8}', ' | '.join(f"{c}: {v['result']}" for c, v in conds.items()))
            print(' ' * 33, conds['intact']['detail'])
    print(f'{time.time() - t0:.0f} s')
