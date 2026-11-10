"""The lab's circuit driving a physics fly body (NeuroMechFly v2, flygym, MuJoCo), in closed loop.

What drives what, every millisecond:

  circuit -> joints     Motor neurons are grouped by leg and muscle (bodyplan.py). Each joint's angle is offset by
                        span * tanh((agonist - antagonist rate) / 12 Hz) on the matching NeuroMechFly degree of
                        freedom. Which direction of each DoF is forward, up or reach is measured on the model at
                        start-up by moving the joint and watching the foot.
  circuit -> rhythm     Each leg has a stepping oscillator (recorded fly steps from flygym). The circuit sets whether
                        it steps, how fast and which way: amplitude and frequency rise with the leg's motor-neuron
                        drive (no drive: the leg stands still), and the step cycle runs forward when the coxa
                        retractors (the forward-walking stance muscles) are driven harder than the protractors, and
                        backward when the protractors are. Legs are coupled into a tripod. The oscillator shape and
                        the coupling are assumed: the circuit extract has no rhythm-generating interneurons.
  body -> circuit       Foot load (ground contact force, per leg) drives the leg's campaniform sensilla; joint
                        movement (femur-tibia angular speed) drives its chordotonal neurons.
  jump muscle           The jump-muscle (TTM) motor neurons drive a force-producing muscle on each middle leg instead
                        of a position offset: every motor spike is a twitch that raises activation by 80% of what is
                        left, decaying in 5 ms. Its maximum torque is assumed: enough for the two middle legs to push
                        at ten times body weight.
  circuit build         Left-right complete: build_optogenetics.py adds every cell of a selected type on the other
                        side too. Pruning to exact counts per side (optogenetics.symmetric) is kept as an ablation:
                        it removes cells a pathway may need.

Tests mirror real optogenetics. Turning cell types (DNa02, DNa01) are lit on one side while the fly walks under
its own forward command (P9 lit on both sides at the same intensity); the comparison is P9 alone. P9, MDN, the
giant fiber and DNg13 are lit on a standing fly; the comparison is no light. Each piece can be switched off
(ablations) to see what it contributes.

Measured: cells, wiring, transmitter predictions, muscle identity of each motor neuron. Assumed: the network
dynamics (model.py), the added physiology (optogenetics.py), every gain and span here, the oscillator, the
coupling, the jump-muscle strength, and the verdict thresholds. None of them is tuned per experiment.

Requires flygym (pip install flygym; Python 3.12):
    .venv/bin/python bodysim.py            # scorecard -> data/body-benchmark.json
"""
import json
import math
import os
from functools import lru_cache

import numpy as np

import bodyplan
import model
import optogenetics as og
from model import BASE

DURATION = 2.0                   # s; light on from 0.5 s, as in optogenetics.py
NEURAL_DT = 0.001
RECORD_EVERY = 0.01
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
LAB_LEG = dict(zip(LEGS, model.LEGS))
# rhythm
A_SCALE = 10.0                   # Hz of leg motor drive for near-full step amplitude
F_MAX, F_SCALE = 12.0, 15.0      # step frequency rises to F_MAX with leg drive
SUPPLIED_HZ = 12.0               # the fixed rhythm used when the rhythm is supplied (ablation)
COUPLING = 10.0                  # tripod coupling strength (1/s)
# feedback
LOAD_GAIN, CHO_GAIN, V_REF = 0.6, 0.6, 20.0   # threshold units at one third of body weight / at 20 rad/s
# jump muscle
JUMP_G, LEVER_MM, TAU_ACT, TWITCH = 10.0, 1.0, 0.005, 0.8   # each motor spike raises activation by 80% of what is left
# verdicts, relative to the comparison run
TURN_DEG, WALK_MM, STRAIGHT_DEG, RISE_MM = 10.0, 2.0, 45.0, 0.05

FULL = dict(matched=False, rhythm='circuit', feedback=True, muscle=True, physiology='both')
ABLATIONS = {
    'full': ('Everything on', {}),
    'pruned': ('Circuit pruned to exact left-right match', {'matched': True}),
    'supplied_rhythm': ('Rhythm supplied at 12 Hz, always walking forward', {'rhythm': 'supplied'}),
    'no_feedback': ('No feedback from the body', {'feedback': False}),
    'no_muscle': ('No jump muscle (position offset only)', {'muscle': False}),
    'connectome_only': ('No added physiology (no electrical synapses, no resting excitability)', {'physiology': 'connectome'}),
}
SETUP = {   # test -> (background cell type lit on both sides from the start, or None)
    'turn': 'P9', 'forward_turn': None, 'backward': None, 'escape': None, None: None,
}


def available():
    try:
        import flygym  # noqa: F401
        return True
    except Exception:
        return False


# --- the physics body -------------------------------------------------------------------------------------------
class Body:
    def __init__(self, muscle=True):
        from flygym import Simulation
        from flygym.anatomy import BodySegment, JointDOF
        from flygym.compose import ActuatorType, FlatGroundWorld
        from flygym.utils.math import Rotation3D
        from flygym_demo.complex_terrain import PreprogrammedSteps, make_locomotion_fly
        self.POS, self.MOTOR = ActuatorType.POSITION, ActuatorType.MOTOR
        self.fly = make_locomotion_fly(name='nmf', add_adhesion=True)
        self.muscle = muscle
        if muscle:
            dofs = [d for d in self.fly.get_actuated_jointdofs_order(self.POS)
                    if str(d.name) in ('lm_coxa-lm_trochanterfemur-pitch', 'rm_coxa-rm_trochanterfemur-pitch')]
            self.fly.add_actuators(dofs, self.MOTOR, forcerange=(-1e3, 1e3))
        world = FlatGroundWorld()
        world.add_fly(self.fly, [0, 0, 0.5], Rotation3D('quat', [1, 0, 0, 0]))
        self.sim = Simulation(world)
        self.steps = PreprogrammedSteps()
        self.order = self.fly.get_actuated_jointdofs_order(self.POS)
        names = [str(d.name) for d in self.order]
        self.slot = {(leg, joint): names.index(tmpl.format(leg=leg)) for leg in LEGS for joint, (tmpl, *_) in DOF.items()}
        self.thorax = self.fly.get_bodysegs_order().index(BodySegment('c_thorax'))
        all_dofs = [str(d.name) for d in self.fly.get_jointdofs_order()]
        self.tibia_dof = [all_dofs.index(f'{leg}_trochanterfemur-{leg}_tibia-pitch') for leg in LEGS]
        m = self.sim.mj_model
        self.weight = float(sum(m.body_mass[i] for i in range(m.nbody) if m.body(i).name.startswith('nmf'))) * 9810.0
        self.tmax = JUMP_G * self.weight * LEVER_MM / 2
        self.signs = self.calibrate()
        if muscle:
            self.motor_order = [str(d.name)[:2] for d in self.fly.get_actuated_jointdofs_order(self.MOTOR)]

    def calibrate(self):
        """Which direction of each DoF moves the foot forward, up, or away from the body (measured, not assumed)."""
        import mujoco as mj
        m, d = self.sim.mj_model, self.sim.mj_data
        self.sim.reset()
        for dof, angle in zip(self.order, self.steps.default_pose_by_dof_order(self.order)):
            d.qpos[m.joint(f'nmf/{dof.name}').qposadr[0]] = angle
        th = m.body('nmf/c_thorax').id
        signs = {}
        for leg in LEGS:
            tip, base = m.body(f'nmf/{leg}_tarsus5').id, m.body(f'nmf/{leg}_coxa').id
            for joint, (tmpl, _, _, test) in DOF.items():
                adr = m.joint(f'nmf/{tmpl.format(leg=leg)}').qposadr[0]
                saved = d.qpos[adr]

                def measure():
                    mj.mj_forward(m, d)
                    local = d.xmat[th].reshape(3, 3).T @ (d.xpos[tip] - d.xpos[th])
                    return {'forward': local[0], 'up': local[2], 'reach': float(np.linalg.norm(d.xpos[tip] - d.xpos[base]))}[test]
                before = measure(); d.qpos[adr] = saved + 0.2; after = measure(); d.qpos[adr] = saved
                signs[(leg, joint)] = 1.0 if after > before else -1.0
        mj.mj_forward(m, d)
        return signs

    def offsets(self, rates):
        off = np.zeros(len(self.order))
        for leg in LEGS:
            for joint, (_, ag, an, _) in DOF.items():
                diff = rates.get((LAB_LEG[leg], ag), 0.0) - rates.get((LAB_LEG[leg], an), 0.0)
                off[self.slot[(leg, joint)]] = self.signs[(leg, joint)] * SPAN[joint] * math.tanh(diff / RATE_SCALE)
        return off

    def sensors(self):
        """Per leg: load (normal ground force / one third of body weight) and femur-tibia angular speed (rad/s)."""
        found, forces, *_ = self.sim.get_ground_contact_info('nmf')
        load = np.abs(forces[:, 0]) * (found > 0) / (self.weight / 3)
        speed = np.abs(self.sim.get_joint_velocities('nmf')[self.tibia_dof])
        return load, speed


@lru_cache(maxsize=2)
def body(muscle=True):
    return Body(muscle)


# --- the circuit side -----------------------------------------------------------------------------------------------
def groups(lab, muscle=True):
    """Motor neurons by (leg, muscle group). Jump-muscle (TTM) cells: the muscle, or middle-leg femur depressors."""
    g = {(leg, grp): list(map(int, idx)) for (leg, grp), idx in lab.muscle_groups.items()}
    ttm = {'LM': [], 'RM': []}
    for i in np.flatnonzero(lab.ttm):
        ttm[('L' if lab.circuit.side[i] < 0 else 'R') + 'M'].append(int(i))
    if not muscle:
        for leg, idx in ttm.items(): g.setdefault((leg, 'femur_depress'), []).extend(idx)
    return g, ttm


def sensory_cells(lab):
    """Per leg: (campaniform sensilla indices, chordotonal organ indices)."""
    c = lab.circuit
    out = []
    for k in range(6):
        idx = c.sensory[k]
        sub = [str(c.nodes[i].get('subclass')) for i in idx]
        out.append((np.array([i for i, s in zip(idx, sub) if 'campaniform' in s], int),
                    np.array([i for i, s in zip(idx, sub) if 'chordotonal' in s], int)))
    return out


@lru_cache(maxsize=4)
def lab_for(matched, path=None):
    return og.Optogenetics(path, matched=matched)


def drive_vector(lab, experiment, side, light, condition, seed, resting):
    c = lab.circuit
    lit = lab.targets(experiment, condition, seed) if experiment else np.zeros(c.n, bool)
    d = np.zeros(c.n)
    d[lit] = light * (1 + og.SIDES[side] * c.side[lit])
    rest = np.where(c.vnc, resting, 0.0)
    return d, rest


# --- one closed-loop run -------------------------------------------------------------------------------------------
def run_once(experiment, side='both', seed=1, condition='intact', config=None, background=None, light=2.5,
             duration=DURATION, path=None):
    cfg = FULL | (config or {})
    phys = og.PHYSIOLOGY[cfg['physiology']]
    lab = lab_for(cfg['matched'], path)
    c = lab.circuit
    b = body(cfg['muscle'])
    sim = b.sim
    lit, rest = drive_vector(lab, experiment, side, light, condition, seed, phys['resting']) if experiment else (np.zeros(c.n), None)
    if rest is None: rest = np.where(c.vnc, phys['resting'], 0.0)
    bg = np.zeros(c.n)
    if background:
        bg, _ = drive_vector(lab, background, 'both', light, 'intact', seed, 0.0)
    gaps = [(a, b2, og.GAP_KICK) for a, b2 in lab.gaps] if phys['gap_junctions'] else []
    net = model.Stepper(c, rest + bg, condition='shuffled' if condition == 'shuffled' else 'intact', seed=seed, gaps=gaps)
    grp, ttm = groups(lab, cfg['muscle'])
    grp_idx = {k: np.array(v, int) for k, v in grp.items() if v}
    pools = [np.array(c.motor[k], int) for k in range(6)]
    pro = [grp_idx.get((LAB_LEG[l], 'coxa_protract'), np.zeros(0, int)) for l in LEGS]
    ret = [grp_idx.get((LAB_LEG[l], 'coxa_retract'), np.zeros(0, int)) for l in LEGS]
    sens = sensory_cells(lab)
    mean = lambda idx: float(net.rates[idx].mean()) if len(idx) else 0.0

    sim.reset()
    default = b.steps.default_pose_by_dof_order(b.order)
    sim.set_actuator_inputs('nmf', b.POS, default)
    sim.set_leg_adhesion_states('nmf', np.ones(6, bool))
    if cfg['muscle']: sim.set_actuator_inputs('nmf', b.MOTOR, np.zeros(len(b.motor_order)))
    sim.warmup()
    substeps = int(round(NEURAL_DT / sim.timestep))
    phases = TRIPOD.copy().astype(float)
    activation = {'LM': 0.0, 'RM': 0.0}
    injected = np.zeros(c.n)
    out, legs_log = [], []
    for step in range(int(round(duration / NEURAL_DT))):
        t = step * NEURAL_DT
        net.drive = rest + bg + (lit if t >= og.LIGHT_ON else 0.0)
        spikes = net.step(injected if cfg['feedback'] else None)
        rates = {k: mean(idx) for k, idx in grp_idx.items()}
        offsets = b.offsets(rates)
        if cfg['rhythm'] == 'supplied':
            amp, freq, direction = np.ones(6), np.full(6, SUPPLIED_HZ), np.ones(6)
        else:
            drive = np.array([mean(p) for p in pools])
            amp = np.tanh(drive / A_SCALE)
            freq = F_MAX * np.tanh(drive / F_SCALE)
            r_, p_ = np.array([mean(i) for i in ret]), np.array([mean(i) for i in pro])
            direction = (r_ - p_) / (r_ + p_ + 1.0)
        # tripod-coupled phase oscillators
        diff = phases[None, :] - phases[:, None] - (TRIPOD[None, :] - TRIPOD[:, None])
        dphi = 2 * np.pi * freq * direction + COUPLING * (amp[None, :] * np.sin(diff)).sum(1) / 6
        phases = phases + dphi * NEURAL_DT
        target = b.steps.get_joint_angles_by_dof_order(phases, amp, b.order) + offsets
        adhesion = np.where(amp < 0.1, True, b.steps.get_adhesion_onoff_by_phase(phases))
        sim.set_actuator_inputs('nmf', b.POS, target)
        sim.set_leg_adhesion_states('nmf', adhesion.astype(bool))
        if cfg['muscle']:
            torques = []
            for leg in b.motor_order:
                L = leg.upper()
                activation[L] -= NEURAL_DT / TAU_ACT * activation[L]
                if ttm[L] and spikes[ttm[L]].any(): activation[L] += TWITCH * (1 - activation[L])
                torques.append(-b.signs[(leg, 'femur')] * activation[L] * b.tmax)   # femur depression
            sim.set_actuator_inputs('nmf', b.MOTOR, np.array(torques))
        for _ in range(substeps): sim.step()
        if cfg['feedback']:
            load, speed = b.sensors()
            injected = np.zeros(c.n)
            for k in range(6):
                camp, cho = sens[k]
                injected[camp] = LOAD_GAIN * min(load[k], 2.0)
                injected[cho] = CHO_GAIN * min(speed[k] / V_REF, 2.0)
        if step % int(round(RECORD_EVERY / NEURAL_DT)) == 0:
            pos = sim.get_body_positions('nmf')[b.thorax]; q = sim.get_body_rotations('nmf')[b.thorax]
            yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
            out.append({'t': round(t, 3), 'x': float(pos[0]), 'y': float(pos[1]), 'z': float(pos[2]), 'yaw': yaw})
            legs_log.append({'t': round(t, 3), 'amp': np.round(amp, 3).tolist(), 'dir': np.round(direction, 3).tolist(),
                             'freq': np.round(freq, 2).tolist()})
    return out, legs_log


def summarise(traj, comparison):
    """Effect of the light: yaw change, forward and lateral travel and height, relative to the comparison run."""
    def window(tr):
        on = [s for s in tr if s['t'] >= og.LIGHT_ON]
        a, b = on[0], on[-1]
        yaws = np.unwrap([s['yaw'] for s in on])
        h = a['yaw']; dx, dy = b['x'] - a['x'], b['y'] - a['y']
        pre = [s['z'] for s in tr if s['t'] < og.LIGHT_ON] or [a['z']]
        return {'yaw_deg': math.degrees(yaws[-1] - yaws[0]), 'forward_mm': dx * math.cos(h) + dy * math.sin(h),
                'lateral_mm': -dx * math.sin(h) + dy * math.cos(h), 'rise_mm': max(s['z'] for s in on) - float(np.mean(pre))}
    lit, base = window(traj), window(comparison)
    return {k + '_lit': v for k, v in lit.items()} | {k: lit[k] - base[k] for k in lit} | {'comparison_' + k: v for k, v in base.items()}


def verdict(test, runs, _single=False):
    """runs: {side: [summary per seed]}. The verdict uses seed means; 'agreement' counts seeds giving the same verdict."""
    seeds = min(len(v) for v in runs.values())
    if seeds > 1 and not _single:
        per_seed = [verdict(test, {s: [v[i]] for s, v in runs.items()}, _single=True)['result'] for i in range(seeds)]
    else:
        per_seed = None
    mean = lambda side, key: float(np.mean([r[key] for r in runs[side]])) if runs.get(side) else None
    out = {k: {s: mean(s, k) for s in og.SIDES} for k in ('yaw_deg', 'forward_mm', 'lateral_mm', 'rise_mm')}
    lit_fwd = {s: mean(s, 'forward_mm_lit') for s in og.SIDES}
    out['forward_mm_lit'] = lit_fwd
    yaw = out['yaw_deg']

    def turning():
        r, l = yaw['right'], yaw['left']
        if r < -TURN_DEG and l > TURN_DEG: return 'match', f'right side lit turns right ({r:+.0f}°), left side lit turns left ({l:+.0f}°)'
        if r > TURN_DEG and l < -TURN_DEG: return 'opposite', f'turns away from the lit side (right {r:+.0f}°, left {l:+.0f}°)'
        if r < 0 < l: return 'weak, right direction', f'turns toward the lit side, below ±{TURN_DEG:.0f}° (right {r:+.0f}°, left {l:+.0f}°)'
        if r > 0 > l: return 'weak, wrong direction', f'turns away from the lit side, below ±{TURN_DEG:.0f}° (right {r:+.0f}°, left {l:+.0f}°)'
        return 'no clear turn', f'right {r:+.0f}°, left {l:+.0f}° (threshold ±{TURN_DEG:.0f}°)'

    if test is None:
        out['result'] = 'exploratory'
        out['detail'] = f"both sides: travels {lit_fwd['both']:+.2f} mm, turns {yaw['both']:+.0f}°"
    elif test == 'turn':
        out['result'], out['detail'] = turning()
        out['detail'] += ' while walking under P9, compared with P9 alone'
    elif test == 'forward_turn':
        walks = lit_fwd['both'] > WALK_MM and abs(yaw['both']) < STRAIGHT_DEG
        t_result, t_detail = turning()
        out['result'] = 'match' if walks and t_result == 'match' else 'partial' if walks or t_result in ('match', 'weak, right direction') else 'mismatch'
        out['detail'] = (f"both sides: travels {lit_fwd['both']:+.2f} mm, turns {yaw['both']:+.0f}° "
                         f"({'walks forward' if walks else 'no straight forward walking'}); one side: {t_detail}")
    elif test == 'backward':
        f = lit_fwd['both']
        out['result'] = 'match' if f < -WALK_MM else 'partial' if f < 0 else 'mismatch'
        out['detail'] = f'travels {f:+.2f} mm during light (backward walking: below −{WALK_MM} mm)'
    elif test == 'escape':
        rise = out['rise_mm']['both']
        out['result'] = 'match' if rise > RISE_MM else 'mismatch'
        out['detail'] = f'body rises {rise:+.3f} mm more than without light (threshold {RISE_MM} mm)'
    if per_seed:
        out['seed_results'] = per_seed
        out['agreement'] = f"{sum(r == out['result'] for r in per_seed)} of {len(per_seed)} seeds"
        out['detail'] += f"; {out['agreement']} agree"
    return out


# --- scorecard -------------------------------------------------------------------------------------------------------
def plan(path=None, seeds=(1, 2, 3)):
    lab = og.Optogenetics(path)
    jobs = []
    for name, e in og.EXPERIMENTS.items():
        if not lab.available[name]: continue
        bg = SETUP[e['test']] if lab.available.get(SETUP[e['test']] or '', 0) else None
        for ablation, (_, cfg) in ABLATIONS.items():
            conditions = list(og.CONDITIONS) if ablation == 'full' else ['intact']
            for condition in conditions:
                for side in og.SIDES:
                    for seed in seeds:
                        jobs.append((name, side, seed, condition, ablation, bg))
    return jobs


def _job(job):
    name, side, seed, condition, ablation, bg = job
    cfg = ABLATIONS[ablation][1]
    test = og.EXPERIMENTS[name]['test']
    traj, legs = run_once(name, side, seed, condition, cfg, background=bg)
    comp, _ = _comparison(bg, seed, condition, ablation)
    return summarise(traj, comp), traj, legs


@lru_cache(maxsize=64)
def _comparison(bg, seed, condition, ablation):
    """The same body and circuit without the tested light: nothing lit, or only the background (P9) lit."""
    traj, legs = run_once(None, 'both', seed, condition, ABLATIONS[ablation][1], background=bg)
    return traj, legs


def scorecard(path=None, seeds=(1, 2, 3), out=None, workers=None):
    jobs = plan(path, seeds)
    workers = workers or max(1, min(8, os.cpu_count() or 2))
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context('spawn').Pool(workers) as pool:
            results = pool.map(_job, jobs, chunksize=1)
    else:
        results = [_job(j) for j in jobs]
    rows, traces = {}, {}
    for job, (summary, traj, legs) in zip(jobs, results):
        name, side, seed, condition, ablation, bg = job
        rows.setdefault(name, {}).setdefault(ablation, {}).setdefault(condition, {}).setdefault(side, []).append(summary)
        if seed == seeds[0] and condition == 'intact' and ablation == 'full':
            traces[f'{name}|{side}'] = {'trace': [{k: round(s[k], 4) for k in ('t', 'x', 'y', 'yaw')} for s in traj],
                                        'legs': legs[::5]}
    comps = {}
    for name in rows:
        bg = SETUP[og.EXPERIMENTS[name]['test']]
        comps[name] = [{k: round(s[k], 4) for k in ('t', 'x', 'y', 'yaw')} for s in _comparison(
            bg if og.Optogenetics(path).available.get(bg or '', 0) else None, seeds[0], 'intact', 'full')[0]]
    lab = lab_for(True, path)
    built = json.loads(lab.path.read_text()).get('mirror_added')
    result = {'source': lab.source, 'matching': lab.matching, 'mirror_added': built, 'seeds': list(seeds), 'conditions': og.CONDITIONS,
              'ablations': {k: v[0] for k, v in ABLATIONS.items()},
              'calibration': {f'{leg} {joint}': s for (leg, joint), s in body(True).signs.items()},
              'parameters': {'span_rad': SPAN, 'rate_scale_hz': RATE_SCALE, 'a_scale_hz': A_SCALE, 'f_max_hz': F_MAX,
                             'f_scale_hz': F_SCALE, 'coupling': COUPLING, 'load_gain': LOAD_GAIN, 'cho_gain': CHO_GAIN,
                             'v_ref': V_REF, 'jump_g': JUMP_G, 'lever_mm': LEVER_MM, 'tau_act_s': TAU_ACT,
                             'jump_torque': body(True).tmax, 'body_weight': body(True).weight, 'duration_s': DURATION,
                             'turn_deg': TURN_DEG, 'walk_mm': WALK_MM, 'straight_deg': STRAIGHT_DEG, 'rise_mm': RISE_MM},
              'experiments': {}, 'traces': traces, 'comparisons': comps}
    for name, by_ab in rows.items():
        e = og.EXPERIMENTS[name]
        result['experiments'][name] = {'expect': e['expect'], 'setup': ('walking under P9' if SETUP[e['test']] else 'standing'),
                                       'ablations': {ab: {c: verdict(e['test'], sides) for c, sides in conds.items()}
                                                     for ab, conds in by_ab.items()}}
    (out or BASE / 'data/body-benchmark.json').write_text(json.dumps(result, indent=1))
    return result


def run(**options):
    """One closed-loop experiment for the page."""
    if not available(): raise ValueError('flygym is not installed here: .venv/bin/pip install -r requirements-body.txt')
    allowed = {'experiment', 'side', 'seed', 'condition', 'ablation'}
    extra = set(options) - allowed
    if extra: raise ValueError('Unknown option: ' + ', '.join(sorted(extra)))
    name = options.get('experiment', 'DNa02')
    if name not in og.EXPERIMENTS: raise ValueError('Unknown experiment')
    ablation = options.get('ablation', 'full')
    if ablation not in ABLATIONS: raise ValueError('Unknown ablation')
    side = options.get('side', 'right')
    if side not in og.SIDES: raise ValueError('Unknown side')
    condition = options.get('condition', 'intact')
    if condition not in og.CONDITIONS: raise ValueError('Unknown condition')
    seed = int(options.get('seed', 1))
    lab = lab_for(FULL['matched'])
    if not lab.available[name]: raise ValueError(f'{name} is not in this circuit')
    bg = SETUP[og.EXPERIMENTS[name]['test']]
    bg = bg if lab.available.get(bg or '', 0) else None
    traj, legs = run_once(name, side, seed, condition, ABLATIONS[ablation][1], background=bg)
    comp, _ = _comparison(bg, seed, condition, ablation)
    return {'experiment': name, 'setup': 'walking under P9' if bg else 'standing', 'effect': summarise(traj, comp),
            'trace': traj, 'comparison': comp, 'legs': legs[::5]}


if __name__ == '__main__':
    import time
    t0 = time.time()
    r = scorecard()
    print('matching:', r['matching'])
    for name, row in r['experiments'].items():
        for ab, conds in row['ablations'].items():
            print(f'{name:12} {ab:16}', ' | '.join(f"{c}: {v['result']}" for c, v in conds.items()))
            if ab == 'full': print(' ' * 30, conds['intact']['detail'])
    print(f'{time.time() - t0:.0f} s')
