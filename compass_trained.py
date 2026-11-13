"""The compass circuit, trained on its own job.

backbone.py used the central complex as a general-purpose network (digits, aiming) and the measured wiring lost to
shuffled wiring. This file does what Lappalainen et al. (Nature 2024) did for the fly's visual system: keep the
measured wiring, learn the handful of numbers the connectome does not give, and train on the job the circuit
evolved for. For the compass that job is keeping track of heading in the dark from the fly's own turning.

The network is the compass core as measured (data/central-complex.json): 152 cells of six types (EPG and EPGt
heading cells, PEN_a and PEN_b rotation cells, PEG, Delta7), every connection between them, each cell's sign from
its transmitter. A connection's strength is its synapse count times one number per pair of cell types. Learned:
36 type-pair strengths, a time constant and a resting drive per type, how strongly turning drives the two
rotation types (opposite on the two sides of the bridge), the cue strength, and one angle (how the left bridge
lines up with the right): 52 numbers. A heading cue is shown for 0.3 s, then the network runs in the dark while
the fly turns; the heading is read from the heading cells as a population vector, with each cell's preferred
direction fixed by the bridge glomerulus it sits in.

Compared, five starting points each:
  wiring        measured; shuffled (within every pair of cell types the partners are swapped, each cell keeping its
                number of connections, so only who-connects-to-whom inside a type is lost); random (same counts)
  what learns   one number per type pair (52 in all) or every synapse separately (thousands)
  reference     a standard recurrent network of the same size with nothing fixed

Then: what the trained network looks like inside (one bump? how wide? does it stay put in the dark?), which
pathways training strengthened, and whether the trained compass can serve as the location signal for recognising
objects (compass.py), the problem a hand-set gain could not solve.

Measured: the cells, their connections and synapse counts, their transmitters, their bridge glomeruli.
Assumed: the unit (a leaky rate cell), that turning drives rotation cells oppositely by side, the left bridge
numbered as the mirror image of the right, the task, and the training method (backpropagation through time,
which the fly does not use: it stands in for evolution and development).

    python3 compass_trained.py     # writes data/compass-trained.json and data/compass-trained-params.json
"""
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DT = 0.02                                # s, the same step as navigation.py
CUE_S, TRAIN_S = 0.3, 4.0
CORE = ('heading', 'rotate', 'sustain', 'inhibit')
SIGN = {'acetylcholine': 1.0, 'glutamate': -1.0, 'gaba': -1.0}
WIRINGS = {'measured': 'Measured wiring', 'shuffled': 'Shuffled within cell types', 'random': 'Random within cell types'}
SHARING = {'types': 'One number per pair of cell types', 'synapses': 'Every synapse separately'}


# --- the circuit --------------------------------------------------------------------------------------------------
def load(path=None):
    return json.loads(Path(path or BASE / 'data/central-complex.json').read_text())


class Circuit:
    def __init__(self, data=None, wiring='measured', seed=0):
        if wiring not in WIRINGS: raise ValueError('wiring must be measured, shuffled or random')
        data = data or load()
        cells = [n for n in data['nodes'] if n.get('group') in CORE]
        self.ids = [n['bodyId'] for n in cells]; ix = {b: i for i, b in enumerate(self.ids)}
        self.n = len(cells)
        self.types = sorted({n['type'] for n in cells}); tix = {t: i for i, t in enumerate(self.types)}
        self.type = np.array([tix[n['type']] for n in cells])
        self.sign = np.array([SIGN.get(n.get('nt'), 1.0) for n in cells])
        glom = [re.search(r'_([LR])(\d)$', str(n.get('instance'))) for n in cells]
        self.side = np.array([0 if not g else (1 if g.group(1) == 'R' else -1) for g in glom])
        self.glom = np.array([int(g.group(2)) if g else 0 for g in glom])
        self.heading = np.flatnonzero(np.array([n['group'] == 'heading' and n['type'] == 'EPG' for n in cells]))
        self.rotate = [np.flatnonzero(self.type == tix[t]) for t in self.types if t.startswith('PEN')]
        pre, post, count = [], [], []
        for a, b, c in data['edges']:
            if a in ix and b in ix: pre.append(ix[a]); post.append(ix[b]); count.append(c)
        pre, post, count = np.array(pre), np.array(post), np.array(count, float)
        if wiring != 'measured':
            rng = np.random.default_rng(100 + seed)
            block = self.type[pre] * len(self.types) + self.type[post]
            for b in np.unique(block):
                e = np.flatnonzero(block == b)
                if wiring == 'shuffled':
                    post[e] = rng.permutation(post[e])                 # every cell keeps its number of partners in this block
                else:
                    tp, tq = divmod(int(b), len(self.types))
                    pre[e] = rng.choice(np.flatnonzero(self.type == tp), len(e)); post[e] = rng.choice(np.flatnonzero(self.type == tq), len(e))
        C = np.zeros((self.n, self.n)); np.add.at(C, (post, pre), count)       # rows: receiving cell
        self.C = C / C.sum(1).mean()                                             # an average cell's inputs sum to one
        self.mask = self.C > 0
        self.pairs = sorted({(int(self.type[i]), int(self.type[j])) for i, j in zip(*np.nonzero(self.mask))})

    def angles(self, offset):
        """Preferred direction of each heading cell: its bridge glomerulus, the left side mirrored and offset."""
        g = self.glom[self.heading]; s = self.side[self.heading]
        return np.where(s > 0, 2 * math.pi * (g - 1) / 8, -2 * math.pi * (g - 1) / 8 + offset)


# --- the model ----------------------------------------------------------------------------------------------------
class Model:
    def __init__(self, circuit, sharing='types', seed=0):
        if sharing not in SHARING: raise ValueError('sharing must be types or synapses')
        self.c, self.sharing = circuit, sharing
        rng = np.random.default_rng(seed)
        T = len(circuit.types)
        self.p = {'strength': rng.normal(0.0, 0.2, (T, T)) if sharing == 'types' else rng.normal(0.0, 0.2, (circuit.n, circuit.n)),
                  'leak': rng.normal(-1.0, 0.1, T), 'rest': rng.normal(0.1, 0.05, T),
                  'turn': rng.normal(0.3, 0.05, len(circuit.rotate)), 'cue': np.array([2.0]), 'offset': np.array([math.pi / 8])}

    def count(self):
        c = self.c
        strength = len(c.pairs) if self.sharing == 'types' else int(c.mask.sum())
        return strength + sum(v.size for k, v in self.p.items() if k != 'strength')

    def weights(self):
        c, s = self.c, self.p['strength']
        A = np.exp(s[c.type[:, None], c.type[None, :]] if self.sharing == 'types' else s)
        return c.C * A * c.sign[None, :], A

    def inputs(self, omega, phi0):
        """omega: (batch, steps) turning in rad/s; phi0: starting heading. Returns the external drive (batch, steps, cells)."""
        c = self.c
        B, S = omega.shape
        u = np.zeros((B, S, c.n))
        for k, cells in enumerate(c.rotate):
            u[:, :, cells] = self.p['turn'][k] * omega[:, :, None] * c.side[cells][None, None, :]
        cue = np.exp(2.0 * (np.cos(c.angles(self.p['offset'][0])[None, :] - phi0[:, None]) - 1))
        n_cue = int(round(CUE_S / DT))
        u[:, :n_cue, c.heading] += self.p['cue'][0] * cue[:, None, :]
        return u, cue, n_cue

    def run(self, omega, phi0, keep=False):
        c = self.c
        W, A = self.weights()
        a = 1 / (1 + np.exp(-self.p['leak']))[c.type]; b = self.p['rest'][c.type]
        u, cue, n_cue = self.inputs(omega, phi0)
        B, S = omega.shape
        V = np.zeros((B, c.n)); Vs, Is = [], []
        R = np.zeros((B, S, c.n))
        for t in range(S):
            r = np.maximum(V, 0); R[:, t] = r
            I = r @ W.T + b + u[:, t]
            if keep: Vs.append(V); Is.append(I)
            V = V + a * (I - V)
        th = c.angles(self.p['offset'][0])
        zx, zy = R[:, :, c.heading] @ np.cos(th), R[:, :, c.heading] @ np.sin(th)
        cache = dict(W=W, A=A, a=a, u=u, cue=cue, n_cue=n_cue, Vs=Vs, Is=Is, th=th) if keep else None
        return R, np.arctan2(zy, zx), (zx, zy), cache

    def loss_and_grad(self, omega, phi, activity=1e-3):
        """Mean of 1 - cos(decoded - true heading) after the cue, plus a small cost on activity. Gradients by
        backpropagation through time, written out by hand."""
        c = self.c
        R, est, (zx, zy), k = self.run(omega, phi[:, 0], keep=True)
        B, S = omega.shape
        valid = np.zeros(S); valid[k['n_cue']:] = 1
        norm = np.sqrt(zx ** 2 + zy ** 2 + 1e-6)
        cosd = (zx * np.cos(phi) + zy * np.sin(phi)) / norm
        M = B * valid.sum()
        loss = float(((1 - cosd) * valid).sum() / M + activity * (R ** 2).mean())
        # d loss / d (zx, zy)
        gx = -(np.cos(phi) / norm - cosd * zx / norm ** 2) * valid / M
        gy = -(np.sin(phi) / norm - cosd * zy / norm ** 2) * valid / M
        dR = 2 * activity * R / R.size
        dR[:, :, c.heading] += gx[:, :, None] * np.cos(k['th']) + gy[:, :, None] * np.sin(k['th'])
        d_offset_dec = (R[:, :, c.heading] * (-gx[:, :, None] * np.sin(k['th']) + gy[:, :, None] * np.cos(k['th'])))[:, :, c.side[c.heading] < 0].sum()
        W, a = k['W'], k['a']
        gW = np.zeros_like(W); gb = np.zeros(c.n); ga = np.zeros(c.n); gu = np.zeros_like(k['u'])
        dV = np.zeros((B, c.n))
        for t in range(S - 1, -1, -1):
            V, I = k['Vs'][t], k['Is'][t]
            dI = dV * a                                                # through V' = V + a (I - V)
            ga += (dV * (I - V)).sum(0)
            r = R[:, t]
            gW += dI.T @ r; gb += dI.sum(0); gu[:, t] = dI
            dr = dI @ W + dR[:, t]
            dV = dV * (1 - a) + dr * (V > 0)
        g = {}
        gA = gW * c.C * c.sign[None, :] * k['A']                       # strengths are exp(parameter)
        T = len(c.types)
        if self.sharing == 'types':
            g['strength'] = np.zeros((T, T)); np.add.at(g['strength'], (c.type[:, None].repeat(c.n, 1), c.type[None, :].repeat(c.n, 0)), gA)
        else:
            g['strength'] = gA * c.mask
        sig = 1 / (1 + np.exp(-self.p['leak']))
        g['leak'] = np.bincount(c.type, weights=ga, minlength=T) * sig * (1 - sig)
        g['rest'] = np.bincount(c.type, weights=gb, minlength=T)
        g['turn'] = np.array([(gu[:, :, cells] * omega[:, :, None] * c.side[cells][None, None, :]).sum() for cells in c.rotate])
        gcue = gu[:, :k['n_cue'], c.heading].sum(1)                    # (batch, heading cells)
        g['cue'] = np.array([(gcue * k['cue']).sum()])
        dcue_dth = k['cue'] * 2.0 * -np.sin(k['th'][None, :] - phi[:, [0]])
        g['offset'] = np.array([d_offset_dec + (self.p['cue'][0] * gcue * dcue_dth)[:, c.side[c.heading] < 0].sum()])
        return loss, g


class Dense:
    """Reference: a standard recurrent network of the same size, nothing fixed, a learned linear readout."""
    def __init__(self, n=152, seed=0):
        rng = np.random.default_rng(seed)
        self.n = n
        self.p = {'W': rng.normal(0, 0.8 / math.sqrt(n), (n, n)), 'b': np.zeros(n), 'turn': rng.normal(0, 0.3, n),
                  'cx': rng.normal(0, 0.5, n), 'cy': rng.normal(0, 0.5, n), 'ox': rng.normal(0, 0.1, n), 'oy': rng.normal(0, 0.1, n)}
        self.a = 0.25

    def count(self): return sum(v.size for v in self.p.values())

    def run(self, omega, phi0, keep=False):
        p = self.p; B, S = omega.shape
        n_cue = int(round(CUE_S / DT))
        cue = np.cos(phi0)[:, None] * p['cx'] + np.sin(phi0)[:, None] * p['cy']
        V = np.zeros((B, self.n)); R = np.zeros((B, S, self.n)); Vs = []
        for t in range(S):
            r = np.tanh(V); R[:, t] = r
            if keep: Vs.append(V)
            I = r @ p['W'].T + p['b'] + omega[:, [t]] * p['turn'] + (cue if t < n_cue else 0)
            V = V + self.a * (I - V)
        zx, zy = R @ p['ox'], R @ p['oy']
        return R, np.arctan2(zy, zx), (zx, zy), dict(Vs=Vs, n_cue=n_cue) if keep else None

    def loss_and_grad(self, omega, phi, activity=1e-3):
        p = self.p
        R, est, (zx, zy), k = self.run(omega, phi[:, 0], keep=True)
        B, S = omega.shape
        valid = np.zeros(S); valid[k['n_cue']:] = 1
        norm = np.sqrt(zx ** 2 + zy ** 2 + 1e-6); cosd = (zx * np.cos(phi) + zy * np.sin(phi)) / norm; M = B * valid.sum()
        loss = float(((1 - cosd) * valid).sum() / M + activity * (R ** 2).mean())
        gx = -(np.cos(phi) / norm - cosd * zx / norm ** 2) * valid / M; gy = -(np.sin(phi) / norm - cosd * zy / norm ** 2) * valid / M
        dR = 2 * activity * R / R.size + gx[:, :, None] * p['ox'] + gy[:, :, None] * p['oy']
        g = {q: np.zeros_like(v) for q, v in p.items()}
        g['ox'] = np.einsum('bs,bsn->n', gx, R); g['oy'] = np.einsum('bs,bsn->n', gy, R)
        dV = np.zeros((B, self.n))
        for t in range(S - 1, -1, -1):
            dI = dV * self.a; r = R[:, t]
            g['W'] += dI.T @ r; g['b'] += dI.sum(0); g['turn'] += (dI * omega[:, [t]]).sum(0)
            if t < k['n_cue']:
                g['cx'] += (dI * np.cos(phi[:, [0]])).sum(0); g['cy'] += (dI * np.sin(phi[:, [0]])).sum(0)
            dV = dV * (1 - self.a) + (dI @ p['W'] + dR[:, t]) * (1 - r ** 2)
        return loss, g


# --- task and training ----------------------------------------------------------------------------------------------
def trajectories(rng, batch, seconds=TRAIN_S, scale=2.0):
    """Turning like a walking fly's: slow wandering, bouts of standing still, and occasional sharp turns."""
    S = int(round(seconds / DT))
    omega = np.zeros((batch, S)); w = rng.normal(0, scale, batch)
    for t in range(S):
        w += (-w / 0.5) * DT + scale * math.sqrt(2 * DT / 0.5) * rng.normal(size=batch)
        omega[:, t] = w
    still = np.repeat(rng.random((batch, S // 25 + 1)) < 0.25, 25, axis=1)[:, :S]        # half-second bouts of no turning
    omega = np.clip(np.where(still, 0.0, omega), -6, 6)
    phi0 = rng.uniform(-math.pi, math.pi, batch)
    phi = phi0[:, None] + np.cumsum(omega, 1) * DT - omega * DT                           # heading at the start of each step
    return omega, phi


def train(model, seed=0, steps=2500, batch=16, lr=0.02, log=None):
    rng = np.random.default_rng(1000 + seed)
    m = {k: np.zeros_like(v) for k, v in model.p.items()}; v = {k: np.zeros_like(x) for k, x in model.p.items()}
    curve = []
    for i in range(1, steps + 1):
        omega, phi = trajectories(rng, batch, TRAIN_S if i <= 0.6 * steps else 2 * TRAIN_S)       # longer runs in the dark late in training
        loss, g = model.loss_and_grad(omega, phi)
        gn = math.sqrt(sum(float((x ** 2).sum()) for x in g.values()))
        clip = min(1.0, 1.0 / (gn + 1e-12))
        rate = lr * (0.1 if i > 0.8 * steps else 1.0)
        for k in model.p:
            gk = g[k] * clip
            m[k] = 0.9 * m[k] + 0.1 * gk; v[k] = 0.999 * v[k] + 0.001 * gk ** 2
            model.p[k] = model.p[k] - rate * (m[k] / (1 - 0.9 ** i)) / (np.sqrt(v[k] / (1 - 0.999 ** i)) + 1e-8)
        if i % 100 == 0 or i == 1:
            curve.append((i, round(loss, 4)))
            if log: print(f'  step {i:5d} loss {loss:.4f}', file=sys.stderr, flush=True)
    return curve


def wrap(x): return np.arctan2(np.sin(x), np.cos(x))


def evaluate(model, seed=0, batch=64):
    """Held-out turning. Error in degrees after 4 s (as trained), after 20 s (five times longer), and standing still."""
    rng = np.random.default_rng(5000 + seed)
    out = {}
    for name, seconds in (('error_deg_4s', 4.0), ('error_deg_20s', 20.0)):
        omega, phi = trajectories(rng, batch, seconds)
        _, est, (zx, zy), _ = model.run(omega, phi[:, 0])
        err = np.abs(wrap(est - phi))
        out[name] = float(np.degrees(err[:, -25:].mean()))
        if seconds == 20.0:
            moving = np.abs(omega[:, 50:-1]) > 0.3
            d_est = wrap(est[:, 51:] - est[:, 50:-1]) / DT
            out['turn_gain'] = float((d_est[moving] * omega[:, 50:-1][moving]).sum() / (omega[:, 50:-1][moving] ** 2).sum())
    omega = np.zeros((batch, int(10 / DT))); phi0 = rng.uniform(-math.pi, math.pi, batch)
    R, est, _, _ = model.run(omega, phi0)
    out['drift_deg_10s_still'] = float(np.degrees(np.abs(wrap(est[:, -1] - phi0)).mean()))
    if isinstance(model, Model):
        c = model.c
        th = c.angles(model.p['offset'][0])
        rel = wrap(th[None, :] - est[:, -1][:, None])                              # each heading cell's direction relative to the bump
        act = R[:, -1][:, c.heading]
        bins = np.linspace(-math.pi, math.pi, 17); mid = (bins[:-1] + bins[1:]) / 2
        prof = np.array([act[(rel >= lo) & (rel < hi)].mean() if ((rel >= lo) & (rel < hi)).any() else 0.0 for lo, hi in zip(bins[:-1], bins[1:])])
        out['bump_profile'] = (prof / max(prof.max(), 1e-9)).round(3).tolist(); out['bump_angles_deg'] = np.degrees(mid).round(1).tolist()
        out['bump_width_deg'] = float((prof > prof.max() / 2).sum() * 22.5)
        out['bump_contrast'] = float((prof.max() - prof.min()) / max(prof.max(), 1e-9))
    return out


def pathways(model):
    """What training did to each type-to-type pathway: learned multiplier, and its share of the receiving type's input."""
    c = model.c; W, A = model.weights(); rows = []
    for tp, tq in c.pairs:
        sub = np.abs(W[np.ix_(c.type == tp, c.type == tq)])
        rows.append({'to': c.types[tp], 'from': c.types[tq], 'multiplier': float(np.exp(model.p['strength'][tp, tq])) if model.sharing == 'types' else None,
                     'sign': float(np.sign(c.sign[c.type == tq][0])), 'input_per_cell': float(sub.sum(1).mean())})
    return sorted(rows, key=lambda r: -r['input_per_cell'])


# --- the trained compass as a location signal (same interface as navigation.Navigator) ----------------------------------
class TrainedCompass:
    def __init__(self, model, heading, rng=None):
        self.m, self.c = model, model.c
        self.W, _ = model.weights()
        self.a = 1 / (1 + np.exp(-model.p['leak']))[self.c.type]; self.b = model.p['rest'][self.c.type]
        self.th = self.c.angles(model.p['offset'][0])
        self.V = np.zeros(self.c.n); self.vector = np.zeros(2); self.rotation_gain = 1.0
        cue = model.p['cue'][0] * np.exp(2.0 * (np.cos(self.th - heading) - 1))
        for _ in range(int(round(CUE_S / DT))): self._step(0.0, cue)

    def _step(self, omega, cue=None):
        u = np.zeros(self.c.n)
        for k, cells in enumerate(self.c.rotate): u[cells] = self.m.p['turn'][k] * omega * self.c.side[cells]
        if cue is not None: u[self.c.heading] += cue
        r = np.maximum(self.V, 0)
        self.V = self.V + self.a * (r @ self.W.T + self.b + u - self.V)

    def decode_heading(self):
        r = np.maximum(self.V[self.c.heading], 0)
        return float(math.atan2(r @ np.sin(self.th), r @ np.cos(self.th))), 1.0

    def step(self, omega, speed, dt=DT):
        self._step(omega)
        est, _ = self.decode_heading()
        self.vector += dt * speed * np.array([math.cos(est), math.sin(est)])
        return est

    def home_vector(self):
        return float(self.vector[0]), float(self.vector[1])


def save_params(model):
    return {'wiring_seed': None, 'sharing': model.sharing, 'params': {k: v.tolist() for k, v in model.p.items()}}


def load_trained(wiring='measured', path=None):
    """The saved trained compass for this wiring (the median of the seeds), or None if not trained yet."""
    f = Path(path or BASE / 'data/compass-trained-params.json')
    if not f.exists(): return None
    d = json.loads(f.read_text()).get(wiring)
    if not d: return None
    m = Model(Circuit(wiring=wiring, seed=d['wiring_seed']), d['sharing'])
    m.p = {k: np.array(v) for k, v in d['params'].items()}
    return m


# --- experiment -------------------------------------------------------------------------------------------------------
def one(wiring, sharing, seed, steps=2500, data=None, log=None):
    if wiring == 'dense':
        best = None
        for lr in (0.02, 0.005, 0.001):                    # the best of three learning rates, so it is not held back by one that suits the others
            model = Dense(seed=seed); before = evaluate(model, seed)['error_deg_4s']
            curve = train(model, seed, steps, lr=lr, log=log); ev = evaluate(model, seed)
            if best is None or ev['error_deg_4s'] < best[1]['error_deg_4s']: best = (model, {'wiring': wiring, 'sharing': sharing, 'seed': seed, 'numbers_learned': model.count(),
                                                                                                'error_deg_untrained': before, 'curve': curve, 'learning_rate': lr, **ev})
        return best
    else:
        model = Model(Circuit(data, wiring, seed), sharing, seed)
    before = evaluate(model, seed)['error_deg_4s']
    curve = train(model, seed, steps, log=log)
    return model, {'wiring': wiring, 'sharing': sharing, 'seed': seed, 'numbers_learned': model.count(), 'error_deg_untrained': before,
                   'curve': curve, **evaluate(model, seed)}


def summarise(rows):
    keys = ['error_deg_untrained', 'error_deg_4s', 'error_deg_20s', 'drift_deg_10s_still', 'turn_gain', 'bump_width_deg', 'bump_contrast']
    out = {k: rows[0][k] for k in ('wiring', 'sharing', 'numbers_learned')}
    for k in keys:
        vals = [r[k] for r in rows if k in r]
        if vals: out[k] = float(np.median(vals)); out[k + '_seeds'] = [round(v, 2) for v in vals]
    out['works'] = f"{sum(r['error_deg_20s'] < 30 for r in rows)} of {len(rows)}"
    out['curve'] = [[rows[0]['curve'][i][0], float(np.median([r['curve'][i][1] for r in rows]))] for i in range(len(rows[0]['curve']))]
    if 'bump_profile' in rows[0]:
        best = min(rows, key=lambda r: r['error_deg_20s']); out['bump_profile'] = best['bump_profile']; out['bump_angles_deg'] = best['bump_angles_deg']
    return out


def _job(args):
    wiring, sharing, seed, steps = args
    model, row = one(wiring, sharing, seed, steps)
    extra = {'params': save_params(model), 'pathways': pathways(model)} if sharing == 'types' else {}
    return row, extra


def _job_keyed(args):
    return tuple(args), _job(args)


def benchmark(seeds=(1, 2, 3, 4, 5), steps=2500, out=None, log=True, workers=None):
    import multiprocessing as mp, os
    start = time.time()
    plan = [(w, 'types') for w in WIRINGS] + [(w, 'synapses') for w in WIRINGS] + [('dense', 'everything')]
    jobs = [(w, sh, s, steps) for w, sh in plan for s in seeds]
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    # finished runs are kept in a side file, so an interrupted benchmark resumes instead of starting over
    cache = BASE / 'data/compass-trained-runs.partial'
    results = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            try: r = json.loads(line)
            except ValueError: continue
            results[tuple(r['job'])] = (r['row'], r['extra'])
    todo = [j for j in jobs if j not in results]
    with cache.open('a') as f:
        def keep(j, r):
            results[j] = r; f.write(json.dumps({'job': list(j), 'row': r[0], 'extra': r[1]}) + '\n'); f.flush()
            if log: report(r[0])
        if workers > 1 and todo:
            with mp.get_context('spawn').Pool(workers) as pool:
                for j, r in pool.imap_unordered(_job_keyed, todo, chunksize=1): keep(j, r)
        else:
            for j in todo: keep(j, _job(j))
    table, saved, paths = [], {}, None
    for wiring, sharing in plan:
        runs = [results[(wiring, sharing, s, steps)] for s in seeds]
        table.append(summarise([r for r, _ in runs]))
        if sharing == 'types':
            order = sorted(runs, key=lambda re: re[0]['error_deg_20s']); row, extra = order[len(order) // 2]      # the median seed
            saved[wiring] = dict(extra['params'], wiring_seed=row['seed'], error_deg_20s=row['error_deg_20s'])
            if wiring == 'measured': paths = order[0][1]['pathways']
    c = Circuit()
    result = {'seeds': list(seeds), 'steps': steps, 'wirings': WIRINGS, 'sharing': SHARING, 'rows': table, 'pathways': paths,
              'circuit': {'cells': c.n, 'types': c.types, 'cells_per_type': {t: int((c.type == i).sum()) for i, t in enumerate(c.types)},
                          'connections': int(c.mask.sum()), 'type_pairs': len(c.pairs), 'heading_cells_read': int(len(c.heading))},
              'task': {'cue_s': CUE_S, 'train_s': TRAIN_S, 'dt_s': DT}, 'minutes': round((time.time() - start) / 60, 1)}
    (out or BASE / 'data/compass-trained.json').write_text(json.dumps(result, indent=1))
    (BASE / 'data/compass-trained-params.json').write_text(json.dumps(saved))
    if cache.exists(): cache.unlink()
    return result


def location_test(seeds=(7, 8, 9), trials=100):
    """Can the trained compass tell the scaffold where its glimpses are? Objects recognised by feature location
    (compass.py), with the location taken from the trained circuit's own heading. Appended to compass-trained.json."""
    import compass as cp
    from learning import LearningCircuit
    lc = LearningCircuit()
    out = {}
    for source in ('true_location', 'compass_idealised', 'compass_trained_measured', 'compass_trained_shuffled', 'bag_of_features'):
        out[source] = cp.average([cp.experiment(lc, s, source, trials=trials) for s in seeds])
    out['labels'] = {**cp.SOURCES, **cp.TRAINED}
    f = BASE / 'data/compass-trained.json'
    d = json.loads(f.read_text()); d['location'] = out; f.write_text(json.dumps(d, indent=1))
    return out


def report(row):
    print(f"{row['wiring']:9} {row['sharing']:10} seed {row['seed']}: untrained {row['error_deg_untrained']:6.1f}°  4 s {row['error_deg_4s']:6.1f}°  20 s {row['error_deg_20s']:6.1f}°  "
          f"gain {row.get('turn_gain', float('nan')):.2f}  still {row['drift_deg_10s_still']:.1f}°", file=sys.stderr, flush=True)


if __name__ == '__main__':
    r = benchmark()
    for row in r['rows']:
        print(f"{row['wiring']:9} {row['sharing']:10} {row['numbers_learned']:6d} numbers: 4 s {row['error_deg_4s']:.1f}°, 20 s {row['error_deg_20s']:.1f}°, works {row['works']}", file=sys.stderr)
    loc = location_test()
    for k, v in loc.items():
        if k != 'labels': print(f"{loc['labels'][k]:<55} 3 modules, 4 glimpses: {v['three_modules'][3]:.2f}  location error {v['location_error_spacings']:.2f}", file=sys.stderr)
