"""The connectome as the backbone of a learning network.

Until now the lab froze the wiring and trained a readout on top, and shuffled wiring did as well as measured.
Here learning happens inside the wiring. The network has exactly the connectome's connections; the strengths are
learned; each cell keeps the sign its transmitter predicts. Three things are varied:

  learning rule
    readout     only the readout learns (what the lab did before)
    reward      one scalar reward: try a small random change in activity, keep it if things got better
                (node perturbation; pure reward and punishment)
    teachers    one teacher signal per output (too high / too low), sent back to the cells along fixed pathways,
                each synapse changing by its own input x its cell's teacher signal (dopamine-style, no backprop
                of the learned weights). Pathways: random, or the original wiring itself.
    backprop    exact gradients inside the same wiring (the upper bound; not biological)
  wiring        measured; shuffled (every cell keeps its number of inputs and outputs); random (same density)
  biology       added to the plain unit one at a time: winner-take-most inhibition, excitability by cell size,
                saturating dendritic branches (inputs grouped by presynaptic type), time constants by cell size;
                and removing the fixed sign per cell

Backbones: the central complex (ring-neuron inputs -> ... -> steering cells) and the leg motor circuit
(descending and sensory inputs -> ... -> motor neurons). Tasks: aiming at the hotter of two objects (reward
task with a decoy), digits from few examples, and classes learned one after another.

Measured: which cells connect, synapse counts (starting strengths), transmitter predictions (signs), cell types
(dendritic branches), input counts (cell size). Assumed: the unit, four update steps, how task inputs reach the
input cells, every learning rule and rate, the tasks.

    python3 backbone.py          # writes data/backbone-benchmark.json (about 25 minutes)
"""
import json
import math
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

import expansion as ex

BASE = Path(__file__).resolve().parent
T = 4                                    # update steps per input
REST = 0.3                               # resting drive: cells sit just above threshold, so inhibition carries signal too
SIGN = {'acetylcholine': 1.0, 'gaba': -1.0, 'glutamate': -1.0}       # the lab's convention; others excitatory
BACKBONES = {
    'central_complex': dict(file='data/central-complex.json', label='Central complex (1,247 cells)',
                            inputs=lambda n: n.get('group') == 'cue', outputs=lambda n: n.get('group') == 'steer'),
    'motor': dict(file='data/optogenetics-circuit.json', label='Leg motor circuit (1,741 cells)',
                  inputs=lambda n: n.get('superclass') in ('descending_neuron', 'vnc_sensory'), outputs=lambda n: n.get('superclass') == 'vnc_motor'),
}
WIRINGS = {'measured': 'Measured wiring', 'shuffled': 'Shuffled (same inputs and outputs per cell)', 'random': 'Random (same density)'}
RULES = {'readout': 'Readout only', 'reward': 'Scalar reward (trial and error)', 'teachers_random': 'Teacher per output, random pathways',
         'teachers_wiring': 'Teacher per output, along the original wiring', 'backprop': 'Backprop inside the wiring (upper bound)'}
BIOLOGY = {'plain': 'Plain unit', 'inhibition': 'Winner-take-most inhibition (top 20%)', 'cell_size': 'Excitability by cell size',
           'dendrites': 'Saturating dendritic branches', 'timescales': 'Time constants by cell size', 'free_signs': 'No fixed sign per cell'}
BRANCHES = 4


# --- the graph ---------------------------------------------------------------------------------------------------
@lru_cache(maxsize=4)
def raw(backbone):
    spec = BACKBONES[backbone]
    data = json.loads((BASE / spec['file']).read_text())
    nodes = data['nodes']; index = {n['bodyId']: i for i, n in enumerate(nodes)}
    e = np.array([(index[a], index[b], c) for a, b, c in data['edges'] if a in index and b in index and a != b])
    sign = np.array([SIGN.get(n.get('nt'), 1.0) for n in nodes])
    types = sorted({str(n.get('type')) for n in nodes}); tix = {t: i for i, t in enumerate(types)}
    branch = np.array([hash_stable(str(n.get('type'))) % BRANCHES for n in nodes])
    return dict(n=len(nodes), pre=e[:, 0], post=e[:, 1], count=e[:, 2].astype(float), sign=sign, branch=branch,
                inputs=np.flatnonzero([spec['inputs'](n) for n in nodes]), outputs=np.flatnonzero([spec['outputs'](n) for n in nodes]),
                types=len(types))


def hash_stable(s):
    h = 0
    for ch in s: h = (h * 131 + ord(ch)) % 1000003
    return h


def wiring(backbone, kind, seed):
    g = dict(raw(backbone)); rng = np.random.default_rng(9000 + seed)
    if kind == 'shuffled':                                  # every cell keeps its in- and out-degree; partners are scrambled
        g['post'] = rng.permutation(g['post'])
    elif kind == 'random':
        g['pre'] = rng.integers(g['n'], size=len(g['pre'])); g['post'] = rng.integers(g['n'], size=len(g['pre']))
    elif kind != 'measured': raise ValueError('Unknown wiring')
    keep = g['pre'] != g['post']
    for k in ('pre', 'post', 'count'): g[k] = g[k][keep]
    return g


class Net:
    def __init__(self, backbone='central_complex', wiring_kind='measured', biology='plain', seed=1, d_in=34, d_out=2):
        if biology not in BIOLOGY: raise ValueError('Unknown biology')
        g = wiring(backbone, wiring_kind, seed)
        self.rng = np.random.default_rng(seed)
        self.n, self.pre, self.post = g['n'], g['pre'], g['post']
        self.inputs, self.outputs = g['inputs'], g['outputs']
        self.biology = biology
        self.dale = biology != 'free_signs'
        self.s = g['sign'][self.pre]
        indeg = np.zeros(self.n); np.add.at(indeg, self.post, g['count'])
        self.a = 1.5 * g['count'] / np.maximum(indeg[self.post], 1.0)          # strengths: synapse share of the cell's input
        self.a0 = self.a.copy()
        # dendritic branches: (cell, branch of the presynaptic cell's type); plain units have one branch per cell
        self.nb = BRANCHES if biology == 'dendrites' else 1
        self.row = self.post * self.nb + (g['branch'][self.pre] if biology == 'dendrites' else 0)
        order = np.lexsort((self.pre, self.row))
        self.csr_order = order
        self.M = csr_matrix((np.ones(len(order)), (self.row[order], self.pre[order])), shape=(self.n * self.nb, self.n))
        self.M.sort_indices()
        assert self.M.nnz <= len(order)
        # duplicate edges (after shuffling) are summed by csr; map every edge to its slot
        self.slot = self._slots()
        size = np.log1p(indeg); size = (size - size.mean()) / (size.std() + 1e-9)
        self.theta = (0.15 * np.clip(size, -2, 2) if biology == 'cell_size' else np.zeros(self.n)) - REST     # large cells need more input
        self.alpha = np.clip(0.75 - 0.2 * size, 0.3, 1.0) if biology == 'timescales' else np.ones(self.n)
        self.k = int(0.2 * self.n) if biology == 'inhibition' else None
        self.B = np.zeros((self.n, d_in))
        self.B[self.inputs] = self.rng.normal(0, 1.0, (len(self.inputs), d_in)) * (self.rng.random((len(self.inputs), d_in)) < 0.3)
        self.R = np.zeros((d_out, len(self.outputs)))
        back = self.rng.permutation(len(self.pre))
        self.rand_w = self.a0[back] * self.rng.choice([-1.0, 1.0], len(self.pre))   # random pathways: same wiring slots, scrambled
        self.adam = {}
        self._set()

    def _slots(self):
        key = self.row.astype(np.int64) * self.n + self.pre
        M = self.M.tocoo(); mkey = M.row.astype(np.int64) * self.n + M.col
        return np.searchsorted(mkey, key)

    def _matrix(self, values):
        data = np.zeros(self.M.nnz); np.add.at(data, self.slot, values)
        return csr_matrix((data, self.M.indices, self.M.indptr), shape=self.M.shape)

    def _set(self):
        self.W = self._matrix(self.s * self.a if self.dale else self.a_signed())
        self.Wt = self.W.T.tocsr()

    def a_signed(self):
        if not hasattr(self, 'w'): self.w = self.s * self.a
        return self.w

    # --- forward -----------------------------------------------------------------------------------------------
    def forward(self, x, noise=None):
        """x: batch x d_in. Returns the output and a trace for learning."""
        drive = x @ self.B.T
        h = np.zeros((len(x), self.n)); tr = []
        for t in range(T):
            u = (self.W @ h.T).T                                            # batch x (n * branches)
            if self.nb > 1:
                ub = np.tanh(u); pre = ub.reshape(len(x), self.n, self.nb).sum(2) + drive - self.theta
            else:
                ub = None; pre = u + drive - self.theta
            if noise is not None: pre = pre + noise[t]
            f = np.tanh(np.maximum(pre, 0.0))
            mask = None
            if self.k:
                cut = np.partition(f, -self.k, axis=1)[:, -self.k][:, None]
                mask = f >= cut; f = f * mask
            new = (1 - self.alpha) * h + self.alpha * f
            tr.append((h, ub, pre, f, mask)); h = new
        return h[:, self.outputs] @ self.R.T, (tr, h)

    # --- backward: exact, or teacher signals along fixed pathways -------------------------------------------------
    def gradients(self, x, err, pathways='true'):
        """err = output - target. pathways: 'true' (backprop), 'wiring' (the original strengths), 'random'."""
        y, (tr, h) = self.forward(x) if not hasattr(self, '_cache') else self._cache
        batch = len(x)
        gR = err.T @ h[:, self.outputs] / batch
        dh = np.zeros((batch, self.n)); dh[:, self.outputs] = err @ self.R
        if pathways == 'true': back = self.Wt
        else:
            vals = self.s * self.a0 if pathways == 'wiring' else self.rand_w
            if pathways not in self._back: self._back[pathways] = self._matrix(vals).T.tocsr()
            back = self._back[pathways]
        g = np.zeros(len(self.pre))
        for hprev, ub, pre, f, mask in reversed(tr):
            df = dh * self.alpha * (1 - f ** 2) * (pre > 0) if mask is None else dh * self.alpha * (1 - np.tanh(np.maximum(pre, 0)) ** 2) * (pre > 0) * mask
            if self.nb > 1:
                du = np.repeat(df, self.nb, axis=1) * (1 - ub ** 2)        # each branch: its cell's signal x its own saturation
            else:
                du = df
            g += (du.T @ hprev)[self.row, self.pre] / batch                         # local: the cell's signal x this input
            dh = dh * (1 - self.alpha) + (back @ du.T).T
        return g, gR

    _back = None

    def step(self, g, gR, lr=0.003, lr_out=0.01):
        self._adam('R', gR, lr_out, lambda d: setattr(self, 'R', self.R - d))
        if g is None: return
        if self.dale:
            self._adam('a', self.s * g, lr, lambda d: setattr(self, 'a', np.maximum(self.a - d, 0.0)))     # strength changes; sign stays
        else:
            self.a_signed(); self._adam('w', g, lr, lambda d: setattr(self, 'w', self.w - d))
        self._set()

    def _adam(self, name, grad, lr, apply, b1=0.9, b2=0.999):
        m, v, t = self.adam.get(name, (0.0, 0.0, 0))
        t += 1; m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad ** 2
        self.adam[name] = (m, v, t)
        apply(lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8))

    # --- learning rules -----------------------------------------------------------------------------------------
    def learn(self, x, target, rule, sigma=0.05, tries=4):
        if self._back is None: self._back = {}
        y, cache = self.forward(x)
        err = y - target
        self._cache = (y, cache)
        if rule == 'readout':
            _, gR = self.gradients(x, err, 'true'); g = None
        elif rule in ('backprop', 'teachers_wiring', 'teachers_random'):
            g, gR = self.gradients(x, err, {'backprop': 'true', 'teachers_wiring': 'wiring', 'teachers_random': 'random'}[rule])
        elif rule == 'reward':
            _, gR = self.gradients(x, err, 'true')
            loss0 = (err ** 2).sum(1)                                       # the punishment, per trial
            g = np.zeros(len(self.pre))
            for _ in range(tries):
                noise = [self.rng.normal(0, sigma, (len(x), self.n)) for _ in range(T)]
                y1, (tr1, _) = self.forward(x, noise)
                delta = (((y1 - target) ** 2).sum(1) - loss0)[:, None]      # worse (+) or better (-) than without the change
                for t, (hprev, ub, pre, f, mask) in enumerate(tr1):
                    credit = delta * noise[t] / sigma ** 2                   # dopamine x the cell's own fluctuation
                    if self.nb > 1: credit = np.repeat(credit, self.nb, axis=1) * (1 - ub ** 2)
                    g += (credit.T @ hprev)[self.row, self.pre] / (len(x) * tries)
        else:
            raise ValueError('Unknown rule')
        del self._cache
        self.step(g, gR)
        return float((err ** 2).mean())

    def predict(self, x):
        return self.forward(x)[0]


# --- tasks ---------------------------------------------------------------------------------------------------------
DIRS = np.arange(16) * 2 * math.pi / 16


def aim_batch(rng, size):
    """Two objects at random offsets; the hotter one is the target. Output: the offset to aim at."""
    pos = rng.uniform(-1, 1, (size, 2, 2)); heat = rng.uniform(0.2, 1.0, (size, 2))
    heat[:, 1] = np.where(np.abs(heat[:, 0] - heat[:, 1]) < 0.15, 1.2 - heat[:, 0], heat[:, 1])     # pushes near ties apart; about 7% remain, so the ceiling is below 100%
    ang = np.arctan2(pos[..., 1], pos[..., 0]); mag = np.linalg.norm(pos, axis=2)
    pop = np.maximum(np.cos(ang[..., None] - DIRS), 0) * mag[..., None]      # 16 direction-tuned inputs per object
    x = np.concatenate([pop[:, 0], pop[:, 1], heat], axis=1)
    target = pos[np.arange(size), heat.argmax(1)]
    return x, target


def task_aim(net_args, rule, seed, checkpoints=(200, 1000, 3000, 8000), batch=32):
    rng = np.random.default_rng(100 + seed)
    net = Net(seed=seed, d_in=34, d_out=2, **net_args)
    xt, yt = aim_batch(np.random.default_rng(999), 600)
    out, seen = [], 0
    for cp in checkpoints:
        while seen < cp:
            x, y = aim_batch(rng, batch); net.learn(x, y, rule); seen += batch
        p = net.predict(xt); err = np.linalg.norm(p - yt, axis=1)
        out.append({'trials': cp, 'hit_rate': float((err < 0.25).mean()), 'error': float(err.mean())})
    return {'curve': out, 'final': out[-1]['hit_rate']}


def digits_split(seed, per_class):
    x, y = ex.load_digits(); rng = np.random.default_rng(200 + seed)
    order = rng.permutation(len(y)); x, y = x[order], y[order]
    test = slice(len(y) - 500, len(y))
    idx = np.concatenate([np.flatnonzero(y[:len(y) - 500] == c)[:per_class] for c in range(10)])
    return x[idx], y[idx], x[test], y[test]


def task_digits(net_args, rule, seed, shots=(5, 20, 50), epochs=30, batch=32):
    out = []
    for per in shots:
        xtr, ytr, xte, yte = digits_split(seed, per)
        net = Net(seed=seed, d_in=64, d_out=10, **net_args); rng = np.random.default_rng(300 + seed)
        onehot = np.eye(10)[ytr]
        for _ in range(epochs):
            order = rng.permutation(len(ytr))
            for i in range(0, len(order), batch): net.learn(xtr[order[i:i + batch]], onehot[order[i:i + batch]], rule)
        out.append({'per_class': per, 'accuracy': float((net.predict(xte).argmax(1) == yte).mean())})
    return {'curve': out, 'final': out[-1]['accuracy']}


def task_continual(net_args, rule, seed, per_class=40, epochs=15, batch=32):
    xtr, ytr, xte, yte = digits_split(seed, per_class)
    net = Net(seed=seed, d_in=64, d_out=10, **net_args); rng = np.random.default_rng(400 + seed)
    onehot = np.eye(10)[ytr]; first, acc_after = {}, []
    for k in range(5):
        cls = (2 * k, 2 * k + 1); idx = np.flatnonzero(np.isin(ytr, cls))
        for _ in range(epochs):
            order = rng.permutation(idx)
            for i in range(0, len(order), batch): net.learn(xtr[order[i:i + batch]], onehot[order[i:i + batch]], rule)
        pred = net.predict(xte).argmax(1)
        for j in range(k + 1):
            m = np.isin(yte, (2 * j, 2 * j + 1)); a = float((pred[m] == yte[m]).mean())
            if j == k: first[j] = a
            if k == 4: acc_after.append(a)
    seen = np.isin(yte, range(10))
    return {'final': float((net.predict(xte).argmax(1) == yte)[seen].mean()),
            'forgetting': float(np.mean([first[j] - acc_after[j] for j in range(4)]))}


TASKS = {'aim': ('Aim at the hotter of two objects (hit rate after 8,000 trials)', task_aim),
         'digits': ('Digits from 50 examples per class (accuracy)', task_digits),
         'continual': ('Digit classes learned in sequence (final accuracy)', task_continual)}


def run(task='aim', backbone='central_complex', wiring_kind='measured', rule='teachers_wiring', biology='plain', seed=1):
    if task not in TASKS: raise ValueError('Unknown task')
    if rule not in RULES: raise ValueError('Unknown rule')
    if backbone not in BACKBONES: raise ValueError('Unknown backbone')
    r = TASKS[task][1](dict(backbone=backbone, wiring_kind=wiring_kind, biology=biology), rule, int(seed))
    return dict(r, task=task, backbone=backbone, wiring=wiring_kind, rule=rule, biology=biology, seed=int(seed))


def _job(args):
    return run(*args)


def benchmark(seeds=(1, 2, 3), out=None, workers=None):
    jobs = []
    for seed in seeds:
        for task in TASKS:
            for rule in RULES:
                for w in WIRINGS: jobs.append((task, 'central_complex', w, rule, 'plain', seed))        # rules x wirings
            for bio in BIOLOGY:
                if bio == 'plain': continue
                for rule in ('teachers_wiring', 'backprop'):
                    for w in ('measured', 'shuffled'): jobs.append((task, 'central_complex', w, rule, bio, seed))   # biology, one at a time
            for rule in ('readout', 'teachers_wiring', 'backprop'):
                for w in WIRINGS: jobs.append((task, 'motor', w, rule, 'plain', seed))                   # a second backbone
    jobs = sorted(set(jobs)); start = time.time()
    import multiprocessing as mp, os
    workers = workers or max(1, min(4, os.cpu_count() or 1))
    # finished runs are kept in a side file, so an interrupted benchmark resumes instead of starting over
    cache = BASE / 'data/backbone-runs.partial'
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line); done[(r['task'], r['backbone'], r['wiring'], r['rule'], r['biology'], r['seed'])] = r
    todo = [j for j in jobs if j not in done]
    with cache.open('a') as f:
        if workers > 1 and todo:
            with mp.get_context('spawn').Pool(workers) as pool:
                for r in pool.imap_unordered(_job, todo, chunksize=1):
                    done[(r['task'], r['backbone'], r['wiring'], r['rule'], r['biology'], r['seed'])] = r; f.write(json.dumps(r) + '\n'); f.flush()
        else:
            for j in todo:
                r = _job(j); done[j] = r; f.write(json.dumps(r) + '\n'); f.flush()
    runs = [done[j] for j in jobs]
    table = {}
    for r in runs: table.setdefault((r['task'], r['backbone'], r['wiring'], r['rule'], r['biology']), []).append(r)
    rows = []
    for (task, bb, w, rule, bio), rs in sorted(table.items()):
        row = dict(task=task, backbone=bb, wiring=w, rule=rule, biology=bio, final=float(np.mean([r['final'] for r in rs])),
                   final_sd=float(np.std([r['final'] for r in rs])), per_seed=[round(r['final'], 3) for r in rs])
        if 'curve' in rs[0]:
            key = 'hit_rate' if task == 'aim' else 'accuracy'
            row['curve'] = [dict(c, **{key: float(np.mean([r['curve'][i][key] for r in rs]))}) for i, c in enumerate(rs[0]['curve'])]
        if 'forgetting' in rs[0]: row['forgetting'] = float(np.mean([r['forgetting'] for r in rs]))
        rows.append(row)
    result = dict(seeds=list(seeds), steps=T, tasks={k: v[0] for k, v in TASKS.items()}, rules=RULES, wirings=WIRINGS, biology=BIOLOGY,
                  backbones={k: dict(label=v['label'], cells=raw(k)['n'], connections=int(len(raw(k)['pre'])), inputs=int(len(raw(k)['inputs'])),
                                     outputs=int(len(raw(k)['outputs']))) for k, v in BACKBONES.items()},
                  rows=rows, minutes=round((time.time() - start) / 60, 1))
    (out or BASE / 'data/backbone-benchmark.json').write_text(json.dumps(result, indent=1))
    if cache.exists(): cache.unlink()
    return result


if __name__ == '__main__':
    r = benchmark()
    for row in r['rows']:
        print(f"{row['task']:10} {row['backbone']:16} {row['wiring']:9} {row['rule']:16} {row['biology']:11} {row['final']:.3f} ± {row['final_sd']:.3f}", file=sys.stderr)
    print(r['minutes'], 'minutes', file=sys.stderr)
