"""Sparse expansion for learning tasks in sequence (class-incremental continual learning).

Borrowed from the fly: a wide random-like expansion (314 projection neurons -> 4,064 Kenyon cells),
a winner-take-most threshold that leaves ~5% of expansion units active, and an associative readout in
which a reward-like signal strengthens only the active units' connections to the correct class.

Question: does this structure reduce forgetting when classes arrive in sequence, and does the
*measured* MaleCNS wiring matter, or only the structure?

Tasks: 8x8 handwritten digits (UCI, bundled as data/digits.npz), learned one task after another.
'digits': five tasks of two digit classes. 'two_shape': five classes that each merge two different
digits (0+5, 1+6, ...), in three tasks, so a single average cannot describe a class. After each task,
every model is tested on all classes seen so far, choosing among all classes (class-incremental).

Measured: PN->KC edges and synapse counts (normalised per Kenyon cell, as in learning.py).
Assumed: pixels drive projection neurons through a fixed random assignment (each pixel ~5 PNs), the 5%
threshold, both readout rules, all learning rates, and the digit task itself. Flies do not read digits.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from learning import LearningCircuit
from model import BASE

TASK_SETS = {
    # Each digit is its own class; five tasks of two classes.
    'digits': {'classes': 10, 'relabel': lambda y: y, 'tasks': [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]},
    # Each class merges two different digit shapes (0+5, 1+6, 2+7, 3+8, 4+9); three tasks.
    'two_shape': {'classes': 5, 'relabel': lambda y: y % 5, 'tasks': [(0, 1), (2, 3), (4,)]},
}
TASKS = TASK_SETS['digits']['tasks']
MODELS = {
    'fly_measured': 'Measured PN→KC, 5% active, associative readout',
    'fly_shuffled': 'Shuffled PN→KC, 5% active, associative readout',
    'random_sparse': 'Random sparse wiring, 5% active, associative readout',
    'dense_topk': 'Dense random projection, 5% active, associative readout',
    'no_sparsity': 'Measured PN→KC, all units active, associative readout',
    'fly_delta': 'Measured PN→KC, 5% active, error-driven readout',
    'pixels_delta': 'No expansion: softmax regression on pixels',
    'pixels_associative': 'No expansion: associative readout on pixels',
    'mlp_backprop': 'Dense MLP (64→512→10), backpropagation',
}
DEFAULT = dict(seed=7, per_class=0, epochs=3, sparsity=0.05, lr=0.05, task='digits')


def load_digits():
    d = np.load(BASE / 'data/digits.npz')
    return d['images'].astype(np.float64) / 16.0, d['labels'].astype(int)


def split(x, y, seed, per_class):
    """Stratified 70/30 split; optionally keep only `per_class` training examples per class."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(y == c)); cut = int(round(len(idx) * 0.7))
        tr.extend(idx[:cut][:per_class] if per_class else idx[:cut]); te.extend(idx[cut:])
    return np.array(tr), np.array(te)


class Expansion:
    """Pixels -> projection neurons -> expansion units -> (optional) top-k sparsification."""
    def __init__(self, circuit, kind, seed, sparsity):
        rng = np.random.default_rng(seed + 500)
        npn = circuit.np
        self.pix_to_pn = rng.permutation(npn) % 64          # each PN listens to one pixel, ~5 PNs per pixel
        if kind in ('fly_measured', 'no_sparsity', 'fly_delta'): w = circuit.encoder('intact', seed)
        elif kind == 'fly_shuffled': w = circuit.encoder('shuffled', seed)
        elif kind == 'random_sparse':                         # same per-KC in-degree and weights, random sources
            base = circuit.encoder('intact', seed).tocsr(); rows, cols, vals = [], [], []
            for k in range(base.shape[0]):
                s, e = base.indptr[k], base.indptr[k + 1]
                if e > s:
                    rows += [k] * (e - s); cols += list(rng.choice(npn, e - s, replace=False)); vals += list(base.data[s:e])
            w = sp.csr_matrix((vals, (rows, cols)), shape=base.shape)
        elif kind == 'dense_topk':
            w = rng.normal(0, 1 / math.sqrt(npn), (circuit.encoder('intact', seed).shape[0], npn))
        else: raise ValueError(kind)
        self.w, self.k_active = w, None if kind == 'no_sparsity' else max(1, int(round(sparsity * w.shape[0])))
        self.n = w.shape[0]

    def __call__(self, images):
        pn = images[:, self.pix_to_pn]                       # (samples, PNs)
        drive = np.asarray((self.w @ pn.T).T) if sp.issparse(self.w) else pn @ self.w.T
        if self.k_active is None: return np.maximum(drive, 0)
        out = np.zeros_like(drive)
        top = np.argpartition(-drive, self.k_active - 1, axis=1)[:, :self.k_active]
        np.put_along_axis(out, top, 1.0, axis=1)             # binary code: the k most driven units
        return out


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


class Associative:
    """Reward-gated Hebbian readout: only the correct class's weights from active units grow.
    Prediction compares each class's summed weight onto the active units, normalised per class."""
    def __init__(self, n): self.W = np.zeros((10, n)); self.count = np.zeros(10)
    def fit_batch(self, z, y, lr):
        for c in np.unique(y):
            m = y == c; self.W[c] += lr * z[m].sum(0); self.count[c] += m.sum()
    def predict(self, z):
        norm = np.linalg.norm(self.W, axis=1); norm[norm == 0] = 1
        s = z @ (self.W / norm[:, None]).T; s[:, self.count == 0] = -np.inf
        return s.argmax(1)
    params = property(lambda self: self.W.size)


class Delta:
    """Error-driven softmax readout (gradient of cross-entropy); updates every class's weights."""
    def __init__(self, n): self.W = np.zeros((10, n)); self.b = np.zeros(10)
    def fit_batch(self, z, y, lr):
        p = softmax(z @ self.W.T + self.b); p[np.arange(len(y)), y] -= 1
        self.W -= lr * p.T @ z / len(y); self.b -= lr * p.mean(0)
    def predict(self, z): return (z @ self.W.T + self.b).argmax(1)
    params = property(lambda self: self.W.size + self.b.size)


class MLP:
    def __init__(self, seed, hidden=512):
        r = np.random.default_rng(seed + 900)
        self.W1 = r.normal(0, math.sqrt(2 / 64), (64, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, math.sqrt(2 / hidden), (hidden, 10)); self.b2 = np.zeros(10)
    def fit_batch(self, x, y, lr):
        h = np.maximum(x @ self.W1 + self.b1, 0); p = softmax(h @ self.W2 + self.b2); p[np.arange(len(y)), y] -= 1; p /= len(y)
        dh = (p @ self.W2.T) * (h > 0)
        self.W2 -= lr * h.T @ p; self.b2 -= lr * p.sum(0); self.W1 -= lr * x.T @ dh; self.b1 -= lr * dh.sum(0)
    def predict(self, x): return (np.maximum(x @ self.W1 + self.b1, 0) @ self.W2 + self.b2).argmax(1)
    params = property(lambda self: self.W1.size + self.b1.size + self.W2.size + self.b2.size)


class Experiment:
    def __init__(self, circuit=None): self.c = circuit or LearningCircuit(); self.x, self.y = load_digits()

    def validate(self, o):
        p = DEFAULT | o
        if p['task'] not in TASK_SETS: raise ValueError('task must be digits or two_shape')
        for k, lo, hi in [('seed', 0, 999999), ('per_class', 0, 125), ('epochs', 1, 20)]:
            v = p[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v) or not lo <= v <= hi: raise ValueError(f'{k} must be an integer in [{lo},{hi}]')
            p[k] = int(v)
        for k, lo, hi in [('sparsity', 0.005, 0.5), ('lr', 1e-4, 1.0)]:
            p[k] = float(p[k])
            if not math.isfinite(p[k]) or not lo <= p[k] <= hi: raise ValueError(f'{k} must be in [{lo},{hi}]')
        return p

    def model(self, kind, p):
        if kind == 'pixels_delta': return None, Delta(64)
        if kind == 'pixels_associative': return None, Associative(64)
        if kind == 'mlp_backprop': return None, MLP(p['seed'])
        feat = Expansion(self.c, kind, p['seed'], p['sparsity'])
        return feat, (Delta if kind == 'fly_delta' else Associative)(feat.n)

    def run(self, **options):
        p = self.validate(options)
        spec = TASK_SETS[p['task']]; tasks = spec['tasks']; ncls = spec['classes']
        tr, te = split(self.x, self.y, p['seed'], p['per_class'])     # split by digit, so both variants use the same images
        labels = spec['relabel'](self.y)
        rng = np.random.default_rng(p['seed'] + 7)
        results, codes = {}, {}
        for kind in MODELS:
            feat, head = self.model(kind, p)
            f_tr = feat(self.x[tr]) if feat else self.x[tr]; f_te = feat(self.x[te]) if feat else self.x[te]
            y_tr, y_te = labels[tr], labels[te]
            lr = p['lr'] * (10 if kind in ('pixels_delta', 'mlp_backprop', 'fly_delta') else 1)   # error-driven rules need larger steps
            R = np.full((len(tasks), len(tasks)), np.nan)
            for i, task in enumerate(tasks):
                idx = np.flatnonzero(np.isin(y_tr, task))
                for _ in range(p['epochs']):
                    order = rng.permutation(idx)
                    for s in range(0, len(order), 16): head.fit_batch(f_tr[order[s:s + 16]], y_tr[order[s:s + 16]], lr)
                pred = head.predict(f_te)
                for j in range(i + 1):
                    m = np.isin(y_te, tasks[j]); R[i, j] = float((pred[m] == y_te[m]).mean())
            final = head.predict(f_te)
            forgetting = [float(np.nanmax(R[:, j]) - R[-1, j]) for j in range(len(tasks) - 1)]
            results[kind] = {'label': MODELS[kind], 'trainable_parameters': int(head.params),
                             'expansion_units': int(feat.n) if feat else 0,
                             'active_fraction': float(f_tr.astype(bool).mean()) if feat else None,
                             'final_accuracy': float((final == y_te).mean()), 'forgetting': float(np.mean(forgetting)),
                             # Entries for tasks not yet learned are null (NaN is not valid JSON for browsers).
                             'accuracy_matrix': [[None if math.isnan(v) else v for v in row] for row in R.tolist()],
                             'seen_accuracy_by_task': [float(np.mean(R[i, :i + 1])) for i in range(len(tasks))]}
            if kind in ('fly_measured', 'dense_topk', 'no_sparsity'):
                means = np.array([f_tr[y_tr == c].mean(0) for c in range(ncls)])
                bins = means > 0.5 * means.max(1, keepdims=True)   # each class's typical code
                codes[kind] = [[float((bins[a] & bins[b]).sum() / max(1, (bins[a] | bins[b]).sum())) for b in range(ncls)] for a in range(ncls)]
        # Upper bound: the MLP trained on all classes together, same total passes.
        joint = MLP(p['seed']); order_all = np.arange(len(tr))
        for _ in range(p['epochs'] * len(tasks)):
            o = rng.permutation(order_all)
            for s in range(0, len(o), 16): joint.fit_batch(self.x[tr][o[s:s + 16]], labels[tr][o[s:s + 16]], p['lr'] * 10)
        return {'parameters': p, 'tasks': tasks, 'classes': ncls, 'train_size': int(len(tr)), 'test_size': int(len(te)),
                'models': results, 'class_code_overlap': codes,
                'joint_upper_bound': float((joint.predict(self.x[te]) == labels[te]).mean()),
                'chance': 1 / ncls,
                'interpretation': 'Class-incremental: no task label at test time. Forgetting is the average drop from each earlier task\'s best '
                                  'accuracy to its final accuracy. The associative readout changes only the correct class\'s weights, so its '
                                  'forgetting comes from new classes competing at test time, not from overwritten weights.'}

    def suite(self, seed=7):
        out = {'defaults': DEFAULT, 'labels': MODELS, 'seeds': 5, 'conditions': {},
               'interpretation': 'Five seeds per condition at default settings; seeds change the split, pixel-to-PN assignment, random '
                                 'wirings and presentation order. Model experiments on a toy dataset, not evidence about fly cognition.'}
        for name, opts in [('digits', {}), ('digits_5_per_class', {'per_class': 5}), ('two_shape', {'task': 'two_shape'})]:
            runs = [self.run(seed=s, **opts) for s in range(seed, seed + 5)]
            out['conditions'][name] = {
                'options': opts, 'parameters': {k: runs[0]['models'][k]['trainable_parameters'] for k in MODELS},
                'joint_upper_bound': float(np.mean([r['joint_upper_bound'] for r in runs])),
                'summary': {k: {m: {'mean': float(np.mean([r['models'][k][m] for r in runs])), 'sd': float(np.std([r['models'][k][m] for r in runs]))}
                                for m in ['final_accuracy', 'forgetting']} for k in MODELS}}
        return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--benchmark', action='store_true'); ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args(); e = Experiment()
    if a.benchmark:
        (BASE / 'data/expansion-benchmark.json').write_text(json.dumps(e.suite(a.seed), indent=1)); print('Wrote data/expansion-benchmark.json')
    else:
        r = e.run(seed=a.seed)
        for k, v in r['models'].items(): print(f"{k:14} final {v['final_accuracy']:.3f}  forgetting {v['forgetting']:.3f}  params {v['trainable_parameters']}")
        print('joint upper bound', round(r['joint_upper_bound'], 3))
