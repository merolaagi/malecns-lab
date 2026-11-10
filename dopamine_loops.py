"""Reward and punishment through the fly's own dopamine wiring.

backbone.py asks whether learning inside the connectome helps, with teaching signals that say how wrong each
output was. This file asks the harder question: can the network learn from reward and punishment alone (it is told
only whether its action was right), and does the fly's measured dopamine wiring deliver that signal well?

The network is the mushroom body exactly as measured (data/mushroom-body.json):

    input -> 314 projection neurons -> 4,064 Kenyon cells (5% active) -> 97 output neurons -> actions
                                                   ^ learned synapses ^
    340 dopamine neurons reach the output neurons, and the output neurons feed back onto the dopamine neurons

The Kenyon-cell -> output-neuron synapses learn, as in the fly. Each output neuron has a fixed random vote for
each action. Varied, one at a time:

1. How dopamine is delivered
   readout        the backbone is frozen; only the votes are learned (what the lab did until now)
   global         one dopamine signal for the whole network, judging random exploration in the output neurons
   per_cell       dopamine delivered separately to every output neuron (ideal routing; no wiring says how)
   loops          dopamine routed through the measured wiring: each dopamine neuron hears the output neurons that
                  feed back onto it and teaches the output neurons it reaches
   loops_shuffled the same with the dopamine neurons' targets shuffled
   backprop       gradient descent on the same synapses, told the right answer (the upper bound)
2. The wiring: measured, shuffled (every cell keeps its number of connections), random (same density).
3. Neuron biology, one ingredient at a time on the 'per_cell' rule (the reward-only rule that learns): sparse coding, cell size setting excitability,
   prediction-error dopamine, depression-only excitatory synapses, connections between output neurons, dendritic
   branches by Kenyon-cell class.

Tasks: aiming at a target among decoys (8 directions), handwritten digits (10), and digits arriving two classes at
a time. The measure is accuracy on held-out examples after a number of trials.

Measured: which cells connect, synapse counts, transmitters, Kenyon-cell classes. Assumed: the rate model, the
learning rules and their rates (one per rule), the input mapping, the action votes, the tasks. Signs of the
dopamine loops are ignored (strengths only).

    python3 dopamine_loops.py      # data/dopamine-loops.json
"""
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

import expansion as ex

BASE = Path(__file__).resolve().parent
SPARSITY = 0.05
RULES = {'readout': 'Frozen backbone, learned readout', 'global': 'One dopamine signal for everything',
         'per_cell': 'Dopamine per output neuron (ideal routing)', 'loops': 'Dopamine through the measured loops',
         'loops_shuffled': 'Dopamine through shuffled loops', 'backprop': 'Backpropagation on the same synapses (told the answer)'}
WIRINGS = {'measured': 'Measured', 'shuffled': 'Shuffled, same connection counts', 'random': 'Random, same density'}
# one rate per rule, the best of a fixed grid on a separate seed (1), by accuracy after 1,000 trials on aim and digits
RATES = {'readout': 0.05, 'global': 0.2, 'per_cell': 0.5, 'loops': 1.0, 'loops_shuffled': 1.0, 'backprop': 1.0}
PLAIN = dict(sparse=True, size=True, prediction_error=True, depression_only=False, lateral=False, dendrites=False, compartment_votes=False)
BIOLOGY = {   # name -> (text, change to the reference model)
    'plain': ('Reference: sparse code, size-scaled cells, prediction-error dopamine', {}),
    'no_sparse': ('Without sparse coding (all Kenyon cells respond)', {'sparse': False}),
    'no_size': ('Without size-scaled excitability (identical cells)', {'size': False}),
    'no_prediction_error': ('Dopamine signals raw reward and punishment, not surprise', {'prediction_error': False}),
    'depression_only': ('Synapses excitatory only, dopamine can only weaken them (as in the fly)', {'depression_only': True}),
    'lateral': ('With the measured connections between output neurons', {'lateral': True}),
    'dendrites': ('With dendritic branches, one per Kenyon-cell class', {'dendrites': True}),
    'compartment_votes': ('Output neurons of one compartment vote together (as valence is organised in the fly)', {'compartment_votes': True}),
}
CHECKPOINTS = [50, 200, 1000, 4000]
TASKS = {'aim': 'Aim at the target among decoys (8 directions)', 'digits': 'Handwritten digits (10 classes)', 'continual': 'Digits, two classes at a time'}


# --- the measured circuit ---------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def circuit(path=None):
    d = json.loads(Path(path or BASE / 'data/mushroom-body.json').read_text())
    by_role = {}
    for n in d['nodes']: by_role.setdefault(n['role'], []).append(n)
    idx = {role: {n['bodyId']: i for i, n in enumerate(ns)} for role, ns in by_role.items()}
    role = {n['bodyId']: n['role'] for n in d['nodes']}
    shape = lambda a, b: (len(by_role[a]), len(by_role[b]))
    M = {k: np.zeros(shape(*k)) for k in (('PN', 'KC'), ('KC', 'MBON'), ('MBON', 'MBON'), ('MBON', 'DAN'), ('DAN', 'MBON'))}
    for pre, post, count in d['edges']:
        k = (role[pre], role[post])
        if k in M: M[k][idx[k[0]][pre], idx[k[1]][post]] += count
    classes = sorted({n['kc_class'] for n in by_role['KC'] if n['kc_class']})
    kc_class = np.array([classes.index(n['kc_class']) if n['kc_class'] in classes else 0 for n in by_role['KC']])
    sign = np.array([1.0 if n['nt'] == 'acetylcholine' else -1.0 for n in by_role['MBON']])
    comp = [tuple(sorted(n['compartments_in'])) for n in by_role['MBON']]
    return dict(pn_kc=M[('PN', 'KC')], kc_mbon=M[('KC', 'MBON')], mbon_mbon=M[('MBON', 'MBON')] * sign[:, None],
                mbon_dan=M[('MBON', 'DAN')], dan_mbon=M[('DAN', 'MBON')], kc_class=kc_class, classes=classes, compartments=comp,
                counts={k: len(v) for k, v in by_role.items()})


def shuffle_targets(W, rng):
    """Every target keeps its number and sizes of inputs; which cells they come from is permuted."""
    out = np.zeros_like(W)
    for j in range(W.shape[1]): out[:, j] = W[rng.permutation(W.shape[0]), j]
    return out


def random_like(W, rng):
    out = np.zeros_like(W)
    nz = W[W > 0]
    out.flat[rng.choice(W.size, len(nz), replace=False)] = rng.permutation(nz)
    return out


def loops(mbon_dan, dan_mbon):
    """S[j, k]: how much of the dopamine reaching output neuron j comes from dopamine neurons that hear output neuron k."""
    F = mbon_dan / np.maximum(mbon_dan.sum(0, keepdims=True), 1e-9)        # each dopamine neuron: whose opinion it hears
    D = dan_mbon / np.maximum(dan_mbon.sum(0, keepdims=True), 1e-9)        # each output neuron: whose dopamine it gets
    return D.T @ F.T


@lru_cache(maxsize=16)
def wiring(condition='measured', seed=0):
    c = circuit(); rng = np.random.default_rng(900 + seed)
    if condition == 'measured':
        w = dict(pn_kc=c['pn_kc'], kc_mbon=c['kc_mbon'], mbon_mbon=c['mbon_mbon'], mbon_dan=c['mbon_dan'], dan_mbon=c['dan_mbon'])
    elif condition == 'shuffled':
        w = dict(pn_kc=shuffle_targets(c['pn_kc'], rng), kc_mbon=shuffle_targets(c['kc_mbon'], rng),
                 mbon_mbon=shuffle_targets(c['mbon_mbon'], rng), mbon_dan=shuffle_targets(c['mbon_dan'], rng), dan_mbon=shuffle_targets(c['dan_mbon'], rng))
    elif condition == 'random':
        w = dict(pn_kc=random_like(c['pn_kc'], rng), kc_mbon=random_like(c['kc_mbon'], rng),
                 mbon_mbon=random_like(np.abs(c['mbon_mbon']), rng) * rng.choice([-1.0, 1.0], (c['mbon_mbon'].shape[0], 1)),
                 mbon_dan=random_like(c['mbon_dan'], rng), dan_mbon=random_like(c['dan_mbon'], rng))
    else:
        raise ValueError('wiring must be measured, shuffled or random')
    w['loops'] = loops(w['mbon_dan'], w['dan_mbon'])
    w['loops_shuffled'] = loops(w['mbon_dan'], w['dan_mbon'][:, rng.permutation(w['dan_mbon'].shape[1])])
    w['kc_class'] = c['kc_class'] if condition == 'measured' else rng.permutation(c['kc_class'])
    w['name'] = condition
    return w


def loop_structure(condition='measured', seed=0):
    """How much of each output neuron's dopamine comes back from itself, and from output neurons of its own compartment."""
    w, c = wiring(condition, seed), circuit()
    same = np.array([[bool(set(a) & set(b)) for b in c['compartments']] for a in c['compartments']])
    out = {}
    for key in ('loops', 'loops_shuffled'):
        S = w[key]; taught = S.sum(1) > 0; tot = np.maximum(S.sum(1), 1e-9)
        out[key] = {'output_neurons_taught': int(taught.sum()), 'of': int(len(taught)),
                    'from_itself': float((np.diag(S) / tot)[taught].mean()),
                    'from_own_compartment': float(((S * same).sum(1) / tot)[taught].mean()) if condition == 'measured' else None}
    return out


# --- tasks --------------------------------------------------------------------------------------------------------
def aim_data(n, rng, decoys=2, noise=0.25):
    """A target direction among weaker decoys on a ring of 32 direction-tuned inputs, plus 32 inputs of pure noise."""
    theta = np.arange(32) * 2 * math.pi / 32
    y = rng.integers(8, size=n)
    ang = (y + rng.uniform(-0.45, 0.45, n)) * 2 * math.pi / 8
    x = np.exp((np.cos(theta[None] - ang[:, None]) - 1) / 0.15)
    for _ in range(decoys):
        d = rng.uniform(0, 2 * math.pi, n)
        x += 0.6 * np.exp((np.cos(theta[None] - d[:, None]) - 1) / 0.15)
    x = np.concatenate([x, rng.random((n, 32))], 1) + noise * rng.normal(size=(n, 64))
    return np.clip(x, 0, None), y


def task_data(task, seed):
    rng = np.random.default_rng(300 + seed)
    if task == 'aim':
        xtr, ytr = aim_data(4000, rng); xte, yte = aim_data(600, rng)
        return xtr, ytr, xte, yte, 8
    x, y = ex.load_digits()
    order = rng.permutation(len(y)); x, y = x[order], y[order]
    return x[:-400], y[:-400], x[-400:], y[-400:], 10


# --- the network --------------------------------------------------------------------------------------------------
BETA = 3.0                               # how decisively scores become choices


class Network:
    def __init__(self, w, actions, seed, cfg=None, d=64):
        self.cfg = PLAIN | (cfg or {})
        rng = np.random.default_rng(700 + seed)
        self.rng, self.w = rng, w
        n_pn = w['pn_kc'].shape[0]
        self.inp = np.zeros((d, n_pn)); self.inp[rng.integers(d, size=n_pn * 3), np.repeat(np.arange(n_pn), 3)] = 1.0   # each input cell samples 3 inputs
        self.mask = w['kc_mbon'] > 0
        self.W = w['kc_mbon'].astype(float).copy()                                                       # start from the measured synapse counts
        self.W0 = self.W.copy()
        self.M = self.W.shape[1]
        self.V = rng.choice([-1.0, 1.0], (self.M, actions))                                              # each output neuron's fixed vote
        if self.cfg['compartment_votes']:                     # one vote per compartment; a cell in several takes the majority
            comps = circuit()['compartments']; cv = {k: rng.choice([-1.0, 1.0], actions) for k in sorted({x for t in comps for x in t})}
            order = np.arange(self.M) if w.get('name') == 'measured' else rng.permutation(self.M)
            for j, t in enumerate(comps):
                if t: self.V[order[j]] = np.sign(sum(cv[x] for x in t) + 1e-6 * rng.normal(size=actions))
        self.Vlearn = np.zeros_like(self.V)
        L = w['mbon_mbon']; self.L = 0.5 * L / max(np.abs(L).sum(0).max(), 1e-9)
        self.k = max(1, round(SPARSITY * self.W.shape[0]))
        self.branch = [np.flatnonzero(w['kc_class'] == b) for b in range(int(w['kc_class'].max()) + 1)]
        self.mu, self.sd = np.zeros(self.M), np.ones(self.M)

    def codes(self, x):
        drive = (x @ self.inp) @ self.w['pn_kc']
        drive = (drive - drive.mean(1, keepdims=True)) / (drive.std(1, keepdims=True) + 1e-9)
        if not self.cfg['sparse']:
            z = np.maximum(drive, 0); return z * (self.k / np.maximum(z.sum(1, keepdims=True), 1e-9))
        z = np.zeros_like(drive)
        np.put_along_axis(z, np.argpartition(-drive, self.k - 1, axis=1)[:, :self.k], 1.0, axis=1)
        return z

    def adapt(self, z):
        """Each output neuron's resting level and sensitivity, set once from its typical input. With size-scaled
        excitability every cell gets its own (a cell with more input needs more to respond); without it all cells
        share one setting, so cells with many inputs dominate."""
        raw = self.summed(z)
        if self.cfg['size']: self.mu, self.sd = raw.mean(0), raw.std(0) + 1e-9
        else: self.mu, self.sd = np.full(self.M, raw.mean()), np.full(self.M, raw.std(0).mean() + 1e-9)

    def summed(self, z):
        if not self.cfg['dendrites']: return z @ self.W
        out = 0
        for idx in self.branch:                               # each branch saturates on its own before the cell sums them
            part = z[:, idx] @ self.W[idx]
            cap = np.maximum(2.0 * self.W0[idx].sum(0) * self.k / self.W.shape[0], 1e-9)     # twice the branch's typical input
            out = out + cap * np.tanh(part / cap)
        return out

    def outputs(self, z):
        m = (self.summed(z) - self.mu) / self.sd
        if self.cfg['lateral']: m = m + m @ self.L
        return m

    def scores(self, z, learned_votes=False):
        m = self.outputs(z)
        return m @ (self.Vlearn if learned_votes else self.V) / math.sqrt(self.M), m

    def learn(self, rule, z, label, lr):
        """One trial: act, get reward or punishment, change synapses. Returns whether the action was right."""
        active = np.flatnonzero(z)
        s, m = self.scores(z[None], learned_votes=(rule == 'readout'))
        p = ex.softmax(s * BETA)[0]
        if rule == 'backprop':
            g = p.copy(); g[label] -= 1                       # told the right answer
            teach = -(self.V @ g)
            r = float(int(p.argmax()) == label)
        else:
            a = int(self.rng.choice(len(p), p=p)); r = float(a == label)
            delta = (r - p[a]) if self.cfg['prediction_error'] else (r - 0.5)
            if rule == 'readout':
                self.Vlearn[:, a] += lr * delta * m[0]; return int(r)
            if rule == 'global':
                noise = self.rng.normal(0, 1.0, self.M)       # exploration in the output neurons; one dopamine signal judges it
                a = int(((m[0] + noise) @ self.V).argmax()); r = float(a == label)
                delta = (r - p[a]) if self.cfg['prediction_error'] else (r - 0.5)
                teach = delta * noise
            elif rule == 'per_cell':
                teach = delta * self.V[:, a]
            else:
                t = self.w[rule] @ self.V[:, a]               # 'loops' or 'loops_shuffled': dopamine through the wiring
                teach = delta * t / (np.abs(t).mean() + 1e-9)
        dW = (lr / self.k) * (z[active][:, None] * (teach * self.sd)[None, :]) * self.mask[active] * (self.k / max(z[active].sum(), 1e-9))
        if self.cfg['depression_only']:
            self.W[active] += 2 * np.minimum(dW, 0) + 0.002 * (self.W0[active] - self.W[active])   # dopamine only weakens; slow recovery
            np.maximum(self.W, 0, out=self.W)
        else:
            self.W[active] += dW
        return int(r)

    def predict(self, z, rule, allowed=None):
        s, _ = self.scores(z, learned_votes=(rule == 'readout'))
        if allowed is not None: s[:, allowed:] = -1e9
        return s.argmax(1)


def run(task='aim', rule='loops', wiring_condition='measured', seed=7, cfg=None, trials=None, lr=None):
    if task not in TASKS: raise ValueError('unknown task')
    if rule not in RULES: raise ValueError('unknown rule')
    xtr, ytr, xte, yte, actions = task_data('digits' if task == 'continual' else task, seed)
    net = Network(wiring(wiring_condition, seed), actions, seed, cfg)
    ztr, zte = net.codes(xtr), net.codes(xte)
    net.adapt(ztr[:500])
    lr = RATES[rule] if lr is None else lr
    trials = trials or CHECKPOINTS[-1]
    rng = np.random.default_rng(500 + seed)
    curve, hits = [], []
    if task == 'continual':
        per_phase = trials // 5; first = None
        for phase in range(5):
            pool = np.flatnonzero((ytr == 2 * phase) | (ytr == 2 * phase + 1))
            for _ in range(per_phase):
                i = int(pool[rng.integers(len(pool))]); hits.append(net.learn(rule, ztr[i], int(ytr[i]), lr))
            pred = net.predict(zte, rule, allowed=2 * phase + 2)          # answers are among the classes met so far
            seen, old = yte < 2 * phase + 2, yte < 2
            acc, first_acc = float((pred == yte)[seen].mean()), float((pred == yte)[old].mean())
            if phase == 0: first = first_acc
            curve.append({'trials': (phase + 1) * per_phase, 'accuracy': acc, 'first_task_accuracy': first_acc})
        return {'task': task, 'rule': rule, 'wiring': wiring_condition, 'curve': curve, 'final': curve[-1]['accuracy'],
                'forgetting': first - curve[-1]['first_task_accuracy'], 'chance': 1 / actions}
    for t in range(1, trials + 1):
        i = int(rng.integers(len(ytr))); hits.append(net.learn(rule, ztr[i], int(ytr[i]), lr))
        if t in CHECKPOINTS: curve.append({'trials': t, 'accuracy': float((net.predict(zte, rule) == yte).mean())})
    return {'task': task, 'rule': rule, 'wiring': wiring_condition, 'curve': curve, 'final': curve[-1]['accuracy'],
            'reward_rate_last_500': float(np.mean(hits[-500:])), 'chance': 1 / actions}


def standard_network(task, seed, trials=None):
    """A standard dense network (64 -> 256 -> actions), backpropagation, told the answer on every trial."""
    import scaffold as sc
    xtr, ytr, xte, yte, actions = task_data('digits' if task == 'continual' else task, seed)
    net = sc.MLPn(64, actions, seed); rng = np.random.default_rng(500 + seed); curve = []
    trials = trials or CHECKPOINTS[-1]
    top = xtr.max() + 1e-9
    if task == 'continual':
        per = trials // 5; first = None
        for phase in range(5):
            pool = np.flatnonzero((ytr == 2 * phase) | (ytr == 2 * phase + 1))
            for _ in range(per):
                i = pool[rng.integers(len(pool), size=1)]; net.fit(xtr[i] / top, ytr[i], 0.05)
            h = np.maximum((xte / top) @ net.W1 + net.b1, 0) @ net.W2 + net.b2; h[:, 2 * phase + 2:] = -1e9
            pred = h.argmax(1)
            acc = float((pred == yte)[yte < 2 * phase + 2].mean()); f = float((pred == yte)[yte < 2].mean())
            if phase == 0: first = f
            curve.append({'trials': (phase + 1) * per, 'accuracy': acc, 'first_task_accuracy': f})
        return {'curve': curve, 'final': curve[-1]['accuracy'], 'forgetting': first - curve[-1]['first_task_accuracy']}
    for t in range(1, trials + 1):
        i = rng.integers(len(ytr), size=1); net.fit(xtr[i] / top, ytr[i], 0.05)
        if t in CHECKPOINTS: curve.append({'trials': t, 'accuracy': float((net.predict(xte / top) == yte).mean())})
    return {'curve': curve, 'final': curve[-1]['accuracy']}


def mean_runs(runs):
    out = {k: v for k, v in runs[0].items() if k not in ('curve', 'final', 'forgetting', 'reward_rate_last_500')}
    out['curve'] = [{'trials': c['trials'], **{k: float(np.mean([r['curve'][i][k] for r in runs])) for k in c if k != 'trials'}} for i, c in enumerate(runs[0]['curve'])]
    for k in ('final', 'forgetting', 'reward_rate_last_500'):
        if k in runs[0]: out[k] = float(np.mean([r[k] for r in runs])); out[k + '_sd'] = float(np.std([r[k] for r in runs]))
    return out


def benchmark(seeds=(7, 8, 9), out=None):
    c = circuit()
    rules = {t: {r: {w: mean_runs([run(t, r, w, s) for s in seeds]) for w in WIRINGS} for r in RULES} for t in TASKS}
    reference = {t: mean_runs([standard_network(t, s) for s in seeds]) for t in TASKS}
    # biology is tested on the ideal-routing rule, the reward-only rule that learns; the loops rule is added for the
    # one ingredient that concerns it (votes organised by compartment)
    biology = {t: {b: mean_runs([run(t, 'per_cell', 'measured', s, cfg) for s in seeds]) for b, (_, cfg) in BIOLOGY.items()} for t in TASKS}
    loops_by_compartment = {t: {r: mean_runs([run(t, r, 'measured', s, {'compartment_votes': True}) for s in seeds]) for r in ('loops', 'loops_shuffled')} for t in TASKS}
    result = {'seeds': list(seeds), 'tasks': TASKS, 'rules': RULES, 'wirings': WIRINGS, 'checkpoints': CHECKPOINTS, 'rates': RATES,
              'cells': c['counts'], 'learned_synapses': int((c['kc_mbon'] > 0).sum()), 'kenyon_classes': c['classes'],
              'by_rule': rules, 'standard_network': reference, 'biology': biology, 'biology_rule': 'per_cell', 'loops_with_compartment_votes': loops_by_compartment, 'biology_text': {k: v[0] for k, v in BIOLOGY.items()},
              'loop_structure': {w: loop_structure(w, seeds[0]) for w in WIRINGS}}
    (out or BASE / 'data/dopamine-loops.json').write_text(json.dumps(result, indent=1))
    return result


if __name__ == '__main__':
    r = benchmark()
    for t in TASKS:
        print(t, file=sys.stderr)
        for rule in RULES:
            print(f"  {rule:15}", '  '.join(f"{w}: {r['by_rule'][t][rule][w]['final']:.2f}" for w in WIRINGS), file=sys.stderr)
        print('  standard net   ', round(r['standard_network'][t]['final'], 2), file=sys.stderr)
        for b in BIOLOGY: print(f"    {b:20} {r['biology'][t][b]['final']:.2f}", file=sys.stderr)
    print(r['loop_structure'], file=sys.stderr)
