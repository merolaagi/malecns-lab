"""A connectome-scaffolded learner with sparse distributed representations: how little data does it need?

Three sources, one model:
  fly connectome   a sparse expansion (measured projection-neuron -> Kenyon-cell wiring, top-k winners), one-shot
                   memory written only on the active cells (mushroom-body style), and a reference frame that tracks
                   position by adding up movements (central-complex style)
  HTM / SDR        codes are sparse distributed representations; prediction-assisted pooling (Byrne 2015,
                   arXiv:1509.08255): cells predicted by the previous step get a head start in the top-k
                   competition, so expected inputs keep their codes and context shapes the code
  Monty            features stored at locations in an object's reference frame, learned from one exploration,
                   recognised by moving a sensor and accumulating evidence, with several modules voting

Five tests, each against baselines trained on the same data:
  few_shot     8x8 digits with 1-50 examples per class; test images also shifted by one pixel; three glimpses with
               and without prediction-assisted pooling
  sequences    next-symbol prediction where the answer depends on a symbol four steps back; prediction-assisted
               context against first- and second-order Markov baselines
  objects      objects made of shared features, told apart only by where features are; reference-frame matching
               (one module, three voting modules) against a bag of features; learned from one exploration
  arithmetic   numbers as locations: learn only the successor relation (19 pairs), then add, subtract and compare
               by moving in the frame; against a backpropagation network trained on 19-400 addition examples
  continual    classes learned in sequence without forgetting (from the sparse-expansion benchmark)

Measured: the PN->KC wiring and synapse counts. Assumed: everything else (pixel->PN assignment, sparsity, the
learning rules, the tasks, the reference-frame geometry). The arithmetic test gives the frame a line; what is
learned is which symbol sits where. Wiring is compared measured vs shuffled vs random, as elsewhere.

    python3 scaffold.py      # writes data/scaffold-benchmark.json
"""
import json
import math

import numpy as np

import expansion as ex
from learning import LearningCircuit
from model import BASE

SHOTS = [1, 2, 5, 10, 50]
WIRINGS = {'fly_measured': 'Measured PN→KC', 'fly_shuffled': 'Shuffled PN→KC', 'random_sparse': 'Random sparse'}
SPARSITY = 0.05
BETA = 1.0                       # prediction-assisted head start, in units of the drive's standard deviation


# --- encoder with prediction-assisted top-k -----------------------------------------------------------------------
class Encoder:
    """Inputs (64 values) -> projection neurons -> expansion units -> top-k code, optionally biased by a prediction."""

    def __init__(self, circuit, wiring, seed, sparsity=SPARSITY):
        self.e = ex.Expansion(circuit, wiring, seed, sparsity)
        self.k = self.e.k_active
        self.n = self.e.n

    def drive(self, x):
        pn = x[:, self.e.pix_to_pn]
        d = np.asarray((self.e.w @ pn.T).T) if hasattr(self.e.w, 'tocsr') else pn @ self.e.w.T
        sd = d.std(1, keepdims=True); sd[sd == 0] = 1
        return (d - d.mean(1, keepdims=True)) / sd

    def code(self, x, bias=None, beta=BETA):
        d = self.drive(x)
        if bias is not None: d = d + beta * bias
        out = np.zeros_like(d)
        top = np.argpartition(-d, self.k - 1, axis=1)[:, :self.k]
        np.put_along_axis(out, top, 1.0, axis=1)
        return out


class Memory:
    """One-shot associative memory: a label's weights grow on the active cells only (nothing else changes)."""

    def __init__(self, n, labels):
        self.W = np.zeros((labels, n)); self.seen = np.zeros(labels, bool)

    def learn(self, z, y):
        for c in np.unique(y): self.W[c] += z[y == c].sum(0); self.seen[c] = True

    def scores(self, z):
        norm = np.linalg.norm(self.W, axis=1); norm[norm == 0] = 1
        s = z @ (self.W / norm[:, None]).T; s[:, ~self.seen] = -np.inf
        return s


def shift(images, rng):
    """Each 8x8 image moved by one pixel in a random direction (zeros fill the edge)."""
    out = np.zeros_like(images)
    for i, img in enumerate(images.reshape(-1, 8, 8)):
        dy, dx = [(0, 1), (0, -1), (1, 0), (-1, 0)][rng.integers(4)]
        moved = np.roll(np.roll(img, dy, 0), dx, 1)
        if dy == 1: moved[0] = 0
        if dy == -1: moved[-1] = 0
        if dx == 1: moved[:, 0] = 0
        if dx == -1: moved[:, -1] = 0
        out[i] = moved.ravel()
    return out


# --- 1. few-shot digits -------------------------------------------------------------------------------------------
def few_shot(circuit, x, y, seed):
    rng = np.random.default_rng(seed + 11)
    rows = []
    _, te = ex.split(x, y, seed, 0)
    x_te, y_te = x[te], y[te]
    x_sh = shift(x_te, rng)
    glimpses = [shift(x_te, np.random.default_rng(seed + 20 + g)) for g in range(2)]
    encoders = {w: Encoder(circuit, w, seed) for w in WIRINGS}
    for shots in SHOTS:
        tr, _ = ex.split(x, y, seed, shots)
        x_tr, y_tr = x[tr], y[tr]
        row = {'shots': shots}
        for w, enc in encoders.items():
            mem = Memory(enc.n, 10); mem.learn(enc.code(x_tr), y_tr)
            row[w] = float((mem.scores(enc.code(x_te)).argmax(1) == y_te).mean())
            row[w + '_shifted'] = float((mem.scores(enc.code(x_sh)).argmax(1) == y_te).mean())
            if w == 'fly_measured':
                # three glimpses (the image, then two shifted views): evidence summed over glimpses, with and without
                # prediction-assisted pooling (each glimpse's code biased by the previous one)
                views = [x_te] + glimpses
                plain = sum(mem.scores(enc.code(v)) for v in views)
                prev, assisted = None, 0
                for v in views:
                    z = enc.code(v, bias=prev); assisted = assisted + mem.scores(z); prev = z
                row['glimpses_plain'] = float((plain.argmax(1) == y_te).mean())
                row['glimpses_assisted'] = float((assisted.argmax(1) == y_te).mean())
        # baselines: nearest neighbour and softmax regression on pixels, a backprop MLP (trained to convergence)
        d = ((x_te[:, None, :] - x_tr[None, :, :]) ** 2).sum(2)
        row['pixels_nearest'] = float((y_tr[d.argmin(1)] == y_te).mean())
        delta = ex.Delta(64)
        for _ in range(200): delta.fit_batch(x_tr, y_tr, 0.5)
        row['pixels_softmax'] = float((delta.predict(x_te) == y_te).mean())
        mlp = ex.MLP(seed)
        for _ in range(300): mlp.fit_batch(x_tr, y_tr, 0.1)
        row['mlp_backprop'] = float((mlp.predict(x_te) == y_te).mean())
        row['mlp_backprop_shifted'] = float((mlp.predict(x_sh) == y_te).mean())
        rows.append(row)
    return rows


# --- 2. sequences with long context ------------------------------------------------------------------------------
def sequences(circuit, seed, exposures=(1, 2, 3, 5), betas=(0.0, 0.5, 1.0, 2.0), decay=0.7):
    """Six sequences share the middle (B C D E); the last symbol depends on the first. Predict the last symbol."""
    rng = np.random.default_rng(seed + 31)
    symbols = 16
    patterns = (rng.random((symbols, 64)) < 0.25).astype(float)
    shared = [1, 2, 3, 4]
    seqs = [[5 + i] + shared + [11 + i] for i in range(5)]
    out = {'exposures': list(exposures), 'betas': list(betas), 'sequences': seqs}
    for w in WIRINGS:
        enc = Encoder(circuit, w, seed)
        for beta in betas:
            accs = []
            for n_exp in exposures:
                mem = Memory(enc.n, symbols)
                for _ in range(n_exp):
                    for s in rng.permutation(len(seqs)):
                        for z, nxt in contexts(enc, patterns, seqs[s], beta, decay): mem.learn(z[None], np.array([nxt]))
                correct = 0
                for seq in seqs:
                    z, nxt = contexts(enc, patterns, seq, beta, decay)[-1]
                    correct += int(mem.scores(z[None]).argmax() == nxt)
                accs.append(correct / len(seqs))
            out[f'{w}|{beta}'] = accs
    # Markov baselines on the same data: the order needed to see the first symbol is 5
    for order in (1, 2, 5):
        correct = 0
        table = {}
        for seq in seqs:
            for t in range(len(seq) - 1):
                table.setdefault(tuple(seq[max(0, t - order + 1):t + 1]), []).append(seq[t + 1])
        for seq in seqs:
            ctx = tuple(seq[max(0, len(seq) - 1 - order):len(seq) - 1])
            options = table.get(ctx, [])
            # ties among equally frequent options are broken at random: expected accuracy 1/len(options)
            correct += (options.count(seq[-1]) / len(options)) if options else 0
        out[f'markov_order_{order}'] = correct / len(seqs)
    out['chance'] = 1 / 5
    return out


def contexts(enc, patterns, seq, beta, decay):
    """Codes along a sequence, each biased by a decaying trace of the previous codes; (code, next symbol) pairs."""
    trace, pairs = None, []
    for t in range(len(seq) - 1):
        z = enc.code(patterns[seq[t]][None], bias=None if trace is None or beta == 0 else trace[None], beta=beta)[0]
        trace = z if trace is None else z + decay * trace
        pairs.append((z, seq[t + 1]))
    return pairs


# --- 3. objects with reference frames ----------------------------------------------------------------------------
def objects(circuit, seed, n_objects=20, features=6, size=4, noise=0.1, max_glimpses=8, trials=200):
    """Objects are 4x4 grids of features drawn from only six, so many objects share the same features; they differ in
    where the features are. Learned from one exploration of every location."""
    rng = np.random.default_rng(seed + 41)
    enc = Encoder(circuit, 'fly_measured', seed)
    pats = (rng.random((features, 64)) < 0.25).astype(float)
    objs = rng.integers(features, size=(n_objects, size, size))
    stored = {(o, r, c): enc.code(pats[objs[o, r, c]][None])[0] for o in range(n_objects) for r in range(size) for c in range(size)}
    bag = np.array([sum(stored[(o, r, c)] for r in range(size) for c in range(size)) for o in range(n_objects)])
    bag /= np.linalg.norm(bag, axis=1, keepdims=True)

    def sense(o, r, c):
        p = pats[objs[o, r, c]].copy()
        flip = rng.random(64) < noise
        p[flip] = 1 - p[flip]
        return enc.code(p[None])[0]

    def path(start):
        locs = [start]
        while len(locs) < max_glimpses:
            r, c = locs[-1]
            moves = [(dr, dc) for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)) if 0 <= r + dr < size and 0 <= c + dc < size]
            dr, dc = moves[rng.integers(len(moves))]
            locs.append((r + dr, c + dc))
        return locs

    results = {k: np.zeros(max_glimpses) for k in ('reference_frame', 'voting_3_modules', 'bag_of_features')}
    k_active = enc.k
    for _ in range(trials):
        target = int(rng.integers(n_objects))
        modules = [path((int(rng.integers(size)), int(rng.integers(size)))) for _ in range(3)]
        obs = [[sense(target, *loc) for loc in m] for m in modules]
        evidence = []
        for m, locs in enumerate(modules):
            # hypotheses: (object, start location); displacements are known (the sensor's own movement)
            ev = np.zeros((n_objects, size, size)); per_glimpse = []
            for g, loc in enumerate(locs):
                dr, dc = loc[0] - locs[0][0], loc[1] - locs[0][1]
                for o in range(n_objects):
                    for r in range(size):
                        for c in range(size):
                            rr, cc = r + dr, c + dc
                            ev[o, r, c] += (obs[m][g] @ stored[(o, rr, cc)]) / k_active if 0 <= rr < size and 0 <= cc < size else -1
                per_glimpse.append(ev.reshape(n_objects, -1).max(1).copy())
            evidence.append(per_glimpse)
        bag_ev = np.zeros(n_objects)
        for g in range(max_glimpses):
            results['reference_frame'][g] += int(evidence[0][g].argmax() == target)
            results['voting_3_modules'][g] += int(sum(e[g] for e in evidence).argmax() == target)
            bag_ev += bag @ obs[0][g]
            results['bag_of_features'][g] += int(bag_ev.argmax() == target)
    return {k: (v / trials).round(3).tolist() for k, v in results.items()} | {
        'glimpses': list(range(1, max_glimpses + 1)), 'chance': 1 / n_objects, 'objects': n_objects, 'features': features}


# --- 4. arithmetic as movement in a reference frame -------------------------------------------------------------
def arithmetic(circuit, seed, n=20, train_sizes=(19, 50, 100, 150)):
    rng = np.random.default_rng(seed + 51)
    enc = Encoder(circuit, 'fly_measured', seed)
    pats = (rng.random((n, 64)) < 0.25).astype(float)
    codes = enc.code(pats)
    noisy = lambda i: enc.code(np.clip(pats[i] + (rng.random(64) < 0.05) * (1 - 2 * pats[i]), 0, 1)[None])[0]
    # learning: only successor pairs (a, a+1), in random order, each seen once. Locations by path integration.
    pairs = [(a, a + 1) for a in range(n - 1)]
    location = {}
    place = {}                                            # location -> stored code (one-shot)
    pending = list(rng.permutation(len(pairs)))
    anchor = pairs[pending[0]][0]
    location[anchor] = 0; place[0] = codes[anchor]
    while pending:
        progressed = False
        for k in list(pending):
            a, b = pairs[k]
            if a in location and b not in location: location[b] = location[a] + 1; place[location[b]] = codes[b]
            elif b in location and a not in location: location[a] = location[b] - 1; place[location[a]] = codes[a]
            if a in location and b in location: pending.remove(k); progressed = True
        if not progressed: break

    def recall(loc):
        if loc not in place: return None
        return int((codes @ place[loc]).argmax())

    def where(code):                                      # recognise a symbol, then look up its location
        sym = int((codes @ code).argmax())
        return location.get(sym)

    tests = [(a, b) for a in range(n) for b in range(n) if a + b < n]
    add = np.mean([recall(where(noisy(a)) + b) == a + b for a, b in tests])
    sub = np.mean([recall(where(noisy(a + b)) - b) == a for a, b in tests])
    cmp_pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    compare = np.mean([(where(noisy(a)) > where(noisy(b))) == (a > b) for a, b in cmp_pairs])
    # baseline: backprop MLP on (code a, code b) -> sum, trained on random addition examples, tested on the rest
    curve = []
    for size in train_sizes:
        order = rng.permutation(len(tests)); tr = [tests[i] for i in order[:min(size, len(tests) - 20)]]
        te = [tests[i] for i in order[min(size, len(tests) - 20):]]
        X = lambda ps: np.array([np.concatenate([codes[a], codes[b]]) for a, b in ps])
        mlp = MLPn(X(tr).shape[1], n, seed)
        for _ in range(400): mlp.fit(X(tr), np.array([a + b for a, b in tr]), 0.05)
        curve.append({'train_examples': len(tr), 'held_out_accuracy': float((mlp.predict(X(te)) == np.array([a + b for a, b in te])).mean())})
    return {'symbols': n, 'successor_pairs_seen': len(pairs), 'placed': len(location),
            'addition': float(add), 'subtraction': float(sub), 'comparison': float(compare),
            'mlp_addition_curve': curve, 'tests': len(tests), 'chance': 1 / n}


class MLPn:
    """Backprop network for the arithmetic baseline (inputs -> 256 ReLU -> classes)."""
    def __init__(self, d, classes, seed, hidden=256):
        r = np.random.default_rng(seed + 61)
        self.W1 = r.normal(0, math.sqrt(2 / d), (d, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, math.sqrt(2 / hidden), (hidden, classes)); self.b2 = np.zeros(classes)

    def fit(self, x, y, lr):
        h = np.maximum(x @ self.W1 + self.b1, 0); p = ex.softmax(h @ self.W2 + self.b2)
        p[np.arange(len(y)), y] -= 1; p /= len(y); dh = (p @ self.W2.T) * (h > 0)
        self.W2 -= lr * h.T @ p; self.b2 -= lr * p.sum(0); self.W1 -= lr * x.T @ dh; self.b1 -= lr * dh.sum(0)

    def predict(self, x):
        return (np.maximum(x @ self.W1 + self.b1, 0) @ self.W2 + self.b2).argmax(1)


# --- 5. continual learning (from the sparse-expansion benchmark) ------------------------------------------------
def continual():
    path = BASE / 'data/expansion-benchmark.json'
    if not path.exists(): return None
    s = json.loads(path.read_text())['conditions']['digits']['summary']
    return {k: s[k] for k in ('fly_measured', 'fly_shuffled', 'random_sparse', 'mlp_backprop') if k in s}


def run(task='few_shot', seed=7, circuit=None):
    circuit = circuit or LearningCircuit()
    if task == 'few_shot':
        x, y = ex.load_digits(); return few_shot(circuit, x, y, seed)
    if task == 'sequences': return sequences(circuit, seed)
    if task == 'objects': return objects(circuit, seed)
    if task == 'arithmetic': return arithmetic(circuit, seed)
    if task == 'continual': return continual()
    raise ValueError('task must be few_shot, sequences, objects, arithmetic or continual')


def mean_rows(runs):
    keys = [k for k in runs[0][0] if k != 'shots']
    return [{'shots': r0['shots'], **{k: float(np.mean([r[i][k] for r in runs])) for k in keys},
             **{k + '_sd': float(np.std([r[i][k] for r in runs])) for k in keys}} for i, r0 in enumerate(runs[0])]


def benchmark(seeds=(7, 8, 9), out=None):
    circuit = LearningCircuit()
    x, y = ex.load_digits()
    few = mean_rows([few_shot(circuit, x, y, s) for s in seeds])
    seq_runs = [sequences(circuit, s) for s in seeds]
    seq = {k: (np.mean([r[k] for r in seq_runs], 0).round(3).tolist() if isinstance(v, list) and k not in ('sequences',) and k not in ('exposures', 'betas')
               else float(np.mean([r[k] for r in seq_runs])) if isinstance(v, float) else v) for k, v in seq_runs[0].items()}
    obj_runs = [objects(circuit, s) for s in seeds]
    obj = {k: (np.mean([r[k] for r in obj_runs], 0).round(3).tolist() if k in ('reference_frame', 'voting_3_modules', 'bag_of_features') else v)
           for k, v in obj_runs[0].items()}
    ar_runs = [arithmetic(circuit, s) for s in seeds]
    ar = {k: (float(np.mean([r[k] for r in ar_runs])) if k in ('addition', 'subtraction', 'comparison') else v) for k, v in ar_runs[0].items()}
    ar['mlp_addition_curve'] = [{'train_examples': c['train_examples'],
                                 'held_out_accuracy': float(np.mean([r['mlp_addition_curve'][i]['held_out_accuracy'] for r in ar_runs]))}
                                for i, c in enumerate(ar_runs[0]['mlp_addition_curve'])]
    result = {'seeds': list(seeds), 'wirings': WIRINGS, 'sparsity': SPARSITY, 'beta': BETA, 'few_shot': few,
              'sequences': seq, 'objects': obj, 'arithmetic': ar, 'continual': continual()}
    (out or BASE / 'data/scaffold-benchmark.json').write_text(json.dumps(result, indent=1))
    return result


if __name__ == '__main__':
    import time
    t0 = time.time()
    r = benchmark()
    print('few-shot (accuracy by examples per class):')
    for row in r['few_shot']:
        print(f"  {row['shots']:>3}: measured {row['fly_measured']:.2f} shuffled {row['fly_shuffled']:.2f} random {row['random_sparse']:.2f}"
              f" | nearest {row['pixels_nearest']:.2f} softmax {row['pixels_softmax']:.2f} mlp {row['mlp_backprop']:.2f}"
              f" | shifted: measured {row['fly_measured_shifted']:.2f} mlp {row['mlp_backprop_shifted']:.2f}"
              f" | glimpses plain {row['glimpses_plain']:.2f} assisted {row['glimpses_assisted']:.2f}")
    s = r['sequences']
    print('sequences (last-symbol accuracy by exposures):', {k: v for k, v in s.items() if '|' in k or k.startswith('markov')})
    o = r['objects']
    print('objects by glimpses:', {k: o[k] for k in ('reference_frame', 'voting_3_modules', 'bag_of_features')})
    a = r['arithmetic']
    print('arithmetic:', {k: a[k] for k in ('placed', 'addition', 'subtraction', 'comparison')}, a['mlp_addition_curve'])
    print(f'{time.time() - t0:.0f} s')
