"""Frame discovery: the scaffold works out its own reference frame from a few relations.

The Scaffold learner did arithmetic from 19 successor pairs, but its frame (a line) was given by hand. Here the
frame is computed from the data. Input: triples (a, relation, b) between symbols, nothing else. Method:

1. Treat every relation type as one kind of movement. Walk the relations (path integration) and give each symbol
   the count of movements of each type that reaches it: a point in Z^R for R relation types.
2. Every relation that closes a loop says "this combination of movements brings you back": a constraint.
3. The frame is Z^R with those combinations set to zero. Diagonalising the constraints (Smith normal form) gives
   its shape: each free direction is an axis, each direction that repeats every n steps is a loop of n, and
   directions that turn out to be combinations of others disappear.

So a line, a loop, a grid or a cylinder is an output, not a choice from a list. Then held-out relations and
multi-step questions are answered by moving in the frame and reading out what is stored there.

Honest limits, measured below: it assumes movements commute (going east then north equals north then east), so a
tree has no such frame (reported as "no consistent frame", detected by two symbols landing on one location); and
one wrong relation corrupts the frame, because every loop is trusted.

Symbols are sparse codes from the measured PN -> Kenyon-cell expansion, recognised through input noise, as in
scaffold.py. Baselines get the same triples: a lookup table that chains known relations, and a backprop network.
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

import scaffold as sc
from learning import LearningCircuit

BASE = Path(__file__).resolve().parent


# --- integer algebra --------------------------------------------------------------------------------------------
def diagonalise(C, R):
    """Smith-style diagonalisation of the constraint rows C (m x R integers).
    Returns (d, Q): Q is an R x R integer matrix with determinant +-1 and, in the coordinates w = v @ Q, the
    constraints say d[i] * w[i] = 0. d[i] = 0 is a free axis, 1 a direction that vanishes, n > 1 a loop of n."""
    A = [[int(x) for x in row] for row in C] or []
    Q = [[int(i == j) for j in range(R)] for i in range(R)]
    m = len(A)

    def colop(j, k, f):                                   # column j -= f * column k
        for row in A: row[j] -= f * row[k]
        for row in Q: row[j] -= f * row[k]

    def colswap(j, k):
        for row in A: row[j], row[k] = row[k], row[j]
        for row in Q: row[j], row[k] = row[k], row[j]

    d = []
    r = 0
    for t in range(R):
        if r >= m: d.append(0); continue
        while True:
            entries = [(abs(A[i][j]), i, j) for i in range(r, m) for j in range(t, R) if A[i][j]]
            if not entries: break
            _, i, j = min(entries)
            A[r], A[i] = A[i], A[r]; colswap(t, j)
            p = A[r][t]; clean = True
            for i in range(r + 1, m):
                f = A[i][t] // p
                if f: A[i] = [x - f * y for x, y in zip(A[i], A[r])]
                if A[i][t]: clean = False
            for j in range(t + 1, R):
                f = A[r][j] // p
                if f: colop(j, t, f)
                if A[r][j]: clean = False
            if clean: break
        if r < m and A[r][t]: d.append(abs(A[r][t])); r += 1
        else: d.append(0)
    return d, Q


class Frame:
    """A discovered reference frame: where each symbol sits, and how each relation moves."""
    def __init__(self, triples, n_symbols):
        self.relations = sorted({r for _, r, _ in triples})
        self.index = {r: i for i, r in enumerate(self.relations)}
        R = len(self.relations)
        nbrs = {s: [] for s in range(n_symbols)}
        for a, r, b in triples:
            nbrs[a].append((b, self.index[r], 1)); nbrs[b].append((a, self.index[r], -1))
        raw, comp = {}, {}
        self.components = 0
        for start in range(n_symbols):
            if start in raw or not nbrs[start]: continue
            raw[start] = (0,) * R; comp[start] = self.components; queue = deque([start])
            while queue:
                a = queue.popleft()
                for b, i, sign in nbrs[a]:
                    if b not in raw:
                        v = list(raw[a]); v[i] += sign; raw[b] = tuple(v); comp[b] = self.components; queue.append(b)
            self.components += 1
        constraints = set()
        for a, r, b in triples:
            c = list(raw[a]); c[self.index[r]] += 1
            c = tuple(x - y for x, y in zip(c, raw[b]))
            if any(c): constraints.add(c)
        self.constraints = len(constraints)
        self.d, self.Q = diagonalise(sorted(constraints), R)
        self.keep = [i for i, x in enumerate(self.d) if x != 1]
        self.raw, self.comp = raw, comp
        self.loc = {s: (comp[s],) + self.coord(v) for s, v in raw.items()}
        self.at = {}
        self.collisions = 0
        for s, l in self.loc.items():
            if l in self.at: self.collisions += 1
            else: self.at[l] = s
        self.placed = len(raw)

    def coord(self, v):
        w = [sum(v[k] * self.Q[k][i] for k in range(len(v))) for i in self.keep]
        return tuple(x % self.d[i] if self.d[i] else x for x, i in zip(w, self.keep))

    def move(self, a, steps):
        """Start at symbol a, make steps = {relation: count}; return the symbol stored there, or None."""
        if a not in self.raw: return None
        v = list(self.raw[a])
        for r, k in steps.items():
            if r not in self.index: return None
            v[self.index[r]] += k
        return self.at.get((self.comp[a],) + self.coord(v))

    @property
    def axes(self): return sum(1 for i in self.keep if self.d[i] == 0)

    @property
    def loops(self): return sorted(self.d[i] for i in self.keep if self.d[i] > 1)

    @property
    def shape(self):
        if self.collisions: return 'no consistent frame'
        if self.components > 1: return 'pieces'
        a, l = self.axes, self.loops
        if a == 0 and not l: return 'point'
        name = {(1, 0): 'line', (2, 0): 'grid', (0, 1): 'loop', (1, 1): 'cylinder', (0, 2): 'torus', (3, 0): '3-D grid'}.get((a, len(l)), f'{a} axes, {len(l)} loops')
        return name + (' of ' + ' x '.join(map(str, l)) if l else '')

    def describe(self):
        return dict(shape=self.shape, axes=self.axes, loops=self.loops, relation_types=len(self.relations), constraints=self.constraints,
                    placed=self.placed, components=self.components, collisions=self.collisions)


# --- worlds -----------------------------------------------------------------------------------------------------
def world(name):
    """(symbols, all true triples, expected shape, truth function for moves, train fraction)."""
    if name == 'numbers':
        n = 20; T = [(a, 'next', a + 1) for a in range(n - 1)]
        return n, T, 'line', lambda a, s: a + s.get('next', 0) if 0 <= a + s.get('next', 0) < n else None, 1.0
    if name == 'clock':
        n = 12; T = [(a, 'next', (a + 1) % n) for a in range(n)]
        return n, T, 'loop of 12', lambda a, s: (a + s.get('next', 0)) % n, 1.0
    if name == 'room':
        w = 5; n = w * w
        T = [(r * w + c, 'east', r * w + c + 1) for r in range(w) for c in range(w - 1)] + [(r * w + c, 'north', (r + 1) * w + c) for r in range(w - 1) for c in range(w)]
        def truth(a, s):
            r, c = divmod(a, w); r += s.get('north', 0); c += s.get('east', 0)
            return r * w + c if 0 <= r < w and 0 <= c < w else None
        return n, T, 'grid', truth, 0.75
    if name == 'numbers_two_relations':
        n = 20; T = [(a, 'next', a + 1) for a in range(n - 1)] + [(a, 'five_on', a + 5) for a in range(n - 5)]
        def truth(a, s):
            b = a + s.get('next', 0) + 5 * s.get('five_on', 0)
            return b if 0 <= b < n else None
        return n, T, 'line', truth, 0.75
    if name == 'week_by_shift':
        days, shifts = 7, 4; n = days * shifts          # a rota: day of week (wraps) x shift of the day (does not)
        T = [(s * days + d, 'tomorrow', s * days + (d + 1) % days) for s in range(shifts) for d in range(days)] + \
            [(s * days + d, 'later', (s + 1) * days + d) for s in range(shifts - 1) for d in range(days)]
        def truth(a, st):
            s, d = divmod(a, days); s += st.get('later', 0); d = (d + st.get('tomorrow', 0)) % days
            return s * days + d if 0 <= s < shifts else None
        return n, T, 'cylinder of 7', truth, 0.8
    if name == 'family_tree':
        n = 15; T = [(a, 'left', 2 * a + 1) for a in range(7)] + [(a, 'right', 2 * a + 2) for a in range(7)]
        def truth(a, s):                                 # only single steps are well defined in a tree
            if s == {'left': 1}: return 2 * a + 1 if a < 7 else None
            if s == {'right': 1}: return 2 * a + 2 if a < 7 else None
            return None
        return n, T, 'no consistent frame', truth, 1.0
    raise ValueError('unknown world')


WORLDS = ['numbers', 'clock', 'room', 'numbers_two_relations', 'week_by_shift', 'family_tree']
LABELS = {'numbers': 'Numbers 0–19, "next"', 'clock': 'Clock hours, "next" (wraps)', 'room': '5 × 5 room, "east" and "north"',
          'numbers_two_relations': 'Numbers, "next" and "five on"', 'week_by_shift': 'Rota: 7 days (wrap) × 4 shifts',
          'family_tree': 'Family tree, "left child" and "right child"'}


def questions(name, n, relations, truth, rng, count=300):
    """Multi-step questions: start anywhere, make a few movements of each type; keep those with a defined answer."""
    out = []
    span = 26 if name == 'clock' else 6
    tries = 0
    while len(out) < count and tries < count * 40:
        tries += 1
        a = int(rng.integers(n))
        steps = {r: int(rng.integers(-span, span + 1)) for r in relations}
        steps = {r: k for r, k in steps.items() if k}
        if sum(abs(k) for k in steps.values()) < 2: continue
        b = truth(a, steps)
        if b is not None: out.append((a, steps, b))
    return out


# --- baselines --------------------------------------------------------------------------------------------------
class Chain:
    """Lookup table of the relations seen, followed link by link (in any order of relation types)."""
    def __init__(self, triples):
        self.f = {(a, r): b for a, r, b in triples}; self.b = {(b, r): a for a, r, b in triples}

    def move(self, a, steps):
        self.seen = set()
        return self._go(a, tuple(sorted((r, k) for r, k in steps.items() if k)))

    def _go(self, a, todo):
        if not todo: return a
        if (a, todo) in self.seen: return None            # already failed from here
        self.seen.add((a, todo))
        for i, (r, k) in enumerate(todo):                 # try any remaining movement that has a known link
            nxt = (self.f if k > 0 else self.b).get((a, r))
            if nxt is None: continue
            left = k - (1 if k > 0 else -1)
            out = self._go(nxt, todo[:i] + (((r, left),) if left else ()) + todo[i + 1:])
            if out is not None: return out
        return None


class Net:
    """Backprop network: (code of a, relation, direction) -> b, trained on the same triples in both directions."""
    def __init__(self, triples, codes, relations, n, seed, steps=500):
        self.codes, self.rel = codes, {r: i for i, r in enumerate(relations)}
        X, Y = [], []
        for a, r, b in triples:
            X.append(self.x(a, r, 1)); Y.append(b); X.append(self.x(b, r, -1)); Y.append(a)
        X, Y = np.array(X), np.array(Y)
        self.net = sc.MLPn(X.shape[1], n, seed)
        for _ in range(steps): self.net.fit(X, Y, 0.05)

    def x(self, a, r, sign):
        one = np.zeros(2 * len(self.rel)); one[2 * self.rel[r] + (sign < 0)] = 1
        return np.concatenate([self.codes[a], one * 20])

    def move(self, a, steps):
        for r, k in steps.items():
            for _ in range(abs(k)): a = int(self.net.predict(self.x(a, r, 1 if k > 0 else -1)[None])[0])
        return a


# --- experiment -------------------------------------------------------------------------------------------------
def trial(circuit, name, seed, fraction=None, corrupt=0, net=True):
    n, T, expected, truth, default = world(name)
    rng = np.random.default_rng(seed * 1000 + sum(map(ord, name)))
    fraction = default if fraction is None else fraction
    order = rng.permutation(len(T)); k = max(1, round(fraction * len(T)))
    train = [T[i] for i in order[:k]]; held = [T[i] for i in order[k:]]
    if corrupt:
        for i in rng.choice(len(train), corrupt, replace=False):
            a, r, b = train[i]; train[i] = (a, r, int(rng.choice([s for s in range(n) if s not in (a, b)])))
    enc = sc.Encoder(circuit, 'fly_measured', seed)
    pats = (rng.random((n, 64)) < 0.25).astype(float)
    codes = enc.code(pats)

    def recognise(s):                                     # a noisy view of the symbol, identified by its sparse code
        p = pats[s].copy(); flip = rng.random(64) < 0.05; p[flip] = 1 - p[flip]
        return int((codes @ enc.code(p[None])[0]).argmax())

    frame = Frame(train, n)
    relations = sorted({r for _, r, _ in T})
    multi = questions(name, n, relations, truth, rng)
    single = [(a, {r: 1}, b) for a, r, b in held]
    learners = {'frame': frame, 'chain_lookup': Chain(train)}
    if net: learners['backprop_net'] = Net(train, codes, relations, n, seed)
    out = dict(world=name, expected=expected, relations_given=len(train), relations_total=len(T), symbols=n,
               correct_shape=frame.shape == expected, **frame.describe())
    for key, model in learners.items():
        score = lambda qs: float(np.mean([model.move(recognise(a), s) == b for a, s, b in qs])) if qs else None
        out[key] = dict(held_out_relations=score(single), multi_step=score(multi))
    return out


def summarise(runs):
    first = runs[0]
    out = {k: first[k] for k in ('world', 'expected', 'relations_given', 'relations_total', 'symbols')}
    out['label'] = LABELS[first['world']]
    out['shapes_found'] = sorted({r['shape'] for r in runs})
    out['correct_shape'] = float(np.mean([r['correct_shape'] for r in runs]))
    for key in ('frame', 'chain_lookup', 'backprop_net'):
        if key not in first: continue
        out[key] = {m: (float(np.mean([r[key][m] for r in runs])) if first[key][m] is not None else None) for m in first[key]}
    return out


def benchmark(seeds=(7, 8, 9), out=None):
    circuit = LearningCircuit()
    worlds = [summarise([trial(circuit, w, s) for s in seeds]) for w in WORLDS]
    # how few relations are enough? (the worlds with spare relations)
    sweep = {}
    for w in ('room', 'numbers_two_relations', 'week_by_shift'):
        rows = []
        for f in (0.55, 0.65, 0.75, 0.85, 1.0):
            rows.append(summarise([trial(circuit, w, s, fraction=f, net=False) for s in seeds + tuple(x + 10 for x in seeds)]))
        sweep[w] = rows
    noise = [summarise([trial(circuit, w, s, corrupt=1, net=False) for s in seeds + tuple(x + 10 for x in seeds)]) for w in ('numbers', 'room')]
    result = dict(seeds=list(seeds), worlds=worlds, sweep=sweep, one_wrong_relation=noise,
                  assumptions=['every relation type is one kind of movement', 'movements commute', 'every relation given is trusted',
                               'symbols are random sparse patterns through the measured PN→KC expansion, recognised through 5% input noise'])
    (out or BASE / 'data/frames-benchmark.json').write_text(json.dumps(result, indent=1))
    return result


if __name__ == '__main__':
    r = benchmark()
    for w in r['worlds']:
        print(f"{w['label']:<44} given {w['relations_given']:>2}/{w['relations_total']:<2} found {', '.join(w['shapes_found']):<22} "
              f"frame {w['frame']}  chain {w['chain_lookup']}  net {w.get('backprop_net')}", file=sys.stderr)
