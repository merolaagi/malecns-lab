"""Magnitude comparison ("choose the larger number") on the measured mushroom-body subset.

Question: does the overlap structure of a sparse distributed (scalar) number code let the
measured PN->KC expansion generalize to numbers never seen in training?

Measured: PN->KC edges and synapse counts, KC->MBON edges, PAM->KC synapse totals (gate).
Assumed: the number encoding, the 95th-percentile KC threshold, the pooled value readout,
+1/-1 outcomes (reward for the larger, punishment for the smaller) applied to both options
each trial, and the learning rule.
Flies are not claimed to encode numbers this way. The encoding is an HTM-style scalar encoder
applied to PN cells in a random order, so PN identity carries no biological meaning here.
"""
import hashlib
import json
import math
from itertools import combinations
import numpy as np
from learning import LearningCircuit
from model import BASE

CONDITIONS = ['scalar', 'onehot', 'shuffled', 'no_plasticity']
LABELS = {'scalar': 'Scalar SDR code', 'onehot': 'One-hot code (no overlap)',
          'shuffled': 'Scalar code, shuffled PN→KC', 'no_plasticity': 'Scalar code, no plasticity'}
DEFAULT = dict(seed=7, numbers=9, width=30, step=8, held_out=[4, 5, 6], train_trials=600,
               learning_rate=.2, temperature=.1, noise=.035, test_exemplars=8, condition='scalar')


class Numerosity:
    def __init__(self, circuit=None):
        self.c = circuit or LearningCircuit()

    def validate(self, options):
        p = DEFAULT | options
        if p['condition'] not in CONDITIONS: raise ValueError('Unknown numerosity condition')
        for k, lo, hi in [('seed', 0, 999999), ('numbers', 3, 12), ('width', 5, 60), ('step', 1, 60),
                          ('train_trials', 0, 3000), ('test_exemplars', 1, 32)]:
            v = float(p[k])
            if not math.isfinite(v) or v != int(v) or not lo <= v <= hi: raise ValueError(f'{k} must be an integer in [{lo},{hi}]')
            p[k] = int(v)
        for k, lo, hi in [('learning_rate', 0, 1), ('temperature', .01, 2), ('noise', 0, .3)]:
            p[k] = float(p[k])
            if not math.isfinite(p[k]) or not lo <= p[k] <= hi: raise ValueError(f'Invalid {k}')
        held = p['held_out']
        if not isinstance(held, list) or any(not isinstance(h, int) or not 1 <= h <= p['numbers'] for h in held):
            raise ValueError('held_out must be a list of numbers within range')
        p['held_out'] = sorted(set(held))
        if p['numbers'] - len(p['held_out']) < 2: raise ValueError('At least two numbers must be trained')
        span = p['width'] * p['numbers'] if p['condition'] == 'onehot' else p['step'] * (p['numbers'] - 1) + p['width']
        if span > self.c.np: raise ValueError(f'Encoding needs {span} PNs but only {self.c.np} exist')
        return p

    def codes(self, p):
        """PN index sets per number. Every number activates exactly `width` PNs, so total input
        activity carries no magnitude cue; only which PNs are active differs."""
        order = np.random.default_rng(p['seed'] + 2000).permutation(self.c.np)
        if p['condition'] == 'onehot':
            return [order[i * p['width']:(i + 1) * p['width']] for i in range(p['numbers'])]
        return [order[i * p['step']:i * p['step'] + p['width']] for i in range(p['numbers'])]

    def run(self, **options):
        p = self.validate(options)
        c = self.c
        rng = np.random.default_rng(p['seed'])
        noise_rng = np.random.default_rng(p['seed'] + 3000)
        w = c.encoder('shuffled' if p['condition'] == 'shuffled' else 'intact', p['seed'])
        codes = self.codes(p)
        templates = np.zeros((p['numbers'], c.np))
        for i, idx in enumerate(codes): templates[i, idx] = 1

        def encode(i):
            pn = np.clip(templates[i] + noise_rng.normal(0, p['noise'], c.np), 0, 1)
            a = w.dot(pn)
            a = np.maximum(a - np.quantile(a, .95), 0)
            mass = a[c.kidx] * c.base
            idx = np.flatnonzero(mass > 0)
            return idx, mass[idx] / max(mass.sum(), 1e-12), set(np.flatnonzero(a).tolist())

        trained = [n for n in range(1, p['numbers'] + 1) if n not in p['held_out']]
        pairs = [(a, b) for a, b in combinations(trained, 2)]
        delta = np.zeros(len(c.km))
        presented = set()
        log = []
        for t in range(p['train_trials']):
            a, b = pairs[rng.integers(len(pairs))]
            if rng.random() < .5: a, b = b, a
            fa, fb = encode(a - 1), encode(b - 1)
            va, vb = float(np.dot(delta[fa[0]], fa[1])), float(np.dot(delta[fb[0]], fb[1]))
            pa = 1 / (1 + math.exp(float(np.clip((vb - va) / p['temperature'], -60, 60))))
            chose_a = rng.random() < pa
            chosen, other = (a, b) if chose_a else (b, a)
            # Full-information feedback (assumption): both options are updated toward their own
            # outcome, +1 for the larger and -1 for the smaller. With chosen-only updates, values
            # drift to "outcome given that I chose it", which is +1 for nearly every number once
            # choices are good, and no magnitude ordering forms.
            if p['condition'] != 'no_plasticity':
                for f, v, outcome in [(fa, va, 1. if a > b else -1.), (fb, vb, 1. if b > a else -1.)]:
                    idx, phi = f[0], f[1]
                    gate = c.gate[c.kidx[idx]]
                    denom = float(np.sum(phi * phi * gate))
                    if denom > 0: delta[idx] = np.clip(delta[idx] + p['learning_rate'] * (outcome - v) * phi * gate / denom, -3, 3)
            presented.update([a, b])
            if t % 10 == 0 or t == p['train_trials'] - 1:
                log.append({'trial': t + 1, 'p_larger': pa if a > b else 1 - pa, 'correct': bool(chosen > other)})
        weights_after_training = hashlib.sha256(delta.tobytes()).hexdigest()

        # Test: every pair, fresh noisy exemplars, no learning.
        ex = [[encode(i) for _ in range(p['test_exemplars'])] for i in range(p['numbers'])]
        vals = np.array([[float(np.dot(delta[e[0]], e[1])) for e in row] for row in ex])
        N = p['numbers']
        matrix = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j: continue
                d = (vals[j][None, :] - vals[i][:, None]) / p['temperature']
                pi = 1 / (1 + np.exp(np.clip(d, -60, 60)))          # P(choose i over j)
                matrix[i][j] = float(pi.mean() if i > j else (1 - pi).mean())  # P(choose the larger)
        held = set(p['held_out'])

        def cat(i, j):
            k = (i + 1 in held) + (j + 1 in held)
            return ['trained_pairs', 'one_novel', 'both_novel'][k]
        groups = {'trained_pairs': [], 'one_novel': [], 'both_novel': []}
        by_distance = {}
        for i in range(N):
            for j in range(i + 1, N):
                groups[cat(i, j)].append(matrix[i][j])
                if cat(i, j) == 'trained_pairs': by_distance.setdefault(j - i, []).append(matrix[i][j])
        jac = [[len(ex[i][0][2] & ex[j][0][2]) / max(1, len(ex[i][0][2] | ex[j][0][2])) for j in range(N)] for i in range(N)]
        pn_overlap = [[len(set(codes[i]) & set(codes[j])) / p['width'] for j in range(N)] for i in range(N)]
        weights_after_test = hashlib.sha256(delta.tobytes()).hexdigest()
        return {
            'parameters': p, 'label': LABELS[p['condition']],
            'trained_numbers': trained, 'presented_in_training': sorted(presented),
            'values': [{'number': n + 1, 'mean': float(vals[n].mean()), 'sd': float(vals[n].std()),
                        'held_out': n + 1 in held} for n in range(N)],
            'p_larger': matrix,
            'accuracy': {k: (float(np.mean(v)) if v else None) for k, v in groups.items()},
            'pairs': {k: len(v) for k, v in groups.items()},
            'distance_curve': [{'distance': d, 'p_larger': float(np.mean(v))} for d, v in sorted(by_distance.items())],
            'kc_jaccard': jac, 'pn_overlap': pn_overlap,
            'training_log': log,
            'weights': {'after_training_sha256': weights_after_training, 'after_test_sha256': weights_after_test,
                        'nonzero_edges': int(np.count_nonzero(delta))},
            'interpretation': 'Model experiment, not evidence that flies count. Chance is 0.5. Generalization to held-out '
                              'numbers can only come from overlap between their codes and trained codes; the one-hot '
                              'control removes that overlap while keeping the same number of active PNs.'}

    DESIGNS = {'interpolate': [2, 5, 8], 'middle_block': [4, 5, 6], 'extrapolate': [8, 9]}

    def suite(self, seed=7):
        rows = []
        for design, held in self.DESIGNS.items():
            for s in range(seed, seed + 5):
                for cond in CONDITIONS:
                    r = self.run(seed=s, condition=cond, held_out=held)
                    rows.append({'design': design, 'held_out': held, 'seed': s, 'condition': cond,
                                 'accuracy': r['accuracy'], 'values': [v['mean'] for v in r['values']]})
        summary = {}
        for design in self.DESIGNS:
            summary[design] = {}
            for cond in CONDITIONS:
                sel = [r for r in rows if r['condition'] == cond and r['design'] == design]
                summary[design][cond] = {k: {'mean': float(np.mean([r['accuracy'][k] for r in sel])),
                                             'sd': float(np.std([r['accuracy'][k] for r in sel]))} for k in sel[0]['accuracy']}
        return {'defaults': DEFAULT, 'labels': LABELS, 'designs': self.DESIGNS, 'runs': rows, 'summary': summary, 'seeds': 5,
                'interpretation': 'Five seeds per condition and held-out design at default settings. Seeds change the PN order '
                                  'used by the encoder, stimulus noise and choice sampling. Model runs, not biological replicates. '
                                  '"one_novel" pairs can be solved without any generalization when held-out numbers sit in the '
                                  'middle of the range, because untrained values stay near 0, the midpoint of the +1/-1 outcomes. '
                                  '"both_novel" pairs are the clean generalization test.'}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--benchmark', action='store_true')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--condition', default='scalar', choices=CONDITIONS)
    a = ap.parse_args()
    exp = Numerosity()
    if a.benchmark:
        (BASE / 'data/numerosity-benchmark.json').write_text(json.dumps(exp.suite(a.seed), indent=1))
        print('Wrote data/numerosity-benchmark.json')
    else:
        r = exp.run(seed=a.seed, condition=a.condition)
        print(json.dumps({'accuracy': r['accuracy'], 'values': [round(v['mean'], 3) for v in r['values']]}, indent=1))
