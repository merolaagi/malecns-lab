"""Seeing an object, remembering it, and recognising a slightly different one.

The path, stage by stage:

  scene       procedurally rendered household objects (paper cup, mug, bottle, wine glass, can), each with
              lighting (level and direction), distance, rotation, gaze, and instance-to-instance shape changes
  eye         721 hexagonal columns at 5 degrees, the layout of the lab's connectome-constrained eye model;
              each column averages the scene over its acceptance angle and adds photoreceptor noise
  lamina      contrast adaptation, as the first visual layer (L1/L2) does: each column's signal is divided by
              the local and global mean, then split into ON, OFF and three edge-orientation channels
  VPN         visual projection neurons from the optic lobe, each pooling one channel over its receptive field
  Kenyon      the measured VPN-to-Kenyon-cell wiring onto visual Kenyon cells (KCg-d, KCab-p), APL keeping the
              code sparse (top-k)
  memory      the lab's compartment memory model (memory.py): dopamine depresses active Kenyon-cell synapses onto
              output neurons in the compartments it reaches, and the output balance is read as approach

Measured: Kenyon cells, output neurons, dopamine neurons, their compartments and every edge among them
(data/mushroom-body.json); the visual inputs onto Kenyon cells and their synapse counts, when
data/visual-memory.json has been built (build_visualmemory.py). Until then a synthetic stand-in wires random
visual inputs onto the measured visual Kenyon cells, and every result says so.

Assumed: the objects and their rendering, the lamina model, each VPN's feature channel and receptive-field size
(columnar cells with annotated optic-lobe coordinates get a measured position; others a random one), the VPN
nonlinearity, the sparseness, and everything memory.py assumes. Fly vision is monochrome here; real flies also
see UV, blue and green.

This is not tbp.monty. Monty learns object models as features at locations in the object's own reference frame.
Nothing here builds an object-centred frame; glimpses add evidence, they do not assemble a model of the object.
"""
import json
import math
import zlib

import numpy as np

import memory
from model import BASE
from vision_eye import hex_xy

DEG_PER_UNIT = 5.0 / math.sqrt(3)            # neighbouring columns are sqrt(3) lattice units apart: 5 degrees
CHANNELS = ['ON', 'OFF', 'edge 0°', 'edge 60°', 'edge 120°']
VISUAL_KC_TYPES = ('KCg-d', 'KCab-p')
H = np.linspace(0, 1, 65)

# Each class: height (cm), radius profile r(h) over normalised height (cm), surface albedo, how see-through it is,
# how shiny, and handle size (mugs only).
CLASSES = {
    'paper cup':  dict(height=9.0,  radius=2.5 + 1.5 * H, albedo=0.85, glass=0.0, specular=0.05, handle=0.0),
    'mug':        dict(height=9.5,  radius=np.full_like(H, 4.0), albedo=0.55, glass=0.0, specular=0.25, handle=1.0),
    'bottle':     dict(height=24.0, radius=np.interp(H, [0, 0.58, 0.78, 1], [3.5, 3.5, 1.2, 1.2]), albedo=0.45, glass=0.0, specular=0.6, handle=0.0),
    'wine glass': dict(height=18.0, radius=np.interp(H, [0, 0.03, 0.04, 0.45, 0.55, 0.75, 1], [3.4, 3.4, 0.4, 0.4, 3.0, 4.0, 3.6]), albedo=0.9, glass=0.75, specular=0.9, handle=0.0),
    'can':        dict(height=12.0, radius=np.interp(H, [0, 0.04, 0.94, 1], [2.9, 3.3, 3.3, 2.7]), albedo=0.7, glass=0.0, specular=0.8, handle=0.0),
}

# Training views come from a familiar range; each test set moves one factor outside it.
TRAIN = dict(light=(0.8, 1.25), light_az=(-25, 25), distance=(10, 14), shape=0.08)   # a fly walking up to it
TEST_SETS = {
    'familiar':     {},
    'dim light':    dict(light=(0.3, 0.5)),
    'bright light': dict(light=(1.6, 2.2)),
    'side light':   dict(light_az=(50, 80)),
    'nearer or farther': dict(distance=(6.5, 8.5, 17, 22)),
    'new kinds':    dict(shape=0.22),
}
CONDITIONS = {
    'intact': 'Visual inputs as built, with the adapting eye',
    'raw_pixels': 'Same inputs, no lamina adaptation: contrast against a fixed reference',
    'shuffled': 'Visual-input destinations permuted among Kenyon cells (source weights and in-degree kept)',
    'random': 'Each visual edge given a random source and a random visual Kenyon cell (weights kept)',
    'olfactory_kcs': 'Same wiring pattern moved onto olfactory Kenyon cells of the same lobe class',
    'assumed_features': 'Measured wiring, but upstream measures ignored: every input fully visual, hashed feature, random field',
}
PROTOCOLS = {
    'absolute': 'Only the trained object, paired with reward',
    'differential': 'The trained object paired with reward, alternating with a contrast object paired with punishment',
}
DEFAULT = dict(seed=7, condition='intact', trained='paper cup', sparsity=0.10, train_views=12, exemplars=8, glimpses=8,
               morph_to='bottle', noise=0.01, protocol='absolute', contrast='mug')


# --- scene and eye ------------------------------------------------------------------------------------------------
class Eye:
    def __init__(self):
        x, y = hex_xy()
        self.xy = np.stack([x, y], 1) * DEG_PER_UNIT                     # column centres in degrees
        n = len(x)
        # 19 sample points per column inside a ~4.5 degree acceptance angle, Gaussian weighted
        ring = [(0, 0)] + [(1.3 * math.cos(a), 1.3 * math.sin(a)) for a in np.arange(6) * math.pi / 3] \
            + [(2.6 * math.cos(a), 2.6 * math.sin(a)) for a in np.arange(12) * math.pi / 6]
        self.offsets = np.array(ring)
        self.weights = np.exp(-np.sum(self.offsets ** 2, 1) / (2 * 1.9 ** 2)); self.weights /= self.weights.sum()
        d = np.hypot(*(self.xy[:, None, :] - self.xy[None, :, :]).transpose(2, 0, 1))
        local = np.exp(-d ** 2 / (2 * (3 * 5.0) ** 2)); self.local = local / local.sum(1, keepdims=True)
        # neighbours along the three hex axes, for edge channels
        self.axes = []
        for angle in (0, 60, 120):
            step = 5.0 * np.array([math.cos(math.radians(angle + 30)), math.sin(math.radians(angle + 30))])
            fwd = self._nearest(self.xy + step); back = self._nearest(self.xy - step)
            self.axes.append((fwd, back))
        self.n = n

    def _nearest(self, points):
        d = np.hypot(*(points[:, None, :] - self.xy[None, :, :]).transpose(2, 0, 1))
        idx = d.argmin(1)
        return np.where(d[np.arange(len(points)), idx] < 2.0, idx, np.arange(len(points)))

    def photoreceptors(self, spec, view, rng=None, noise=0.01):
        pts = self.xy[:, None, :] + self.offsets[None, :, :]
        lum = luminance(spec, view, pts[..., 0], pts[..., 1])
        signal = lum @ self.weights
        if rng is not None and noise:
            signal = signal + rng.normal(0, noise, signal.shape) * np.sqrt(np.maximum(signal, 0) + 0.02)
        return np.maximum(signal, 0)

    def lamina(self, signal, adapt=True):
        """ON, OFF and edge channels. Adapting: contrast against the local and global mean. Not adapting:
        contrast against a fixed reference, the background luminance under the familiar light."""
        if adapt:
            mean = 0.5 * (self.local @ signal) + 0.5 * signal.mean()
            c = (signal - mean) / (mean + 0.02)
        else:
            c = (signal - 0.35) / 0.35
        channels = [np.maximum(c, 0), np.maximum(-c, 0)]
        for fwd, back in self.axes: channels.append(np.abs(c[fwd] - c[back]) / 2)
        return np.stack(channels)


def spec(name, rng=None, jitter=0.0, morph_to=None, morph=0.0):
    """An object instance: class geometry, optionally morphed toward another class, with shape jitter."""
    a = CLASSES[name]
    s = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in a.items()}
    if morph_to and morph:
        b = CLASSES[morph_to]
        for k in s: s[k] = (1 - morph) * a[k] + morph * b[k]
    if rng is not None and jitter:
        s['height'] *= 1 + rng.uniform(-jitter, jitter)
        s['radius'] = s['radius'] * (1 + rng.uniform(-jitter, jitter)) * (1 + rng.uniform(-jitter, jitter) * (H - 0.5))
        s['albedo'] = float(np.clip(s['albedo'] * (1 + rng.uniform(-jitter, jitter) * 1.5), 0.05, 1))
    return s


def luminance(s, v, x, y):
    """Luminance at visual-field points (degrees) for object s seen under view v."""
    d = v['distance']; scale = d / 57.2958
    X = (x - v['gaze'][0]) * scale
    Y = (y - v['gaze'][1]) * scale + s['height'] / 2
    h = Y / s['height']
    r = np.interp(np.clip(h, 0, 1), H, s['radius'])
    inside = (h >= 0) & (h <= 1) & (np.abs(X) < r)
    az = math.radians(v['light_az'])
    background = 0.35 * v['light'] * (1 + 0.15 * math.sin(az) * x / 60)
    u = np.clip(X / np.maximum(r, 1e-6), -1, 1)
    ndotl = u * math.sin(az) + np.sqrt(1 - u ** 2) * math.cos(az)
    half = np.array([math.sin(az / 2), math.cos(az / 2)])
    ndoth = np.maximum(u * half[0] + np.sqrt(1 - u ** 2) * half[1], 0)
    body = v['light'] * (s['albedo'] * (0.25 + 0.75 * np.maximum(ndotl, 0)) + s['specular'] * ndoth ** 20)
    body = s['glass'] * background + (1 - s['glass']) * body
    out = np.where(inside, body, background)
    if s['handle'] > 0.05:
        side = math.sin(math.radians(v['rotation']))
        if abs(side) > 0.25:
            mid = float(np.interp(0.55, H, s['radius']))
            hx = math.copysign(mid + 1.4 * s['handle'], side)
            dx = (X - hx) / max(abs(side), 0.3); dy = Y - 0.55 * s['height']
            rr = np.hypot(dx, dy)
            ring = (rr < 2.2 * s['handle']) & (rr > 1.2 * s['handle']) & ~inside
            out = np.where(ring, v['light'] * s['albedo'] * (0.25 + 0.4 * max(math.cos(az), 0)), out)
    return out


def draw(rng, ranges):
    r = TRAIN | ranges
    def pick(span):
        if len(span) == 4: span = span[:2] if rng.random() < 0.5 else span[2:]
        return float(rng.uniform(*span))
    return dict(light=pick(r['light']), light_az=pick(r['light_az']) * (1 if rng.random() < 0.5 else -1),
                distance=pick(r['distance']), rotation=float(rng.uniform(0, 360)),
                gaze=(float(rng.uniform(-6, 6)), float(rng.uniform(-6, 6)))), r['shape']


def glimpse(view, rng):
    """Another look at the same object: a saccade re-fixates it at a new spot near the centre of view, and the
    fly's own movement changes the distance a little. Lighting and the object stay the same."""
    v = dict(view)
    v['gaze'] = (float(rng.uniform(-6, 6)), float(rng.uniform(-6, 6)))   # re-fixation lands somewhere new
    v['distance'] = view['distance'] * float(rng.uniform(0.95, 1.05))
    return v


# --- visual projection neurons --------------------------------------------------------------------------------------
def stable(text, salt=''):
    return zlib.crc32(f'{salt}{text}'.encode())


MIX_TO_CHANNELS = {'ON': [1, 0, 0, 0, 0], 'OFF': [0, 1, 0, 0, 0], 'luminance': [0.5, 0.5, 0, 0, 0],
                   'form': [0, 0, 1 / 3, 1 / 3, 1 / 3]}


class Projection:
    """Each visual input pools the lamina channels over a Gaussian receptive field, then rectifies and saturates.

    Measured, when build_visualmemory.py has added upstream measures:
      visual_share  scales the image-driven response; inputs that carry no optic-lobe signal stay silent
      channel_mix   how the cell weighs ON, OFF, luminance and form channels
      rf            receptive-field centre, size and eye, from the columnar cells one or two hops up
    Assumed otherwise (or with use_measured=False): one channel per type chosen by a hash of the type name, a random
    centre, a 10-30 degree field, full visual drive. Left-eye cells see the mirror image."""

    def __init__(self, nodes, eye, seed=0, use_measured=True):
        self.eye = eye
        V = len(nodes)
        rng = np.random.default_rng(seed)
        self.weights = np.zeros((V, len(CHANNELS))); self.gain = np.ones(V)
        centres = np.zeros((V, 2)); sigma = np.zeros(V); self.left = np.zeros(V, bool)
        extent = np.abs(eye.xy).max(0)
        self.measured_positions = self.measured_features = 0
        for i, n in enumerate(nodes):
            mix = n.get('channel_mix') if use_measured else None
            if mix:
                self.weights[i] = sum(np.array(MIX_TO_CHANNELS[k]) * float(v) for k, v in mix.items() if k in MIX_TO_CHANNELS)
                self.measured_features += 1
            else:
                self.weights[i, stable(n.get('type'), 'ch') % len(CHANNELS)] = 1.0
            if use_measured and 'visual_share' in n: self.gain[i] = float(n['visual_share'])
            rf = n.get('rf') if use_measured else None
            side = (rf or {}).get('side') or n.get('somaSide')
            self.left[i] = str(side) == 'L'
            if rf:
                centres[i] = np.array([rf['x'], rf['y']]) * extent * 0.9
                sigma[i] = math.hypot(float(rf['spread']) * float(extent.mean()), 5.0)
                self.measured_positions += 1
            elif n.get('hex'):
                h = n['hex']; px = np.array([h[0] - 0.5 * h[1], h[1] * math.sqrt(3) / 2])
                centres[i] = np.clip(px / 30, -1, 1) * extent * 0.9; sigma[i] = 6.0; self.measured_positions += 1
            else:
                centres[i] = eye.xy[rng.integers(eye.n)]
                sigma[i] = 5.0 * (2 + stable(n.get('type'), 'rf') % 5)   # 10 to 30 degrees, shared within a type
        d2 = np.sum((centres[:, None, :] - eye.xy[None, :, :]) ** 2, 2)
        rf = np.exp(-d2 / (2 * sigma[:, None] ** 2)); self.rf = rf / rf.sum(1, keepdims=True)
        self.mirror = eye._nearest(eye.xy * np.array([-1, 1]))
        self.mean_share = float(self.gain.mean())

    def respond(self, channels):
        out = np.empty(len(self.gain))
        for side, chans in ((False, channels), (True, channels[:, self.mirror])):
            m = self.left == side
            pooled = self.rf[m] @ chans.T                         # cells x channels
            out[m] = np.sum(pooled * self.weights[m], 1)
        return self.gain * np.tanh(np.maximum(out - 0.02, 0) / 0.25)


# --- circuit --------------------------------------------------------------------------------------------------------
def visual_inputs(mb_data):
    path = BASE / 'data/visual-memory.json'
    if path.exists():
        d = json.loads(path.read_text())
        return d['nodes'], d['edges'], 'measured'
    return (*synthetic_inputs(mb_data), 'synthetic')


def synthetic_inputs(mb_data, seed=0, cells=120, types=12, inputs_per_kc=6):
    """Stand-in until build_visualmemory.py has run: random visual inputs onto the measured visual Kenyon cells."""
    rng = np.random.default_rng(seed)
    nodes = [{'bodyId': 9_000_000 + i, 'type': f'synthVPN{i % types:02d}', 'somaSide': 'L' if rng.random() < 0.5 else 'R',
              'hex': None, 'role': 'VPN'} for i in range(cells)]
    kcs = [n['bodyId'] for n in mb_data['nodes'] if n['role'] == 'KC' and str(n['type']).startswith(VISUAL_KC_TYPES)]
    edges = []
    for k in kcs:
        for i in rng.choice(cells, inputs_per_kc, replace=False): edges.append([nodes[int(i)]['bodyId'], k, int(rng.integers(3, 16))])
    return nodes, edges


def rewire(edges, condition, mb_nodes, rng):
    if condition in ('intact', 'raw_pixels', 'assumed_features'): return edges
    if condition == 'shuffled':
        targets = rng.permutation([e[1] for e in edges])
        return [[a, int(t), w] for (a, _, w), t in zip(edges, targets)]
    if condition == 'random':
        sources = sorted({e[0] for e in edges}); targets = sorted({e[1] for e in edges})
        return [[int(rng.choice(sources)), int(rng.choice(targets)), w] for _, _, w in edges]
    if condition == 'olfactory_kcs':
        kind = {n['bodyId']: (n.get('kc_class') or str(n['type'])[:4]) for n in mb_nodes if n['role'] == 'KC'}
        visual = {n['bodyId'] for n in mb_nodes if n['role'] == 'KC' and str(n['type']).startswith(VISUAL_KC_TYPES)}
        pools = {}
        for b, k in kind.items():
            if b not in visual: pools.setdefault(k, []).append(b)
        mapping = {}
        for k, members in pools.items(): pools[k] = list(rng.permutation(members))
        for b in sorted({e[1] for e in edges}):
            pool = pools.get(kind.get(b)) or pools[max(pools, key=lambda k: len(pools[k]))]
            mapping[b] = int(pool.pop()) if pool else b
        return [[a, mapping[b], w] for a, b, w in edges]
    raise ValueError('Unknown condition')


class VisualMushroomBody(memory.MushroomBody):
    """The compartment memory model with visual projection neurons in place of olfactory ones."""

    def __init__(self, data, sparsity=0.10, **options):
        super().__init__(data, sparsity=sparsity, **options)
        self.receivers = np.flatnonzero(self.pn_kc.sum(1) > 0)
        self.stimuli, self.codes = {}, {}

    def present(self, name, vpn):
        self.stimuli[name] = vpn; self.codes.pop(name, None); self.naive.pop(name, None)

    def odour(self, name):
        return self.stimuli[name]

    def kenyon(self, name):
        if name not in self.codes:
            drive = self.pn_kc[self.receivers] @ self.stimuli[name]
            k = max(1, int(round(self.sparsity * len(self.receivers))))
            code = np.zeros(self.pn_kc.shape[0])
            if drive.max() > 0: code[self.receivers[np.argsort(-drive)[:k]]] = 1.0
            self.codes[name] = code
        return self.codes[name]

    def score(self, name):
        naive = self.naive_outputs(name)
        return float(self.valence @ (self.outputs(name) - naive) / (np.abs(naive).sum() + 1e-9))


class Lab:
    """Eye, projection neurons and mushroom body for one seed and condition."""

    def __init__(self, seed=7, condition='intact', sparsity=0.10, noise=0.01, mb_data=None):
        if condition not in CONDITIONS: raise ValueError('Unknown condition')
        self.rng = np.random.default_rng(seed)
        mb = mb_data or memory.load()
        vpn_nodes, vpn_edges, self.source = visual_inputs(mb)
        vpn_edges = rewire([list(e) for e in vpn_edges], condition, mb['nodes'], np.random.default_rng(seed + 1))
        nodes = [dict(n, role='PN') for n in vpn_nodes]
        nodes += [dict(n) for n in mb['nodes'] if n['role'] in ('KC', 'MBON', 'DAN')]
        roles = {n['bodyId']: n['role'] for n in mb['nodes']}
        edges = [e for e in mb['edges'] if roles.get(e[0]) != 'PN'] + vpn_edges
        self.mb = VisualMushroomBody({'nodes': nodes, 'edges': edges}, sparsity=sparsity, seed=seed)
        self.eye = EYE()
        self.vpn = Projection(vpn_nodes, self.eye, seed=seed, use_measured=condition != 'assumed_features')
        self.adapt = condition != 'raw_pixels'
        self.noise, self.condition, self.counter = noise, condition, 0
        self.vpn_count = len(vpn_nodes)

    def sense(self, s, view):
        signal = self.eye.photoreceptors(s, view, self.rng, self.noise)
        vpn = self.vpn.respond(self.eye.lamina(signal, self.adapt))
        self.counter += 1
        name = f's{self.counter}'
        self.mb.present(name, vpn)
        return name, signal, vpn

    def teach(self, name, reinforcer='reward'):
        self.mb.teach(name, reinforcer, memory.DEFAULT)


_EYE = None
def EYE():
    global _EYE
    if _EYE is None: _EYE = Eye()
    return _EYE


# --- experiments ----------------------------------------------------------------------------------------------------
def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg): return None
    greater = (pos[:, None] > neg[None, :]).mean(); ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def jaccard(a, b):
    union = np.sum((a > 0) | (b > 0))
    return float(np.sum((a > 0) & (b > 0)) / union) if union else 0.0


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def validate(options):
    p = DEFAULT | {k: v for k, v in options.items() if v is not None}
    if p['condition'] not in CONDITIONS: raise ValueError('Unknown condition')
    if p['protocol'] not in PROTOCOLS: raise ValueError('Unknown protocol')
    for key in ('trained', 'morph_to', 'contrast'):
        if p[key] not in CLASSES: raise ValueError(f'{key} must be one of: {", ".join(CLASSES)}')
    p['seed'] = int(p['seed']); p['sparsity'] = float(p['sparsity']); p['noise'] = float(p['noise'])
    if not 0.01 <= p['sparsity'] <= 0.5: raise ValueError('sparsity must be between 0.01 and 0.5')
    if not 0 <= p['noise'] <= 0.2: raise ValueError('noise must be between 0 and 0.2')
    for key, lo, hi in (('train_views', 1, 40), ('exemplars', 2, 20), ('glimpses', 1, 12)):
        p[key] = int(p[key])
        if not lo <= p[key] <= hi: raise ValueError(f'{key} must be between {lo} and {hi}')
    return p


def train(lab, p):
    rng = lab.rng
    if p['protocol'] == 'differential' and p['contrast'] == p['trained']: raise ValueError('contrast must differ from trained')
    for _ in range(p['train_views']):
        view, jitter = draw(rng, {})
        name, _, _ = lab.sense(spec(p['trained'], rng, jitter), view)
        lab.teach(name, 'reward')
        if p['protocol'] == 'differential':
            view, jitter = draw(rng, {})
            name, _, _ = lab.sense(spec(p['contrast'], rng, jitter), view)
            lab.teach(name, 'punishment')


def run(mb_data=None, **options):
    p = validate(options)
    lab = Lab(p['seed'], p['condition'], p['sparsity'], p['noise'], mb_data)
    train(lab, p)
    rng = np.random.default_rng(p['seed'] + 101)                     # test draws independent of training draws
    G = p['glimpses']
    sets, glimpse_curve = {}, {g: [] for g in range(1, G + 1)}
    for set_name, ranges in TEST_SETS.items():
        scores = {}
        for cls in CLASSES:
            per = []
            for _ in range(p['exemplars']):
                view, jitter = draw(rng, ranges)
                s = spec(cls, rng, jitter)
                views = [view] + [glimpse(view, rng) for _ in range(G - 1)]
                per.append([lab.mb.score(lab.sense(s, v)[0]) for v in views])
            scores[cls] = np.array(per)
        trained = scores[p['trained']]
        others = np.concatenate([scores[c] for c in CLASSES if c != p['trained']])
        sets[set_name] = {
            'auc_one_look': auc(trained[:, 0], others[:, 0]),
            'auc_all_glimpses': auc(trained.mean(1), others.mean(1)),
            'mean_score': {c: float(scores[c][:, 0].mean()) for c in CLASSES},
            'pairwise_auc': {c: auc(trained[:, 0], scores[c][:, 0]) for c in CLASSES if c != p['trained']},
        }
        for g in range(1, G + 1):
            glimpse_curve[g].append(auc(trained[:, :g].mean(1), others[:, :g].mean(1)))
    fam = sets['familiar']['mean_score'][p['trained']] or 1e-9
    for v in sets.values():
        v['relative_score'] = {c: s / fam for c, s in v['mean_score'].items()}
    return {'parameters': p, 'source': lab.source, 'condition_text': CONDITIONS[p['condition']], 'protocol_text': PROTOCOLS[p['protocol']],
            'circuit': {'visual_inputs': lab.vpn_count, 'receiving_kcs': int(len(lab.mb.receivers)),
                        'active_kcs': int(max(1, round(p['sparsity'] * len(lab.mb.receivers)))),
                        'measured_positions': lab.vpn.measured_positions, 'measured_features': lab.vpn.measured_features,
                        'mean_visual_share': round(lab.vpn.mean_share, 3)},
            'test_sets': sets,
            'glimpses': [{'glimpses': g, 'auc': float(np.mean(v))} for g, v in glimpse_curve.items()],
            'morph': morph_curve(lab, p), 'similarity': similarity(lab, p)}


def morph_curve(lab, p, steps=9):
    """Memory response along a continuous morph from the trained object to another, with the overlap of the
    eye's VPN code and the Kenyon-cell code against the unmorphed object seen in the same view."""
    rng = np.random.default_rng(p['seed'] + 202)
    rows = []
    views = [draw(rng, {})[0] for _ in range(p['exemplars'])]
    base = []
    for view in views:
        name, _, vpn = lab.sense(spec(p['trained']), view)
        base.append((vpn, lab.mb.kenyon(name).copy()))
    for m in np.linspace(0, 1, steps):
        score, vpn_overlap, kc_overlap = [], [], []
        for view, (bv, bk) in zip(views, base):
            name, _, vpn = lab.sense(spec(p['trained'], morph_to=p['morph_to'], morph=float(m)), view)
            score.append(lab.mb.score(name)); vpn_overlap.append(cosine(vpn, bv)); kc_overlap.append(jaccard(lab.mb.kenyon(name), bk))
        rows.append({'morph': float(m), 'score': float(np.mean(score)), 'vpn_overlap': float(np.mean(vpn_overlap)),
                     'kc_overlap': float(np.mean(kc_overlap))})
    top = rows[0]['score'] or 1e-9
    for r in rows: r['relative'] = r['score'] / top
    return {'to': p['morph_to'], 'rows': rows}


def similarity(lab, p, n=6):
    """How alike two objects look at each stage: mean VPN cosine and Kenyon-cell Jaccard between exemplars,
    and how much a change of light alone moves each code."""
    rng = np.random.default_rng(p['seed'] + 303)
    codes = {}
    for cls in CLASSES:
        codes[cls] = []
        for _ in range(n):
            view, jitter = draw(rng, {})
            name, _, vpn = lab.sense(spec(cls, rng, jitter), view)
            codes[cls].append((vpn, lab.mb.kenyon(name).copy()))
    names = list(CLASSES)
    vpn_m = [[0.0] * len(names) for _ in names]; kc_m = [[0.0] * len(names) for _ in names]
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            pairs = [(x, y) for xi, x in enumerate(codes[a]) for yi, y in enumerate(codes[b]) if a != b or xi < yi]
            vpn_m[i][j] = float(np.mean([cosine(x[0], y[0]) for x, y in pairs]))
            kc_m[i][j] = float(np.mean([jaccard(x[1], y[1]) for x, y in pairs]))
    # light alone: same object, same view, light level 0.4 vs 1.0 vs 2.0
    light = []
    for _ in range(n):
        view, _ = draw(rng, {}); s = spec(p['trained'])
        got = []
        for level in (0.4, 1.0, 2.0):
            name, _, vpn = lab.sense(s, dict(view, light=level)); got.append((vpn, lab.mb.kenyon(name).copy()))
        light.append(got)
    light_change = {f'{a} vs 1.0': {'vpn': float(np.mean([cosine(g[i][0], g[1][0]) for g in light])),
                                    'kc': float(np.mean([jaccard(g[i][1], g[1][1]) for g in light]))}
                    for i, a in ((0, '0.4'), (2, '2.0'))}
    return {'classes': names, 'vpn': vpn_m, 'kc': kc_m, 'light_change': light_change}


def view(mb_data=None, **options):
    """Render one object as a person and as the fly would see it, and the trained memory's response to it."""
    allowed = {'cls', 'light', 'light_az', 'distance', 'rotation', 'morph', 'morph_to', 'seed', 'trained', 'condition', 'protocol', 'contrast'}
    extra = set(options) - allowed
    if extra: raise ValueError('Unknown option: ' + ', '.join(sorted(extra)))
    cls = options.get('cls', 'paper cup')
    if cls not in CLASSES: raise ValueError('Unknown object')
    p = validate({k: options[k] for k in ('seed', 'trained', 'condition', 'morph_to', 'protocol', 'contrast') if k in options})
    v = dict(light=float(options.get('light', 1.0)), light_az=float(options.get('light_az', 0)),
             distance=float(options.get('distance', 12)), rotation=float(options.get('rotation', 90)), gaze=(0.0, 0.0))
    if not (0.05 <= v['light'] <= 4 and -90 <= v['light_az'] <= 90 and 4 <= v['distance'] <= 60):
        raise ValueError('light 0.05-4, light_az -90-90, distance 4-60')
    morph = float(options.get('morph', 0))
    if not 0 <= morph <= 1: raise ValueError('morph must be between 0 and 1')
    lab = Lab(p['seed'], p['condition'], p['sparsity'], p['noise'], mb_data)
    train(lab, p)
    s = spec(cls, morph_to=p['morph_to'], morph=morph)
    name, signal, vpn = lab.sense(s, v)
    # what a person would see: 0.6 degree pixels over the same field
    xs = np.linspace(-66, 66, 110); ys = np.linspace(66, -66, 110)
    gx, gy = np.meshgrid(xs, ys)
    image = luminance(s, v, gx, gy)
    channels = lab.eye.lamina(signal, lab.adapt)
    return {'columns': np.round(lab.eye.xy, 2).tolist(), 'photoreceptors': np.round(signal, 4).tolist(),
            'contrast': np.round(channels[0] - channels[1], 4).tolist(),
            'image': {'width': 110, 'height': 110, 'extent_deg': 66, 'values': np.round(image.ravel(), 3).tolist()},
            'vpn_active': int(np.sum(vpn > 0.05)), 'vpn_total': len(vpn), 'kc_active': int(lab.mb.kenyon(name).sum()),
            'score': lab.mb.score(name), 'trained': p['trained'], 'source': lab.source}


def benchmark(seeds=(7, 8, 9, 10, 11), conditions=tuple(CONDITIONS), out=None):
    mb = memory.load()
    runs = {c: [run(mb_data=mb, seed=s, condition=c) for s in seeds] for c in conditions}
    differential = [run(mb_data=mb, seed=s, protocol='differential') for s in seeds]
    def summary(rs):
        sets = {k: {'auc_one_look': float(np.mean([r['test_sets'][k]['auc_one_look'] for r in rs])),
                    'auc_one_look_sd': float(np.std([r['test_sets'][k]['auc_one_look'] for r in rs])),
                    'auc_all_glimpses': float(np.mean([r['test_sets'][k]['auc_all_glimpses'] for r in rs])),
                    'pairwise_auc': {c: float(np.mean([r['test_sets'][k]['pairwise_auc'][c] for r in rs]))
                                     for c in rs[0]['test_sets'][k]['pairwise_auc']}}
                for k in TEST_SETS}
        glimpses = [{'glimpses': g['glimpses'], 'auc': float(np.mean([r['glimpses'][i]['auc'] for r in rs]))}
                    for i, g in enumerate(rs[0]['glimpses'])]
        morph = [{'morph': row['morph'], **{k: float(np.mean([r['morph']['rows'][i][k] for r in rs]))
                                            for k in ('relative', 'vpn_overlap', 'kc_overlap')}}
                 for i, row in enumerate(rs[0]['morph']['rows'])]
        sim = rs[0]['similarity']
        mean_matrix = lambda key: np.mean([r['similarity'][key] for r in rs], 0).round(4).tolist()
        light = {k: {m: float(np.mean([r['similarity']['light_change'][k][m] for r in rs])) for m in ('vpn', 'kc')}
                 for k in sim['light_change']}
        return {'test_sets': sets, 'glimpses': glimpses, 'morph': morph,
                'similarity': {'classes': sim['classes'], 'vpn': mean_matrix('vpn'), 'kc': mean_matrix('kc'), 'light_change': light}}
    result = {'seeds': list(seeds), 'conditions': CONDITIONS, 'source': runs[conditions[0]][0]['source'],
              'circuit': runs[conditions[0]][0]['circuit'], 'parameters': DEFAULT,
              'summary': {c: summary(rs) for c, rs in runs.items()},
              'differential': summary(differential), 'protocols': PROTOCOLS}
    # paired differences against the intact circuit, per test set, one look
    paired = {}
    for c in conditions:
        if c == 'intact': continue
        paired[c] = {}
        for k in TEST_SETS:
            diffs = [runs[c][i]['test_sets'][k]['auc_one_look'] - runs['intact'][i]['test_sets'][k]['auc_one_look'] for i in range(len(seeds))]
            paired[c][k] = {'mean': float(np.mean(diffs)), 'sd': float(np.std(diffs)), 'better': int(sum(d > 0 for d in diffs)), 'n': len(diffs)}
    result['paired_vs_intact'] = paired
    out = out or BASE / 'data/objects-benchmark.json'
    out.write_text(json.dumps(result, indent=1))
    return result


if __name__ == '__main__':
    import sys
    if '--benchmark' in sys.argv:
        r = benchmark()
        for c, s in r['summary'].items():
            print(c.ljust(14), '  '.join(f"{k}:{v['auc_one_look']:.2f}" for k, v in s['test_sets'].items()),
                  '| 8 glimpses', round(s['glimpses'][-1]['auc'], 2))
    else:
        r = run()
        print(json.dumps({k: r[k] for k in ('source', 'circuit')}, indent=1))
        for k, v in r['test_sets'].items(): print(k.ljust(18), round(v['auc_one_look'], 3), round(v['auc_all_glimpses'], 3))
        print('glimpses', [round(g['auc'], 3) for g in r['glimpses']])
        print('morph', [(round(m['morph'], 2), round(m['relative'], 2), round(m['vpn_overlap'], 2), round(m['kc_overlap'], 2)) for m in r['morph']['rows']])
