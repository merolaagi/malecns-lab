"""Milestone 2 of vision-to-walking: eye responses.

Runs a connectome-constrained model of the fly optic lobe (flyvis 1.2.0, Lappalainen et al.,
Nature 2024) on standard stimuli and saves compact responses of its 34 output cell types.

The model simulates one eye: 721 hexagonal columns. Responses of the other eye to world motion in
direction theta equal this eye's responses to the mirrored direction (180 - theta), so milestone 3
composes both eyes from one simulation.

Direction convention: angles are in the model's hex pixel plane (0 = +x, 90 = +y). Which of these
is front-to-back is not assumed. T4a is anatomically the front-to-back motion detector, so its
preferred direction in the trained model defines front-to-back for milestone 3, and the tuning of
T4b-d against T4a is reported as a check.

Heavy optional dependency: pip install -r requirements-vision.txt (installs PyTorch).
"""
import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]
MOTION_TYPES = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
STIM = dict(dt=0.02, grey=0.5, motion=2.0, period=12.0, temporal_hz=2.0, contrast=1.0,
            loom_start_radius=1.0, loom_end_radius=20.0, loom_duration=1.0)


def hex_xy(extent=15):
    """Pixel coordinates of flyvis hex columns, same order as flyvis.utils.hex_utils.get_hex_coords."""
    u, v = [], []
    for q in range(-extent, extent + 1):
        for r in range(max(-extent, -extent - q), min(extent, extent - q) + 1):
            u.append(q); v.append(r)
    u, v = np.array(u, float), np.array(v, float)
    return 3 / 2 * u, math.sqrt(3) * (v + u / 2)   # flyvis hex_to_pixel, mode="default", size=1


def stimuli(p=STIM):
    """Return (names, movies[n, frames, hexals], time axis, onset time). Luminance in [0, 1], grey 0.5."""
    x, y = hex_xy()
    n_grey, n_mov = round(p['grey'] / p['dt']), round(p['motion'] / p['dt'])
    t = np.arange(n_grey + n_mov) * p['dt']
    names, movies = [], []
    for deg in DIRECTIONS:
        a = math.radians(deg)
        proj = x * math.cos(a) + y * math.sin(a)
        m = np.full((len(t), len(x)), 0.5)
        tm = t[n_grey:] - p['grey']
        # Pattern drifts toward +direction: phase = k*proj - w*t.
        m[n_grey:] = 0.5 + 0.5 * p['contrast'] * np.sin(2 * np.pi * (proj[None, :] / p['period'] - p['temporal_hz'] * tm[:, None]))
        m[:n_grey] = 0.5 + 0.5 * p['contrast'] * np.sin(2 * np.pi * proj[None, :] / p['period'])  # static until onset
        names.append(f'grating_{deg}'); movies.append(m)
    r = np.hypot(x, y)
    loom = np.full((len(t), len(x)), 0.5)
    for i, ti in enumerate(t):
        if ti < p['grey']: continue
        frac = min(1, (ti - p['grey']) / p['loom_duration'])
        loom[i, r <= p['loom_start_radius'] + frac * (p['loom_end_radius'] - p['loom_start_radius'])] = 0.0
    names.append('loom_dark'); movies.append(loom)
    for name, level in [('flash_on', 1.0), ('flash_off', 0.0)]:
        m = np.full((len(t), len(x)), 0.5); m[n_grey:] = level
        names.append(name); movies.append(m)
    return names, np.stack(movies), t, p['grey']


def tuning(responses, directions=DIRECTIONS):
    """Vector-sum preferred direction (degrees) and direction selectivity index from mean responses."""
    r = np.maximum(np.asarray(responses, float), 0)
    if r.sum() <= 0: return {'preferred_deg': None, 'dsi': 0.0}
    ang = np.radians(directions)
    vx, vy = (r * np.cos(ang)).sum(), (r * np.sin(ang)).sum()
    return {'preferred_deg': float(math.degrees(math.atan2(vy, vx)) % 360), 'dsi': float(math.hypot(vx, vy) / r.sum())}


def angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


FLYVIS_DATA = BASE / 'flyvis-data'


def ensure_pretrained():
    """Download flyvis pretrained models into ./flyvis-data on first use (git-ignored)."""
    if (FLYVIS_DATA / 'results' / 'flow' / '0000' / '000').exists(): return
    import subprocess, sys
    cli = Path(sys.executable).parent / 'flyvis'
    print('Downloading flyvis pretrained models (first run only)…', flush=True)
    subprocess.run([str(cli), 'download-pretrained', '--skip_large_files'], check=True, env=os.environ.copy())


def load_networks(n_models, untrained):
    import torch
    from flyvis import Network, NetworkView
    nets = []
    if untrained:
        net = Network(); net.eval()
        for q in net.parameters(): q.requires_grad = False
        nets.append(('untrained-initialization', net))
        return nets, torch
    ensure_pretrained()
    for i in range(n_models):
        name = f'flow/0000/{i:03d}'
        nets.append((name, NetworkView(name).init_network()))
    return nets, torch


def run(n_models=5, untrained=False, out=BASE / 'data/eye-responses.json', every=2):
    os.environ.setdefault('FLYVIS_ROOT_DIR', str(FLYVIS_DATA))   # must be set before flyvis is imported
    import flyvis
    names, movies, t, onset = stimuli()
    nets, torch = load_networks(n_models, untrained)
    x = torch.tensor(movies, dtype=torch.float32)[:, :, None, :]
    conn = nets[0][1].connectome
    types = np.array([s.decode() for s in conn.nodes.type[:]])
    central = conn.central_cells_index[:]
    unique = [s.decode() for s in conn.unique_cell_types[:]]
    outputs = [s.decode() for s in conn.output_cell_types[:]]
    base = t < onset
    motion = t >= onset + 0.5   # skip the first 0.5 s of motion (transient)
    models = []
    for name, net in nets:
        idx_of = {ct: np.flatnonzero(types == ct) for ct in outputs}
        means = {ct: np.zeros((len(names), len(t))) for ct in outputs}
        cents = {ct: np.zeros((len(names), len(t))) for ct in MOTION_TYPES}
        for s in range(len(names)):   # one stimulus at a time keeps memory modest
            with torch.no_grad():
                xs = x[s:s + 1]
                init = net.fade_in_state(1.0, STIM['dt'], xs[:, 0])
                act = net.simulate(xs, STIM['dt'], initial_state=init).cpu().numpy()[0]   # (frames, nodes)
            for ct in outputs:
                means[ct][s] = act[:, idx_of[ct]].mean(axis=1)
                if ct in MOTION_TYPES: cents[ct][s] = act[:, central[unique.index(ct)]]
            del act
        per = {}
        for ct in outputs:
            mean = means[ct]
            per[ct] = {'mean': np.round(mean[:, ::every], 5).tolist(),
                       'central': np.round(cents[ct][:, ::every], 5).tolist() if ct in MOTION_TYPES else None,
                       'motion_response': [float(mean[s, motion].mean() - mean[s, base].mean()) for s in range(len(names))]}
        tun = {ct: tuning([per[ct]['motion_response'][names.index(f'grating_{d}')] for d in DIRECTIONS]) for ct in MOTION_TYPES}
        models.append({'name': name, 'cell_types': per, 'tuning': tun})
    # Ensemble summary of motion tuning and the anatomical-consistency check.
    summary = {}
    for ct in MOTION_TYPES:
        resp = np.mean([[m['cell_types'][ct]['motion_response'][names.index(f'grating_{d}')] for d in DIRECTIONS] for m in models], axis=0)
        summary[ct] = tuning(resp) | {'mean_response_by_direction': [float(v) for v in resp]}
    checks = {}
    if summary['T4a']['preferred_deg'] is not None:
        a = summary['T4a']['preferred_deg']
        for ct, expect in [('T4b', 180), ('T4c', 90), ('T4d', 90), ('T5a', 0), ('T5b', 180), ('T5c', 90), ('T5d', 90)]:
            pd = summary[ct]['preferred_deg']
            checks[ct] = None if pd is None else {'angle_from_T4a': round(angle_diff(pd, a), 1), 'expected': expect,
                                                  'consistent': angle_diff(angle_diff(pd, a), expect) <= 45}
        c, d = summary['T4c']['preferred_deg'], summary['T4d']['preferred_deg']
        if c is not None and d is not None:
            checks['T4c_vs_T4d'] = {'angle': round(angle_diff(c, d), 1), 'expected': 180, 'consistent': angle_diff(angle_diff(c, d), 180) <= 45}
    result = {
        'model': 'flyvis', 'flyvis_version': getattr(flyvis, '__version__', '1.2.0'), 'torch_version': torch.__version__,
        'citation': 'Lappalainen et al., Connectome-constrained networks predict neural activity across the fly visual system, Nature 634, 1132-1140 (2024)',
        'pretrained': not untrained, 'models': [m['name'] for m in models], 'built_at': datetime.now(timezone.utc).isoformat(),
        'stimulus_parameters': STIM, 'stimuli': names, 'time': np.round(t[::every], 4).tolist(), 'onset': onset,
        'directions_deg': DIRECTIONS, 'output_cell_types': outputs,
        'direction_convention': 'Angles in the model hex pixel plane (0 = +x). Front-to-back is defined by T4a preferred direction.',
        'tuning_summary': summary, 'anatomical_checks': checks,
        'per_model': models}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, separators=(',', ':')))
    print(json.dumps({'models': result['models'],
                      'tuning': {k: {'preferred_deg': None if v['preferred_deg'] is None else round(v['preferred_deg']), 'dsi': round(v['dsi'], 2)} for k, v in summary.items()},
                      'anatomical_checks': checks, 'wrote': str(out)}, indent=1))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', type=int, default=5, help='Number of pretrained flyvis models (ensemble flow/0000) to run')
    ap.add_argument('--untrained', action='store_true', help='Smoke test with an untrained network; no download needed')
    ap.add_argument('--out', type=Path, default=None)
    a = ap.parse_args()
    out = a.out or BASE / ('data/eye-responses-untrained.json' if a.untrained else 'data/eye-responses.json')
    run(a.models, a.untrained, out)
