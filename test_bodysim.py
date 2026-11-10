import json
import math
import unittest

import numpy as np

import bodysim as bs
import model
import optogenetics as og


def traj(yaw_end=0.0, forward=10.0, rise=0.0, lateral=0.0):
    """A synthetic trajectory along +x, turning to yaw_end (deg), shifting sideways and rising during the light."""
    out = []
    for i in range(21):
        t = i * 0.1
        f = max(0.0, (t - og.LIGHT_ON) / (2.0 - og.LIGHT_ON))
        out.append({'t': t, 'x': forward * f, 'y': lateral * f, 'z': 0.5 + (rise if t >= og.LIGHT_ON else 0),
                    'yaw': math.radians(yaw_end) * f})
    return out


class VerdictTest(unittest.TestCase):
    def test_summary_is_relative_to_the_comparison(self):
        s = bs.summarise(traj(yaw_end=-30, rise=0.1), traj(forward=0))
        self.assertAlmostEqual(s['yaw_deg'], -30, places=5)
        self.assertAlmostEqual(s['rise_mm'], 0.1, places=5)
        self.assertGreater(s['forward_mm'], 5)

    def test_turn_verdicts(self):
        mk = lambda r, l: {'right': [bs.summarise(traj(yaw_end=r), traj())], 'left': [bs.summarise(traj(yaw_end=l), traj())],
                           'both': [bs.summarise(traj(), traj())]}
        self.assertEqual(bs.verdict('turn', mk(-20, 20))['result'], 'match')
        self.assertEqual(bs.verdict('turn', mk(20, -20))['result'], 'opposite')
        self.assertEqual(bs.verdict('turn', mk(-5, 5))['result'], 'weak, right direction')
        self.assertEqual(bs.verdict('turn', mk(5, -5))['result'], 'weak, wrong direction')

    def test_walking_backward_and_escape_verdicts(self):
        still = traj(forward=0)
        runs = lambda lit: {s: [bs.summarise(lit, still)] for s in og.SIDES}
        self.assertEqual(bs.verdict('backward', runs(traj(forward=-5)))['result'], 'match')
        self.assertEqual(bs.verdict('backward', runs(traj(forward=-1)))['result'], 'partial')
        self.assertEqual(bs.verdict('backward', runs(traj(forward=5)))['result'], 'mismatch')
        p9 = {'both': [bs.summarise(traj(forward=5), still)], 'right': [bs.summarise(traj(forward=5, yaw_end=-20), still)],
              'left': [bs.summarise(traj(forward=5, yaw_end=20), still)]}
        self.assertEqual(bs.verdict('forward_turn', p9)['result'], 'match')
        self.assertEqual(bs.verdict('escape', runs(traj(forward=0, rise=0.3)))['result'], 'match')
        self.assertEqual(bs.verdict('escape', runs(still))['result'], 'mismatch')


class CircuitSideTest(unittest.TestCase):
    def test_stepper_matches_simulate_rates(self):
        c = model.Circuit()
        d = np.zeros(c.n); d[c.dn] = 2.0
        s = model.Stepper(c, d, seed=1)
        for _ in range(2500): s.step()
        r = c.simulate(duration=3, drive=2.0, mode='tonic', seed=1, feedback=0)   # stimulus from 0.5 s: 2.5 s of drive
        stepper = np.mean([s.total[idx].mean() / 2.5 for idx in c.motor])
        self.assertAlmostEqual(stepper, r['metrics']['motor_mean_hz'] * 3 / 2.5, delta=2.5)

    def test_gap_kicks_make_the_partner_follow(self):
        c = model.Circuit()
        a, b = int(np.flatnonzero(c.dn)[0]), int(np.flatnonzero(c.vnc)[0])
        d = np.zeros(c.n); d[a] = 3.0
        s = model.Stepper(c, d, seed=1, gaps=[(a, b, 1.2)])
        for _ in range(500): s.step()
        self.assertGreater(s.total[b], 0.5 * s.total[a])

    def test_symmetric_circuit_is_matched(self):
        data = json.loads((og.BASE / 'data/circuit.json').read_text())
        out, report = og.symmetric(data)
        pools = {}
        for n in out['nodes']:
            if n.get('type') and og.side_of(n) in ('L', 'R'):
                key = (n['type'], n.get('subclass'), n.get('entryNerve'))
                pools.setdefault(key, {'L': 0, 'R': 0})[og.side_of(n)] += 1
        unmatched = [k for k, v in pools.items() if v['L'] != v['R'] and not any(k[0] in e['types'] for e in og.EXPERIMENTS.values())]
        self.assertEqual(unmatched, [])
        ids = {n['bodyId'] for n in out['nodes']}
        self.assertTrue(all(a in ids and b in ids for a, b, _ in out['edges']))
        self.assertEqual(report['kept'], len(out['nodes']))

    def test_groups_and_sensors(self):
        lab = og.Optogenetics(og.BASE / 'data/circuit.json')
        g, ttm = bs.groups(lab, muscle=True)
        self.assertTrue({leg for leg, _ in g} <= set(model.LEGS))
        sens = bs.sensory_cells(lab)
        self.assertEqual(len(sens), 6)
        self.assertGreater(sum(len(a) + len(b) for a, b in sens), 20)


class CouplingTest(unittest.TestCase):
    def test_uniform_and_measured_coupling(self):
        lab = og.Optogenetics(og.BASE / 'data/circuit.json')
        self.assertTrue(np.all(bs.coupling_matrix(lab, 'uniform') == bs.COUPLING))
        K = bs.coupling_matrix(lab, 'measured')
        off = ~np.eye(6, dtype=bool)
        self.assertTrue(np.all(np.diag(K) == 0))
        self.assertAlmostEqual(K[off].mean(), bs.COUPLING, places=6)
        links = bs.leg_links_from_circuit(lab)
        self.assertEqual(np.array(links['excitatory']).shape, (6, 6))

    def test_measured_coupling_uses_build_links(self):
        lab = og.Optogenetics(og.BASE / 'data/circuit.json')
        M = np.ones((6, 6)); M[0, 1] = 7.0                    # leg 0 (LF) reaches leg 1 (LM) strongly
        lab.leg_links = {'excitatory': M.tolist(), 'inhibitory': np.zeros((6, 6)).tolist()}
        K = bs.coupling_matrix(lab, 'measured')
        self.assertGreater(K[1, 0], K[0, 1])                    # LM is pulled by LF more than the reverse


class CoordinationRulesTest(unittest.TestCase):
    def test_modes(self):
        lab = bs.lab_for(False)
        self.assertIsNone(bs.coordination_rules(lab, 'uniform')); self.assertIsNone(bs.coordination_rules(lab, 'measured'))
        ind = bs.coordination_rules(lab, 'independent')
        self.assertFalse(ind['share']); self.assertEqual(float(ind['K'].sum()), 0.0)
        tri = bs.coordination_rules(lab, 'tripod_sync')
        self.assertTrue(tri['share']); self.assertAlmostEqual(tri['offset'][0, 1], np.pi)
        with self.assertRaises(ValueError): bs.coordination_rules(lab, 'nope')

    def test_signed_rule_comes_from_the_links_and_shuffling_changes_it(self):
        lab = bs.lab_for(False)
        s = bs.coordination_rules(lab, 'signed'); sh = bs.coordination_rules(lab, 'signed_shuffled', seed=1)
        off = ~np.eye(6, dtype=bool)
        self.assertTrue(set(np.unique(s['offset'][off])) <= {0.0, np.pi})
        self.assertAlmostEqual(float(s['T'][off].mean()), 1.0)
        self.assertAlmostEqual(float(s['K'][off].mean()), bs.COUPLING)
        self.assertFalse(np.allclose(s['T'], sh['T']))
        self.assertAlmostEqual(float(sh['T'][off].mean()), 1.0)                # same links, different pairs


@unittest.skipUnless(bs.available(), 'flygym not installed')
class PhysicsTest(unittest.TestCase):
    def test_body_calibration_and_offsets(self):
        b = bs.body(True)
        self.assertEqual(len(b.signs), 24)
        off = b.offsets({('RF', 'coxa_protract'): 50.0})
        self.assertEqual(np.count_nonzero(off), 1)
        self.assertGreater(b.tmax, 0)

    def test_unlit_fly_stands_and_supplied_rhythm_walks(self):
        still, _ = bs.run_once(None, duration=0.6, path=og.BASE / 'data/circuit.json')
        self.assertLess(abs(still[-1]['x'] - still[0]['x']), 0.5)
        walk, _ = bs.run_once(None, duration=0.6, config={'rhythm': 'supplied'}, path=og.BASE / 'data/circuit.json')
        self.assertGreater(walk[-1]['x'] - walk[0]['x'], 1.0)
        with self.assertRaises(ValueError): bs.run(experiment='nope')


if __name__ == '__main__':
    unittest.main()
