import math
import unittest

import numpy as np

import bodysim as bs
import optogenetics as og


def traj(yaw_end=0.0, forward=10.0, rise=0.0, lateral=0.0):
    """A straight synthetic trajectory along +x, turning to yaw_end (deg) and shifting sideways during the light."""
    out = []
    for i in range(21):
        t = i * 0.1
        f = max(0.0, (t - og.LIGHT_ON) / (2.0 - og.LIGHT_ON))
        out.append({'t': t, 'x': forward * t / 2.0, 'y': lateral * f, 'z': 0.5 + (rise if t >= og.LIGHT_ON else 0),
                    'yaw': math.radians(yaw_end) * f})
    return out


class BodySimTest(unittest.TestCase):
    def test_summary_is_relative_to_the_unlit_body(self):
        s = bs.summarise(traj(yaw_end=-30, rise=0.1), traj())
        self.assertAlmostEqual(s['yaw_deg'], -30, places=5)
        self.assertAlmostEqual(s['rise_mm'], 0.1, places=5)
        self.assertAlmostEqual(s['forward_mm'], 0, delta=0.6)

    def test_turn_verdicts(self):
        mk = lambda r, l: {'right': [bs.summarise(traj(yaw_end=r), traj())], 'left': [bs.summarise(traj(yaw_end=l), traj())],
                           'both': [bs.summarise(traj(), traj())]}
        self.assertEqual(bs.verdict('turn', mk(-20, 20))['result'], 'match')
        self.assertEqual(bs.verdict('turn', mk(20, -20))['result'], 'opposite')
        self.assertEqual(bs.verdict('turn', mk(-5, 5))['result'], 'weak, right direction')
        self.assertEqual(bs.verdict('turn', mk(5, -5))['result'], 'weak, wrong direction')
        self.assertEqual(bs.verdict('turn', mk(5, 5))['result'], 'no clear turn')

    def test_backward_and_escape_verdicts(self):
        runs = lambda lit: {s: [bs.summarise(lit, traj())] for s in og.SIDES}
        self.assertEqual(bs.verdict('backward', runs(traj(forward=-5)))['result'], 'match')
        self.assertEqual(bs.verdict('backward', runs(traj(forward=4)))['result'], 'partial')
        self.assertEqual(bs.verdict('backward', runs(traj(forward=10)))['result'], 'mismatch')
        self.assertEqual(bs.verdict('escape', runs(traj(rise=0.2)))['result'], 'match')
        self.assertEqual(bs.verdict('escape', runs(traj()))['result'], 'mismatch')

    def test_groups_cover_legs_and_jump_muscle(self):
        lab = og.Optogenetics(og.BASE / 'data/circuit.json')
        g = bs.groups(lab)
        legs = {k.split('|')[0] for k in g}
        self.assertTrue(legs <= set(bs.LAB_LEG.values()))
        self.assertTrue(all(isinstance(i, int) for v in g.values() for i in v))
        rates, lit = bs.motor_rates(lab, og.validate(dict(experiment='DNa02', side='right', duration=bs.DURATION)))
        self.assertEqual(lit, 2)                                   # both DNa02 are targeted; the bias lights one side
        self.assertGreater(len(rates), 50)
        self.assertTrue(all(isinstance(k, tuple) and len(k) == 2 for k in rates[-1]))

    @unittest.skipUnless(bs.available(), 'flygym not installed')
    def test_physics_body(self):
        b = bs.body()
        self.assertEqual(len(b.signs), 24)
        self.assertTrue(set(b.signs.values()) <= {-1.0, 1.0})
        off = b.offsets({('RF', 'coxa_protract'): 50.0})
        self.assertEqual(np.count_nonzero(off), 1)
        self.assertAlmostEqual(abs(off).max(), bs.SPAN['coxa'] * math.tanh(50 / bs.RATE_SCALE))
        tr = b.run([], 'walking', duration=0.3)
        self.assertGreater(tr[-1]['x'] - tr[0]['x'], 0.5)          # the supplied rhythm walks forward
        with self.assertRaises(ValueError): bs.run(experiment='nope')


if __name__ == '__main__':
    unittest.main()
