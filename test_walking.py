import unittest

import walking as wk


def traj(dx, yaw=0.0):
    return [{'t': 0.0, 'x': 0, 'y': 0, 'z': 1, 'yaw': 0}, {'t': 0.5, 'x': 0, 'y': 0, 'z': 1, 'yaw': 0}, {'t': 2.0, 'x': dx, 'y': 0, 'z': 1, 'yaw': yaw}]


LEGS = [{'t': t, 'amp': [1, 1, 1, 1, 1, 0.0], 'dir': [1] * 6, 'freq': [8, 8, 10, 10, 8, 0]} for t in (0.8, 1.0)]


class WalkingTest(unittest.TestCase):
    def test_measure(self):
        m = wk.measure(traj(6.0, 0.2), LEGS)
        self.assertAlmostEqual(m['forward_mm'], 6.0); self.assertAlmostEqual(m['yaw_deg'], 11.459, places=2)
        self.assertEqual(m['legs_stepping'], 5); self.assertGreater(m['frequency_spread'], 0)

    def test_verdict_respects_the_commanded_direction(self):
        fwd = wk.verdict('P9', [wk.measure(traj(6.0), LEGS)] * 3)
        self.assertTrue(fwd['walks_all']); self.assertEqual(fwd['travel_mm'], 6.0)
        back = wk.verdict('MDN', [wk.measure(traj(6.0), LEGS)] * 3)
        self.assertFalse(back['walks_all']); self.assertEqual(back['travel_in_commanded_direction_mm'], -6.0)
        self.assertTrue(wk.verdict('MDN', [wk.measure(traj(-6.0), LEGS)] * 2)['walks_all'])
        spin = wk.verdict('P9', [wk.measure(traj(6.0, 2.0), LEGS)])
        self.assertFalse(spin['walks_all'])                                    # travels but turns more than 45 degrees
        still = [dict(wk.measure(traj(0.0), LEGS), frequency_spread=None)]
        self.assertIsNone(wk.verdict('P9', still)['frequency_spread'])

    def test_link_balance(self):
        b = wk.link_balance()
        self.assertEqual(b['pairs'], 30); self.assertEqual(len(b['net_balance_from_row_to_column']), 6)
        self.assertTrue(0 <= b['pairs_matching_tripod'] <= 30)


if __name__ == '__main__':
    unittest.main()
