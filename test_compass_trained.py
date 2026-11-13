import math
import unittest

import numpy as np

import compass_trained as ct


class CircuitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ct.load(); cls.c = ct.Circuit(cls.data)

    def test_core(self):
        c = self.c
        self.assertEqual(c.n, 152); self.assertEqual(len(c.types), 6); self.assertEqual(len(c.heading), 46)
        self.assertEqual(sorted(len(r) for r in c.rotate), [20, 22])
        self.assertTrue(np.all(c.sign[c.type == c.types.index('Delta7')] < 0))            # the inhibitory type
        th = c.angles(0.3)
        right = c.side[c.heading] > 0
        self.assertTrue(np.all((th[right] >= 0) & (th[right] < 2 * math.pi)))

    def test_shuffle_keeps_each_cells_partner_count_within_type_pairs(self):
        s = ct.Circuit(self.data, 'shuffled', 3)
        self.assertFalse(np.array_equal(s.mask, self.c.mask))
        np.testing.assert_allclose(s.C.sum(0), self.c.C.sum(0), rtol=1e-9)             # every cell sends exactly the synapses it did
        T = len(self.c.types)
        for tp in range(T):                                                              # and each type pair carries the same total
            for tq in range(T):
                blk = np.ix_(self.c.type == tp, self.c.type == tq)
                self.assertAlmostEqual(float(s.C[blk].sum()), float(self.c.C[blk].sum()), places=9)
        self.assertEqual(ct.Circuit(self.data, 'random', 3).n, 152)
        with self.assertRaises(ValueError): ct.Circuit(self.data, 'nope')


class GradientTest(unittest.TestCase):
    def check(self, model):
        rng = np.random.default_rng(0)
        omega, phi = ct.trajectories(rng, 2, 0.8)
        _, g = model.loss_and_grad(omega, phi)
        for k, v in model.p.items():
            for _ in range(3):
                ix = tuple(rng.integers(s) for s in v.shape)
                old = v[ix]; v[ix] = old + 1e-5; up, _ = model.loss_and_grad(omega, phi); v[ix] = old - 1e-5; down, _ = model.loss_and_grad(omega, phi); v[ix] = old
                num = (up - down) / 2e-5
                if abs(num) + abs(g[k][ix]) > 1e-7: self.assertLess(abs(num - g[k][ix]) / (abs(num) + abs(g[k][ix])), 1e-3, k)

    def test_hand_written_gradients_match_finite_differences(self):
        self.check(ct.Model(ct.Circuit(), 'types', 1)); self.check(ct.Dense(40, 1))

    def test_counts(self):
        c = ct.Circuit()
        self.assertEqual(ct.Model(c, 'types').count(), 52)
        self.assertEqual(ct.Model(c, 'synapses').count(), int(c.mask.sum()) + 16)


class TrainingTest(unittest.TestCase):
    def test_measured_wiring_learns_and_the_compass_interface_works(self):
        model, row = ct.one('measured', 'types', 1, steps=150)
        self.assertLess(row['error_deg_4s'], row['error_deg_untrained'] - 20)
        nav = ct.TrainedCompass(model, heading=1.0)
        self.assertLess(abs(ct.wrap(nav.decode_heading()[0] - 1.0)), 0.6)
        for _ in range(25): nav.step(0.0, 8.0)
        x, y = nav.home_vector()
        self.assertAlmostEqual(math.hypot(x, y), 4.0, delta=0.6)
        self.assertEqual(len(ct.pathways(model)), 36)

    def test_trajectories(self):
        omega, phi = ct.trajectories(np.random.default_rng(1), 4, 2.0)
        self.assertEqual(omega.shape, (4, 100)); self.assertLessEqual(np.abs(omega).max(), 6.0)
        np.testing.assert_allclose(phi[:, 1] - phi[:, 0], omega[:, 0] * ct.DT)


if __name__ == '__main__':
    unittest.main()
