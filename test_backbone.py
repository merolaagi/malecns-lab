import unittest

import numpy as np

import backbone as bb


class WiringTest(unittest.TestCase):
    def test_shuffle_keeps_every_cells_connection_counts(self):
        m, s = bb.wiring('central_complex', 'measured', 1), bb.wiring('central_complex', 'shuffled', 1)
        n = m['n']
        self.assertTrue(np.array_equal(np.bincount(bb.raw('central_complex')['pre'], minlength=n), np.bincount(bb.raw('central_complex')['pre'][np.argsort(np.argsort(bb.raw('central_complex')['pre']))], minlength=n)))
        self.assertGreater(len(s['pre']), 0.95 * len(m['pre']))
        self.assertFalse(np.array_equal(m['post'][:200], s['post'][:200]))
        with self.assertRaises(ValueError): bb.wiring('central_complex', 'nope', 1)

    def test_backbones_have_inputs_and_outputs(self):
        for name in bb.BACKBONES:
            g = bb.raw(name)
            self.assertGreater(len(g['inputs']), 10); self.assertGreater(len(g['outputs']), 10)


class NetTest(unittest.TestCase):
    def test_forward_and_signs(self):
        net = bb.Net(seed=1)
        x, y = bb.aim_batch(np.random.default_rng(0), 8)
        out, _ = net.forward(x)
        self.assertEqual(out.shape, (8, 2))
        for _ in range(3): net.learn(x, y, 'backprop')
        self.assertTrue(np.all(net.a >= 0))                                  # strengths change, the cell's sign does not
        W = net.W.tocoo(); pre_sign = bb.raw('central_complex')['sign'][W.col]
        self.assertTrue(np.all(W.data * pre_sign >= -1e-12))

    def test_free_signs_can_flip(self):
        net = bb.Net(biology='free_signs', seed=1)
        x, y = bb.aim_batch(np.random.default_rng(0), 16)
        w0 = net.a_signed().copy()
        for _ in range(20): net.learn(x, y, 'backprop')
        self.assertFalse(np.allclose(w0, net.w))

    def test_backprop_learns_the_training_batch_and_readout_only_touches_the_readout(self):
        x, y = bb.aim_batch(np.random.default_rng(0), 32)
        net = bb.Net(seed=1)
        first = net.learn(x, y, 'backprop')
        for _ in range(150): last = net.learn(x, y, 'backprop')
        self.assertLess(last, 0.3 * first)
        net = bb.Net(seed=1); a0 = net.a.copy()
        for _ in range(5): net.learn(x, y, 'readout')
        self.assertTrue(np.array_equal(a0, net.a)); self.assertTrue(np.any(net.R != 0))

    def test_every_rule_and_unit_runs(self):
        x, y = bb.aim_batch(np.random.default_rng(0), 8)
        for rule in bb.RULES: self.assertTrue(np.isfinite(bb.Net(seed=1).learn(x, y, rule)), rule)
        for bio in bb.BIOLOGY:
            net = bb.Net(biology=bio, seed=1); self.assertTrue(np.isfinite(net.learn(x, y, 'teachers_wiring')), bio)
        with self.assertRaises(ValueError): bb.Net(biology='nope')
        with self.assertRaises(ValueError): bb.run(task='nope')
        with self.assertRaises(ValueError): bb.run(rule='nope')

    def test_aim_target_is_the_hotter_object(self):
        x, y = bb.aim_batch(np.random.default_rng(1), 200)
        self.assertEqual(x.shape, (200, 34)); self.assertGreater(float((np.abs(x[:, 32] - x[:, 33]) >= 0.15).mean()), 0.9)   # a few near ties remain


if __name__ == '__main__':
    unittest.main()
