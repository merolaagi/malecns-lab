import unittest

import numpy as np

import dopamine_loops as dl


class WiringTest(unittest.TestCase):
    def test_circuit(self):
        c = dl.circuit()
        self.assertEqual(c['kc_mbon'].shape, (4064, 97)); self.assertEqual(c['pn_kc'].shape[0], 314)
        self.assertEqual(len(c['classes']), 3)

    def test_shuffles_keep_counts(self):
        m, s, r = (dl.wiring(k, 3) for k in ('measured', 'shuffled', 'random'))
        self.assertTrue(np.allclose(m['kc_mbon'].sum(0), s['kc_mbon'].sum(0)))          # every output neuron keeps its input
        self.assertAlmostEqual(m['kc_mbon'].sum(), r['kc_mbon'].sum())
        self.assertFalse(np.array_equal(m['kc_mbon'], s['kc_mbon']))
        with self.assertRaises(ValueError): dl.wiring('nope', 3)

    def test_measured_loops_return_to_their_own_compartment_more_than_shuffled(self):
        s = dl.loop_structure('measured', 7)
        self.assertGreater(s['loops']['from_own_compartment'], 1.5 * s['loops_shuffled']['from_own_compartment'])
        self.assertEqual(s['loops']['of'], 97)
        S = dl.wiring('measured', 7)['loops']
        self.assertTrue(np.all(S >= 0)); self.assertLessEqual(float(S.sum(1).max()), 1.0 + 1e-9)


class LearningTest(unittest.TestCase):
    def test_codes_are_sparse(self):
        net = dl.Network(dl.wiring('measured', 7), 8, 7)
        z = net.codes(np.random.default_rng(0).random((5, 64)))
        self.assertTrue(np.all(z.sum(1) == net.k))
        dense = dl.Network(dl.wiring('measured', 7), 8, 7, {'sparse': False}).codes(np.random.default_rng(0).random((5, 64)))
        self.assertGreater((dense > 0).sum(1).min(), 5 * net.k)

    def test_backprop_and_ideal_dopamine_learn_above_chance(self):
        b = dl.run('aim', 'backprop', 'measured', 7, trials=600)
        p = dl.run('aim', 'per_cell', 'measured', 7, trials=1000)
        self.assertGreater(b['final'], 0.35); self.assertGreater(p['final'], 0.2)       # chance is 0.125
        self.assertEqual(b['chance'], 0.125)

    def test_readout_leaves_the_backbone_alone_and_depression_only_never_strengthens(self):
        w = dl.wiring('measured', 7); x, y, *_ = dl.task_data('aim', 7)
        net = dl.Network(w, 8, 7); z = net.codes(x[:50]); net.adapt(z)
        for i in range(50): net.learn('readout', z[i], int(y[i]), 0.05)
        self.assertTrue(np.array_equal(net.W, net.W0)); self.assertTrue(np.any(net.Vlearn != 0))
        net = dl.Network(w, 8, 7, {'depression_only': True}); z = net.codes(x[:50]); net.adapt(z)
        for i in range(50): net.learn('per_cell', z[i], int(y[i]), 0.5)
        self.assertTrue(np.all(net.W <= net.W0 + 1e-9)); self.assertTrue(np.all(net.W >= 0)); self.assertTrue(np.any(net.W < net.W0))
        self.assertTrue(np.all(net.W[~net.mask] == 0))                                   # no synapse appears where none was measured

    def test_continual_and_validation(self):
        r = dl.run('continual', 'per_cell', 'measured', 7, trials=500)
        self.assertEqual(len(r['curve']), 5); self.assertIn('forgetting', r)
        with self.assertRaises(ValueError): dl.run('nope')
        with self.assertRaises(ValueError): dl.run('aim', 'nope')


if __name__ == '__main__':
    unittest.main()
