import json
import math
import unittest

import numpy as np

import theory as th
from learning import LearningCircuit


class AnalyticTests(unittest.TestCase):
    def test_overlap_curve_endpoints_and_monotonicity(self):
        for f in [0.01, 0.05, 0.2]:
            self.assertAlmostEqual(th.overlap_exact(1e-6, f), 1.0, places=2)     # identical inputs
            self.assertAlmostEqual(th.overlap_exact(math.pi / 2, f), f, places=3)  # orthogonal: chance overlap
            vals = [th.overlap_exact(t, f) for t in np.linspace(0.05, math.pi / 2, 12)]
            self.assertTrue(all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])))

    def test_asymptotic_matches_exact_at_ninety_degrees(self):
        for f in [0.01, 0.05, 0.2]:
            self.assertAlmostEqual(th.overlap_asymptotic(math.pi / 2, f), f, places=6)
            self.assertAlmostEqual(th.overlap_asymptotic(1e-6, f), 1.0, places=3)

    def test_capacity_model_behaviour(self):
        acc = [th.capacity_accuracy(c, 5.0, 0.5, 1.0, 0.3) for c in [10, 50, 200, 400]]
        self.assertTrue(all(a >= b for a, b in zip(acc, acc[1:])))              # more classes, never better
        self.assertGreater(th.capacity_accuracy(10, 50.0, 0.5, 1.0, 0.3), 0.99)  # well separated
        self.assertLess(th.capacity_accuracy(400, 1.0, 0.5, 1.0, 0.5), 0.2)      # not separated

    def test_pearson(self):
        self.assertAlmostEqual(th.pearson([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
        self.assertTrue(math.isnan(th.pearson([1, 1, 1, 1], [1, 2, 3, 4])))


class NormalisedReadoutTests(unittest.TestCase):
    def test_normalised_predict_ignores_bias_and_scale(self):
        h = th.DeltaNormalised(4); h.W = np.array([[1., 0, 0, 0], [10., 0, 0, 0], [0, 1., 0, 0]]); h.b = np.array([0., 50., 0.])
        z = np.array([[1., 0, 0, 0]])
        self.assertEqual(h.predict(z)[0], 0)          # class 1's larger weights and bias do not win
        plain = th.Delta(4); plain.W, plain.b = h.W.copy(), h.b.copy()
        self.assertEqual(plain.predict(z)[0], 1)      # the unnormalised readout does prefer it


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.r = th.Theory(LearningCircuit()).run(seed=7)

    def test_strict_json_and_structure(self):
        json.loads(json.dumps(self.r, allow_nan=False))
        self.assertEqual(set(self.r['verdicts']), {'P1', 'P2', 'P3', 'P4'})
        self.assertTrue(all(isinstance(v['supported'], bool) for v in self.r['verdicts'].values()))
        self.assertEqual(len(self.r['P1']['points']), len(th.SPARSITIES) * len(th.WIRINGS))

    def test_regression_findings(self):
        # Regression checks on the documented conclusions, not independent evidence.
        self.assertLess(self.r['P1']['pearson_r'], 0.6)                          # overlap does not explain forgetting
        self.assertGreater(self.r['P1']['recency_bias_share'], 0.05)             # normalising at test time helps
        self.assertLess(self.r['P1']['associative_forgetting_mean'], 0.15)
        self.assertLess(self.r['P3']['wirings']['dense_topk']['rmse_vs_exact'], 0.05)
        self.assertGreater(self.r['P3']['wirings']['fly_measured']['rmse_vs_exact'],
                           self.r['P3']['wirings']['dense_topk']['rmse_vs_exact'])
        skews = [v['wrong_score_skew'] for k, v in sorted(self.r['P4']['curves'].items(), key=lambda kv: float(kv[0]))]
        self.assertTrue(skews[0] > skews[-1])                                    # sparsest codes are the most skewed
        for v in self.r['P4']['curves'].values():
            acc = [p['simulated'] for p in v['points']]
            self.assertTrue(all(a >= b - 0.02 for a, b in zip(acc, acc[1:])))    # accuracy falls as classes are added

    def test_validation(self):
        for bad in [{'seed': 1.5}, {'seed': -1}, {'seed': 'x'}]:
            with self.assertRaises(ValueError): th.Theory(LearningCircuit()).run(**bad)


if __name__ == '__main__':
    unittest.main()
