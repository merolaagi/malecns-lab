import unittest

import numpy as np

from expansion import Associative, Expansion, Experiment, MODELS
from learning import LearningCircuit


class ExpansionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = LearningCircuit(); cls.e = Experiment(cls.c)
        cls.r = cls.e.run(seed=7)

    def test_reproducible(self):
        self.assertEqual(self.e.run(seed=7, per_class=5), self.e.run(seed=7, per_class=5))

    def test_sparsity_and_codes(self):
        m = self.r['models']
        for k in ['fly_measured', 'fly_shuffled', 'random_sparse', 'dense_topk']:
            self.assertAlmostEqual(m[k]['active_fraction'], 0.05, delta=0.002)
        self.assertGreater(m['no_sparsity']['active_fraction'], 0.3)

    def test_random_sparse_keeps_in_degree(self):
        a = Expansion(self.c, 'fly_measured', 7, .05).w; b = Expansion(self.c, 'random_sparse', 7, .05).w
        np.testing.assert_array_equal(np.diff(a.indptr), np.diff(b.indptr))
        self.assertFalse(np.array_equal(a.indices, b.indices))

    def test_associative_rule_touches_only_the_correct_class(self):
        h = Associative(6); z = np.array([[1, 0, 1, 0, 0, 0.]]); h.fit_batch(z, np.array([3]), 0.5)
        self.assertEqual(np.count_nonzero(h.W), 2); self.assertTrue((h.W[3, [0, 2]] == 0.5).all())
        self.assertEqual(h.predict(z)[0], 3)

    def test_class_incremental_protocol(self):
        for k, v in self.r['models'].items():
            R = v['accuracy_matrix']
            self.assertTrue(all(R[i][j] is None for i in range(5) for j in range(i + 1, 5)), k)   # never tested before learned
            self.assertTrue(all(R[i][j] is not None for i in range(5) for j in range(i + 1)), k)
        self.assertEqual(self.e.run(seed=7, per_class=5)['train_size'], 50)

    def test_regression_saved_findings(self):
        # Regression checks on the documented results, not independent evidence.
        m = self.r['models']
        self.assertGreater(m['mlp_backprop']['forgetting'], 0.8)
        self.assertLess(m['fly_measured']['forgetting'], 0.2)
        self.assertLess(abs(m['fly_measured']['final_accuracy'] - m['pixels_associative']['final_accuracy']), 0.06)
        t = self.e.run(seed=7, task='two_shape')['models']
        self.assertGreater(t['dense_topk']['final_accuracy'], t['no_sparsity']['final_accuracy'] + 0.03)
        self.assertEqual(set(t), set(MODELS))

    def test_results_are_strict_json(self):
        import json
        json.loads(json.dumps(self.r, allow_nan=False))

    def test_validation(self):
        for bad in [{'task': 'x'}, {'seed': 1.5}, {'per_class': 500}, {'sparsity': 0.9}, {'epochs': 0}]:
            with self.assertRaises(ValueError): self.e.run(**bad)


if __name__ == '__main__':
    unittest.main()
