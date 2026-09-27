import json
import unittest

import numpy as np

import spectral


class MatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.data = spectral.graphs()

    def test_graphs_present(self):
        self.assertIn('locomotion', self.data)
        pre, post, count, sign, n, label = self.data['locomotion']
        self.assertEqual(len(pre), len(post)); self.assertEqual(len(pre), len(count))
        self.assertEqual(n, 807)

    def test_incoming_normalisation_bounds_the_radius(self):
        # Rows of absolute weights sum to 1, so no eigenvalue can exceed 1 in magnitude.
        for randomisation in spectral.RANDOMISATIONS:
            W = spectral.matrix('locomotion', randomisation, 'incoming_l1', 7, self.data)
            rows = np.abs(W).sum(1).A.ravel()
            self.assertTrue(np.allclose(rows[rows > 0], 1.0, atol=1e-9), randomisation)
            self.assertLessEqual(spectral.spectrum(W)['radius'], 1.0 + 1e-6, randomisation)

    def test_randomisations_preserve_what_they_claim(self):
        pre, post, count, sign, n, _ = self.data['locomotion']
        degree = spectral.matrix('locomotion', 'degree_shuffle', 'log_raw', 7, self.data).tocoo()
        self.assertEqual(degree.nnz > 0, True)
        weight = spectral.matrix('locomotion', 'weight_shuffle', 'log_raw', 7, self.data).tocoo()
        measured = spectral.matrix('locomotion', 'measured', 'log_raw', 7, self.data).tocoo()
        self.assertEqual(sorted(zip(measured.row.tolist(), measured.col.tolist())),
                         sorted(zip(weight.row.tolist(), weight.col.tolist())))   # topology kept
        # Parallel edges merge when the matrix is assembled, so totals match only approximately.
        self.assertLess(abs(np.abs(measured.data).sum() - np.abs(weight.data).sum()) / np.abs(measured.data).sum(), 0.05)
        with self.assertRaises(ValueError): spectral.matrix('locomotion', 'nonsense', 'log_raw', 7, self.data)
        with self.assertRaises(ValueError): spectral.matrix('locomotion', 'measured', 'nonsense', 7, self.data)

    def test_spectrum_measures(self):
        symmetric = np.array([[0.0, 0.5], [0.5, 0.0]])
        import scipy.sparse as sp
        result = spectral.spectrum(sp.csr_matrix(symmetric))
        self.assertAlmostEqual(result['radius'], 0.5, places=6)
        self.assertAlmostEqual(result['henrici'], 0.0, places=6)      # a symmetric matrix is normal
        self.assertLessEqual(result['transient_gain'], 1.01)          # and cannot amplify transiently


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = spectral.graphs()
        cls.result = spectral.compare('locomotion', 'incoming_l1', 7, cls.data)

    def test_measured_beats_rewired_but_not_weight_shuffled(self):
        # The documented finding: the near-unit radius follows the measured topology, not its weights.
        conditions = self.result['conditions']
        self.assertGreater(conditions['measured']['radius'], 0.9)
        for rewired in ['degree_shuffle', 'rewire_targets', 'full_random', 'reciprocal_random']:
            self.assertLess(conditions[rewired]['radius'], 0.5, rewired)
        self.assertGreater(conditions['weight_shuffle']['radius'], 0.9)
        self.assertGreater(self.result['radius_ratio_vs_rewired'], 2.0)

    def test_reciprocity_alone_does_not_explain_it(self):
        conditions = self.result['conditions']
        self.assertGreater(conditions['reciprocal_random']['reciprocity'], 0.2)   # matched on reciprocity
        self.assertLess(conditions['reciprocal_random']['radius'], 0.5)           # and still far below

    def test_the_effect_depends_on_normalisation(self):
        scaled = spectral.compare('locomotion', 'global_scale', 7, self.data)
        self.assertLess(scaled['radius_ratio_vs_rewired'], 1.5)      # no measured advantage once rescaled

    def test_leading_eigenvector_is_localised(self):
        self.assertLess(self.result['conditions']['measured']['participation_ratio'], 0.1)
        self.assertGreater(self.result['conditions']['degree_shuffle']['participation_ratio'], 0.3)
        self.assertTrue(self.result['conditions']['measured']['leading_named'])

    def test_closed_blocks_reported(self):
        closed = self.result['closed']
        self.assertGreater(closed['self_contained_cells'], 100)      # subset selection leaves many
        self.assertLessEqual(closed['self_contained_cells'], closed['cells_with_input'])

    def test_controls_run(self):
        thresholds = spectral.threshold_control('locomotion', 'incoming_l1', thresholds=(1, 10))
        self.assertEqual(len(thresholds['rows']), 2)
        self.assertIn('ratio_vs_rewired', thresholds['rows'][0])
        coverage = spectral.coverage_control('locomotion', 'incoming_l1', minimums=(0.0, 0.1))
        self.assertEqual(len(coverage['rows']), 2)
        self.assertGreater(coverage['rows'][0]['cells_kept'], coverage['rows'][1]['cells_kept'])

    def test_validation_and_json(self):
        with self.assertRaises(ValueError): spectral.compare('nonsense')
        json.loads(json.dumps(self.result, allow_nan=False))


if __name__ == '__main__':
    unittest.main()
