import unittest
from numerosity import Numerosity, CONDITIONS
from learning import LearningCircuit


class NumerosityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.e = Numerosity(LearningCircuit())

    def test_reproducible(self):
        self.assertEqual(self.e.run(seed=3, train_trials=100), self.e.run(seed=3, train_trials=100))

    def test_held_out_numbers_never_trained(self):
        for held in [[2, 5, 8], [4, 5, 6], [8, 9]]:
            r = self.e.run(held_out=held, train_trials=200)
            self.assertFalse(set(held) & set(r['presented_in_training']))

    def test_every_number_activates_same_pn_count(self):
        for cond in ['scalar', 'onehot']:
            p = self.e.validate({'condition': cond})
            self.assertTrue(all(len(c) == p['width'] for c in self.e.codes(p)))

    def test_codes_differ_only_in_overlap(self):
        s = self.e.run(condition='scalar', train_trials=0)['pn_overlap']
        o = self.e.run(condition='onehot', train_trials=0)['pn_overlap']
        self.assertGreater(s[3][4], 0.5)
        self.assertEqual(o[3][4], 0)

    def test_testing_never_changes_weights(self):
        r = self.e.run(train_trials=100)
        self.assertEqual(r['weights']['after_training_sha256'], r['weights']['after_test_sha256'])

    def test_no_plasticity_and_untrained_are_exactly_chance(self):
        for opts in [{'condition': 'no_plasticity'}, {'train_trials': 0}]:
            r = self.e.run(**opts)
            self.assertTrue(all(v == 0.5 for v in r['accuracy'].values() if v is not None))
            self.assertEqual(r['weights']['nonzero_edges'], 0)

    def test_regression_interpolation_needs_overlap(self):
        # Regression check on the saved finding, not independent evidence.
        s = self.e.run(condition='scalar', held_out=[2, 5, 8])['accuracy']['both_novel']
        o = self.e.run(condition='onehot', held_out=[2, 5, 8])['accuracy']['both_novel']
        self.assertGreater(s, 0.8)
        self.assertLess(abs(o - 0.5), 0.15)

    def test_input_validation(self):
        for bad in [{'condition': 'x'}, {'held_out': [0]}, {'held_out': [1, 2, 3, 4, 5, 6, 7, 8]}, {'width': 60, 'numbers': 12, 'condition': 'onehot'}, {'seed': 1.5}]:
            with self.assertRaises(ValueError): self.e.run(**bad)


if __name__ == '__main__':
    unittest.main()
