import json
import unittest

import numpy as np

import inhibition as inh


class PlacementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: cls.graph = inh.load_graph()
        except ValueError as e: raise unittest.SkipTest(str(e))
        cls.rng = np.random.default_rng(0)
        cls.sub = inh.subsample(cls.graph, 400, np.random.default_rng(1))

    def test_placements_keep_the_inhibitory_count(self):
        measured = inh.place(self.sub, 'measured', self.rng)
        for placement in inh.PLACEMENTS:
            flags = inh.place(self.sub, placement, np.random.default_rng(2))
            expected = 0 if placement == 'none' else int(measured.sum())
            self.assertEqual(int(flags.sum()), expected, placement)
        with self.assertRaises(ValueError): inh.place(self.sub, 'nonsense', self.rng)

    def test_hub_and_antihub_placements_differ_in_throughput(self):
        out = np.bincount(self.sub['pre'].astype(int), weights=self.sub['weight'], minlength=self.sub['n'])
        hubs = inh.place(self.sub, 'hubs', self.rng)
        antihubs = inh.place(self.sub, 'antihubs', self.rng)
        self.assertGreater(out[hubs].sum(), out[antihubs].sum())

    def test_scaling_modes(self):
        flags = inh.place(self.sub, 'measured', self.rng)
        natural, radius = inh.matrix(self.sub, flags, 0.9, 'natural')
        per_condition, _ = inh.matrix(self.sub, flags, 0.9, 'per_condition')
        self.assertAlmostEqual(float(np.max(np.abs(np.linalg.eigvals(per_condition)))), 0.9, places=6)
        shared, _ = inh.matrix(self.sub, flags, 0.9, 'shared', reference=radius)
        np.testing.assert_allclose(shared, per_condition, atol=1e-9)       # reference is its own radius here

    def test_measured_inhibition_lowers_the_radius(self):
        # The spectral finding this module was built to test.
        flags = {p: inh.place(self.sub, p, np.random.default_rng(5)) for p in ['measured', 'random', 'none']}
        radii = {p: inh.matrix(self.sub, f, 0.9, 'natural')[1] for p, f in flags.items()}
        self.assertLess(radii['measured'], radii['none'])
        self.assertLess(radii['measured'], radii['random'])


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try: cls.graph = inh.load_graph()
        except ValueError as e: raise unittest.SkipTest(str(e))
        cls.result = inh.run('measured', graph=cls.graph, cells=200, steps=500, delays=8)

    def test_run_shape_and_reproducibility(self):
        again = inh.run('measured', graph=self.graph, cells=200, steps=500, delays=8)
        self.assertEqual(self.result['memory_by_delay'], again['memory_by_delay'])
        self.assertEqual(len(self.result['memory_by_delay']), 8)
        self.assertTrue(0 <= self.result['memory_capacity'] <= 8)
        json.loads(json.dumps(self.result, allow_nan=False))

    def test_memory_decays_with_delay(self):
        scores = self.result['memory_by_delay']
        self.assertGreater(scores[0], scores[-1])
        self.assertGreater(scores[0], 0.5)

    def test_paired_comparison_is_paired(self):
        result = inh.paired('measured', ['random'], seeds=3, graph=self.graph, cells=200, steps=500, delays=8)
        row = result['rows']['random']
        self.assertEqual(row['seeds'], 3)
        self.assertLessEqual(row['favours_reference'], 3)
        self.assertEqual(inh.paired('measured', ['measured'], seeds=2, graph=self.graph, cells=200, steps=500,
                                    delays=8)['rows']['measured']['mean_difference'], 0.0)

    def test_validation(self):
        for bad in [{'cells': 10}, {'radius': 9}, {'leak': 0}, {'scaling': 'x'}]:
            with self.assertRaises(ValueError): inh.run('measured', graph=self.graph, **bad)
        with self.assertRaises(ValueError): inh.run('nonsense', graph=self.graph)


if __name__ == '__main__':
    unittest.main()
