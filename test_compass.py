import math
import unittest

import numpy as np

import compass as cp
from learning import LearningCircuit


class WalkerTest(unittest.TestCase):
    def walk(self, source, noise=0.0):
        w = cp.Walker(source, np.random.default_rng(3), noise)
        o = w.start((0, 0)); out = []
        for cell in [(0, 1), (1, 1), (1, 0), (2, 0)]:
            e = w.go(cell); out.append((e[0] - o[0], e[1] - o[1]))
        return out

    def test_true_location(self):
        self.assertEqual(self.walk('true_location')[-1], (2.0, 0.0))

    def test_published_compass_integrates_the_path(self):
        r, c = self.walk('compass_idealised')[-1]
        self.assertLess(math.hypot(r - 2, c - 0), 0.25)

    def test_without_rotation_cells_the_position_is_wrong(self):
        r, c = self.walk('compass_no_rotation')[-1]
        self.assertGreater(math.hypot(r - 2, c - 0), 1.0)

    def test_paths(self):
        self.assertEqual(len(set(cp.serpentine(4))), 16)
        p = cp.random_path((0, 0), 4, 6, np.random.default_rng(0))
        self.assertTrue(all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(p, p[1:])))


class ExperimentTest(unittest.TestCase):
    def test_compass_matches_true_location_and_beats_no_locations(self):
        c = LearningCircuit()
        kw = dict(trials=25, max_glimpses=5, n_objects=10)
        true = cp.experiment(c, 7, 'true_location', **kw)
        ring = cp.experiment(c, 7, 'compass_idealised', **kw)
        none = cp.experiment(c, 7, 'compass_no_rotation', **kw)
        self.assertGreater(ring['three_modules'][-1], 0.8)
        self.assertLess(abs(ring['three_modules'][-1] - true['three_modules'][-1]), 0.2)
        self.assertLess(none['three_modules'][-1], ring['three_modules'][-1] - 0.3)
        self.assertLess(ring['location_error_spacings'], 0.3)


if __name__ == '__main__':
    unittest.main()
