import json
import math
import unittest

import numpy as np

import navigation as nav


class RingTests(unittest.TestCase):
    def test_compass_tracks_a_constant_turn(self):
        for omega in (0.5, 2.0, -3.0):
            n = nav.Navigator(); heading = 0.0; errors = []
            for i in range(int(6 / nav.DT)):
                heading += omega * nav.DT
                estimate = n.step(omega, 0.0)
                if i > int(1 / nav.DT):
                    errors.append(abs(math.degrees(math.atan2(math.sin(estimate - heading), math.cos(estimate - heading)))))
            self.assertLess(np.mean(errors), 5.0, omega)
            self.assertEqual(n.bumps(), 1)

    def test_rotation_removed_leaves_the_bump_behind(self):
        # Short enough that the true heading does not wrap past the stationary bump.
        n = nav.Navigator(condition='no_rotation'); heading = 0.0
        for _ in range(int(0.7 / nav.DT)):
            heading += 2.0 * nav.DT; n.step(2.0, 0.0)
        error = abs(math.degrees(math.atan2(math.sin(n.decode_heading()[0] - heading), math.cos(n.decode_heading()[0] - heading))))
        self.assertGreater(error, 45.0)

    def test_inhibition_removed_loses_the_single_bump(self):
        intact, broken = nav.Navigator(), nav.Navigator(condition='no_inhibition')
        for _ in range(int(2 / nav.DT)): intact.step(1.0, 5.0); broken.step(1.0, 5.0)
        self.assertEqual(intact.bumps(), 1)
        self.assertLess(broken.heading_cells.max(), intact.heading_cells.max())

    def test_home_vector_matches_the_travelled_path(self):
        n = nav.Navigator(); x = y = 0.0
        for _ in range(int(4 / nav.DT)):                      # straight line, no turning
            x += math.cos(0.0) * 6.0 * nav.DT; y += math.sin(0.0) * 6.0 * nav.DT
            n.step(0.0, 6.0)
        vx, vy = n.home_vector()
        self.assertLess(math.hypot(vx - x, vy - y), 1.0)

    def test_steering_points_at_the_goal(self):
        n = nav.Navigator()
        self.assertAlmostEqual(n.steer(0.0, 0.0), 0.0, places=6)      # already aligned
        self.assertGreater(n.steer(math.pi / 2, 0.0), 0.5)            # goal to the left turns left
        self.assertLess(n.steer(-math.pi / 2, 0.0), -0.5)


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.runs = [nav.run(seed=s) for s in range(7, 12)]

    def mean(self, runs, key): return float(np.mean([r['metrics'][key] for r in runs]))

    def test_reproducible_and_strict_json(self):
        self.assertEqual(nav.run(seed=3)['trace'], nav.run(seed=3)['trace'])
        json.loads(json.dumps(self.runs[0], allow_nan=False))

    def test_intact_navigator_gets_home(self):
        self.assertLess(self.mean(self.runs, 'closest_during_homing_mm'), 15.0)
        self.assertGreater(self.mean(self.runs, 'homed'), 0.5)
        self.assertLess(self.mean(self.runs, 'mean_heading_error_deg'), 15.0)
        self.assertGreater(self.mean(self.runs, 'furthest_distance_mm'), 30.0)   # it really did travel

    def test_every_control_fails_to_get_home(self):
        # Regression on the documented result: each control removes one mechanism and homing stops.
        for condition in ['no_rotation', 'no_inhibition', 'no_vector', 'shuffled_columns', 'random_goal']:
            runs = [nav.run(seed=s, condition=condition) for s in range(7, 12)]
            self.assertEqual(self.mean(runs, 'homed'), 0.0, condition)
            self.assertGreater(self.mean(runs, 'closest_during_homing_mm'), 40.0, condition)

    def test_compass_drift_degrades_homing_in_order(self):
        closest = [self.mean([nav.run(seed=s, compass_noise=n) for s in range(7, 10)], 'closest_during_homing_mm')
                   for n in (0.0, 0.3, 0.6)]
        self.assertTrue(closest[0] < closest[1] < closest[2], closest)
        self.assertLess(closest[0], 1.0)                      # noiseless integration homes exactly

    def test_measured_connectivity_reports_missing_data_clearly(self):
        from pathlib import Path
        if (nav.BASE / 'data/central-complex.json').exists(): return self.skipTest('measured subset present')
        with self.assertRaises(ValueError) as e: nav.measured_offsets()
        self.assertIn('build_centralcomplex.py', str(e.exception))

    def test_validation(self):
        for bad in [{'connectivity': 'x'}, {'condition': 'x'}, {'speed': 99}, {'seed': 1.5}, {'compass_noise': 9}]:
            with self.assertRaises(ValueError): nav.run(**bad)


if __name__ == '__main__':
    unittest.main()
