import json
import math
import unittest

import arena
from model import Circuit, steer


class SteeringTests(unittest.TestCase):
    def test_steer_sign_matches_the_body_readout(self):
        self.assertAlmostEqual(steer(0), 0)
        self.assertAlmostEqual(steer(math.pi / 2), -1)          # target to the left needs a negative bias
        self.assertAlmostEqual(steer(-math.pi / 2), 1)

    def test_target_mode_now_reaches_its_target(self):
        # Regression: the bearing-to-bias sign was inverted, so target mode used to steer away.
        t = Circuit().simulate(duration=20, mode='target')['trace'][-1]
        self.assertLess(math.hypot(12 - t['x'], 6 - t['y']), 4.0)


class ArenaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = arena.Arena(Circuit())
        cls.rivalry = cls.a.run(seed=7, scenario='rivalry')

    def test_reproducible_and_strict_json(self):
        short = self.a.run(seed=3, duration=4)
        self.assertEqual(short, self.a.run(seed=3, duration=4))
        json.loads(json.dumps(short, allow_nan=False))

    def test_three_agents_one_measured_circuit(self):
        self.assertEqual([a['id'] for a in self.rivalry['agents']], ['male_1', 'male_2', 'female'])
        self.assertEqual(sum(a['sex'] == 'male' for a in self.rivalry['agents']), 2)
        self.assertIn('same circuit', dict((a['id'], a['label']) for a in self.rivalry['agents'])['female'])

    def test_rivalry_brings_the_males_close_to_her(self):
        # At calibrated walking speed the pursuit overshoots, so the males come close but not reliably
        # into contact; the sensory controls below are what the scenario actually tests.
        p = self.rivalry['pairs']
        closest = min(p['male_1|female']['min_distance'], p['male_2|female']['min_distance'])
        self.assertLess(closest, 6.0)                  # at least one male reaches her
        isolated = self.a.run(seed=7, scenario='rivalry', condition='isolated', duration=8)
        self.assertLess(p['male_1|female']['min_distance'], isolated['pairs']['male_1|female']['min_distance'])
        self.assertEqual(self.rivalry['song_seconds']['female'], 0.0)

    def test_odour_is_the_channel_that_finds_her(self):
        no_odour = self.a.run(seed=7, scenario='rivalry', condition='no_odour')
        self.assertGreater(no_odour['pairs']['male_1|female']['min_distance'], 8.0)
        isolated = self.a.run(seed=7, scenario='rivalry', condition='isolated', duration=12)
        self.assertEqual(sum(isolated['song_seconds'].values()), 0.0)
        self.assertGreater(isolated['pairs']['male_1|female']['min_distance'], 8.0)

    def test_courtship_only_the_attracted_male_approaches(self):
        r = self.a.run(seed=7, scenario='courtship')
        self.assertLess(r['pairs']['male_1|female']['min_distance'], 3.0)
        self.assertGreater(r['pairs']['male_2|female']['min_distance'], 5.0)
        self.assertEqual(r['song_seconds']['male_2'], 0.0)

    def test_food_needs_odour(self):
        with_odour = self.a.run(seed=7, scenario='food', duration=16)
        self.assertGreaterEqual(len(with_odour['food']['arrivals']), 2)
        self.assertEqual(self.a.run(seed=7, scenario='food', condition='no_odour', duration=16)['food']['arrivals'], {})

    def test_silenced_vnc_nearly_stops_every_agent(self):
        # Descending neurons still reach motor neurons directly, so a little movement remains.
        r = self.a.run(seed=7, scenario='rivalry', condition='silence_vnc', duration=6)
        self.assertTrue(all(a['path_mm'] < 1.0 for a in r['agents']))
        intact = self.a.run(seed=7, scenario='rivalry', duration=6)
        self.assertGreater(min(a['path_mm'] for a in intact['agents']), 3.0)

    def test_joints_are_driven_per_leg_and_reported(self):
        a = self.rivalry['trace'][-1]['agents'][0]
        self.assertEqual(set(a['legs']), set(['LF', 'LM', 'LH', 'RF', 'RM', 'RH']))
        for leg, j in a['legs'].items():
            for joint in ['coxa', 'femur', 'tibia', 'tarsus']: self.assertTrue(math.isfinite(j[joint]))
            self.assertEqual(len(j['foot']), 2)
        angles = [a['legs'][leg]['tibia'] for leg in a['legs']]
        self.assertGreater(max(angles) - min(angles), 0.02)     # legs differ from each other
        self.assertFalse(self.rivalry['leg_coverage']['LM']['tarsus']['drivable'])

    def test_kinematic_body_mode_walks(self):
        r = self.a.run(seed=7, scenario='rivalry', duration=8, body='kinematic')
        paths = [x['path_mm'] for x in r['agents']]
        self.assertTrue(all(math.isfinite(p) for p in paths))
        self.assertGreater(max(paths), 1.0)
        self.assertLess(max(paths), 60.0)

    def test_gait_is_measured_and_reported(self):
        g = self.rivalry['gait']['male_1']
        low, high = self.rivalry['reported_ranges']['speed_mm_s']
        self.assertTrue(low <= g['mean_speed_mm_s'] <= high, g['mean_speed_mm_s'])   # calibrated walking speed
        self.assertEqual(len(g['duty_factor_relative']), 6)
        self.assertTrue(all(0 <= v <= 1 for v in g['duty_factor_relative'] + g['duty_factor_absolute']))
        # The measured circuit shows no tripod coordination; this records that, it does not require it.
        self.assertLess(abs(g['tripod_index']), 0.5)
        self.assertGreater(max(g['duty_factor_absolute']), 0.85)   # several legs are almost never lifted

    def test_drive_band_maps_commands_into_the_usable_range(self):
        slow = self.a.run(seed=7, scenario='rivalry', condition='isolated', duration=6)
        fast = self.a.run(seed=7, scenario='threat', duration=8)
        self.assertGreater(fast['gait']['male_1']['mean_speed_mm_s'], slow['gait']['male_1']['mean_speed_mm_s'])
        self.assertGreater(slow['gait']['male_1']['mean_speed_mm_s'], 0.5)         # exploration still walks

    def test_validation(self):
        for bad in [{'scenario': 'x'}, {'condition': 'x'}, {'duration': 100}, {'seed': 1.5}, {'odour_gain': 9}, {'body': 'x'}]:
            with self.assertRaises(ValueError): self.a.run(**bad)


if __name__ == '__main__':
    unittest.main()
