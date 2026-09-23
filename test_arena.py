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

    def test_rivalry_brings_both_males_to_her_and_produces_song(self):
        p = self.rivalry['pairs']
        self.assertLess(p['male_1|female']['min_distance'], 2.0)
        self.assertLess(p['male_2|female']['min_distance'], 2.0)
        self.assertGreater(self.rivalry['both_males_near_female_s'], 1.0)
        self.assertGreater(sum(self.rivalry['song_seconds'].values()), 1.0)
        self.assertEqual(self.rivalry['song_seconds']['female'], 0.0)

    def test_odour_is_the_channel_that_finds_her(self):
        no_odour = self.a.run(seed=7, scenario='rivalry', condition='no_odour')
        self.assertGreater(no_odour['pairs']['male_1|female']['min_distance'], 3.0)
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
        self.assertEqual(len(with_odour['food']['arrivals']), 3)
        self.assertEqual(self.a.run(seed=7, scenario='food', condition='no_odour', duration=16)['food']['arrivals'], {})

    def test_silenced_vnc_nearly_stops_every_agent(self):
        # Descending neurons still reach motor neurons directly, so a little movement remains.
        r = self.a.run(seed=7, scenario='rivalry', condition='silence_vnc', duration=6)
        self.assertTrue(all(a['path_mm'] < 1.0 for a in r['agents']))
        intact = self.a.run(seed=7, scenario='rivalry', duration=6)
        self.assertGreater(min(a['path_mm'] for a in intact['agents']), 3.0)

    def test_validation(self):
        for bad in [{'scenario': 'x'}, {'condition': 'x'}, {'duration': 100}, {'seed': 1.5}, {'odour_gain': 9}]:
            with self.assertRaises(ValueError): self.a.run(**bad)


if __name__ == '__main__':
    unittest.main()
