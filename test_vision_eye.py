import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import vision_eye as ve


class StimulusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.names, cls.movies, cls.t, cls.onset = ve.stimuli()

    def test_shapes_and_range(self):
        self.assertEqual(self.movies.shape, (11, len(self.t), 721))
        self.assertTrue(((self.movies >= 0) & (self.movies <= 1)).all())

    def test_gratings_are_grey_on_average_and_static_before_onset(self):
        pre = self.t < self.onset
        for i in range(8):
            self.assertLess(abs(self.movies[i].mean() - 0.5), 0.05)
            self.assertTrue(np.allclose(self.movies[i][pre], self.movies[i][0]))

    def test_grating_drifts_toward_its_direction(self):
        p, x, _ = ve.STIM, *ve.hex_xy()
        speed = p['period'] * p['temporal_hz']            # pixel units per second
        m, k = self.movies[self.names.index('grating_0')], np.flatnonzero(self.t >= self.onset + 0.2)[0]
        # The pattern at time t+dt equals the pattern at time t shifted by +speed*dt along x.
        shift = speed * p['dt']
        expected = 0.5 + 0.5 * np.sin(2 * np.pi * ((x - shift) / p['period'] - p['temporal_hz'] * (self.t[k] - self.onset)))
        self.assertTrue(np.allclose(m[k + 1], expected, atol=1e-6))

    def test_loom_grows_monotonically(self):
        dark = (self.movies[self.names.index('loom_dark')] == 0).sum(axis=1)
        self.assertEqual(dark[self.t < self.onset].max(), 0)
        self.assertTrue((np.diff(dark) >= 0).all())
        self.assertGreater(dark[-1], 100)

    def test_hex_layout_matches_flyvis(self):
        x, y = ve.hex_xy()
        self.assertEqual(len(x), 721)
        self.assertAlmostEqual(x.max(), 22.5)
        self.assertAlmostEqual(y.max(), 15 * 3 ** .5)


class TuningTests(unittest.TestCase):
    def test_cosine_tuning_recovers_preferred_direction(self):
        for pref in [0, 90, 225]:
            r = [max(0, np.cos(np.radians(d - pref))) for d in ve.DIRECTIONS]
            t = ve.tuning(r)
            self.assertLess(ve.angle_diff(t['preferred_deg'], pref), 1e-6)
            self.assertGreater(t['dsi'], 0.5)

    def test_flat_and_empty_responses(self):
        self.assertAlmostEqual(ve.tuning([1] * 8)['dsi'], 0, places=9)
        self.assertEqual(ve.tuning([0] * 8), {'preferred_deg': None, 'dsi': 0.0})

    def test_angle_diff(self):
        self.assertEqual(ve.angle_diff(350, 10), 20)
        self.assertEqual(ve.angle_diff(0, 180), 180)


@unittest.skipUnless(os.environ.get('MALECNS_TEST_FLYVIS') == '1', 'set MALECNS_TEST_FLYVIS=1 to run the slow flyvis smoke test')
class FlyvisSmokeTest(unittest.TestCase):
    def test_untrained_network_runs_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / 'eye.json'
            ve.run(untrained=True, out=out)
            r = json.loads(out.read_text())
            self.assertEqual(len(r['output_cell_types']), 34)
            self.assertEqual(set(r['tuning_summary']), set(ve.MOTION_TYPES))
            self.assertFalse(r['pretrained'])


if __name__ == '__main__':
    unittest.main()
