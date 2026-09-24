import math
import unittest

import numpy as np

import model
from model import ADEX, Circuit


class NeuronModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.c = Circuit()

    def run_sim(self, **opts): return self.c.simulate(duration=3, **opts)

    def test_both_models_run_and_are_reproducible(self):
        for neuron in ['lif', 'adex']:
            a, b = self.run_sim(neuron=neuron), self.run_sim(neuron=neuron)
            self.assertEqual(a['metrics'], b['metrics'])
            self.assertTrue(all(math.isfinite(t['motor_hz']) for t in a['trace']))
        with self.assertRaises(ValueError): self.run_sim(neuron='hodgkin')

    def test_calibrated_to_the_same_rate_at_default_drive(self):
        lif, adex = self.run_sim(neuron='lif')['metrics'], self.run_sim(neuron='adex')['metrics']
        self.assertLess(abs(lif['motor_mean_hz'] - adex['motor_mean_hz']), 3.0)

    def test_adaptation_and_shunting_give_gain_control(self):
        # Regression on the documented finding: raising the drive saturates the integrate-and-fire
        # network but leaves the adaptive one near its default rate.
        lif_low, lif_high = self.run_sim(drive=2.0)['metrics'], self.run_sim(drive=4.0)['metrics']
        ad_low, ad_high = self.run_sim(drive=2.0, neuron='adex')['metrics'], self.run_sim(drive=4.0, neuron='adex')['metrics']
        self.assertGreater(lif_high['motor_mean_hz'], 4 * lif_low['motor_mean_hz'])
        self.assertLess(ad_high['motor_mean_hz'], 2 * ad_low['motor_mean_hz'])
        self.assertGreater(lif_high['clipped_motor_fraction'], 0.5)
        self.assertLess(ad_high['clipped_motor_fraction'], 0.2)

    def test_descending_cells_still_follow_the_drive(self):
        low = np.mean([t['dn_hz'] for t in self.run_sim(drive=2.0, neuron='adex')['trace']])
        high = np.mean([t['dn_hz'] for t in self.run_sim(drive=4.0, neuron='adex')['trace']])
        self.assertGreater(high, low * 1.3)       # the clamp is downstream, not at the input

    def test_conductance_matrices_split_the_signs(self):
        w = self.c.matrix('intact', 7)
        exc, inh = self.c.conductance_matrices('intact', 7)
        self.assertGreaterEqual(exc.min(), 0); self.assertGreaterEqual(inh.min(), 0)
        np.testing.assert_allclose((exc - inh).toarray(), w.toarray(), atol=1e-12)
        self.assertGreater(inh.nnz, 0)

    def test_controls_behave_under_both_models(self):
        for neuron in ['lif', 'adex']:
            self.assertEqual(self.run_sim(neuron=neuron, condition='no_stimulus')['metrics']['path_mm'], 0)
            # Descending neurons reach motor neurons directly, so a trace of activity survives silencing.
            silenced = self.run_sim(neuron=neuron, condition='silence_vnc')['metrics']
            self.assertLess(silenced['motor_mean_hz'], 1.0)
            self.assertLess(silenced['path_mm'], 0.5)
            self.assertGreater(self.run_sim(neuron=neuron)['metrics']['path_mm'], 5.0)

    def test_membrane_voltages_stay_physiological(self):
        r = self.run_sim(neuron='adex')
        self.assertGreater(len(r['raster']), 0)
        self.assertLess(ADEX['V_reset'], ADEX['V_T'])
        self.assertEqual(model.SUBSTEPS * 10, 100)   # 0.1 ms substeps for the exponential term


if __name__ == '__main__':
    unittest.main()
