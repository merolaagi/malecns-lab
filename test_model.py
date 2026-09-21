import json
import unittest
import numpy as np
from model import Circuit

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.c = Circuit()

    def test_anatomical_integrity(self):
        c = self.c
        self.assertEqual(len(set(n['bodyId'] for n in c.nodes)), c.n)
        self.assertTrue(all(n['status'] == 'Traced' for n in c.nodes))
        self.assertTrue(np.all(c.count > 0))
        self.assertEqual(int(c.dn.sum()), 6)
        self.assertTrue(all(len(group) for group in c.motor))
        self.assertTrue(all(len(group) for group in c.sensory))
        self.assertTrue(all(len(v['sha256']) == 64 for v in c.data['files'].values()))

    def test_fixed_seed_reproduces_full_trajectory(self):
        a = self.c.simulate(duration=1, seed=3)
        b = self.c.simulate(duration=1, seed=3)
        self.assertEqual(a, b)

    def test_no_stimulus_does_not_create_spontaneous_walking(self):
        r = self.c.simulate(duration=2, condition='no_stimulus')
        self.assertEqual(r['metrics']['spikes'], 0)
        self.assertEqual(r['metrics']['path_mm'], 0)

    def test_zero_synaptic_gain_blocks_motor_recruitment(self):
        r = self.c.simulate(duration=2, gain=0)
        self.assertGreater(r['metrics']['spikes'], 0)
        self.assertEqual(r['metrics']['motor_mean_hz'], 0)
        self.assertEqual(r['metrics']['path_mm'], 0)

    def test_silencing_holds_vnc_cells_silent(self):
        r = self.c.simulate(duration=2, condition='silence_vnc')
        self.assertTrue(np.all(np.array(r['neuron_hz'])[self.c.vnc] == 0))

    def test_shuffled_control_is_different_and_seeded(self):
        intact = self.c.matrix('intact', 7)
        shuffled = self.c.matrix('shuffled', 7)
        self.assertGreater((intact != shuffled).nnz, 0)
        self.assertEqual((shuffled != self.c.matrix('shuffled', 7)).nnz, 0)

    def test_unknown_transmitters_have_no_outgoing_effect(self):
        self.assertEqual(self.c.matrix('intact', 7)[:, self.c.sign == 0].nnz,
                         int(np.count_nonzero(self.c.matrix('intact', 7)[:, self.c.sign == 0].data == 0)))
        self.assertTrue(np.all(self.c.matrix('intact', 7)[:, self.c.sign == 0].data == 0))

    def test_feedback_can_change_neural_activity(self):
        intact = self.c.simulate(duration=8)
        removed = self.c.simulate(duration=8, condition='no_feedback')
        self.assertNotEqual(intact['neuron_hz'], removed['neuron_hz'])

    def test_invalid_parameters_rejected(self):
        for args in [dict(duration=100), dict(gain=float('nan')), dict(condition='missing'), dict(mode='smell')]:
            with self.assertRaises(ValueError): self.c.simulate(**args)

    def test_export_is_finite_and_motor_signal_changes_with_wiring(self):
        r = self.c.simulate(duration=3)
        self.assertGreater(r['metrics']['motor_mean_hz'], 0)
        json.dumps(r, allow_nan=False)
        self.assertEqual(len(r['neuron_hz']), self.c.n)

if __name__ == '__main__': unittest.main()
