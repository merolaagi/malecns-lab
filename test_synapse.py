import json
import unittest

from model import Circuit
from synapse import MEMBRANE, RECEPTORS, Synapses


class SynapseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = Circuit(); cls.s = Synapses(cls.c)
        cls.result = cls.s.run(body_id=10360, duration_ms=200, rate_hz=60)

    def test_partners_match_the_measured_edges(self):
        rows = self.s.partners(10360)
        expected = [(pre, count) for pre, post, count in self.c.data['edges'] if post == 10360]
        self.assertEqual(len(rows), len(expected))
        self.assertEqual(sum(r['synapses'] for r in rows), sum(c for _, c in expected))
        self.assertEqual(rows, sorted(rows, key=lambda r: -r['synapses']))
        with self.assertRaises(ValueError): self.s.partners(1)

    def test_receptors_follow_the_transmitter(self):
        self.assertEqual(RECEPTORS['acetylcholine']['reversal'], 0.0)
        self.assertEqual(RECEPTORS['glutamate']['reversal'], -70.0)      # insect inhibitory glutamate
        for row in self.s.partners(10360):
            if row['transmitter'] in RECEPTORS: self.assertEqual(row['receptor'], RECEPTORS[row['transmitter']]['name'])
            else: self.assertEqual(row['receptor'], 'no receptor assigned')

    def test_reproducible_bounded_and_strict_json(self):
        again = self.s.run(body_id=10360, duration_ms=200, rate_hz=60)
        self.assertEqual(self.result['voltage_mv'], again['voltage_mv'])
        self.assertTrue(all(-85 <= v <= 5 for v in self.result['voltage_mv']))   # between the reversal potentials
        json.loads(json.dumps(self.result, allow_nan=False))

    def test_silent_input_leaves_the_cell_at_rest(self):
        quiet = self.s.run(body_id=10360, duration_ms=100, rate_hz=0)
        self.assertAlmostEqual(quiet['mean_voltage_mv'], MEMBRANE['rest_mv'], places=3)
        self.assertEqual(quiet['spikes_ms'], [])

    def test_probability_mode_spreads_and_recovers_conductance(self):
        dominant = self.s.weights(10360, 'dominant', 1.0, 1.0)
        split = self.s.weights(10360, 'probability', 1.0, 1.0)
        rows = self.s.partners(10360)
        labelled = sum(r['synapses'] for r in rows if r['transmitter'] in RECEPTORS) * 0.03
        self.assertAlmostEqual(sum(g['weight'] for g in dominant.values()), labelled, places=6)
        # Probability mode also uses cells whose consensus label sits outside the receptor set but whose
        # measured probabilities include one, so its total is at least the labelled total.
        self.assertGreaterEqual(sum(g['weight'] for g in split.values()) + 1e-9, labelled)
        with_any = sum(r['synapses'] for r in rows if r['probabilities']) * 0.03
        self.assertAlmostEqual(sum(g['weight'] for g in split.values()), with_any, places=6)
        self.assertGreater(len(split), len(dominant))      # conductance reaches more receptor types

    def test_conductance_inhibition_shunts(self):
        result = self.s.shunting(body_id=10360, duration_ms=300, rate_hz=60, excitation_scale=2)
        self.assertGreater(result['depolarisation_alone_mv'], 0)
        self.assertLess(result['ratio'], 1.0)

    def test_drive_raises_the_passive_membrane(self):
        low = self.s.run(body_id=10360, duration_ms=300, rate_hz=10, inhibition_scale=0, spiking=False)
        high = self.s.run(body_id=10360, duration_ms=300, rate_hz=90, inhibition_scale=0, spiking=False)
        self.assertGreater(high['mean_voltage_mv'], low['mean_voltage_mv'])
        self.assertGreater(high['spike_rate_hz'], -1)

    def test_validation(self):
        for bad in [{'duration_ms': 5}, {'rate_hz': 500}, {'excitation_scale': 9}, {'mode': 'x'}]:
            with self.assertRaises(ValueError): self.s.run(body_id=10360, **bad)


if __name__ == '__main__':
    unittest.main()
