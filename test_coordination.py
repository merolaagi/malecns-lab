import unittest

import coordination as co
import model


class CoordinationTest(unittest.TestCase):
    def test_evenness(self):
        legs = {g: {'amp': 0.9, 'freq_hz': 10.0} for g in model.LEGS}
        self.assertEqual(co.evenness(legs), 0.0)
        legs['LF'] = {'amp': 0.0, 'freq_hz': 0.0}               # a standing leg is left out
        self.assertEqual(co.evenness(legs), 0.0)
        legs['RF'] = {'amp': 0.9, 'freq_hz': 4.0}
        self.assertGreater(co.evenness(legs), 0.2)

    def test_chosen_light(self):
        rows = [dict(circuit='x', experiment='P9', side='both', light=l, rear_amp=a, rear_dir=d)
                for l, a, d in [(2.5, 0.0, 0.0), (3.5, 0.6, 0.9), (5.0, 0.9, 0.9)]]
        self.assertEqual(co.chosen_light(rows, 'x'), 3.5)
        self.assertEqual(co.chosen_light(rows, 'y'), max(co.LIGHTS))

    def test_circuits_listed(self):
        self.assertIn('optogenetics', co.available_circuits())


if __name__ == '__main__':
    unittest.main()
