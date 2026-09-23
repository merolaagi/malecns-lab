import json
import unittest

import numpy as np

import bodyplan
from model import BASE


class BodyPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = json.loads((BASE / 'data/circuit.json').read_text())['nodes']
        cls.idx = bodyplan.index(cls.nodes)

    def test_muscle_names_map_to_joints(self):
        self.assertEqual(bodyplan.muscle_of('Ti flexor MN'), 'tibia_flex')
        self.assertEqual(bodyplan.muscle_of('Acc. ti flexor MN'), 'tibia_flex')
        self.assertEqual(bodyplan.muscle_of('Sternal anterior rotator MN'), 'coxa_protract')
        self.assertIsNone(bodyplan.muscle_of('MNml80'))
        self.assertIsNone(bodyplan.muscle_of(None))

    def test_every_motor_neuron_is_placed_once(self):
        placed = sum(len(v) for leg in self.idx['groups'].values() for v in leg.values())
        motor = sum(1 for n in self.nodes if n['superclass'] == 'vnc_motor')
        self.assertEqual(placed + len(self.idx['unmatched']), motor)
        all_idx = [int(i) for leg in self.idx['groups'].values() for v in leg.values() for i in v]
        self.assertEqual(len(all_idx), len(set(all_idx)))

    def test_coverage_reports_what_can_be_driven(self):
        cov = bodyplan.coverage(self.idx)
        self.assertEqual(set(cov), set(bodyplan.LEGS))
        for leg in bodyplan.LEGS:
            for joint in ['coxa', 'femur', 'tibia']:
                self.assertTrue(cov[leg][joint]['drivable'], f'{leg} {joint}')
        # Tarsus levator motor neurons are only annotated for the front legs.
        self.assertTrue(cov['LF']['tarsus']['drivable'] and cov['RF']['tarsus']['drivable'])
        self.assertFalse(any(cov[leg]['tarsus']['drivable'] for leg in ['LM', 'LH', 'RM', 'RH']))

    def test_angles_follow_the_antagonist_difference(self):
        n = len(self.nodes)
        rest = bodyplan.angles(self.idx, np.zeros(n))
        self.assertAlmostEqual(rest['LF']['coxa'], 0.0)
        self.assertAlmostEqual(rest['LF']['tibia'], bodyplan.JOINTS['tibia'][2])
        rates = np.zeros(n); rates[self.idx['groups']['LF']['tibia_extend']] = 60
        extended = bodyplan.angles(self.idx, rates)
        self.assertGreater(extended['LF']['tibia'], rest['LF']['tibia'] + 0.4)
        self.assertAlmostEqual(extended['LM']['tibia'], rest['LM']['tibia'])   # only that leg changed
        rates = np.zeros(n); rates[self.idx['groups']['LF']['tibia_flex']] = 60
        self.assertLess(bodyplan.angles(self.idx, rates)['LF']['tibia'], rest['LF']['tibia'] - 0.4)

    def test_foot_position_and_stance(self):
        n = len(self.nodes)
        lifted = dict(bodyplan.angles(self.idx, np.zeros(n))['LF']); lifted['femur'] = 0.8
        planted = dict(lifted); planted['femur'] = -0.3
        fx, fy, up = bodyplan.foot('LF', lifted); px, py, down = bodyplan.foot('LF', planted)
        self.assertFalse(up); self.assertTrue(down)
        self.assertLess(abs(fy), abs(py))                      # a lifted leg projects shorter from above
        self.assertGreater(bodyplan.foot('RF', planted)[1], 0)  # right legs to one side, left to the other
        self.assertLess(py, 0)
        self.assertGreater(bodyplan.foot('LF', planted)[0], bodyplan.foot('LH', planted)[0])


if __name__ == '__main__':
    unittest.main()
