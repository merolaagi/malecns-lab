import unittest

import numpy as np

import diagnose as dg
import model


class DiagnoseTest(unittest.TestCase):
    def test_glutamate_sign_switch(self):
        inh, exc = dg.lab('inhibitory'), dg.lab('excitatory')
        glu = np.array([str(n.get('nt')) == 'glutamate' for n in exc.circuit.nodes])
        self.assertTrue(glu.any())
        self.assertTrue(np.all(inh.circuit.sign[glu] == -1))
        self.assertTrue(np.all(exc.circuit.sign[glu] == 1))
        self.assertTrue(np.all(inh.circuit.sign[~glu] == exc.circuit.sign[~glu]))

    def test_leg_state(self):
        legs = dg.leg_state(dg.lab('inhibitory'), 'DNa02', 'right', 2.5, 1)
        self.assertEqual(set(legs), set(model.LEGS))
        for v in legs.values():
            self.assertTrue(0 <= v['amp'] <= 1 and -1 <= v['dir'] <= 1)

    def test_pick_skips_signs_without_direction(self):
        rows = [dict(glutamate='inhibitory', experiment='P9', side='both', light=l, mean_amp=a, rear_dir=d)
                for l, a, d in [(2.5, 0.0, 0.0), (5.0, 0.8, 0.9)]]
        rows += [dict(glutamate='excitatory', experiment='P9', side='both', light=3.5, mean_amp=1.0, rear_dir=-0.02)]
        self.assertEqual(dg.pick(rows), {'inhibitory': 5.0})


if __name__ == '__main__':
    unittest.main()
