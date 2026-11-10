import unittest

import numpy as np

import scaffold as sc
from learning import LearningCircuit


class ScaffoldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = LearningCircuit()
        cls.enc = sc.Encoder(cls.c, 'fly_measured', 7)

    def test_codes_are_sparse_and_prediction_biases_them(self):
        x = np.random.default_rng(0).random((3, 64))
        z = self.enc.code(x)
        self.assertTrue(np.all(z.sum(1) == self.enc.k))
        y = np.random.default_rng(1).random((3, 64))
        plain, assisted = self.enc.code(y), self.enc.code(y, bias=z, beta=5.0)
        overlap = lambda a, b: (a * b).sum(1).mean()
        self.assertGreater(overlap(assisted, z), overlap(plain, z))   # the predicted cells win more often

    def test_one_shot_memory(self):
        x = np.random.default_rng(2).random((5, 64))
        mem = sc.Memory(self.enc.n, 5); mem.learn(self.enc.code(x), np.arange(5))
        self.assertTrue(np.all(mem.scores(self.enc.code(x)).argmax(1) == np.arange(5)))
        self.assertTrue(np.all(np.isinf(sc.Memory(self.enc.n, 3).scores(self.enc.code(x)))))

    def test_shift_moves_one_pixel(self):
        img = np.zeros((1, 64)); img[0, 27] = 1
        moved = sc.shift(img, np.random.default_rng(0))
        self.assertEqual(moved.sum(), 1); self.assertNotEqual(int(moved.argmax()), 27)

    def test_context_solves_long_dependency(self):
        r = sc.sequences(self.c, 7, exposures=(1,), betas=(0.0, 1.0))
        self.assertAlmostEqual(r['fly_measured|0.0'][0], 0.2)
        self.assertEqual(r['fly_measured|1.0'][0], 1.0)
        self.assertEqual(r['markov_order_5'], 1.0)

    def test_reference_frame_arithmetic(self):
        r = sc.arithmetic(self.c, 7, train_sizes=(19,))
        self.assertEqual(r['placed'], 20)
        self.assertEqual(r['addition'], 1.0)
        self.assertEqual(r['comparison'], 1.0)

    def test_objects_need_locations(self):
        r = sc.objects(self.c, 7, trials=30, max_glimpses=5)
        self.assertGreater(r['voting_3_modules'][-1], r['bag_of_features'][-1])
        self.assertEqual(len(r['reference_frame']), 5)

    def test_unknown_task(self):
        with self.assertRaises(ValueError): sc.run('nope', circuit=self.c)


if __name__ == '__main__':
    unittest.main()
