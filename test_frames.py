import unittest

import numpy as np

import frames as fr
from learning import LearningCircuit


class AlgebraTest(unittest.TestCase):
    def shape(self, triples, n): return fr.Frame(triples, n).shape

    def test_line_loop_grid(self):
        self.assertEqual(self.shape([(a, 'n', a + 1) for a in range(9)], 10), 'line')
        self.assertEqual(self.shape([(a, 'n', (a + 1) % 7) for a in range(7)], 7), 'loop of 7')
        grid = [(r * 3 + c, 'e', r * 3 + c + 1) for r in range(3) for c in range(2)] + [(r * 3 + c, 'u', (r + 1) * 3 + c) for r in range(2) for c in range(3)]
        self.assertEqual(self.shape(grid, 9), 'grid')

    def test_dependent_relation_collapses_to_a_line(self):
        t = [(a, 'one', a + 1) for a in range(11)] + [(a, 'three', a + 3) for a in range(9)]
        f = fr.Frame(t, 12)
        self.assertEqual(f.shape, 'line'); self.assertEqual(f.axes, 1)
        self.assertEqual(f.move(2, {'three': 2, 'one': -1}), 7)

    def test_torus_and_cylinder(self):
        n, m = 4, 6
        t = [(i * m + j, 'a', ((i + 1) % n) * m + j) for i in range(n) for j in range(m)] + [(i * m + j, 'b', i * m + (j + 1) % m) for i in range(n) for j in range(m)]
        f = fr.Frame(t, n * m)
        self.assertEqual((f.axes, np.prod(f.loops)), (0, 24)); self.assertEqual(f.collisions, 0)
        self.assertEqual(f.move(0, {'a': 5, 'b': -7}), 1 * m + 5)
        self.assertEqual(fr.world('week_by_shift')[2], fr.Frame(fr.world('week_by_shift')[1], 28).shape)

    def test_tree_has_no_consistent_frame_and_pieces_are_reported(self):
        n, T, expected, *_ = fr.world('family_tree')
        self.assertEqual(fr.Frame(T, n).shape, 'no consistent frame')
        self.assertEqual(fr.Frame([(0, 'n', 1), (2, 'n', 3)], 4).shape, 'pieces')

    def test_moves_fill_in_missing_relations(self):
        n, T, _, truth, _ = fr.world('room')
        held = [t for t in T if t == (6, 'east', 7)]; train = [t for t in T if t != (6, 'east', 7)]
        f = fr.Frame(train, n)
        self.assertEqual(f.move(6, {'east': 1}), 7)
        self.assertIsNone(fr.Chain(train).move(6, {'east': 1}))
        self.assertEqual(fr.Chain(train).move(6, {'east': 1, 'north': 1}), 12)       # the lookup can route around, in some order
        self.assertIsNone(f.move(4, {'east': 1}))                                    # off the edge: nothing stored there

    def test_diagonalise_identity(self):
        d, Q = fr.diagonalise([(3, -1)], 2)
        self.assertEqual(sorted(d), [0, 1]); self.assertEqual(abs(Q[0][0] * Q[1][1] - Q[0][1] * Q[1][0]), 1)


class TrialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.c = LearningCircuit()

    def test_worlds(self):
        for w in ('numbers', 'clock'):
            r = fr.trial(self.c, w, 7, net=False)
            self.assertTrue(r['correct_shape'], w); self.assertEqual(r['frame']['multi_step'], 1.0)
        r = fr.trial(self.c, 'room', 7, fraction=0.85, net=False)
        self.assertTrue(r['correct_shape'])
        self.assertGreater(r['frame']['held_out_relations'], 0.7); self.assertEqual(r['chain_lookup']['held_out_relations'], 0.0)

    def test_one_wrong_relation_is_detected(self):
        self.assertFalse(fr.trial(self.c, 'room', 7, corrupt=1, net=False)['correct_shape'])


if __name__ == '__main__':
    unittest.main()
