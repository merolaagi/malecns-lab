import json
import unittest

import memory as mem
from tests_support.mbsynth import synthetic


class MemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.data = synthetic()

    def curve(self, reinforcer, spaced, **kw):
        return [c['preference'] for c in mem.forgetting(self.data, reinforcer, spaced, **kw)['curve']]

    def test_valence_comes_from_the_teachers(self):
        mb = mem.MushroomBody(self.data)
        by_compartment = {n['canonical_in'][0]: v for n, v in zip(mb.mbon_nodes, mb.valence)}
        self.assertEqual(by_compartment['γ1'], 1.0)      # punishment-taught: approach-promoting
        self.assertEqual(by_compartment['γ4'], -1.0)         # reward-taught: avoidance-promoting

    def test_punishment_teaches_avoidance_and_reward_approach(self):
        self.assertLess(self.curve('punishment', True)[0], -0.1)
        self.assertGreater(self.curve('reward', True)[0], 0.1)

    def test_spaced_training_lasts_longer_than_massed(self):
        massed, spaced = self.curve('punishment', False), self.curve('punishment', True)
        self.assertAlmostEqual(massed[-1], 0.0, places=3)     # labile trace gone after four days
        self.assertLess(spaced[-1], -0.01)                    # consolidated trace remains
        self.assertTrue(all(abs(a) >= abs(b) - 1e-9 for a, b in zip(spaced, spaced[1:])))

    def test_no_consolidation_behaves_like_massed(self):
        self.assertAlmostEqual(self.curve('punishment', True, condition='no_consolidation')[-1], 0.0, places=3)

    def test_extinction_needs_the_loops_and_recovers(self):
        intact = mem.extinction(self.data)
        self.assertTrue(intact['extinguished'])
        self.assertTrue(intact['spontaneous_recovery'])       # a separate, faster-fading trace
        cut = mem.extinction(self.data, condition='no_loops')
        self.assertFalse(cut['extinguished'])

    def test_ablation_is_compartment_specific(self):
        rows = {r['blocked']: r for r in mem.ablation(self.data, 'punishment')['rows']}
        self.assertLess(abs(rows['γ1']['memory']), abs(rows['none']['memory']))       # a punishment compartment
        self.assertAlmostEqual(rows['γ5']['memory'], rows['none']['memory'], places=6)  # a reward compartment
        self.assertEqual(rows['γ1']['teachers'], ['PPL101'])

    def test_shuffled_wiring_changes_the_memory(self):
        intact = self.curve('punishment', True)[0]
        shuffled = self.curve('punishment', True, condition='shuffled')[0]
        self.assertNotAlmostEqual(intact, shuffled, places=3)

    def test_reproducible_json_and_validation(self):
        result = mem.forgetting(self.data, 'punishment', True)
        self.assertEqual(result, mem.forgetting(self.data, 'punishment', True))
        json.loads(json.dumps(mem.extinction(self.data), allow_nan=False))
        for bad in [{'sparsity': 0.9}, {'fast_rate': 2}, {'loop_gain': -1}]:
            with self.assertRaises(ValueError): mem.forgetting(self.data, **bad)
        with self.assertRaises(ValueError): mem.MushroomBody(self.data, condition='nonsense')

    def test_subcompartments_collapse_to_the_fifteen(self):
        self.assertEqual(mem.canonical('γ1pedc'), 'γ1'); self.assertEqual(mem.canonical('γ1p'), 'γ1')
        self.assertEqual(mem.canonical("β'2m"), "β'2"); self.assertEqual(mem.canonical("α'3a"), "α'3")
        self.assertEqual(mem.canonical('α2s'), 'α2'); self.assertIsNone(mem.canonical('α lobe'))

    def test_traces_only_decay_after_training(self):
        # Regression: summed dopamine drive saturated the traces, so memory appeared to grow after training.
        for reinforcer in ('punishment', 'reward'):
            curve = [abs(v) for v in self.curve(reinforcer, False)]
            self.assertTrue(all(a >= b - 1e-9 for a, b in zip(curve, curve[1:])), reinforcer)

    def test_teaching_signal_is_bounded_by_compartment_size(self):
        data = synthetic()
        for n in [n for n in data['nodes'] if n['role'] == 'DAN'][:2]:          # clone teachers many times
            for k in range(30):
                clone = dict(n, bodyId=n['bodyId'] * 100 + k); data['nodes'].append(clone)
        mb = mem.MushroomBody(data)
        mb.teach('A', 'punishment', mem.validate({}))
        self.assertLessEqual(float((mb.fast + mb.slow).max()), 1.0 + 1e-9)

    def test_missing_file_is_reported(self):
        from pathlib import Path
        with self.assertRaises(ValueError) as e: mem.load(Path('/nonexistent/mushroom-body.json'))
        self.assertIn('build_mushroombody.py', str(e.exception))


if __name__ == '__main__':
    unittest.main()
