import unittest

import numpy as np

import objects
from tests_support.mbsynth import synthetic


def small_mb():
    data = synthetic(seed=1, kcs=200, pns=30)
    for n in data['nodes']:                      # make a third of the Kenyon cells visual types
        if n['role'] == 'KC' and n['bodyId'] % 3 == 0:
            n['type'] = 'KCg-d' if n['kc_class'] == 'γ' else 'KCab-p'
    return data


class ObjectMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mb = small_mb()
        cls._inputs = objects.visual_inputs
        objects.visual_inputs = lambda mb: (*objects.synthetic_inputs(mb, cells=40, types=8), 'synthetic')

    @classmethod
    def tearDownClass(cls):
        objects.visual_inputs = cls._inputs

    def test_eye_has_flyvis_layout(self):
        eye = objects.EYE()
        self.assertEqual(eye.n, 721)
        self.assertAlmostEqual(np.sort(np.hypot(*(eye.xy - eye.xy[0]).T))[1], 5.0, places=6)

    def test_classes_render_differently(self):
        eye, view = objects.EYE(), dict(light=1, light_az=0, distance=12, rotation=90, gaze=(0, 0))
        images = {c: eye.photoreceptors(objects.spec(c), view, noise=0) for c in objects.CLASSES}
        for a in images:
            for b in images:
                if a < b: self.assertGreater(np.abs(images[a] - images[b]).sum(), 0.5, (a, b))

    def test_adaptation_removes_light_level_and_raw_does_not(self):
        eye, s = objects.EYE(), objects.spec('paper cup')
        base = dict(light_az=0, distance=12, rotation=0, gaze=(0, 0))
        dim, bright = (eye.photoreceptors(s, dict(base, light=l), noise=0) for l in (0.5, 2.0))
        adapted = [eye.lamina(x, True).ravel() for x in (dim, bright)]
        raw = [eye.lamina(x, False).ravel() for x in (dim, bright)]
        adapted_change, raw_change = np.abs(adapted[0] - adapted[1]).max(), np.abs(raw[0] - raw[1]).max()
        self.assertLess(adapted_change, 0.1)            # not exactly zero: the semi-saturation constant
        self.assertGreater(raw_change, 0.5)
        self.assertLess(adapted_change * 10, raw_change)

    def test_morph_endpoints_are_the_classes(self):
        a = objects.spec('paper cup', morph_to='bottle', morph=1.0)
        np.testing.assert_allclose(a['radius'], objects.CLASSES['bottle']['radius'])
        self.assertEqual(a['height'], objects.CLASSES['bottle']['height'])

    def test_mug_handle_depends_on_rotation(self):
        eye, s = objects.EYE(), objects.spec('mug')
        base = dict(light=1, light_az=0, distance=10, gaze=(0, 0))
        front, side = (eye.photoreceptors(s, dict(base, rotation=r), noise=0) for r in (0, 90))
        self.assertGreater(np.abs(front - side).sum(), 0.1)
        cup = objects.spec('paper cup')
        a, b = (eye.photoreceptors(cup, dict(base, rotation=r), noise=0) for r in (0, 90))
        np.testing.assert_allclose(a, b)

    def test_rewiring_keeps_weights_and_counts(self):
        nodes, edges = objects.synthetic_inputs(self.mb, cells=40, types=8)
        rng = np.random.default_rng(0)
        visual = {n['bodyId'] for n in self.mb['nodes'] if n['role'] == 'KC' and str(n['type']).startswith(objects.VISUAL_KC_TYPES)}
        for condition in ('shuffled', 'random', 'olfactory_kcs'):
            out = objects.rewire(edges, condition, self.mb['nodes'], rng)
            self.assertEqual(len(out), len(edges))
            self.assertEqual(sorted(w for *_, w in out), sorted(w for *_, w in edges))
        moved = objects.rewire(edges, 'olfactory_kcs', self.mb['nodes'], rng)
        self.assertFalse({b for _, b, _ in moved} & visual)
        with self.assertRaises(ValueError): objects.rewire(edges, 'nope', self.mb['nodes'], rng)

    def test_codes_are_sparse_and_only_in_receivers(self):
        lab = objects.Lab(3, 'intact', 0.1, 0.01, self.mb)
        name, _, vpn = lab.sense(objects.spec('can'), dict(light=1, light_az=0, distance=12, rotation=0, gaze=(0, 0)))
        code = lab.mb.kenyon(name)
        self.assertEqual(int(code.sum()), max(1, round(0.1 * len(lab.mb.receivers))))
        self.assertTrue(set(np.flatnonzero(code)) <= set(lab.mb.receivers))
        self.assertTrue(np.all((vpn >= 0) & (vpn <= 1)))

    def test_run_reports_every_test_set(self):
        r = objects.run(mb_data=self.mb, exemplars=3, glimpses=2, train_views=4)
        self.assertEqual(r['source'], 'synthetic')
        self.assertEqual(set(r['test_sets']), set(objects.TEST_SETS))
        for v in r['test_sets'].values():
            self.assertTrue(0 <= v['auc_one_look'] <= 1)
            self.assertEqual(len(v['pairwise_auc']), len(objects.CLASSES) - 1)
        self.assertAlmostEqual(r['morph']['rows'][0]['relative'], 1.0)
        self.assertEqual(len(r['glimpses']), 2)
        self.assertEqual(len(r['similarity']['kc']), len(objects.CLASSES))

    def test_reward_raises_response_to_trained_object(self):
        lab = objects.Lab(5, 'intact', 0.1, 0.0, self.mb)
        view = dict(light=1, light_az=0, distance=12, rotation=0, gaze=(0, 0))
        before = lab.mb.score(lab.sense(objects.spec('paper cup'), view)[0])
        objects.train(lab, objects.validate({'train_views': 6}))
        after = lab.mb.score(lab.sense(objects.spec('paper cup'), view)[0])
        self.assertAlmostEqual(before, 0.0, places=9)
        self.assertGreater(after, 0)

    def test_differential_protocol_and_validation(self):
        r = objects.run(mb_data=self.mb, exemplars=2, glimpses=1, train_views=3, protocol='differential', contrast='mug')
        self.assertIn('punishment', r['protocol_text'])
        for bad in ({'condition': 'x'}, {'trained': 'spoon'}, {'sparsity': 0.9}, {'protocol': 'x'}, {'glimpses': 0}):
            with self.assertRaises(ValueError): objects.validate(bad)
        with self.assertRaises(ValueError):
            objects.run(mb_data=self.mb, protocol='differential', contrast='paper cup', train_views=1, exemplars=2)

    def test_measured_features_drive_projection(self):
        eye = objects.EYE()
        nodes = [{'type': 'a', 'somaSide': 'R', 'visual_share': 1.0, 'channel_mix': {'ON': 1, 'OFF': 0, 'luminance': 0, 'form': 0},
                  'rf': {'x': 0, 'y': 0, 'spread': 0.05, 'side': 'R'}},
                 {'type': 'b', 'somaSide': 'R', 'visual_share': 1.0, 'channel_mix': {'ON': 0, 'OFF': 1, 'luminance': 0, 'form': 0},
                  'rf': {'x': 0, 'y': 0, 'spread': 0.05, 'side': 'R'}},
                 {'type': 'c', 'somaSide': 'R', 'visual_share': 0.0, 'channel_mix': {'ON': 1, 'OFF': 0, 'luminance': 0, 'form': 0},
                  'rf': {'x': 0, 'y': 0, 'spread': 0.05, 'side': 'R'}}]
        p = objects.Projection(nodes, eye)
        self.assertEqual((p.measured_features, p.measured_positions), (3, 3))
        view = dict(light=1, light_az=0, distance=8, rotation=0, gaze=(0, 0))
        bright = p.respond(eye.lamina(eye.photoreceptors(objects.spec('paper cup'), view, noise=0)))   # pale cup on grey
        self.assertGreater(bright[0], 0.1); self.assertLess(bright[1], 0.02); self.assertEqual(bright[2], 0)
        assumed = objects.Projection(nodes, eye, use_measured=False)
        self.assertEqual(assumed.measured_features, 0); self.assertTrue(np.all(assumed.gain == 1))

    def test_view(self):
        v = objects.view(mb_data=self.mb, cls='mug', light=0.6, distance=10)
        self.assertEqual(len(v['columns']), 721)
        self.assertEqual(len(v['image']['values']), 110 * 110)
        with self.assertRaises(ValueError): objects.view(mb_data=self.mb, cls='spoon')
        with self.assertRaises(ValueError): objects.view(mb_data=self.mb, bogus=1)


if __name__ == '__main__':
    unittest.main()
