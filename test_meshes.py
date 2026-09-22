import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import gcs
import meshes


def sphere(n=300, r=1000.0):
    th, ph = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, n), indexing='ij')
    v = np.stack([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r * np.cos(th)], -1).reshape(-1, 3).astype('<f4')
    i = np.arange(n * n).reshape(n, n); a, b, c, d = i[:-1, :-1], i[1:, :-1], i[1:, 1:], i[:-1, 1:]
    t = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]).astype('<u4')
    return v, t


class MeshTests(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        v, t = sphere(20)
        v2, t2 = meshes.decode(meshes.encode(v, t))
        np.testing.assert_array_equal(v, v2); np.testing.assert_array_equal(t, t2)

    def test_simplify_reaches_target_and_keeps_shape(self):
        v, t = sphere(300)                                   # ~178k triangles
        sv, st = meshes.simplify(v, t, 5000)
        self.assertLessEqual(len(st), 5000); self.assertGreater(len(st), 500)
        self.assertTrue((st < len(sv)).all())
        self.assertTrue(((st[:, 0] != st[:, 1]) & (st[:, 1] != st[:, 2]) & (st[:, 0] != st[:, 2])).all())
        self.assertEqual(len(np.unique(np.sort(st, axis=1), axis=0)), len(st))
        radii = np.linalg.norm(sv, axis=1)
        self.assertLess(abs(np.median(radii) - 1000) / 1000, 0.03)       # still a sphere of the same size
        np.testing.assert_allclose(sv.max(0), v.max(0), rtol=0.05)

    def test_small_meshes_untouched(self):
        v, t = sphere(20)
        sv, st = meshes.simplify(v, t, 10_000)
        self.assertIs(sv, v); self.assertIs(st, t)

    def test_get_assembles_fragments_simplifies_and_caches(self):
        v, t = sphere(120); half = len(t) // 2
        # Two fragments that each carry all vertices but half the triangles.
        frags = {'a.ngmesh': meshes.encode(v, t[:half]), 'b.ngmesh': meshes.encode(v, t[half:])}
        calls = []
        def fake(path, **kw):
            calls.append(path)
            if path.endswith(':0'): return json.dumps({'fragments': list(frags)}).encode()
            return frags[path.split('/')[-1]]
        with tempfile.TemporaryDirectory() as d:
            old_cache, old_fetch, old_layers = gcs.CACHE, gcs.fetch, dict(meshes.LAYERS)
            gcs.CACHE = Path(d); gcs.fetch = fake; meshes.LAYERS['neuron'] = (meshes.LAYERS['neuron'][0], 2000)
            try:
                body, info = meshes.get('neuron', '10360')
                self.assertEqual(info['triangles_original'], len(t)); self.assertLessEqual(info['triangles'], 2000)
                v2, t2 = meshes.decode(body); self.assertEqual(len(t2), info['triangles'])
                n = len(calls); body2, _ = meshes.get('neuron', 10360)
                self.assertEqual(body, body2); self.assertEqual(len(calls), n)          # served from cache
            finally:
                gcs.CACHE, gcs.fetch = old_cache, old_fetch; meshes.LAYERS.clear(); meshes.LAYERS.update(old_layers)

    def test_input_checks(self):
        for layer, sid in [('em', 1), ('neuron', 'abc'), ('neuron', -1), ('../x', 1)]:
            with self.assertRaises(gcs.GCSError): meshes.get(layer, sid)


if __name__ == '__main__':
    unittest.main()
