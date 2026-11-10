import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.feather as feather
except ImportError:
    pa = None


def row(b, t, sup, sub=None, side='L'):
    return dict(bodyId=b, type=t, instance=f'{t}_{side}', superclass=sup, subclass=sub, somaSide=side, rootSide=side,
                entryNerve=None, exitNerve=None, status='Traced', somaLocation=None, synonyms=None, group=None, mancType=None)


@unittest.skipIf(pa is None, 'pyarrow not installed')
class BuildCoordinationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.raw, self.out = Path(self.tmp.name) / 'raw', Path(self.tmp.name) / 'out'
        self.raw.mkdir(); (self.out / 'data').mkdir(parents=True)
        legs = [(1, 'fl', 'L'), (2, 'fl', 'R'), (3, 'ml', 'L'), (4, 'ml', 'R'), (5, 'hl', 'L'), (6, 'hl', 'R')]
        rows = [row(b, f'MN{b}', 'vnc_motor', sub, side) for b, sub, side in legs]
        rows += [row(10 + k, f'IN{10 + k}', 'vnc_intrinsic', side=legs[k][2]) for k in range(6)]   # premotor, one per leg
        rows += [row(20, 'LINK', 'vnc_intrinsic'), row(21, 'ONESEG', 'vnc_intrinsic'), row(31, 'ROUTE', 'vnc_intrinsic'),
                 row(30, 'DNp09', 'descending_neuron')]
        feather.write_feather(pa.Table.from_pylist(rows), self.raw / 'annotations.feather')
        edges = [(10 + k, legs[k][0], 20) for k in range(6)]      # premotor -> own leg's motor neurons
        edges += [(20, 10, 15), (20, 12, 15), (21, 10, 30), (10, 20, 10), (30, 31, 6), (31, 10, 6)]
        feather.write_feather(pa.table({'body_pre': [e[0] for e in edges], 'body_post': [e[1] for e in edges],
                                        'weight': [e[2] for e in edges]}), self.raw / 'weights.feather')
        nts = [dict(body=b['bodyId'], consensus_nt='acetylcholine', predicted_nt_confidence=0.9) for b in rows]
        feather.write_feather(pa.Table.from_pylist(nts), self.raw / 'neurotransmitters.feather')
        base = {'nodes': [dict(r, seed=r['bodyId'] == 30, ttm=False, nt='acetylcholine') for r in rows if r['bodyId'] in (30, 1, 2, 3, 4, 5, 6)],
                'edges': [], 'target_types': ['DNp09'], 'found': {'DNp09': 1}}
        (self.out / 'data/optogenetics-circuit.json').write_text(json.dumps(base))

    def tearDown(self):
        self.tmp.cleanup()

    def test_build(self):
        import build_coordination as bc
        with mock.patch.object(bc, 'BASE', self.out), contextlib.redirect_stdout(io.StringIO()):
            bc.build(self.raw)
        d = json.loads((self.out / 'data/coordination-circuit.json').read_text())
        why = {n['bodyId']: n['added_as'] for n in d['nodes']}
        self.assertEqual({b for b, w in why.items() if w == 'premotor'}, {10, 11, 12, 13, 14, 15})
        self.assertEqual(why[20], 'linking')                 # reaches front (10) and middle (12) premotor cells
        self.assertNotIn(21, why)                            # one segment only
        self.assertEqual(why[31], 'p9_route')                # P9 -> 31 -> front-leg premotor 10
        L = np.array(d['leg_links']['excitatory'])
        self.assertAlmostEqual(L[0, 1], 0.1 * 10 * 15)       # LF premotor -> 20 -> LM premotor
        self.assertEqual(L[0, 0], 0)
        self.assertEqual(next(n for n in d['nodes'] if n['bodyId'] == 10)['premotor_of'], ['LF'])

    def test_probe(self):
        import build_coordination as bc
        buffer = io.StringIO()
        with mock.patch.object(bc, 'BASE', self.out), contextlib.redirect_stdout(buffer):
            bc.probe(self.raw)
        self.assertIn('premotor cells per leg', buffer.getvalue())


if __name__ == '__main__':
    unittest.main()
