import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pyarrow as pa
import pyarrow.feather as feather

import build_visualmemory as bv


def raw_tables(folder):
    rows = [
        dict(bodyId=1, type='KCg-d', status='Traced', instance='KCg-d_R', somaSide='R', superclass='cb_intrinsic', **{'class': 'Kenyon_Cell'}, assignedOlHex1=None, assignedOlHex2=None),
        dict(bodyId=2, type='KCab-p', status='Traced', instance='KCab-p_R', somaSide='R', superclass='cb_intrinsic', **{'class': 'Kenyon_Cell'}, assignedOlHex1=None, assignedOlHex2=None),
        dict(bodyId=3, type='KCg-m', status='Traced', instance='KCg-m_R', somaSide='R', superclass='cb_intrinsic', **{'class': 'Kenyon_Cell'}, assignedOlHex1=None, assignedOlHex2=None),
        dict(bodyId=10, type='MeVPMe1', status='Traced', instance='a', somaSide='R', superclass='visual_projection', **{'class': 'ME'}, assignedOlHex1=12, assignedOlHex2=20),
        dict(bodyId=11, type='LoVP9', status='Traced', instance='b', somaSide='L', superclass='visual_projection', **{'class': 'LO'}, assignedOlHex1=None, assignedOlHex2=None),
        dict(bodyId=12, type='APL', status='Traced', instance='APL_R', somaSide='R', superclass='cb_intrinsic', **{'class': 'x'}, assignedOlHex1=None, assignedOlHex2=None),
        dict(bodyId=13, type='DA1_lPN', status='Traced', instance='c', somaSide='R', superclass='cb_intrinsic', **{'class': 'ALPN'}, assignedOlHex1=None, assignedOlHex2=None),
        dict(bodyId=14, type='ER4d', status='Traced', instance='ER4d_R', somaSide='R', superclass='cb_intrinsic', **{'class': 'CX'}, assignedOlHex1=None, assignedOlHex2=None),
    ]
    feather.write_feather(pa.Table.from_pylist(rows), folder / 'annotations.feather')
    edges = [(10, 1, 9), (10, 3, 2), (11, 2, 3), (11, 1, 1), (12, 1, 50), (13, 3, 20), (13, 1, 1), (10, 14, 7)]
    table = pa.table({'body_pre': [e[0] for e in edges], 'body_post': [e[1] for e in edges], 'weight': [e[2] for e in edges]})
    feather.write_feather(table, folder / 'weights.feather')


class BuildVisualMemoryTest(unittest.TestCase):
    def test_build_selects_visual_inputs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / 'raw', Path(tmp) / 'out'
            raw.mkdir(); (out / 'data').mkdir(parents=True)
            raw_tables(raw)
            with mock.patch.object(bv, 'BASE', out), contextlib.redirect_stdout(io.StringIO()):
                bv.build(raw)
            d = json.loads((out / 'data/visual-memory.json').read_text())
        # 10 has 9 synapses onto visual KCs; 11 only 4 (below threshold); APL is mushroom body; the PN has 1
        self.assertEqual([n['bodyId'] for n in d['nodes']], [10])
        self.assertEqual(d['nodes'][0]['hex'], [12, 20])
        self.assertEqual(sorted(map(tuple, d['edges'])), [(10, 1, 9), (10, 3, 2)])   # all its edges onto any KC
        self.assertEqual(d['visual_kcs'], [1, 2])

    def test_probe_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp); raw_tables(raw)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer): bv.probe(raw)
        text = buffer.getvalue()
        self.assertIn('visual Kenyon cells: 2', text)
        self.assertIn('ring neurons', text)


if __name__ == '__main__':
    unittest.main()
