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
        # optic lobe upstream of input 10: two columnar ON/OFF cells with coordinates, and a lobula cell without
        dict(bodyId=20, type='Mi1', status='Traced', instance='Mi1', somaSide='R', superclass='ol_intrinsic', **{'class': None}, assignedOlHex1=5, assignedOlHex2=5),
        dict(bodyId=21, type='Tm2', status='Traced', instance='Tm2', somaSide='R', superclass='ol_intrinsic', **{'class': None}, assignedOlHex1=25, assignedOlHex2=25),
        dict(bodyId=22, type='LC10a', status='Traced', instance='LC10', somaSide='R', superclass='visual_projection', **{'class': None}, assignedOlHex1=None, assignedOlHex2=None),
        dict(bodyId=23, type='Tm1', status='Traced', instance='Tm1', somaSide='R', superclass='ol_intrinsic', **{'class': None}, assignedOlHex1=25, assignedOlHex2=5),
        dict(bodyId=24, type='SMP001', status='Traced', instance='SMP', somaSide='R', superclass='cb_intrinsic', **{'class': None}, assignedOlHex1=None, assignedOlHex2=None),
    ]
    feather.write_feather(pa.Table.from_pylist(rows), folder / 'annotations.feather')
    edges = [(10, 1, 9), (10, 3, 2), (11, 2, 3), (11, 1, 1), (12, 1, 50), (13, 3, 20), (13, 1, 1), (10, 14, 7),
             (20, 10, 30), (21, 10, 10), (22, 10, 20), (24, 10, 40), (23, 22, 12)]
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

    def test_upstream_measures(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / 'raw', Path(tmp) / 'out'
            raw.mkdir(); (out / 'data').mkdir(parents=True)
            raw_tables(raw)
            with mock.patch.object(bv, 'BASE', out), contextlib.redirect_stdout(io.StringIO()):
                bv.build(raw)
            n = json.loads((out / 'data/visual-memory.json').read_text())['nodes'][0]
        # 60 of 100 annotated input synapses come from optic-lobe cells
        self.assertAlmostEqual(n['visual_share'], 0.6)
        mix = n['channel_mix']
        self.assertAlmostEqual(sum(mix.values()), 1.0, places=3)
        # Mi1 30 ON; Tm2 10 OFF; LC10 20 split half form, half its derived input (Tm1, OFF)
        self.assertAlmostEqual(mix['ON'], 30 / 60, places=3)
        self.assertAlmostEqual(mix['OFF'], 20 / 60, places=3)
        self.assertAlmostEqual(mix['form'], 10 / 60, places=3)
        self.assertEqual(n['rf']['side'], 'R')
        self.assertEqual(n['rf']['weight'], 60)
        self.assertGreater(n['rf']['spread'], 0)

    def test_channel_of(self):
        self.assertEqual(bv.channel_of('T4a', 'ol_intrinsic'), 'ON')
        self.assertEqual(bv.channel_of('Tm1', 'ol_intrinsic'), 'OFF')
        self.assertEqual(bv.channel_of('Tm16', 'ol_intrinsic'), 'form')      # not Tm1
        self.assertEqual(bv.channel_of('Dm8a', 'ol_intrinsic'), 'luminance')
        self.assertIsNone(bv.channel_of('SMP001', 'cb_intrinsic'))

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
