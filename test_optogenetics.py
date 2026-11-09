import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import optogenetics as og


class OptogeneticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lab = og.Optogenetics(og.BASE / 'data/circuit.json')

    def test_original_circuit_has_steering_types_only(self):
        self.assertEqual(self.lab.source, 'original')
        self.assertEqual(self.lab.available['DNa02'], 2)
        self.assertEqual(self.lab.available['MDN'], 0)
        with self.assertRaises(ValueError): self.lab.stimulate(experiment='MDN')

    def test_only_target_cells_are_lit_and_dn_mask_is_restored(self):
        before = self.lab.circuit.dn.copy()
        r = self.lab.stimulate(experiment='DNa02', side='right', duration=1.5, seed=2)
        self.assertEqual(r['lit_cells'], 2)
        np.testing.assert_array_equal(self.lab.circuit.dn, before)
        self.assertGreater(r['metrics']['target_hz'], 0)

    def test_one_side_lit_drives_that_side(self):
        right = self.lab.stimulate(experiment='DNa02', side='right', duration=2, seed=3)['metrics']
        left = self.lab.stimulate(experiment='DNa02', side='left', duration=2, seed=3)['metrics']
        self.assertGreater(right['lit_side_share'], 0.6)
        self.assertGreater(left['lit_side_share'], 0.6)

    def test_no_light_no_movement(self):
        m = self.lab.stimulate(experiment='DNa02', side='both', light=0, duration=1.5, seed=1)['metrics']
        self.assertEqual(m['speed_mm_s'], 0.0)

    def test_sham_lights_interneurons_on_the_same_sides(self):
        real = self.lab.targets('DNa02', 'intact', 1); sham = self.lab.targets('DNa02', 'sham', 1)
        self.assertEqual(real.sum(), sham.sum())
        self.assertTrue(np.all(self.lab.circuit.vnc[sham]))
        self.assertEqual(sorted(self.lab.circuit.side[real]), sorted(self.lab.circuit.side[sham]))

    def test_verdicts(self):
        base = dict(speed_mm_s=3.0, front_hz=1.0, hind_hz=1.0, lit_side_share=None, ttm_cells=0, ttm_hz=None, ttm_latency_s=None)
        runs = lambda r, l, b=0: {'right': [base | {'turn_deg': r, 'lit_side_share': 0.8}], 'left': [base | {'turn_deg': l, 'lit_side_share': 0.8}],
                                  'both': [base | {'turn_deg': b}]}
        self.assertEqual(og.verdict('DNa02', runs(-40, 40))['result'], 'match')
        self.assertEqual(og.verdict('DNa02', runs(40, -40))['result'], 'opposite')
        self.assertEqual(og.verdict('DNa02', runs(5, -5))['result'], 'no clear turn')
        self.assertEqual(og.verdict('P9', runs(-40, 40, 0))['result'], 'match')
        r = runs(0, 0); r['both'] = [base | {'turn_deg': 0, 'front_hz': 1.0, 'hind_hz': 3.0}]
        self.assertEqual(og.verdict('MDN', r)['result'], 'proxy match')
        self.assertEqual(og.verdict('Giant fiber', runs(0, 0))['result'], 'not testable')
        r = runs(0, 0); r['both'] = [base | {'turn_deg': 0, 'ttm_cells': 2, 'ttm_hz': 0.0, 'ttm_latency_s': None}]
        self.assertEqual(og.verdict('Giant fiber', r)['result'], 'no TTM activity')
        r['both'] = [base | {'turn_deg': 0, 'ttm_cells': 2, 'ttm_hz': 80.0, 'ttm_latency_s': 0.004}]
        self.assertEqual(og.verdict('Giant fiber', r)['result'], 'match')

    def test_validation(self):
        for bad in ({'experiment': 'x'}, {'side': 'up'}, {'condition': 'x'}, {'light': 9}, {'duration': 0.2}):
            with self.assertRaises(ValueError): og.validate(bad)


try:
    import pyarrow as pa
    import pyarrow.feather as feather
except ImportError:
    pa = None


@unittest.skipIf(pa is None, 'pyarrow not installed')
class BuildOptogeneticsTest(unittest.TestCase):
    def tables(self, folder):
        def row(b, t, sup, sub=None, side='R', nerve=None):
            return dict(bodyId=b, type=t, instance=f'{t}_{side}', superclass=sup, subclass=sub, somaSide=side, rootSide=side,
                        entryNerve=nerve, exitNerve=None, status='Traced', somaLocation=None, synonyms=None, group=None, mancType=None)
        rows = [row(1, 'MDN', 'descending_neuron'), row(2, 'DNp01', 'descending_neuron'), row(3, 'DNa02', 'descending_neuron', side='L'),
                row(10, 'IN01', 'vnc_intrinsic'), row(20, 'Ti flexor MN', 'vnc_motor', 'hl'), row(21, 'TTMn', 'vnc_motor', 'wing'),
                row(30, 'CO', 'vnc_sensory', 'chordotonal organ', nerve='MetaLN'), row(40, 'Other', 'vnc_intrinsic')]
        feather.write_feather(pa.Table.from_pylist(rows), folder / 'annotations.feather')
        edges = [(1, 10, 20), (10, 20, 9), (2, 21, 3), (30, 10, 6), (3, 10, 7), (40, 20, 2)]
        feather.write_feather(pa.table({'body_pre': [e[0] for e in edges], 'body_post': [e[1] for e in edges],
                                        'weight': [e[2] for e in edges]}), folder / 'weights.feather')
        feather.write_feather(pa.Table.from_pylist([dict(body=b, consensus_nt='acetylcholine', predicted_nt_confidence=0.9)
                                                    for b in (1, 2, 3, 10, 20, 21, 30, 40)]), folder / 'neurotransmitters.feather')

    def test_build(self):
        import build_optogenetics as bo
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / 'raw', Path(tmp) / 'out'
            raw.mkdir(); (out / 'data').mkdir(parents=True)
            self.tables(raw)
            with mock.patch.object(bo, 'BASE', out), contextlib.redirect_stdout(io.StringIO()):
                bo.build(raw)
            d = json.loads((out / 'data/optogenetics-circuit.json').read_text())
            ids = {n['bodyId'] for n in d['nodes']}
            self.assertEqual(ids, {1, 2, 3, 10, 20, 21, 30})          # 40 unselected; TTMn kept though not a leg MN
            self.assertTrue(next(n for n in d['nodes'] if n['bodyId'] == 21)['ttm'])
            self.assertEqual(d['found'], {'MDN': 1, 'DNp01': 1, 'DNa02': 1})
            lab = og.Optogenetics(out / 'data/optogenetics-circuit.json')
            self.assertEqual(lab.source, 'optogenetics')
            self.assertEqual(int(lab.ttm.sum()), 1)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer): bo.probe(raw)
            self.assertIn('TTM motor neurons', buffer.getvalue())


if __name__ == '__main__':
    unittest.main()
