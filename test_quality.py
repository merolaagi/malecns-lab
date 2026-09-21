import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather

import build_quality


class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = Path(cls.tmp.name)
        loco = {'nodes': [{'bodyId': 1, 'superclass': 'vnc_motor', 'nt': 'unclear'},
                          {'bodyId': 2, 'superclass': 'vnc_intrinsic', 'nt': 'gaba'}],
                'edges': [[2, 1, 30], [1, 2, 5]]}
        learn = {'nodes': [{'bodyId': 3, 'role': 'PN'}, {'bodyId': 4, 'role': 'KC'}], 'edges': [[3, 4, 12]]}
        (d / 'loco.json').write_text(json.dumps(loco)); (d / 'learn.json').write_text(json.dumps(learn))
        feather.write_feather(pa.table({'body': [1, 2, 3, 4, 99], 'pre': [50, 80, 400, 20, 1], 'post': [120, 60, 10, 40, 1],
                                        'downstream': [20, 70, 300, 30, 1], 'synweight': [0] * 5}), d / 'body-stats.feather')
        rows = {'body': [], 'x': []}
        probs = {k: [] for k in ['acetylcholine', 'gaba', 'glutamate', 'dopamine']}
        def add(body, p, n):
            for _ in range(n):
                rows['body'].append(body); rows['x'].append(0)
                for k, v in zip(probs, p): probs[k].append(v)
        add(1, [0.1, 0.1, 0.8, 0.0], 8); add(1, [0.6, 0.2, 0.2, 0.0], 2)   # unclear aggregate, mostly glutamate
        add(2, [0.2, 0.7, 0.1, 0.0], 5)
        add(3, [0.9, 0.05, 0.05, 0.0], 4)
        add(99, [1, 0, 0, 0], 3)                                              # not in the lab: must be ignored
        table = pa.table({'point_id': list(range(len(rows['body']))), 'body': rows['body'], 'x': rows['x'],
                          **{f'nt_{k}_prob': v for k, v in probs.items()}})
        feather.write_feather(table, d / 'tbar-neurotransmitters.feather')
        cls.r = build_quality.build(d, d / 'q.json', {'locomotion': d / 'loco.json', 'learning': d / 'learn.json'})

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_coverage_is_captured_over_total(self):
        c = self.r['cells']
        self.assertAlmostEqual(c['1']['coverage_in'], 30 / 120)
        self.assertAlmostEqual(c['2']['coverage_out'], 30 / 70, places=3)
        self.assertAlmostEqual(c['4']['coverage_in'], 12 / 40)
        self.assertEqual(c['3']['captured_in'], 0)

    def test_synapse_level_transmitters(self):
        n = self.r['cells']['1']['nt']
        self.assertEqual(n['tbars'], 10)
        self.assertEqual(n['dominant'], 'glutamate'); self.assertAlmostEqual(n['dominant_share'], 0.8)
        self.assertAlmostEqual(n['p_positive'], (8 * .1 + 2 * .6) / 10)
        self.assertAlmostEqual(n['p_negative'], (8 * .9 + 2 * .4) / 10)
        self.assertEqual(self.r['cells']['3']['nt']['dominant'], 'acetylcholine')

    def test_cells_without_tbars_have_no_nt(self):
        self.assertIsNone(self.r['cells']['4']['nt'])

    def test_only_lab_cells_are_reported(self):
        self.assertEqual(set(self.r['cells']), {'1', '2', '3', '4'})

    def test_summary_counts_resolved_unclear_cells(self):
        s = self.r['summary']['locomotion']['vnc_motor']
        self.assertEqual((s['aggregate_unclear'], s['unclear_with_dominant_share_over_0.6']), (1, 1))
        self.assertEqual(self.r['transmitters'], ['acetylcholine', 'gaba', 'glutamate', 'dopamine'])


if __name__ == '__main__':
    unittest.main()
