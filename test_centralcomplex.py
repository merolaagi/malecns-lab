import json
import tempfile
import unittest
from pathlib import Path

import navigation as nav
from build_centralcomplex import GROUPS, column_of, group_of

# Instance strings taken verbatim from the probe on the real annotations.
CASES = [
    ('EPG(PB08)_R2', 1, 'glomerulus'), ('EPG(PB08)_L4', 11, 'glomerulus'), ('EPG(PB08)_R8', 7, 'glomerulus'),
    ('PEN_a(PB06a)_L3', 10, 'glomerulus'), ('PEN_b(PB06b)_R8', 7, 'glomerulus'), ('PEG(PB07)_R1', 0, 'glomerulus'),
    ('PFNv(PB05)_L4_C3', 11, 'glomerulus'), ('PFNa(PB03)_R9_C1', 0, 'glomerulus'),
    ('PFL2(PB12b)_L3_C7', 10, 'glomerulus'), ('Delta7(PB15)_L3R6_R', 8, 'glomerulus_mean'),
    ('FC2A_C2_L', 2, 'fb_column'), ('hDeltaB_08_C7', 12, 'fb_column'), ('hDeltaH_14_C8', 14, 'fb_column'),
]
UNPARSEABLE = ['ER5_L', 'ExR1_R', 'ER2_c_R', 'PFNp_c', None, '']


class ColumnParsingTests(unittest.TestCase):
    def test_real_instance_strings(self):
        for instance, column, rule in CASES:
            self.assertEqual(column_of(instance), (column, rule), instance)

    def test_type_names_are_not_mistaken_for_glomeruli(self):
        # 'ER5' and 'ExR1' contain R5 and R1; they are type names, not bridge glomeruli.
        for instance in UNPARSEABLE:
            self.assertEqual(column_of(instance), (None, None), instance)

    def test_sides_land_on_opposite_halves_of_the_ring(self):
        right = [column_of(f'EPG(PB08)_R{i}')[0] for i in range(1, 9)]
        left = [column_of(f'EPG(PB08)_L{i}')[0] for i in range(1, 9)]
        self.assertEqual(right, list(range(0, 8)))
        self.assertEqual(left, list(range(8, 16)))

    def test_group_assignment(self):
        self.assertEqual(group_of('EPG'), ('EPG', 'heading'))
        self.assertEqual(group_of('PEN_a(PEN1)'), ('PEN', 'rotate'))
        self.assertEqual(group_of('hDeltaB'), ('hDelta', 'goal'))
        self.assertIsNone(group_of('DNa02'))
        self.assertIsNone(group_of(None))
        self.assertEqual(set(GROUPS) & {'EPG', 'PEN', 'PFL'}, {'EPG', 'PEN', 'PFL'})


class MeasuredOffsetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        nodes, edges = [], []
        for c in range(16):
            nodes.append({'bodyId': 1000 + c, 'group': 'heading', 'column': c, 'type': 'EPG', 'somaSide': 'R' if c < 8 else 'L'})
            nodes.append({'bodyId': 2000 + c, 'group': 'rotate', 'column': c, 'type': 'PEN_a(PEN1)', 'somaSide': 'L'})
            nodes.append({'bodyId': 3000 + c, 'group': 'rotate', 'column': c, 'type': 'PEN_b(PEN2)', 'somaSide': 'R'})
            edges += [[2000 + c, 1000 + (c + 1) % 16, 20], [3000 + c, 1000 + (c - 1) % 16, 20], [1000 + c, 2000 + c, 15]]
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name) / 'cx.json'
        cls.path.write_text(json.dumps({'nodes': nodes, 'edges': edges, 'column_rules': {'glomerulus': len(nodes)}}))

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_offsets_recover_the_wiring(self):
        report = nav.measured_report(self.path)
        self.assertEqual(report['columns_parsed'], 48)
        self.assertAlmostEqual(report['group_pairs']['heading to rotate']['mean_offset_columns'], 0.0, places=2)
        # Pooling the two rotation classes cancels their opposite shifts; the per-type split keeps them.
        self.assertAlmostEqual(report['group_pairs']['rotate to heading']['mean_offset_columns'], 0.0, places=2)
        self.assertEqual(report['rotation_offsets_by_type']['PEN_a(PEN1) L'], 1.0)
        self.assertEqual(report['rotation_offsets_by_type']['PEN_b(PEN2) R'], -1.0)
        self.assertTrue(report['rotation_opposite_by_side'])

    def test_circular_mean_offset(self):
        profile = [0.0] * 16; profile[1] = 1.0
        self.assertAlmostEqual(nav.circular_mean_offset(profile), 1.0, places=6)
        profile = [0.0] * 16; profile[15] = 1.0
        self.assertAlmostEqual(nav.circular_mean_offset(profile), -1.0, places=6)

    def test_too_few_columns_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cx.json'
            path.write_text(json.dumps({'nodes': [{'bodyId': 1, 'group': 'heading', 'column': None}], 'edges': []}))
            with self.assertRaises(ValueError) as e: nav.measured_offsets(path)
            self.assertIn('--probe', str(e.exception))


if __name__ == '__main__':
    unittest.main()
