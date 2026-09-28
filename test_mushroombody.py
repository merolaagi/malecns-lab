import json
import pathlib
import tempfile
import unittest

import pyarrow as pa
import pyarrow.feather as feather

import build_mushroombody as mb

# Instance strings in the formats found in the lab's own learning subset.
CASES = [
    ("MBON01(y5B'2a)_R", 'MBON', (['γ5', "β'2a"], [])),
    ('MBON06(B1>a)_R', 'MBON', (['β1'], ['α lobe'])),
    ('MBON05(y4>y1y2)_R', 'MBON', (['γ4'], ['γ1', 'γ2'])),
    ('MBON11(y1pedc>a/B)_L', 'MBON', (['γ1pedc'], ['α lobe', 'β lobe'])),
    ("MBON12(y2a'1)_R", 'MBON', (['γ2', "α'1"], [])),
    ('PAM10(B1)_R', 'DAN', ([], ['β1'])),
    ('PAM07(y4<y1y2)_L', 'DAN', (['γ1', 'γ2'], ['γ4'])),
    ('PPL101(y1pedc)_R', 'DAN', ([], ['γ1pedc'])),
    ('APL_R', 'APL', ([], [])),
    ("MBON26(b'2d)_R", 'MBON', (["β'2d"], [])),                  # lowercase b for beta
    ('PPL101(y1ped)_R', 'DAN', ([], ['γ1pedc'])),                 # 'ped' abbreviates the pedunculus
    ("PPL105(a'2a2)_R", 'DAN', ([], ["α'2", 'α2'])),
    ('PPL107_R', 'DAN', ([], [])),                                # no compartment in the name
]


class ParsingTests(unittest.TestCase):
    def test_instances(self):
        for instance, role, expected in CASES:
            self.assertEqual(mb.parse_instance(instance, role), expected, instance)

    def test_adjacent_compartments_are_not_merged(self):
        self.assertEqual(mb.compartments('y1y2y3'), ['γ1', 'γ2', 'γ3'])

    def test_roles_and_classes(self):
        self.assertEqual(mb.role_of('PPL101'), 'DAN')
        self.assertEqual(mb.role_of('PPL201'), 'DAN_calyx')          # calyx and lateral horn, not lobe teachers
        self.assertEqual(mb.role_of('PAM07'), 'DAN')
        self.assertEqual(mb.role_of('APL'), 'APL')
        self.assertIsNone(mb.role_of('SMP001'))
        self.assertEqual(mb.kc_class("KCa'b'-ap1"), "α'/β'")
        self.assertEqual(mb.kc_class('KCab-s'), 'α/β')
        self.assertEqual(mb.kc_class('KCg-m'), 'γ')


class InferenceTests(unittest.TestCase):
    def test_unnamed_teacher_is_placed_by_kenyon_cell_overlap(self):
        node = lambda body, role, cin=(), cout=(): {'bodyId': body, 'role': role, 'compartments_in': list(cin), 'compartments_out': list(cout)}
        nodes = {1: node(1, 'KC'), 2: node(2, 'KC'), 3: node(3, 'KC'), 4: node(4, 'KC'),
                 10: node(10, 'MBON', ['γ1pedc']), 11: node(11, 'MBON', ['α3']),
                 20: node(20, 'DAN'), 21: node(21, 'DAN_calyx')}
        edges = [[1, 10, 5], [2, 10, 5], [3, 11, 5], [4, 11, 5],     # KCs 1-2 feed gamma1pedc, 3-4 feed alpha3
                 [20, 1, 4], [20, 2, 4], [21, 3, 4]]                 # the unnamed teacher targets KCs 1-2
        mb.infer_compartments(nodes, edges)
        self.assertEqual(nodes[20]['compartments_out'], ['γ1pedc'])
        self.assertTrue(nodes[20]['compartment_source'].startswith('inferred'))
        self.assertEqual(nodes[10]['compartment_source'], 'name')
        self.assertEqual(nodes[21]['compartments_out'], [])          # calyx dopamine cells are never placed


class BuildTests(unittest.TestCase):
    def test_end_to_end_on_synthetic_tables(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            raw = pathlib.Path(raw_dir)
            rows = [{'bodyId': 1, 'type': 'KCg-m', 'instance': 'KCg-m_R', 'status': 'Traced', 'somaSide': 'R'},
                    {'bodyId': 2, 'type': 'KCg-m', 'instance': 'KCg-m_R', 'status': 'Traced', 'somaSide': 'R'},
                    {'bodyId': 10, 'type': 'MBON11', 'instance': 'MBON11(y1pedc>a/B)_R', 'status': 'Traced', 'somaSide': 'R'},
                    {'bodyId': 20, 'type': 'PPL101', 'instance': 'PPL101(y1pedc)_R', 'status': 'Traced', 'somaSide': 'R'},
                    {'bodyId': 21, 'type': 'PAM10', 'instance': 'PAM10(B1)_R', 'status': 'Traced', 'somaSide': 'R'},
                    {'bodyId': 30, 'type': 'APL', 'instance': 'APL_R', 'status': 'Traced', 'somaSide': 'R'},
                    {'bodyId': 99, 'type': 'SMP001', 'instance': 'SMP001_R', 'status': 'Traced', 'somaSide': 'R'}]
            feather.write_feather(pa.Table.from_pylist(rows), raw / 'annotations.feather')
            edges = [(1, 10, 20), (2, 10, 15), (20, 1, 8), (10, 20, 6), (1, 2, 30), (30, 1, 9), (10, 99, 40)]
            feather.write_feather(pa.Table.from_pylist([{'body_pre': a, 'body_post': b, 'weight': w} for a, b, w in edges]),
                                  raw / 'weights.feather')
            feather.write_feather(pa.Table.from_pylist([{'body': b, 'consensus_nt': 'acetylcholine'} for b in (1, 2, 10)]
                                                       + [{'body': 20, 'consensus_nt': 'dopamine'}]), raw / 'neurotransmitters.feather')
            old = mb.BASE; mb.BASE = pathlib.Path(out_dir); (mb.BASE / 'data').mkdir()
            try:
                mb.build(raw)
                out = json.loads((mb.BASE / 'data/mushroom-body.json').read_text())
            finally:
                mb.BASE = old
            self.assertEqual(out['roles']['KC'], 2); self.assertEqual(out['roles']['DAN'], 2)
            self.assertEqual(out['roles']['APL'], 1); self.assertEqual(out['roles']['MBON_target'], 1)
            self.assertEqual(out['kc_to_kc_synapses'], 30)            # summarised, not stored
            self.assertNotIn([1, 2, 30], out['edges'])
            self.assertIn('MBON->DAN', out['edge_kinds'])             # the loop the old subset lacked
            self.assertIn('MBON->MBON_target', out['edge_kinds'])
            self.assertEqual(out['compartments']['γ1pedc'], {'teachers': {'PPL101': 1}, 'readers': {'MBON11': 1}})
            nodes = {n['bodyId']: n for n in out['nodes']}
            self.assertEqual(nodes[20]['nt'], 'dopamine')
            self.assertEqual(nodes[1]['kc_class'], 'γ')


if __name__ == '__main__':
    unittest.main()
