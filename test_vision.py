import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

import build_vision

# bodyId: (type, side, superclass)
BODIES = {1: ('T4a', 'L', 'ol_intrinsic'), 2: ('T4a', 'L', 'ol_intrinsic'), 3: ('T5a', 'L', 'ol_intrinsic'),
          10: ('LPi', 'L', 'ol_intrinsic'), 20: ('X', 'R', 'cb_intrinsic'), 21: ('Y', 'R', 'cb_intrinsic'),
          30: ('Z', 'L', 'cb_intrinsic'), 100: ('DNa02', 'L', 'descending_neuron'), 101: ('DNa01', 'R', 'descending_neuron'),
          999: (None, None, None)}
EDGES = [(1, 10, 10), (2, 10, 10), (999, 10, 20),       # LPi_L input 40: T4a_L gives 20 -> 0.5
         (10, 100, 30), (30, 100, 10),                  # DNa02_L input 40: LPi_L gives 30 -> 0.75
         (3, 20, 5), (20, 21, 8), (999, 21, 8),         # X_R <- T5a_L 1.0; Y_R <- X_R 0.5
         (21, 101, 4), (999, 101, 4)]                   # DNa01_R <- Y_R 0.5
NT = {1: 'acetylcholine', 2: 'acetylcholine', 3: 'acetylcholine', 10: 'gaba', 20: 'glutamate', 21: 'acetylcholine',
      30: 'acetylcholine', 100: 'acetylcholine', 101: 'acetylcholine'}


def write_tables(d):
    ann = [{'bodyId': b, 'type': t, 'instance': t, 'superclass': s, 'subclass': None, 'somaSide': side, 'rootSide': side,
            'status': 'Traced'} for b, (t, side, s) in BODIES.items()]
    feather.write_feather(pa.Table.from_pylist(ann), d / 'annotations.feather')
    feather.write_feather(pa.table({'body_pre': [e[0] for e in EDGES], 'body_post': [e[1] for e in EDGES], 'weight': [e[2] for e in EDGES]}), d / 'weights.feather')
    feather.write_feather(pa.Table.from_pylist([{'body': b, 'consensus_nt': n, 'predicted_nt_confidence': .9} for b, n in NT.items()]), d / 'neurotransmitters.feather')


class VisionBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = Path(cls.tmp.name)
        write_tables(d)
        cls.r = build_vision.build(d, d / 'out.json', keep=10)
        cls.g = {g['id']: g for g in cls.r['groups']}
        cls.name = {g['id']: (g['type'], g['side']) for g in cls.r['groups']}

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def infl(self, out, dn):
        return next(i for i in self.r['influence'] if self.name[i['output']] == out and self.name[i['dn']] == dn)

    def test_two_hop_influence_is_product_of_input_fractions(self):
        i = self.infl(('T4a', 'L'), ('DNa02', 'L'))
        self.assertAlmostEqual(i['two_hop'], 0.5 * 0.75)
        self.assertEqual(i['direct'], 0)

    def test_three_hop_counts_untyped_inputs_in_denominator(self):
        i = self.infl(('T5a', 'L'), ('DNa01', 'R'))
        self.assertAlmostEqual(i['three_hop'], 1.0 * 0.5 * 0.5)
        self.assertAlmostEqual(i['total'], 0.25)

    def test_unconnected_routes_are_zero(self):
        self.assertEqual(self.infl(('T4a', 'L'), ('DNa01', 'R'))['total'], 0)

    def test_intermediates_ranked_and_signed(self):
        mids = [g for g in self.r['groups'] if g['role'] == 'intermediate']
        names = [(g['type'], g['side']) for g in mids]
        self.assertIn(('LPi', 'L'), names); self.assertIn(('X', 'R'), names); self.assertIn(('Y', 'R'), names)
        self.assertNotIn(('Z', 'L'), names)  # feeds a DN but receives nothing from vision
        lpi = next(g for g in mids if g['type'] == 'LPi')
        self.assertEqual((lpi['nt'], lpi['sign']), ('gaba', -1))

    def test_edges_are_measured_synapse_counts(self):
        e = {(self.name[x['pre']], self.name[x['post']]): x for x in self.r['edges']}
        self.assertEqual(e[(('T4a', 'L'), ('LPi', 'L'))]['synapses'], 20)
        self.assertAlmostEqual(e[(('LPi', 'L'), ('DNa02', 'L'))]['input_fraction'], 0.75)

    def test_output_file_and_provenance(self):
        saved = json.loads((Path(self.tmp.name) / 'out.json').read_text())
        self.assertEqual(saved['dataset'], 'male-cns:v1.0')
        self.assertEqual(set(saved['files']), {'annotations', 'weights', 'neurotransmitters'})


if __name__ == '__main__':
    unittest.main()
