import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np

import build_regions

# Neuron 1 listens in A and talks in B; neuron 2 listens half in A, half in B and talks in C.
NEURONS = {1: {'A': {'post': 10, 'downstream': 0}, 'B': {'post': 0, 'downstream': 40}, 'SUPER': {'post': 10, 'downstream': 40}},
           2: {'A': {'post': 5}, 'B': {'post': 5}, 'C': {'downstream': 20}},
           3: {'A': {'post': 7}},                                   # no outputs: contributes no flow
           4: {'C': {'post': 3, 'downstream': 6}}}


def fake_backend(page_calls):
    def query(cypher):
        last = int(re.search(r'n.bodyId > (-?\d+)', cypher).group(1)); limit = int(re.search(r'LIMIT (\d+)', cypher).group(1))
        ids = [b for b in sorted(NEURONS) if b > last][:limit]; page_calls.append(ids)
        return {'bodyId': ids, 'roiInfo': [json.dumps(NEURONS[b]) for b in ids]}
    return {'query': query, 'primary_rois': lambda: ['A', 'B', 'C', 'EMPTY'], 'hierarchy': lambda: {'SUPER': {'A': {}, 'B': {}}}}


class RegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = Path(cls.tmp.name)
        (d / 'loco.json').write_text(json.dumps({'nodes': [{'bodyId': 1}, {'bodyId': 99}]}))
        (d / 'learn.json').write_text(json.dumps({'nodes': [{'bodyId': 2}]}))
        cls.calls = []
        build_regions.PAGE, old = 2, build_regions.PAGE        # force pagination
        try: cls.r = build_regions.build(fake_backend(cls.calls), d / 'regions.json', {'locomotion': d / 'loco.json', 'learning': d / 'learn.json'})
        finally: build_regions.PAGE = old

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def F(self, a, b): r = self.r['rois']; return self.r['flow'][r.index(a)][r.index(b)]

    def test_pagination_reads_every_neuron_once(self):
        self.assertEqual(sum(len(c) for c in self.calls), 4)
        self.assertEqual(self.r['traced_neurons'], 4)

    def test_flow_routes_output_by_input_fraction(self):
        self.assertAlmostEqual(self.F('A', 'B'), 40)            # neuron 1: all input in A, 40 outputs in B
        self.assertAlmostEqual(self.F('A', 'C'), 10)            # neuron 2: half its input in A, 20 outputs in C
        self.assertAlmostEqual(self.F('B', 'C'), 10)
        self.assertAlmostEqual(self.F('C', 'C'), 6)             # neuron 4
        self.assertAlmostEqual(self.F('B', 'B'), 0)

    def test_neuron_counts(self):
        r = self.r['rois']; N = lambda a, b: self.r['neurons'][r.index(a)][r.index(b)]
        self.assertEqual(N('A', 'B'), 1)     # neuron 1
        self.assertEqual(N('A', 'C'), 1)     # neuron 2
        self.assertEqual(N('C', 'C'), 1)     # neuron 4
        self.assertEqual(N('B', 'A'), 0)     # neuron 3 has no outputs

    def test_super_regions_and_empty_regions_excluded(self):
        self.assertNotIn('SUPER', self.r['rois'])               # would double count
        self.assertNotIn('EMPTY', self.r['rois'])

    def test_totals_and_lab_profiles(self):
        t = {x['roi']: x for x in self.r['roi_totals']}
        self.assertEqual(t['A']['input_connections'], 22); self.assertEqual(t['A']['neurons_with_input'], 3)
        p = self.r['profiles']
        self.assertEqual(set(p), {'1', '2'}); self.assertEqual(self.r['lab_cells_total'], 3)
        self.assertEqual(p['2']['input'][0]['fraction'], 0.5)
        self.assertEqual(p['1']['output'], [{'roi': 'B', 'count': 40, 'fraction': 1.0}])


if __name__ == '__main__':
    unittest.main()
