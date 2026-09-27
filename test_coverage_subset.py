import json
import pathlib
import tempfile
import unittest

import pyarrow as pa
import pyarrow.feather as feather

import build_coverage_subset as builder


def synthetic(directory):
    """A well-observed core, cells fed mostly from outside, and the outside cells themselves."""
    core, outside, poor = list(range(1, 41)), list(range(100, 160)), list(range(200, 220))
    feather.write_feather(pa.Table.from_pylist(
        [{'bodyId': b, 'type': f'T{b}', 'instance': f'T{b}_R', 'superclass': 'x', 'somaSide': 'R', 'status': 'Traced'}
         for b in core + outside + poor]), directory / 'annotations.feather')
    edges = []
    for i, b in enumerate(core):
        for j in range(6): edges.append({'body_pre': core[(i + j + 1) % len(core)], 'body_post': b, 'weight': 15})
    for i, b in enumerate(poor):
        edges.append({'body_pre': core[i % len(core)], 'body_post': b, 'weight': 3})
        for j in range(5): edges.append({'body_pre': outside[(i * 3 + j) % len(outside)], 'body_post': b, 'weight': 20})
    feather.write_feather(pa.Table.from_pylist(edges), directory / 'weights.feather')
    feather.write_feather(pa.Table.from_pylist(
        [{'body': b, 'post': 90} for b in core] + [{'body': b, 'post': 103} for b in poor]
        + [{'body': b, 'post': 50} for b in outside]), directory / 'body-stats.feather')
    feather.write_feather(pa.Table.from_pylist(
        [{'body': b, 'consensus_nt': 'acetylcholine'} for b in core + outside + poor]), directory / 'neurotransmitters.feather')
    return core, outside, poor


class CoverageTests(unittest.TestCase):
    def test_coverage_counts_only_sources_inside_the_subset(self):
        found = {1: {2: 30, 99: 70}}
        totals = {1: 100}
        self.assertAlmostEqual(builder.coverage_of(found, totals, {1, 2})[1], 0.3)
        self.assertAlmostEqual(builder.coverage_of(found, totals, {1, 2, 99})[1], 1.0)
        self.assertEqual(builder.coverage_of(found, {}, {1, 2}), {})

    def test_build_keeps_the_observed_core_and_drops_the_rest(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            raw = pathlib.Path(raw_dir)
            core, outside, poor = synthetic(raw)
            seed = pathlib.Path(out_dir) / 'seed.json'
            seed.write_text(json.dumps({'nodes': [{'bodyId': b} for b in core + poor]}))
            out_path = pathlib.Path(out_dir) / 'coverage.json'
            old_seeds, old_base = builder.SEEDS, builder.BASE
            builder.SEEDS, builder.BASE = [seed.name], pathlib.Path(out_dir)
            try:
                result = builder.build(raw, target=0.5, budget=200, out_path=out_path)
            finally:
                builder.SEEDS, builder.BASE = old_seeds, old_base
            kept = {n['bodyId'] for n in result['nodes']}
            self.assertEqual(kept, set(core))                       # the observed core survives
            self.assertFalse(kept & set(poor))                      # cells fed from outside do not
            self.assertFalse(kept & set(outside))                   # nor do the outside cells themselves
            self.assertGreaterEqual(result['summary']['min_coverage'], 0.5)
            self.assertTrue(all(n['nt'] == 'acetylcholine' for n in result['nodes']))
            self.assertTrue(all(a in kept and b in kept for a, b, _ in result['edges']))
            json.loads(json.dumps(result, allow_nan=False))
            self.assertEqual(json.loads(out_path.read_text())['summary'], result['summary'])

    def test_target_above_what_the_data_supports_is_reported(self):
        with tempfile.TemporaryDirectory() as raw_dir, tempfile.TemporaryDirectory() as out_dir:
            raw = pathlib.Path(raw_dir)
            core, outside, poor = synthetic(raw)
            seed = pathlib.Path(out_dir) / 'seed.json'
            seed.write_text(json.dumps({'nodes': [{'bodyId': b} for b in poor]}))
            old_seeds, old_base = builder.SEEDS, builder.BASE
            builder.SEEDS, builder.BASE = [seed.name], pathlib.Path(out_dir)
            try:
                with self.assertRaises(SystemExit) as raised:
                    builder.build(raw, target=0.95, budget=5, out_path=pathlib.Path(out_dir) / 'x.json')
                self.assertIn('coverage', str(raised.exception))
            finally:
                builder.SEEDS, builder.BASE = old_seeds, old_base


if __name__ == '__main__':
    unittest.main()
