import json
import shutil
import subprocess
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from server import Handler, NEURONS

BASE = Path(__file__).resolve().parent


class NeuronInputTests(unittest.TestCase):
    def test_kenyon_cell_inputs_match_measured_edges(self):
        d = NEURONS.describe('learning', 57729)
        self.assertEqual([x['synapses'] for x in d['inputs']], [38, 25, 22, 20, 17, 13, 1])
        rule = {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1}
        for x in d['inputs']:   # +1 by assumption, or the consensus transmitter once quality.json has it
            q = NEURONS.quality.get(str(x['bodyId'])) or {}
            c = q.get('consensus_nt')
            self.assertEqual(x['sign'], rule.get(c, 0) if c not in (None, 'unclear', 'unknown') else 1)
        self.assertGreater(len(d['modulatory']), 0)
        self.assertEqual(len(d['number_codes']['codes']['scalar']), 9)

    def test_locomotion_signs_follow_model_rule(self):
        d = NEURONS.describe('locomotion', 10360)
        rule = {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1}
        self.assertTrue(all(x['sign'] == rule.get(x['nt'], 0) for x in d['inputs']))
        total = sum(1 for a, b, w in NEURONS.loco.data['edges'] if b == 10360)
        self.assertEqual(len(d['inputs']), total)

    def test_bad_requests(self):
        for circuit, cid in [('nope', 1), ('learning', 'abc'), ('learning', 1)]:
            with self.assertRaises(ValueError): NEURONS.describe(circuit, cid)


class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=cls.s.serve_forever, daemon=True).start()
        cls.url = f'http://127.0.0.1:{cls.s.server_address[1]}'

    @classmethod
    def tearDownClass(cls): cls.s.shutdown(); cls.s.server_close()

    def test_pages_and_api(self):
        for path in ['/neuron', '/neuron.js', '/neuron-core.js', '/workbench.css', '/numerosity', '/numerosity.js', '/api/numerosity-benchmark']:
            with urllib.request.urlopen(self.url + path) as r: self.assertEqual(r.status, 200)
        with urllib.request.urlopen(self.url + '/api/neuron?circuit=learning&id=random&seed=5') as r:
            self.assertEqual(json.loads(r.read())['cell']['role'], 'KC')
        req = urllib.request.Request(self.url + '/api/numerosity', data=json.dumps({'train_trials': 50}).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as r: self.assertIn('accuracy', json.loads(r.read()))

    def test_unknown_numerosity_parameter_rejected(self):
        req = urllib.request.Request(self.url + '/api/numerosity', data=b'{"evil": 1}', headers={'Content-Type': 'application/json'})
        with self.assertRaises(urllib.error.HTTPError) as e: urllib.request.urlopen(req)
        self.assertEqual(e.exception.code, 400)

    @unittest.skipUnless(shutil.which('node'), 'node not installed')
    def test_neuron_core_logic(self):
        r = subprocess.run(['node', '--test', 'neuron-core.test.js'], cwd=BASE, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == '__main__':
    unittest.main()
