import json
import shutil
import subprocess
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from server import Handler

BASE = Path(__file__).resolve().parent

class AtlasRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close()

    def get(self, path):
        with urllib.request.urlopen(f'http://127.0.0.1:{self.port}{path}') as r:
            return r.status, r.headers['Content-Type'], r.read()

    def test_atlas_assets_are_served(self):
        for path, mime in [('/atlas', 'text/html'), ('/atlas.js', 'text/javascript'),
                           ('/atlas-core.js', 'text/javascript'), ('/atlas.css', 'text/css')]:
            status, ctype, body = self.get(path)
            self.assertEqual(status, 200)
            self.assertTrue(ctype.startswith(mime))
            self.assertGreater(len(body), 100)

    def test_atlas_reads_the_same_circuit_as_the_model(self):
        status, _, body = self.get('/api/circuit')
        data = json.loads(body)
        self.assertEqual((len(data['nodes']), len(data['edges'])), (807, 21161))

    @unittest.skipUnless(shutil.which('node'), 'node not installed')
    def test_atlas_core_logic(self):
        r = subprocess.run(['node', '--test', 'atlas-core.test.js'], cwd=BASE, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

if __name__ == '__main__':
    unittest.main()
