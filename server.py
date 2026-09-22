"""Local-only research UI. Uses an OS-assigned port and persists its assignment."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from model import Circuit
from learning import LearningCircuit
from numerosity import Numerosity
from arithmetic import Arithmetic
from explorer import Explorer
from membrane import run_membrane
from sdr_math import SDRArithmetic
from vision import run as run_vision
from expansion import Experiment as ExpansionExperiment
from neuron_inputs import NeuronIndex
from urllib.parse import urlparse, parse_qs
import gcs
import meshes

BASE = Path(__file__).resolve().parent
CIRCUIT = Circuit()
LEARNING = LearningCircuit()
NUMEROSITY = Numerosity(LEARNING)
MATH = Arithmetic(LEARNING)
EXPLORER = Explorer()
EXPANSION = ExpansionExperiment(LEARNING)
NEURONS = NeuronIndex(CIRCUIT, LEARNING)
LOCK = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def send(self, status, body, content_type='application/json'):
        payload = json.dumps(body, allow_nan=False).encode() if content_type == 'application/json' else body
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        url = urlparse(self.path); path = url.path
        if path.startswith('/api/mesh/'):
            parts = path[len('/api/mesh/'):].split('/')
            try:
                if len(parts) != 2: raise gcs.GCSError('Use /api/mesh/<layer>/<id>')
                body, info = meshes.get(parts[0], parts[1])
            except gcs.GCSError as e: return self.send(e.status, {'error': str(e)})
            except ValueError as e: return self.send(502, {'error': f'Mesh could not be decoded: {e}'})
            self.send_response(200); self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('X-Mesh-Info', json.dumps(info)); self.send_header('Cache-Control', 'public, max-age=86400')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body); return
        if path.startswith('/api/gcs/'):
            try: body = gcs.fetch(path[len('/api/gcs/'):])
            except gcs.GCSError as e: return self.send(e.status, {'error': str(e)})
            self.send_response(200); self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Cache-Control', 'public, max-age=86400'); self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if path == '/api/explorer': return self.send(200, EXPLORER.graph())
        if path == '/api/skeleton':
            try:
                body_id = int(parse_qs(url.query).get('id', ['10360'])[0])
                return self.send(200, EXPLORER.skeleton(body_id))
            except (ValueError, OSError) as e: return self.send(400, {'error': str(e)})
        if path in ('/api/quality', '/api/regions'):
            name = path.split('/')[-1]; q = BASE / f'data/{name}.json'
            if not q.exists(): return self.send(404, {'error': f'No data/{name}.json yet. Run build_{name}.py (see README).'})
            body = q.read_bytes()
            self.send_response(200); self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body); return
        if path == '/api/neuron':
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                circuit = q.get('circuit', 'learning')
                cell = NEURONS.random_kc(q.get('seed', 0)) if q.get('id') == 'random' and circuit == 'learning' else q.get('id')
                return self.send(200, NEURONS.describe(circuit, cell))
            except ValueError as e: return self.send(400, {'error': str(e)})
        if path == '/api/learning-circuit':
            return self.send(200, LEARNING.data)
        if path == '/api/circuit':
            return self.send(200, CIRCUIT.data)
        files = {'/expansion': ('expansion.html', 'text/html; charset=utf-8'),
                 '/expansion.js': ('expansion.js', 'text/javascript; charset=utf-8'),
                 '/api/expansion-benchmark': ('data/expansion-benchmark.json', 'application/json; charset=utf-8'),
                 '/3d': ('viewer3d.html', 'text/html; charset=utf-8'),
                 '/vendor/three.min.js': ('vendor/three.min.js', 'text/javascript; charset=utf-8'),
                 '/viewer3d.js': ('viewer3d.js', 'text/javascript; charset=utf-8'),
                 '/ng.js': ('ng.js', 'text/javascript; charset=utf-8'),
                 '/regions': ('regions.html', 'text/html; charset=utf-8'),
                 '/regions.js': ('regions.js', 'text/javascript; charset=utf-8'),
                 '/vision': ('vision.html', 'text/html; charset=utf-8'),
                 '/vision.js': ('vision.js', 'text/javascript; charset=utf-8'),
                 '/vision.css': ('vision.css', 'text/css; charset=utf-8'),
                 '/api/vision-example': ('data/vision-example.json', 'application/json; charset=utf-8'),
                 '/api/vision-benchmark': ('data/vision-benchmark.json', 'application/json; charset=utf-8'),
                 '/api/vision-circuit': ('data/vision-circuit.json', 'application/json; charset=utf-8'),
                 '/explore': ('explore.html', 'text/html; charset=utf-8'),
                 '/explore.js': ('explore.js', 'text/javascript; charset=utf-8'),
                 '/explore.css': ('explore.css', 'text/css; charset=utf-8'),
                 '/api/sdr-benchmark': ('data/sdr-benchmark.json', 'application/json; charset=utf-8'),
                 '/math': ('math.html', 'text/html; charset=utf-8'),
                 '/math.js': ('math.js', 'text/javascript; charset=utf-8'),
                 '/api/math-benchmark': ('data/math-benchmark.json', 'application/json; charset=utf-8'),
                 '/neuron': ('neuron.html', 'text/html; charset=utf-8'),
                 '/neuron.js': ('neuron.js', 'text/javascript; charset=utf-8'),
                 '/neuron-core.js': ('neuron-core.js', 'text/javascript; charset=utf-8'),
                 '/workbench.css': ('workbench.css', 'text/css; charset=utf-8'),
                 '/numerosity': ('numerosity.html', 'text/html; charset=utf-8'),
                 '/numerosity.js': ('numerosity.js', 'text/javascript; charset=utf-8'),
                 '/api/numerosity-benchmark': ('data/numerosity-benchmark.json', 'application/json; charset=utf-8'),
                 '/atlas': ('atlas.html', 'text/html; charset=utf-8'),
                 '/atlas.js': ('atlas.js', 'text/javascript; charset=utf-8'),
                 '/atlas-core.js': ('atlas-core.js', 'text/javascript; charset=utf-8'),
                 '/atlas.css': ('atlas.css', 'text/css; charset=utf-8'),
                 '/learning': ('learning.html', 'text/html; charset=utf-8'),
                 '/learning.js': ('learning.js', 'text/javascript; charset=utf-8'),
                 '/learning.css': ('learning.css', 'text/css; charset=utf-8'),
                 '/api/learning-benchmark': ('data/learning-benchmark.json', 'application/json; charset=utf-8'),
                 '/': ('index.html', 'text/html; charset=utf-8'),
                 '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                 '/style.css': ('style.css', 'text/css; charset=utf-8'),
                 '/api/benchmark': ('data/benchmark.json', 'application/json; charset=utf-8')}
        if path not in files: return self.send(404, {'error': 'Not found'})
        name, mime = files[path]
        file = BASE / name
        if not file.exists(): return self.send(404, {'error': 'No saved benchmark yet'})
        self.send(200, file.read_bytes(), mime)

    def do_POST(self):
        if self.path not in ['/api/run', '/api/learn', '/api/numerosity', '/api/math', '/api/membrane', '/api/sdr', '/api/vision', '/api/expansion']: return self.send(404, {'error': 'Not found'})
        # The app is same-origin. Reject browser requests originating elsewhere.
        # Behind an HTTPS proxy such as a Cloudflare tunnel the page origin is https://<host>,
        # so both schemes are accepted for the same Host; other sites are still refused.
        origin = self.headers.get('Origin')
        host = self.headers.get('Host', '')
        if origin and origin not in ('http://' + host, 'https://' + host): return self.send(403, {'error': 'Origin not allowed'})
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 8192: raise ValueError('Invalid request size')
            options = json.loads(self.rfile.read(size))
            if not isinstance(options, dict): raise ValueError('Expected JSON object')
            allowed = {'/api/run': {'duration','drive','bias','gain','feedback','condition','seed','mode'},
                       '/api/learn': {'seed','train_trials','probe_trials','learning_rate','temperature','retention','delay','condition','rewarded_odor'},
                       '/api/numerosity': {'seed','numbers','width','step','held_out','train_trials','learning_rate','temperature','noise','test_exemplars','condition'},
                       '/api/math': {'seed','epochs','learning_rate','condition'},
                       '/api/membrane': {'current','inhibition','context','block_na','body_id'},
                       '/api/sdr': {'seed','epochs','learning_rate','condition','encoding'},
                       '/api/vision': {'seed','epochs'},
                       '/api/expansion': {'seed','per_class','epochs','sparsity','lr','task'}}[self.path]
            if set(options) - allowed: raise ValueError('Unknown parameter')
        except (ValueError, TypeError) as e: return self.send(400, {'error': str(e)})
        if not LOCK.acquire(blocking=False): return self.send(409, {'error': 'An experiment is already running'})
        try:
            if self.path == '/api/membrane' and int(options.get('body_id', 10360)) not in EXPLORER.nodes:
                raise ValueError('Unknown selected neuron')
            if self.path == '/api/sdr':
                encoding = options.pop('encoding', 'scalar_sdr')
                result = SDRArithmetic(LEARNING, encoding).run(**options)
            else:
                run = {'/api/run': CIRCUIT.simulate, '/api/learn': LEARNING.run, '/api/numerosity': NUMEROSITY.run,
                       '/api/math': MATH.run, '/api/membrane': run_membrane, '/api/vision': run_vision, '/api/expansion': EXPANSION.run}[self.path]
                result = run(**options)
            self.send(200, result)
        except (ValueError, TypeError, OverflowError) as e:
            self.send(400, {'error': str(e)})
        finally:
            LOCK.release()

if __name__ == '__main__':
    runtime = BASE / '.runtime.json'
    old = json.loads(runtime.read_text()) if runtime.exists() else {}
    # bind itself is the atomic port check; never kill another listener.
    try:
        server = ThreadingHTTPServer(('127.0.0.1', old.get('port', 0)), Handler)
    except OSError:
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    port = server.server_address[1]
    runtime.write_text(json.dumps({'pid': os.getpid(), 'port': port, 'project': str(BASE)}))
    print(f'MaleCNS Lab: http://127.0.0.1:{port}', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
