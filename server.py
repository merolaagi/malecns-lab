"""Local-only research UI. Uses an OS-assigned port and persists its assignment."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from model import Circuit
from learning import LearningCircuit

BASE = Path(__file__).resolve().parent
CIRCUIT = Circuit()
LEARNING = LearningCircuit()
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
        path = self.path.split('?')[0]
        if path == '/api/learning-circuit':
            return self.send(200, LEARNING.data)
        if path == '/api/circuit':
            return self.send(200, CIRCUIT.data)
        files = {'/atlas': ('atlas.html', 'text/html; charset=utf-8'),
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
        if self.path not in ['/api/run', '/api/learn']: return self.send(404, {'error': 'Not found'})
        # The app is same-origin. Reject browser requests originating elsewhere.
        origin = self.headers.get('Origin')
        expected = 'http://' + self.headers.get('Host', '')
        if origin and origin != expected: return self.send(403, {'error': 'Origin not allowed'})
        try:
            size = int(self.headers.get('Content-Length', 0))
            if not 0 < size <= 8192: raise ValueError('Invalid request size')
            options = json.loads(self.rfile.read(size))
            if not isinstance(options, dict): raise ValueError('Expected JSON object')
            allowed = ({'duration','drive','bias','gain','feedback','condition','seed','mode'} if self.path == '/api/run'
                       else {'seed','train_trials','probe_trials','learning_rate','temperature','retention','delay','condition','rewarded_odor'})
            if set(options) - allowed: raise ValueError('Unknown parameter')
        except (ValueError, TypeError) as e: return self.send(400, {'error': str(e)})
        if not LOCK.acquire(blocking=False): return self.send(409, {'error': 'An experiment is already running'})
        try:
            result = CIRCUIT.simulate(**options) if self.path == '/api/run' else LEARNING.run(**options)
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
