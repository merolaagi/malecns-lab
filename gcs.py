"""Read-only, whitelisted access to the public MaleCNS bucket (gs://flyem-male-cns) with a local cache.

Used by the 3D viewer (meshes, region meshes, segment properties) so the browser never depends on the
bucket's cross-origin settings. Only listed prefixes are allowed, paths are validated, files are capped
at 25 MB, and gzip-encoded objects (common for Neuroglancer precomputed data) are decompressed once.
"""
import gzip
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote
from urllib.request import Request, urlopen

BASE = Path(__file__).resolve().parent
ROOT = 'https://storage.googleapis.com/flyem-male-cns/'
CACHE = BASE / 'data/gcs-cache'
MAX_BYTES = 25 * 1024 * 1024
ALLOWED = ('v1.0/segmentation/meshes-malecns/', 'v1.0/segmentation/skeletons-malecns/',
           'rois/fullbrain-roi-v5/', 'rois/malecns-vnc-neuropil-roi-v0/', 'rois/fullbrain-major-shells/',
           'rois/vnc-neuropil-shell-v2/', 'rois/brain-shell-v2.2/')
SAFE_DIR = re.compile(r'^[A-Za-z0-9_.@+\-]+$')          # directory components: strict
BAD_NAME = re.compile(r'[\x00-\x1f\x7f/\\]')              # file names: anything printable except slashes


class GCSError(ValueError):
    def __init__(self, message, status=400):
        super().__init__(message); self.status = status


def validate(path):
    """Directories must be plain; the final file name may hold any printable characters (mesh fragment
    names from the bucket are not always plain), but never path separators or '..'."""
    path = unquote(path).lstrip('/')
    parts = path.split('/')
    bad = (len(parts) < 2 or any(p in ('', '.', '..') for p in parts)
           or not all(SAFE_DIR.match(p) for p in parts[:-1]) or BAD_NAME.search(parts[-1]))
    if bad: raise GCSError(f'Invalid path: {path[:200]}')
    if not path.startswith(ALLOWED):
        raise GCSError('Path not in the allowed MaleCNS layers', 403)
    return path


def url_for(path):
    return ROOT + quote(path, safe='/')


def cache_path(path):
    head, name = path.rsplit('/', 1)
    return CACHE / head / quote(name, safe='')


def fetch(path, opener=urlopen, max_bytes=None, cache=True):
    """Return bytes for a bucket path, from cache when possible. Large raw files can skip the cache."""
    path = validate(path)
    limit = max_bytes or MAX_BYTES
    local = cache_path(path)
    if cache and local.is_file(): return local.read_bytes()
    try:
        with opener(Request(url_for(path), headers={'Accept-Encoding': 'gzip'}), timeout=60) as r:
            data = r.read(limit + 1)
            encoding = (r.headers.get('Content-Encoding') or '').lower() if getattr(r, 'headers', None) else ''
    except HTTPError as e:
        raise GCSError(f'Not found in the MaleCNS bucket: {path}' if e.code == 404 else f'Bucket error {e.code}', 404 if e.code == 404 else 502)
    except (URLError, TimeoutError) as e:
        raise GCSError(f'Bucket unreachable: {e}', 502)
    if len(data) > limit: raise GCSError(f'File exceeds the {limit // (1024 * 1024)} MB limit', 413)
    if encoding == 'gzip' or data[:2] == b'\x1f\x8b':
        try: data = gzip.decompress(data)
        except OSError: pass            # not actually gzip; keep raw bytes
    if cache:
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_suffix(local.suffix + '.part'); tmp.write_bytes(data); tmp.replace(local)
    return data
