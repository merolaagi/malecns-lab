import gzip
import io
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

import gcs


class FakeResponse(io.BytesIO):
    def __init__(self, data, encoding=None):
        super().__init__(data); self.headers = {'Content-Encoding': encoding} if encoding else {}
    def __enter__(self): return self
    def __exit__(self, *a): return False


class GCSTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.old = gcs.CACHE; gcs.CACHE = Path(self.tmp.name); self.calls = []
    def tearDown(self):
        gcs.CACHE = self.old; self.tmp.cleanup()

    def opener(self, payload, encoding=None, error=None):
        def open_(req, timeout=None):
            self.calls.append(req.full_url)
            if error: raise HTTPError(req.full_url, error, 'x', {}, None)
            return FakeResponse(payload, encoding)
        return open_

    def test_path_rules(self):
        self.assertEqual(gcs.validate('rois/fullbrain-roi-v5/mesh/12:0'), 'rois/fullbrain-roi-v5/mesh/12:0')
        self.assertEqual(gcs.validate('v1.0/segmentation/meshes-malecns/single-res-meshes/10360%3A0'), 'v1.0/segmentation/meshes-malecns/single-res-meshes/10360:0')
        for bad in ['rois/fullbrain-roi-v5/../../secret', 'em/em-clahe-jpeg/info', 'rois/fullbrain-roi-v5/', 'rois/fullbrain-roi-v5/a b', 'https://evil.example/x']:
            with self.assertRaises(gcs.GCSError): gcs.validate(bad)

    def test_fetch_decompresses_gzip_and_caches(self):
        body = b'{"@type":"neuroglancer_legacy_mesh"}'
        a = gcs.fetch('rois/fullbrain-roi-v5/mesh/info', opener=self.opener(gzip.compress(body), 'gzip'))
        b = gcs.fetch('rois/fullbrain-roi-v5/mesh/info', opener=self.opener(b'unused'))
        self.assertEqual(a, body); self.assertEqual(b, body); self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.calls[0].startswith('https://storage.googleapis.com/flyem-male-cns/rois/'))

    def test_plain_bytes_kept(self):
        self.assertEqual(gcs.fetch('rois/fullbrain-roi-v5/mesh/1:0', opener=self.opener(b'\x01\x02\x03')), b'\x01\x02\x03')

    def test_errors(self):
        with self.assertRaises(gcs.GCSError) as e: gcs.fetch('rois/fullbrain-roi-v5/mesh/404:0', opener=self.opener(b'', error=404))
        self.assertEqual(e.exception.status, 404)
        old = gcs.MAX_BYTES; gcs.MAX_BYTES = 4
        try:
            with self.assertRaises(gcs.GCSError) as e: gcs.fetch('rois/fullbrain-roi-v5/mesh/big:0', opener=self.opener(b'123456'))
            self.assertEqual(e.exception.status, 413)
        finally: gcs.MAX_BYTES = old
        self.assertFalse(any(gcs.CACHE.rglob('*:0')))   # failures are never cached


if __name__ == '__main__':
    unittest.main()
