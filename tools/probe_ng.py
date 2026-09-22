"""One-time probe of the Neuroglancer layers the 3D viewer uses. Run on a machine with internet access:
    .venv/bin/python tools/probe_ng.py
Writes data/ng-probe.json (commit and push it so the formats can be checked)."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gcs

out = {}
def get(path, parse=True):
    try:
        b = gcs.fetch(path)
        if not parse: return {'ok': True, 'bytes': len(b), 'head_hex': b[:16].hex()}
        try: return {'ok': True, 'json': json.loads(b)}
        except ValueError: return {'ok': True, 'bytes': len(b), 'head_hex': b[:16].hex()}
    except gcs.GCSError as e: return {'ok': False, 'error': str(e)}

for layer in ['v1.0/segmentation/meshes-malecns/single-res-meshes', 'rois/fullbrain-roi-v5', 'rois/malecns-vnc-neuropil-roi-v0',
              'rois/fullbrain-major-shells', 'rois/vnc-neuropil-shell-v2']:
    info = get(layer + '/info'); out[layer + '/info'] = info
    j = info.get('json') or {}
    for key in ['mesh', 'segment_properties']:
        sub = j.get(key)
        if isinstance(sub, str):
            out[f'{layer}/{sub}/info'] = get(f'{layer}/{sub}/info')
manifest = get('v1.0/segmentation/meshes-malecns/single-res-meshes/10360:0'); out['neuron 10360 manifest'] = manifest
frags = (manifest.get('json') or {}).get('fragments') or []
if frags: out['neuron 10360 first fragment'] = get('v1.0/segmentation/meshes-malecns/single-res-meshes/' + frags[0], parse=False)
Path(gcs.BASE / 'data/ng-probe.json').write_text(json.dumps(out, indent=1)[:2_000_000])
for k, v in out.items():
    print(('OK   ' if v.get('ok') else 'FAIL ') + k + ('' if v.get('ok') else '  ' + v['error']))
print('Wrote data/ng-probe.json')
