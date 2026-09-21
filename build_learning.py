"""Extract measured layered olfactory/MB edges. No inferred edges added."""
import json
from pathlib import Path
import numpy as np
import pyarrow.feather as f
from model import BASE

def build(raw):
    rows=f.read_table(raw/'annotations.feather').to_pylist()
    groups={}
    for r in rows:
        if r['status']!='Traced': continue
        role={'ALPN':'PN','Kenyon_Cell':'KC','MBON':'MBON'}.get(r['class'])
        if r['class']=='DAN' and str(r['type']).startswith('PAM'): role='PAM'
        if role: groups[r['bodyId']]={k:r[k] for k in ['bodyId','type','instance','somaSide','class']}|{'role':role}
    t=f.read_table(raw/'weights.feather',memory_map=True)
    pre,post,count=[t[k].to_numpy() for k in ['body_pre','body_post','weight']]
    edges=[]
    for source,target in [('PN','KC'),('KC','MBON'),('PAM','KC')]:
        a=[i for i,r in groups.items() if r['role']==source];b=[i for i,r in groups.items() if r['role']==target]
        mask=np.isin(pre,a)&np.isin(post,b)&(count>0)
        edges.extend([[int(x),int(y),int(w)] for x,y,w in zip(pre[mask],post[mask],count[mask])])
    used={i for e in edges for i in e[:2]}
    provenance=json.loads((BASE/'data/circuit.json').read_text())
    out={'dataset':'male-cns:v1.0','source':provenance['source'],'files':provenance['files'],
         'license':'CC-BY-4.0','selection':'Traced ALPN, Kenyon_Cell, MBON and PAM DAN annotations; retain only measured PN→KC, KC→MBON and PAM→KC edges; omit isolated cells. Both hemispheres. Other pathways are not simulated.',
         'nodes':[groups[i] for i in sorted(used)],'edges':edges}
    (BASE/'data/learning-circuit.json').write_text(json.dumps(out,separators=(',',':')))
    print({role:sum(n['role']==role for n in out['nodes']) for role in ['PN','KC','MBON','PAM']},'edges',len(edges),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);build(p.parse_args().raw)
