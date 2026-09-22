"""Extract a bounded spatial patch of actual MaleCNS visual neurons and edges."""
import json, math
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
from model import BASE

def build(raw):
 rows=feather.read_table(raw/'annotations.feather').to_pylist()
 types=['L1','L2','L3','Mi1','Mi4','Mi9','C2','C3','Tm1','Tm2','Tm9']
 candidates=[r for r in rows if r['status']=='Traced' and r['somaSide']=='R' and r['type'] in types and r['assignedOlHex1'] is not None and r['assignedOlHex2'] is not None and math.isfinite(r['assignedOlHex1']) and math.isfinite(r['assignedOlHex2'])]
 # The oblique annotation axes form the hex distance max(|dq|,|dr|,|dq-dr|).
 centers=sorted({(int(r['assignedOlHex1']),int(r['assignedOlHex2'])) for r in candidates if r['type']=='L1'})
 median=np.median(centers,axis=0);center=min(centers,key=lambda p:np.sum((np.array(p)-median)**2))
 columns=[p for p in centers if max(abs(p[0]-center[0]),abs(p[1]-center[1]),abs((p[0]-center[0])-(p[1]-center[1])))<=2]
 columns=sorted(columns); lookup={p:i for i,p in enumerate(columns)}
 nodes=[]
 for r in candidates:
  p=(int(r['assignedOlHex1']),int(r['assignedOlHex2']))
  if p in lookup:nodes.append({k:r[k] for k in ['bodyId','type','instance','statusLabel']}|{'column':lookup[p],'hex':list(p)})
 nodes.sort(key=lambda n:(types.index(n['type']),n['column'],n['bodyId']))
 ids=np.array([n['bodyId'] for n in nodes]);edges=[]
 with pa.memory_map(str(raw/'weights.feather'),'r') as file:
  reader=pa.ipc.open_file(file)
  for k in range(reader.num_record_batches):
   batch=reader.get_batch(k);a=batch.column('body_pre').to_numpy();b=batch.column('body_post').to_numpy();w=batch.column('weight').to_numpy()
   mask=np.isin(a,ids)&np.isin(b,ids)&(w>0)
   edges.extend([[int(x),int(y),int(z)] for x,y,z in zip(a[mask],b[mask],w[mask])])
   if k%600==0:print('batch',k,'edges',len(edges),flush=True)
 provenance=json.loads((BASE/'data/circuit.json').read_text())
 out={'dataset':'male-cns:v1.0','license':'CC-BY-4.0','source':provenance['source'],'files':provenance['files'],'nodes':nodes,'edges':edges,'columns':[{'hex':list(p),'x':float((p[0]-center[0]-.5*(p[1]-center[1]))/2),'y':float(math.sqrt(3)/2*(p[1]-center[1])/2)} for p in columns],'types':types,'center':list(center),'scope':'Right optic-lobe radius-2 patch; 11 selected types, traced cells with assignedOlHex coordinates. Induced measured edges only. Outside cells and connections omitted. Column mapping is anatomical; optical sampling and projection are simplified engineering assumptions.'}
 (BASE/'data/vision-circuit.json').write_text(json.dumps(out,separators=(',',':')))
 print('DONE',len(nodes),len(edges),len(columns),flush=True)
if __name__=='__main__':
 import sys
 build(Path(sys.argv[1]))
