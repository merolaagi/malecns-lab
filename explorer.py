"""Measured subset navigation and actual SWC skeleton retrieval."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
from model import BASE

class Explorer:
 def __init__(self):
  self.nodes={};self.edges={}
  for filename,subset in [('circuit.json','motor'),('learning-circuit.json','learning')]:
   data=json.loads((BASE/'data'/filename).read_text())
   for n in data['nodes']:
    self.nodes[n['bodyId']]={**n,'subset':subset}
   for a,b,w in data['edges']:self.edges[(a,b)]=w
  self.incoming={i:[] for i in self.nodes};self.outgoing={i:[] for i in self.nodes}
  for (a,b),w in self.edges.items():self.incoming[b].append((a,w));self.outgoing[a].append((b,w))
 def graph(self):
  return {'dataset':'male-cns:v1.0','nodes':list(self.nodes.values()),'edges':[[a,b,w] for (a,b),w in self.edges.items()],
          'scope':'Union of the motor and learning subsets. Not the full CNS: connections to cells outside these subsets are omitted.',
          'source':'https://male-cns.janelia.org/download/','license':'CC-BY-4.0'}
 def skeleton(self,body_id):
  body_id=int(body_id)
  if body_id not in self.nodes:raise ValueError('Select a neuron from the loaded measured subsets')
  directory=BASE/'data/skeletons';directory.mkdir(exist_ok=True)
  path=directory/f'{body_id}.swc'
  url=f'https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/{body_id}.swc'
  if not path.exists():
   try:
    with urlopen(url,timeout=15) as r:payload=r.read(12*1024*1024+1)
    if len(payload)>12*1024*1024:raise ValueError('Skeleton exceeds viewer download limit')
    path.write_bytes(payload)
   except (URLError,TimeoutError) as e:raise ValueError('Official skeleton unavailable; graph connections remain available.') from e
  payload=path.read_bytes();points=[]
  for line in payload.decode().splitlines():
   if not line.strip() or line.startswith('#'):continue
   a=line.split()
   if len(a)<7:continue
   points.append([int(a[0]),float(a[2])*.008,float(a[3])*.008,float(a[4])*.008,int(a[6])])
  return {'bodyId':body_id,'points':points,'units':'micrometers','source':url,'sha256':hashlib.sha256(payload).hexdigest(),
          'note':'Official coarse reconstructed centerline; 8-nm source coordinates converted to micrometers. Branch type and synapse sites are not assigned. No electrical propagation is inferred from this shape.'}
