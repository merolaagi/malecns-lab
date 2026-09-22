"""Trainable visual graph experiment. Anatomy constrains edges; dynamics are supplied.
NumPy BPTT/Adam, no external ML runtime or downloaded learned weights.
"""
import json, math, time, hashlib
import numpy as np
from model import BASE

NAMES={'graph_scalar':'Measured graph · 1 state/node','graph_vector':'Measured graph · 2 states/node','shuffled_vector':'Rewired graph · 2 states/node','dense_rnn':'Dense recurrent baseline',
 'eye_graph':'Measured graph + trainable eye (retinotopic)','eye_dense':'Measured graph + trainable eye (unrestricted)'}
EYE_KINDS=('eye_graph','eye_dense')

def circuit():return json.loads((BASE/'data/vision-circuit.json').read_text())

def scenes(columns,seed,size,steps=10):
 """Independent sequences: moving textured Gaussian disks, some expanding.
 Coordinates/velocities are image-plane quantities, not recovered 3D geometry.
 """
 rng=np.random.default_rng(seed);xy=np.array([[c['x'],c['y']] for c in columns]);x=np.empty((size,steps,len(xy)),np.float32);y=np.empty((size,3),np.float32);params=[]
 for k in range(size):
  vx,vy=rng.uniform(-.075,.075,2);cx,cy=rng.uniform(-.5,.5,2);radius=rng.uniform(.25,.52);growth=rng.uniform(-.015,.10);phase=rng.uniform(0,2*np.pi);frequency=rng.uniform(3,7)
  p=dict(cx=float(cx),cy=float(cy),vx=float(vx),vy=float(vy),radius=float(radius),growth=float(growth),phase=float(phase),frequency=float(frequency));params.append(p)
  for t in range(steps):
   dx=xy[:,0]-(cx+vx*t);dy=xy[:,1]-(cy+vy*t);r=radius*np.exp(growth*t)
   envelope=np.exp(-(dx*dx+dy*dy)/(2*r*r));texture=.65+.35*np.cos(frequency*dx+phase)*np.cos(frequency*dy-phase)
   x[k,t]=2*np.clip(envelope*texture+rng.normal(0,.015,len(xy)),0,1)-1
  # Synthetic looming hazard: expanding disk reaches the central sensor region in 4 more steps.
  future=steps-1+4;fr=radius*np.exp(growth*future);fc=math.hypot(cx+vx*future,cy+vy*future)
  y[k]=[vx/.075,vy/.075,float(growth>.025 and fr>fc+.45)]
 return x,y,params

def rewire(mask,seed):
 """Directed double-edge swaps preserve every node's in/out degree and self loops."""
 rng=np.random.default_rng(seed);m=mask.copy();target,source=np.where(m);pairs=[(int(a),int(b)) for a,b in zip(target,source) if a!=b];swaps=0
 if len(pairs)<2:return m,swaps
 for _ in range(len(pairs)*30):
  i,j=rng.choice(len(pairs),2,replace=False);a,b=pairs[i];c,d=pairs[j]
  if a==c or b==d or a==d or c==b or m[a,d] or m[c,b]:continue
  m[a,b]=m[c,d]=0;m[a,d]=m[c,b]=1;pairs[i]=(a,d);pairs[j]=(c,b);swaps+=1
 return m,swaps

def hex_neighbors(columns,radius=1):
 """Columns within `radius` hex steps, using the annotation axes' distance max(|dq|,|dr|,|dq-dr|)."""
 h=np.array([c['hex'] for c in columns]);dq=h[:,None,0]-h[None,:,0];dr=h[:,None,1]-h[None,:,1]
 return (np.maximum(np.maximum(np.abs(dq),np.abs(dr)),np.abs(dq-dr))<=radius).astype(np.float32)

class Net:
 def __init__(self,data,kind,seed,budget=None):
  self.kind=kind;self.d=1 if kind in ['graph_scalar','dense_rnn'] else 2;self.dense=kind=='dense_rnn';self.p={};self.m={};self.v={};self.step=0;self.swaps=0
  rng=np.random.default_rng(seed);C=len(data['columns'])
  if self.dense:
   # Same recurrent activation/time unrolling; unrestricted learned input/hidden mixing.
   self.n=min(range(2,256),key=lambda h:abs(h*h+h+h+C*h+3*h+3-budget));self.mask=np.ones((self.n,self.n),np.float32);self.input_mask=np.ones((C,self.n),np.float32);self.out_idx=np.arange(self.n);self.input_idx=None
  else:
   self.n=len(data['nodes']);ids={r['bodyId']:i for i,r in enumerate(data['nodes'])};self.mask=np.zeros((self.n,self.n),np.float32)
   for a,b,w in data['edges']:self.mask[ids[b],ids[a]]=1
   if kind=='shuffled_vector':self.mask,self.swaps=rewire(self.mask,seed+900)
   self.input_idx=np.array([r['column'] for r in data['nodes']]);self.input_mask=np.array([r['type'] in ['L1','L2','L3'] for r in data['nodes']],np.float32)[:,None]
   self.out_idx=np.array([i for i,r in enumerate(data['nodes']) if r['type'] in ['Tm1','Tm2','Tm9']])
  n,d=self.n,self.d
  # Same target-wise random initialization for real/rewired graphs, same per-row norm.
  # Synapse counts are deliberately not treated as physiological weights.
  self.p['W']=np.zeros((n,n),np.float32)
  for i in range(n):
   neighbors=np.flatnonzero(self.mask[i]);self.p['W'][i,neighbors]=rng.normal(0,.35/max(1,np.sqrt(len(neighbors))),len(neighbors))
  self.p['U']=(np.tile(np.eye(d)*.4,(n,1,1))+rng.normal(0,.025,(n,d,d))).astype(np.float32)
  self.p['b']=np.zeros((n,d),np.float32)
  self.p['G']=(rng.normal(0,.25,(C,n)) if self.dense else rng.normal(.6,.1,(n,d))*self.input_mask).astype(np.float32)
  self.p['O']=rng.normal(0,.15/np.sqrt(len(self.out_idx)*d),(len(self.out_idx)*d,3)).astype(np.float32);self.p['c']=np.zeros(3,np.float32)
  self.count=int(self.mask.sum()+n*d*d+n*d+(C*n if self.dense else self.input_mask.sum()*d)+self.p['O'].size+3)
  # Trainable compound-eye front end (eye kinds only). Created after all random draws, so every other
  # parameter matches graph_vector exactly, and at initialization the eye is an identity: the model
  # starts as graph_vector. Photoreceptor adaptation per column (shared parameters):
  #   y_t = a*y_(t-1) + (1-a)*x_t,  r_t = g*(x_t - k*y_t)   (a = sigmoid(E0), k = E1, g = E2)
  # then a column-to-column projection u_t = P r_t into L1/L2/L3. eye_graph restricts P to each column
  # and its six hex neighbours (retinotopy, as in neural superposition); eye_dense allows any column.
  self.eye=kind in EYE_KINDS
  if self.eye:
   self.pmask=hex_neighbors(data['columns']) if kind=='eye_graph' else np.ones((C,C),np.float32)
   self.p['P']=np.eye(C,dtype=np.float32);self.p['E']=np.array([0.,0.,1.],np.float32)
   self.S=np.zeros((n,C),np.float32);self.S[np.arange(n),self.input_idx]=1
   self.count+=int(self.pmask.sum()+3)
 def eye_forward(self,x):
  a=1/(1+math.exp(-float(self.p['E'][0])));k=self.p['E'][1];g=self.p['E'][2];B,T,C=x.shape
  y=x[:,0].copy();dya=np.zeros_like(y);r=np.empty_like(x);ys=np.empty_like(x);dyas=np.empty_like(x)
  for t in range(T):
   dya=(y-x[:,t])+a*dya;y=a*y+(1-a)*x[:,t];ys[:,t]=y;dyas[:,t]=dya;r[:,t]=g*(x[:,t]-k*y)
  return r@self.p['P'].T,(x,r,ys,dyas,a)
 def eye_backward(self,du,cache,grads):
  x,r,ys,dyas,a=cache;k=self.p['E'][1];g=self.p['E'][2]
  grads['P']+=np.einsum('btc,btd->cd',du,r,optimize=True);dr=du@self.p['P']
  grads['E'][0]+=float(np.sum(dr*(-g*k*dyas)))*a*(1-a);grads['E'][1]+=float(np.sum(dr*(-g*ys)));grads['E'][2]+=float(np.sum(dr*(x-k*ys)))
 def reset_eye(self):
  self.p['P'][:]=np.eye(len(self.p['P']));self.p['E'][:]=[0,0,1]
 def forward(self,x,cache=False):
  if self.eye:x=self.eye_forward(x)[0]
  B,T,C=x.shape;n,d=self.n,self.d;h=np.zeros((B,n,d),np.float32);states=[h];acts=[];p=self.p
  for t in range(T):
   flat=h.transpose(0,2,1).reshape(B*d,n);msg=(flat@p['W'].T).reshape(B,d,n).transpose(0,2,1)
   local=np.einsum('bni,nij->bnj',h,p['U'],optimize=True)
   inp=(x[:,t]@p['G'])[:,:,None] if self.dense else x[:,t,self.input_idx,None]*p['G'][None]
   a=np.tanh(msg+local+inp+p['b']);h=.5*h+.5*a;states.append(h);acts.append(a)
  out=h[:,self.out_idx].reshape(B,-1)@p['O']+p['c']
  return (out,(states,acts)) if cache else out
 def loss_grad(self,x,y):
  raw=x;cache=None
  if self.eye:x,cache=self.eye_forward(raw);du=np.zeros_like(x)
  eye,self.eye=self.eye,False   # forward on the already-transformed input
  try:out,(states,acts)=self.forward(x,True)
  finally:self.eye=eye
  B,T,C=x.shape;n,d=self.n,self.d;p=self.p
  motion=np.mean((out[:,:2]-y[:,:2])**2);logits=out[:,2];bce=np.mean(np.logaddexp(0,logits)-y[:,2]*logits);loss=motion+.5*bce
  delta=np.zeros_like(out);delta[:,:2]=(out[:,:2]-y[:,:2])/B;delta[:,2]=.5*(1/(1+np.exp(-np.clip(logits,-40,40)))-y[:,2])/B
  grads={k:np.zeros_like(v) for k,v in p.items()};grads['O']=states[-1][:,self.out_idx].reshape(B,-1).T@delta;grads['c']=delta.sum(0)
  dh=np.zeros_like(states[-1]);dh[:,self.out_idx]=(delta@p['O'].T).reshape(B,len(self.out_idx),d)
  for t in range(T-1,-1,-1):
   dz=.5*dh*(1-acts[t]**2);prev=states[t];zf=dz.transpose(0,2,1).reshape(B*d,n);hf=prev.transpose(0,2,1).reshape(B*d,n)
   grads['W']+=zf.T@hf;grads['U']+=np.einsum('bni,bnj->nij',prev,dz,optimize=True);grads['b']+=dz.sum(0)
   if self.dense:grads['G']+=x[:,t].T@dz[:,:,0]
   else:
    grads['G']+=(x[:,t,self.input_idx,None]*dz).sum(0)
    if self.eye:du[:,t]=(dz*p['G'][None]).sum(2)@self.S
   dh=.5*dh+(zf@p['W']).reshape(B,d,n).transpose(0,2,1)+np.einsum('bnj,nij->bni',dz,p['U'],optimize=True)
  grads['W']*=self.mask
  if not self.dense:grads['G']*=self.input_mask
  if self.eye:self.eye_backward(du,cache,grads);grads['P']*=self.pmask
  return float(loss),grads
 def update(self,grads,lr=.004):
  self.step+=1;norm=math.sqrt(sum(float((g*g).sum()) for g in grads.values()));scale=min(1,2/max(norm,1e-12))
  for k,g in grads.items():
   g=g*scale
   if k not in self.m:self.m[k]=np.zeros_like(g);self.v[k]=np.zeros_like(g)
   self.m[k]=.9*self.m[k]+.1*g;self.v[k]=.999*self.v[k]+.001*g*g
   self.p[k]-=lr*(self.m[k]/(1-.9**self.step))/(np.sqrt(self.v[k]/(1-.999**self.step))+1e-8)
  self.p['W']*=self.mask
  if not self.dense:self.p['G']*=self.input_mask
  if self.eye:self.p['P']*=self.pmask
 def digest(self):return hashlib.sha256(b''.join(self.p[k].tobytes() for k in sorted(self.p))).hexdigest()

def metrics(pred,y):
 prob=1/(1+np.exp(-np.clip(pred[:,2],-40,40)));direction=np.arctan2(pred[:,1],pred[:,0]);truth=np.arctan2(y[:,1],y[:,0]);err=np.abs((direction-truth+np.pi)%(2*np.pi)-np.pi)
 return {'motion_rmse':float(np.sqrt(np.mean((pred[:,:2]-y[:,:2])**2))*.075),'direction_error_deg':float(np.mean(err)*180/np.pi),'hazard_accuracy':float(np.mean((prob>=.5)==y[:,2])),'hazard_brier':float(np.mean((prob-y[:,2])**2))}

def run(seed=401,epochs=35,save_weights=False):
 if isinstance(seed,bool) or not isinstance(seed,int) or not 0<=seed<=999999:raise ValueError('seed must be an integer from 0 to 999999')
 if isinstance(epochs,bool) or not isinstance(epochs,int) or not 0<=epochs<=80:raise ValueError('epochs must be an integer from 0 to 80')
 started=time.time();data=circuit();train=scenes(data['columns'],seed+10000,256);val=scenes(data['columns'],seed+20000,64);test=scenes(data['columns'],seed+30000,128)
 kinds=list(NAMES);budget=Net(data,'graph_vector',seed).count;result={'seed':seed,'epochs':epochs,'train_size':256,'validation_size':64,'test_size':128,'steps':10,'models':[],'demonstrations':[],'circuit':data,'protocol':'256 training / 64 validation / 128 independent test clips. Fixed epochs; validation is displayed but not used to select weights. Test evaluated before and after training, never optimized. All models see identical scenes and minibatch orders. No pretrained checkpoint. Rates and dynamics are engineered, not measured physiology.'}
 for kind in kinds:
  net=Net(data,kind,seed,budget);initial=metrics(net.forward(test[0]),test[1]);history=[];rng=np.random.default_rng(seed+40000)
  for epoch in range(epochs):
   order=rng.permutation(len(train[0]));losses=[]
   for start in range(0,len(order),32):
    indices=order[start:start+32];loss,grad=net.loss_grad(train[0][indices],train[1][indices]);net.update(grad);losses.append(loss)
   if epoch==0 or (epoch+1)%5==0 or epoch==epochs-1:history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':metrics(net.forward(val[0]),val[1])})
  before=net.digest();pred=net.forward(test[0]);after=net.digest();demoout,(states,_)=net.forward(test[0][:12],True)
  entry={'kind':kind,'name':NAMES[kind],'parameters':net.count,'nodes':net.n,'state_size':net.d,'initial_test':initial,'train':metrics(net.forward(train[0]),train[1]),'test':metrics(pred,test[1]),'history':history,'predictions':pred.tolist(),'test_weight_hash_before':before,'test_weight_hash_after':after,'rewiring_swaps':net.swaps,'unsupported_nonzero':int(np.count_nonzero(net.p['W']*(1-net.mask))),'demo_activity':np.stack(states[1:],axis=1).tolist(),'learned_edges':[[int(j),int(i),float(net.p['W'][i,j])] for i,j in zip(*np.where(net.mask))] if not net.dense else []}
  if not net.dense:
   entry['node_models']={k:net.p[k].tolist() for k in ['U','G','b']}
   saved=net.p['W'].copy();net.p['W'][:]=0
   entry['edges_ablated_test']=metrics(net.forward(test[0]),test[1]);net.p['W'][:]=saved
   entry['ablation_restored_hash']=net.digest()
  if net.eye:
   P=net.p['P'];off=net.pmask-np.eye(len(P))
   entry['eye']={'adaptation_a':float(1/(1+math.exp(-float(net.p['E'][0])))),'adaptation_k':float(net.p['E'][1]),'gain':float(net.p['E'][2]),
                 'projection_self_mean':float(np.mean(np.diag(P))),'projection_other_mean_abs':float(np.abs(P*off).sum()/max(off.sum(),1)),
                 'projection_allowed_entries':int(net.pmask.sum()),'projection':P.tolist()}
   entry['eye_input_demo']=net.eye_forward(test[0][:12])[0].tolist()
   saved={k:net.p[k].copy() for k in ['P','E']};net.reset_eye()
   entry['eye_reset_test']=metrics(net.forward(test[0]),test[1])
   for k,v in saved.items():net.p[k][:]=v
   entry['eye_restored_hash']=net.digest()
  result['models'].append(entry)
  if save_weights:np.savez_compressed(BASE/f'data/vision-{seed}-{kind}.npz',**net.p,mask=net.mask,seed=seed,epochs=epochs)
  print(kind,entry['test'],'seconds',round(time.time()-started,1),flush=True)
 for i in range(12):result['demonstrations'].append({'index':i,'scene':test[2][i],'retina':test[0][i].tolist(),'target':test[1][i].tolist()})
 mean=train[1].mean(0);baseline=np.tile(mean,(len(test[1]),1));baseline[:,2]=np.log(np.clip(mean[2],1e-6,1-1e-6)/(1-np.clip(mean[2],1e-6,1-1e-6)))
 result['constant_baseline']=metrics(baseline,test[1]);result['test_targets']=test[1].tolist();result['hazard_prevalence']=float(test[1][:,2].mean());result['elapsed_seconds']=time.time()-started;result['anatomy_note']='This spatial patch omits T4/T5 and all exterior connections. Inputs enter L1/L2/L3 at their annotated columns; the eye_* models add a trainable photoreceptor adaptation stage and column projection in front, an engineered stand-in for photoreceptor physiology and neural superposition. Readout pools Tm1/Tm2/Tm9. Hex sampling is a planar local approximation, not an optical eye model. Learned signs and weights are unconstrained; synapse counts do not initialize strengths.'
 return result

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=35);p.add_argument('--seed',type=int,default=401);p.add_argument('--benchmark',action='store_true');a=p.parse_args()
 if a.benchmark:
  runs=[]
  for seed in [401,402,403]:
   r=run(seed,a.epochs,save_weights=True)
   if seed==401:(BASE/'data/vision-example.json').write_text(json.dumps(r,separators=(',',':')))
   runs.append({'seed':seed,'models':[{k:(v if k!='eye' else {kk:vv for kk,vv in v.items() if kk!='projection'}) for k,v in m.items() if k not in ['demo_activity','learned_edges','predictions','node_models','eye_input_demo']} for m in r['models']],'constant_baseline':r['constant_baseline'],'hazard_prevalence':r['hazard_prevalence']})
  (BASE/'data/vision-benchmark.json').write_text(json.dumps({'epochs':a.epochs,'runs':runs,'note':'Exploratory simulation, three seeds, fixed defaults. No biological validation or claim of connectome superiority.'},indent=2))
 else:
  r=run(a.seed,a.epochs,save_weights=True);(BASE/'data/vision-example.json').write_text(json.dumps(r,separators=(',',':')))
