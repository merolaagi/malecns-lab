"""SDR-inspired encoding comparison; NOT Monty or a full HTM implementation."""
import json
import numpy as np
from arithmetic import Arithmetic
from model import BASE

class SDRArithmetic(Arithmetic):
 def __init__(self,circuit=None,encoding='scalar_sdr'):
  super().__init__(circuit)
  if encoding not in ['scalar_sdr','random_sdr','permuted_scalar']:raise ValueError('Unknown SDR encoding')
  self.encoding=encoding
 def prepare(self,seed,condition):
  pairs,train,test,unused,support,groups,tie=super().prepare(seed,condition)
  c=self.c;rng=np.random.default_rng(seed+99000);half=c.np//2;width=8;stride=2
  code=np.zeros((2,10,c.np))
  for slot in range(2):
   population=np.arange(0,half) if slot==0 else np.arange(half,c.np)
   order=rng.permutation(population)
   digit_order=rng.permutation(10) if self.encoding=='permuted_scalar' else np.arange(10)
   for digit in range(10):
    active=(rng.choice(population,width,replace=False) if self.encoding=='random_sdr'
            else order[int(digit_order[digit])*stride:int(digit_order[digit])*stride+width])
    code[slot,digit,active]=1
  self.codes=code;self.demo=[]
  encoder=c.encoder('shuffled' if condition=='shuffled_wiring' else 'intact',seed)
  features=[]
  for a,b in pairs:
   scores=encoder.dot(code[0,a]+code[1,b]);active=np.argsort(scores,kind='stable')[-max(1,int(c.nk*.02)):]
   kc=np.zeros(c.nk);kc[active[scores[active]>0]]=1
   if condition=='silence_kc':kc[:]=0
   idx=np.flatnonzero(kc);f=kc[idx,None]*support[idx]
   f/=np.maximum(np.sqrt((f*f).sum(axis=0)),1e-12)
   features.append((idx,f))
   if (a,b) in [(2,3),(2,4)]:
    self.demo.append({'a':int(a),'b':int(b),'pn_active':np.flatnonzero(code[0,a]+code[1,b]).tolist(),
                      'kc_active':idx.tolist(),'kc_body_ids':[c.groups['KC'][i]['bodyId'] for i in idx]})
  return pairs,train,test,features,support,groups,tie
 def decoder_comparison(self,result,seed,condition):
  # Alternative engineered ridge readouts; NOT the local plasticity model above.
  pairs,train,test,features,support,groups,tie=self.prepare(seed,condition)
  x=np.zeros((100,self.c.nk))
  for i,(idx,f) in enumerate(features):x[i,idx]=f.sum(axis=1)
  x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
  direct=np.array([self.codes[0,a]+self.codes[1,b] for a,b in pairs])
  direct/=np.maximum(np.linalg.norm(direct,axis=1,keepdims=True),1e-12)
  truth=np.array([p['correct'] for p in result['problems']])
  teacher=np.array([p['teacher_label'] for p in result['problems'] if p['split']=='train'])
  output=[]
  for name,features in [('KC feature ridge readout',x),('Direct input SDR ridge baseline',direct)]:
   # Center using training rows only. Fixed lambda; no tuning on held-out rows.
   mean=features[train].mean(axis=0);ymean=teacher.mean();xt=features[train]-mean
   alpha=np.linalg.solve(xt@xt.T+.1*np.eye(train.sum()),teacher-ymean)
   weights=xt.T@alpha
   prediction=(features-mean)@weights+ymean
   if condition=='no_learning' or result['parameters']['epochs']==0:prediction[:]=0
   rounded=np.clip(np.floor(prediction+.5),0,18).astype(int)
   output.append({'name':name,'train_accuracy':float(np.mean(rounded[train]==truth[train])),
                  'test_accuracy':float(np.mean(rounded[test]==truth[test])),
                  'test_mae':float(np.mean(np.abs(prediction[test]-truth[test]))),
                  'predictions':prediction.tolist(),'rounded_predictions':rounded.tolist(),
                  'lambda':.1,'fit':'One closed-form supervised ridge fit on 80 training examples; independent of epoch count except epochs=0. A decoder baseline, not biological synaptic learning.'})
  return output
 def run(self,seed=101,epochs=100,learning_rate=.15,condition='intact'):
  r=super().run(seed=seed,epochs=epochs,learning_rate=learning_rate,condition=condition)
  r['parameters']['encoding']=self.encoding
  r['decoder_comparison']=self.decoder_comparison(r,seed,condition)
  r['sdr']={'pn_cells':self.c.np,'kc_cells':self.c.nk,'active_bits_per_operand':8,'kc_sparsity_target':.02,
            'digit_codes':[[np.flatnonzero(v).tolist() for v in slot] for slot in self.codes],
            'examples':self.demo,'scope':'Scalar code overlap, binary top-k population coding and overlap inspection are inspired by SDR principles. No Monty package, HTM temporal memory, learned dendritic segments, or cortical-column model is implemented.',
            'references':['https://github.com/thousandbrainsproject/tbp.monty','https://arxiv.org/html/1509.08255v2','https://seanpedersen.github.io/posts/sparse-distributed-representations/']}
  return r
if __name__=='__main__':
 from learning import LearningCircuit
 c=LearningCircuit();out={'runs':[],'seeds':[101,102,103],
 'note':'Preselected settings: 100 epochs, learning rate .15, 8 active input bits per operand, 2% KC winners. Held-out questions are not used for optimization. New seeds relative to original arithmetic benchmark. Encoding/order/shuffle comparisons; no Monty integration claim.'}
 for encoding,condition in [('random_sdr','intact'),('scalar_sdr','intact'),('permuted_scalar','intact'),('scalar_sdr','shuffled_wiring'),('scalar_sdr','no_learning')]:
  for seed in out['seeds']:
   r=SDRArithmetic(c,encoding).run(seed=seed,condition=condition)
   out['runs'].append({k:r[k] for k in ['parameters','summary','baseline','audit','decoder_comparison']})
 (BASE/'data/sdr-benchmark.json').write_text(json.dumps(out,indent=2))
 example=SDRArithmetic(c).run();(BASE/'data/sdr-example.json').write_text(json.dumps(example,indent=2))
 print([(r['parameters']['encoding'],r['parameters']['condition'],r['summary']['test_accuracy']) for r in out['runs']],flush=True)
