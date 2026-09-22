"""Addition learning probe; targets are computed only by the teacher/evaluator.

Artificial digit codes -> measured PN/KC projection -> sparse KC features ->
plastic gains on measured KC/MBON support -> 19 arbitrarily assigned answer pools.
No claim of biological numeracy, arithmetic rules, or integration with walking.
"""
import hashlib
import math
import json
import numpy as np
from learning import LearningCircuit
from model import BASE

CONDITIONS=['intact','no_learning','shuffled_labels','shuffled_wiring','silence_kc']
class Arithmetic:
 def __init__(self,circuit=None):self.c=circuit or LearningCircuit()
 def prepare(self,seed,condition):
  c=self.c;rng=np.random.default_rng(seed+2300)
  pairs=np.array([(a,b) for a in range(10) for b in range(10)])
  candidates=[(a,b) for a in range(10) for b in range(a+1,10) if (a,b) not in [(0,1),(8,9)]]
  held={candidates[i] for i in rng.choice(len(candidates),10,replace=False)}
  test=np.array([tuple(sorted(p)) in held for p in pairs]);train=~test
  # Each operand slot gets its own half of the PN input population; digit codes
  # have no numeric magnitude or sum feature. Codes are generated without targets.
  code=np.zeros((2,10,c.np))
  half=c.np//2
  for slot in range(2):
   population=np.arange(0,half) if slot==0 else np.arange(half,c.np)
   for digit in range(10):code[slot,digit,rng.choice(population,max(1,len(population)//5),replace=False)]=1
  encoder=c.encoder('shuffled' if condition=='shuffled_wiring' else 'intact',seed)
  # Answer identity is a randomly assigned artificial MBON grouping, never inferred valence.
  groups=rng.permutation(len(c.groups['MBON']))%19
  support=np.zeros((c.nk,19))
  np.add.at(support,(c.kidx,groups[c.km[:,1]]),c.base)
  features=[]
  for a,b in pairs:
   kc=encoder.dot(code[0,a]+code[1,b]);kc=np.maximum(kc-np.quantile(kc,.95),0)
   if condition=='silence_kc':kc[:]=0
   idx=np.flatnonzero(kc>0)
   f=kc[idx,None]*support[idx]
   # Fixed per-example, per-output L2 normalization; does not use teacher labels.
   f/=np.maximum(np.sqrt((f*f).sum(axis=0)),1e-12)
   features.append((idx,f))
  # Tiny fixed, label-independent tie breaker for top-1 reporting, not learning.
  tie=rng.uniform(-1e-9,1e-9,(100,19))
  return pairs,train,test,features,support,groups,tie
 def run(self,seed=7,epochs=100,learning_rate=.15,condition='intact'):
  for k,v,lo,hi in [('seed',seed,0,999999),('epochs',epochs,0,300)]:
   if not math.isfinite(float(v)) or int(v)!=float(v) or not lo<=int(v)<=hi:raise ValueError(f'Invalid {k}')
  seed=int(seed);epochs=int(epochs);learning_rate=float(learning_rate)
  if not math.isfinite(learning_rate) or not 0<=learning_rate<=1:raise ValueError('Invalid learning rate')
  if condition not in CONDITIONS:raise ValueError('Unknown arithmetic condition')
  pairs,train,test,features,support,groups,tie=self.prepare(seed,condition)
  # Teacher labels, NEVER provided to the input encoder or prediction function.
  truth=pairs.sum(axis=1);teacher=truth.copy();rng=np.random.default_rng(seed+7300)
  if condition=='shuffled_labels':teacher[train]=rng.permutation(teacher[train])
  delta=np.zeros_like(support);history=[]
  def predict(i):
   idx,f=features[i];logits=(delta[idx]*f).sum(axis=0)
   exp=np.exp(logits-logits.max());return exp/exp.sum()
  def assess(epoch):
   probs=np.array([predict(i) for i in range(100)]);pred=(probs+tie).argmax(axis=1)
   history.append({'epoch':epoch,'train_accuracy':float(np.mean(pred[train]==truth[train])),
                   'teacher_accuracy':float(np.mean(pred[train]==teacher[train])),
                   'test_accuracy':float(np.mean(pred[test]==truth[test])),
                   'test_true_probability':float(probs[np.flatnonzero(test),truth[test]].mean())})
   return probs,pred
  initial_probs,initial_pred=assess(0)
  for epoch in range(1,epochs+1):
   for i in rng.permutation(np.flatnonzero(train)):
    prob=predict(i)
    if condition not in ['no_learning','silence_kc']:
     idx,f=features[i];error=-prob;error[teacher[i]]+=1
     # Gains shared across existing edges from a KC to MBONs in an answer pool.
     delta[idx]=np.clip(delta[idx]+learning_rate*f*error*self.c.gate[idx,None],-.95,8)
   if epoch%5==0 or epoch==epochs:assess(epoch)
  frozen_before=hashlib.sha256(delta.tobytes()).hexdigest()
  probs=np.array([predict(i) for i in range(100)]);pred=(probs+tie).argmax(axis=1)
  frozen_after=hashlib.sha256(delta.tobytes()).hexdigest()
  majority=int(np.bincount(truth[train],minlength=19).argmax())
  rows=[{'a':int(a),'b':int(b),'correct':int(truth[i]),'teacher_label':int(teacher[i]) if train[i] else None,
         'prediction':int(pred[i]),'initial_prediction':int(initial_pred[i]),'split':'train' if train[i] else 'unseen',
         'confidence':float(probs[i,pred[i]]),'probabilities':probs[i].tolist()} for i,(a,b) in enumerate(pairs)]
  return {'parameters':{'seed':seed,'epochs':epochs,'learning_rate':learning_rate,'condition':condition},
          'history':history,'problems':rows,'summary':history[-1],
          'baseline':{'uniform_random_accuracy':1/19,'majority_answer':majority,'majority_test_accuracy':float(np.mean(truth[test]==majority))},
          'audit':{'training_pairs':int(train.sum()),'unseen_pairs':int(test.sum()),'test_weights_before':frozen_before,
                   'test_weights_after':frozen_after,'updated_supported_gains':int(np.count_nonzero(delta)),
                   'unsupported_nonzero':int(np.count_nonzero(delta[support==0])),
                   'reverse_pairs_held_out_together':True,'test_feedback_used':False,
                   'answer_pool_assignments':[{'bodyId':n['bodyId'],'answer':int(groups[i])} for i,n in enumerate(self.c.groups['MBON'])]},
          'provenance':{'dataset':self.c.data['dataset'],'files':self.c.data['files']},
          'interpretation':'Supervised addition classifier using artificial digit encoding and output pools on measured pathway support. Training accuracy measures memorization; unseen-pair accuracy tests limited interpolation. No out-of-range generalization or biological math ability is established. Test labels score predictions only and never update weights.'}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');args=p.parse_args()
 model=Arithmetic()
 if args.benchmark:
  out={'runs':[],'seeds':[7,8,9],'conditions':CONDITIONS,'note':'Default 100 epochs; no hyperparameter selection using held-out results. Three synthetic seeds, not biological replicates.'}
  for seed in out['seeds']:
   for condition in CONDITIONS:
    r=model.run(seed=seed,condition=condition);out['runs'].append({k:r[k] for k in ['parameters','summary','baseline','audit']})
  path=BASE/'data/math-benchmark.json'
 else:out=model.run();path=BASE/'data/math-example.json'
 path.write_text(json.dumps(out,indent=2));print(path)
