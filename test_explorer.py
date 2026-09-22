import copy
import unittest
import numpy as np
from explorer import Explorer
from membrane import run_membrane
from sdr_math import SDRArithmetic

class ExplorerTests(unittest.TestCase):
 def test_measured_graph_and_skeleton(self):
  e=Explorer();g=e.graph()
  self.assertGreater(len(g['nodes']),5000)
  self.assertTrue(all(a in e.nodes and b in e.nodes and w>0 for a,b,w in g['edges']))
  s=e.skeleton(10360)
  self.assertGreater(len(s['points']),1000)
  np.testing.assert_allclose(s['points'][0][1:4],np.array([32960,19584,15424])*.008)
  self.assertEqual(len(s['sha256']),64)
  self.assertEqual(len(e.incoming[10360]),sum(b==10360 for a,b,w in g['edges']))
  with self.assertRaises(ValueError):e.skeleton(-1)

class MembraneTests(unittest.TestCase):
 def test_excitation_rest_and_channel_block(self):
  active=run_membrane();rest=run_membrane(current=0);blocked=run_membrane(block_na=True)
  self.assertGreater(len(active['spike_times_ms']),0)
  self.assertEqual(rest['spike_times_ms'],[])
  self.assertEqual(blocked['spike_times_ms'],[])
  self.assertTrue(all(x==0 for x in blocked['sodium_current']))
  for r in [active,rest,blocked]:
   self.assertTrue(np.isfinite(r['voltage_mv']).all())
   for gate in ['m','h','n']:self.assertTrue(all(0<=v<=1 for v in r[gate]))
 def test_context_and_inhibition(self):
  self.assertEqual(run_membrane(context=3)['plateau_current'],0)
  self.assertEqual(run_membrane(context=4)['plateau_current'],4)
  self.assertEqual(run_membrane(inhibition=10)['spike_times_ms'],[])
 def test_validation(self):
  for opts in [dict(current=float('nan')),dict(context=9),dict(block_na='yes')]:
   with self.assertRaises(ValueError):run_membrane(**opts)

class SDRTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=SDRArithmetic()
 def test_sparsity_and_overlap(self):
  p,train,test,f,*_=self.m.prepare(101,'intact');codes=self.m.codes
  np.testing.assert_equal(codes.sum(axis=2),8)
  self.assertEqual(np.dot(codes[0,3],codes[0,4]),6)
  self.assertEqual(np.dot(codes[0,0],codes[0,9]),0)
  self.assertTrue(all(len(idx)<=int(self.m.c.nk*.02) for idx,x in f))
  self.assertEqual(train.sum(),80);self.assertEqual(test.sum(),20)
 def test_unseen_labels_cannot_train_decoder(self):
  r=self.m.run(seed=101,epochs=1);changed=copy.deepcopy(r)
  for p in changed['problems']:
   if p['split']=='unseen':p['correct']=999
  a=self.m.decoder_comparison(r,101,'intact');b=self.m.decoder_comparison(changed,101,'intact')
  for x,y in zip(a,b):np.testing.assert_equal(x['predictions'],y['predictions'])
 def test_no_learning_and_reproducibility(self):
  r=self.m.run(seed=101,epochs=1,condition='no_learning')
  self.assertTrue(all(p['prediction']==p['initial_prediction'] for p in r['problems']))
  self.assertTrue(all(v==0 for d in r['decoder_comparison'] for v in d['predictions']))
  self.assertEqual(r,self.m.run(seed=101,epochs=1,condition='no_learning'))

if __name__=='__main__':unittest.main()
