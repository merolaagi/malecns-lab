import unittest
import numpy as np
from learning import LearningCircuit

class LearningTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.c=LearningCircuit()
 def test_reproducible(self):self.assertEqual(self.c.run(seed=8),self.c.run(seed=8))
 def test_tests_never_change_weights_or_deliver_rewards(self):
  r=self.c.run()
  for t in r['trials']:
   if not t['training']:
    self.assertEqual(t['reward'],0);self.assertEqual(t['weight_change_l2'],0)
  for c in r['checkpoints']:
   if c['phase'] not in ['Acquisition','Reversal']:self.assertEqual(c['before_sha256'],c['after_sha256'])
 def test_no_reward_information_in_baseline(self):
  a=self.c.run(rewarded_odor='A');b=self.c.run(rewarded_odor='B')
  x=[(t['p_a'],t['choice']) for t in a['trials'] if t['phase']=='Baseline']
  y=[(t['p_a'],t['choice']) for t in b['trials'] if t['phase']=='Baseline']
  self.assertEqual(x,y)
 def test_disabled_mechanisms_cannot_learn(self):
  for condition in ['no_plasticity','silence_kc','no_reward','no_dopamine']:
   r=self.c.run(condition=condition)
   self.assertEqual(r['weight_change']['nonzero_edges'],0)
   self.assertTrue(all(t['p_a']==.5 for t in r['trials']))
 def test_zero_retention_erases_memory(self):
  r=self.c.run(retention=0,delay=1)
  self.assertEqual(r['summary'][3]['mean_probability'],.5)
 def test_acquisition_and_reversal_with_both_initial_rewards(self):
  for odor in ['A','B']:
   r=self.c.run(rewarded_odor=odor)
   self.assertGreater(r['summary'][2]['mean_probability'],.8)
   self.assertGreater(r['summary'][5]['mean_probability'],.8)
 def test_side_layout_counterbalanced_and_choice_mapping_correct(self):
  r=self.c.run()
  for s in r['summary']:
   rows=[t for t in r['trials'] if t['phase']==s['phase']]
   self.assertLessEqual(abs(sum(t['odor_a_side']=='left' for t in rows)-len(rows)/2),.5)
  for t in r['trials']:
   self.assertEqual(t['choice_side']==t['odor_a_side'],t['choice']=='A')
 def test_no_edges_created_by_plasticity(self):
  r=self.c.run();existing={(e[0],e[1]) for e in self.c.data['edges']}
  for e in r['weight_change']['largest_changes']:self.assertIn((e['pre'],e['post']),existing)
  self.assertGreaterEqual(r['weight_change']['min_multiplier'],.05-1e-10)
 def test_shuffle_changes_encoder_and_is_reproducible(self):
  a=self.c.encoder('intact',7);b=self.c.encoder('shuffled',7)
  self.assertGreater((a!=b).nnz,0)
  self.assertEqual((b!=self.c.encoder('shuffled',7)).nnz,0)
 def test_invalid_parameters(self):
  for p in [dict(seed=-1),dict(train_trials=10.5),dict(temperature=0),dict(retention=float('nan')),dict(rewarded_odor='C'),dict(condition='x')]:
   with self.assertRaises(ValueError):self.c.run(**p)
if __name__=='__main__':unittest.main()
