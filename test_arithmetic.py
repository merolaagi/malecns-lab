import unittest
import numpy as np
from arithmetic import Arithmetic
class ArithmeticTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.m=Arithmetic()
 def test_split_and_reversals_do_not_leak(self):
  r=self.m.run(epochs=2);train={(p['a'],p['b']) for p in r['problems'] if p['split']=='train'};test={(p['a'],p['b']) for p in r['problems'] if p['split']=='unseen'}
  self.assertEqual(len(train),80);self.assertEqual(len(test),20);self.assertFalse(train&test)
  self.assertTrue(all((b,a) in test for a,b in test))
  self.assertEqual(r['audit']['test_weights_before'],r['audit']['test_weights_after'])
 def test_no_learning_and_zero_rate_hold_initial_predictions(self):
  for opts in [dict(condition='no_learning'),dict(learning_rate=0),dict(condition='silence_kc')]:
   r=self.m.run(epochs=10,**opts);self.assertEqual(r['audit']['updated_supported_gains'],0)
   self.assertTrue(all(p['prediction']==p['initial_prediction'] for p in r['problems']))
 def test_labels_do_not_change_encoding_or_split(self):
  a=self.m.prepare(7,'intact');b=self.m.prepare(7,'shuffled_labels')
  np.testing.assert_equal(a[1],b[1])
  for (idx,x),(idx2,y) in zip(a[3],b[3]):np.testing.assert_equal(idx,idx2);np.testing.assert_equal(x,y)
 def test_shuffled_labels_only_change_teacher_training_answers(self):
  a=self.m.run(epochs=0);b=self.m.run(epochs=0,condition='shuffled_labels')
  self.assertTrue(any(x['teacher_label']!=y['teacher_label'] for x,y in zip(a['problems'],b['problems'])))
  self.assertTrue(all(x['initial_prediction']==y['initial_prediction'] for x,y in zip(a['problems'],b['problems'])))
  self.assertTrue(all(p['teacher_label'] is None for p in b['problems'] if p['split']=='unseen'))
 def test_reproducible_and_no_new_edges(self):
  a=self.m.run(epochs=5);self.assertEqual(a,self.m.run(epochs=5));self.assertEqual(a['audit']['unsupported_nonzero'],0)
 def test_teacher_fit_improves_without_asserting_generalization(self):
  r=self.m.run(epochs=30);self.assertGreater(r['summary']['teacher_accuracy'],r['history'][0]['teacher_accuracy'])
 def test_invalid_input(self):
  for p in [dict(epochs=1.5),dict(seed=-1),dict(condition='x'),dict(learning_rate=float('nan'))]:
   with self.assertRaises(ValueError):self.m.run(**p)
if __name__=='__main__':unittest.main()
