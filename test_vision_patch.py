import unittest
import json
from model import BASE
import numpy as np
from vision import Net,circuit,scenes,run
class VisionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.data=circuit()
 def test_anatomy(self):
  d=self.data;ids={n['bodyId'] for n in d['nodes']};self.assertEqual(len(d['columns']),19)
  self.assertTrue(all(a in ids and b in ids and w>0 for a,b,w in d['edges']))
  self.assertTrue(all(n['hex']==d['columns'][n['column']]['hex'] for n in d['nodes']))
 def test_matched_rewiring(self):
  a=Net(self.data,'graph_vector',401);b=Net(self.data,'shuffled_vector',401)
  for axis in [0,1]:np.testing.assert_equal(a.mask.sum(axis),b.mask.sum(axis))
  np.testing.assert_equal(a.mask.diagonal(),b.mask.diagonal());self.assertGreater(b.swaps,0);self.assertFalse(np.array_equal(a.mask,b.mask))
  for k in ['U','G','O','b','c']:np.testing.assert_equal(a.p[k],b.p[k])
  np.testing.assert_allclose(np.linalg.norm(a.p['W'],axis=1),np.linalg.norm(b.p['W'],axis=1),atol=1e-7)
 def test_gradients(self):
  x,y,_=scenes(self.data['columns'],991,3,steps=3);x=x.astype(float);y=y.astype(float);budget=Net(self.data,'graph_vector',401).count
  for kind in ['graph_scalar','graph_vector','dense_rnn']:
   n=Net(self.data,kind,401,budget);n.p={k:v.astype(float) for k,v in n.p.items()};loss,g=n.loss_grad(x,y)
   wi=tuple(np.argwhere(n.mask)[0]);gi=(0,0) if n.dense else (int(np.flatnonzero(n.input_mask[:,0])[0]),0)
   for key,idx in [('W',wi),('U',(0,0,0)),('b',(0,0)),('G',gi),('O',(0,0)),('c',(2,))]:
    old=n.p[key][idx];eps=1e-4;n.p[key][idx]=old+eps;hi=n.loss_grad(x,y)[0];n.p[key][idx]=old-eps;lo=n.loss_grad(x,y)[0];n.p[key][idx]=old
    self.assertAlmostEqual(g[key][idx],(hi-lo)/(2*eps),delta=3e-5,msg=f'{kind}/{key}')
 def test_masks_evaluation_and_independent_scenes(self):
  n=Net(self.data,'graph_vector',401);x,y,_=scenes(self.data['columns'],10401,8);other=scenes(self.data['columns'],30401,8)[0];self.assertFalse(np.array_equal(x,other))
  for _ in range(3):loss,g=n.loss_grad(x,y);n.update(g)
  self.assertEqual(np.count_nonzero(n.p['W']*(1-n.mask)),0);self.assertEqual(np.count_nonzero(n.p['G']*(1-n.input_mask)),0)
  before=n.digest();a=n.forward(x);y[:]=999;np.testing.assert_equal(a,n.forward(x));self.assertEqual(before,n.digest())
 def test_budget_validation(self):
  count=Net(self.data,'graph_vector',401).count;dense=Net(self.data,'dense_rnn',401,count);self.assertLess(abs(dense.count-count)/count,.04)
  for opts in [dict(seed=-1),dict(epochs=81),dict(epochs=1.5)]:
   with self.assertRaises(ValueError):run(**opts)
 def test_saved_weights_reproduce_predictions(self):
  r=json.loads((BASE/'data/vision-example.json').read_text());x,y,_=scenes(self.data['columns'],r['seed']+30000,128)
  for m in r['models']:
   n=Net(self.data,m['kind'],r['seed'],3134)
   with np.load(BASE/f"data/vision-{r['seed']}-{m['kind']}.npz") as checkpoint:
    for k in n.p:n.p[k]=checkpoint[k]
   np.testing.assert_allclose(n.forward(x),m['predictions'],atol=1e-4)  # float32 BLAS rounding differs across machines (~4e-6)
   self.assertEqual(m['test_weight_hash_before'],m['test_weight_hash_after'])
   if 'ablation_restored_hash' in m:self.assertEqual(m['ablation_restored_hash'],m['test_weight_hash_after'])
 def test_exported_node_computation_matches_activity(self):
  r=json.loads((BASE/'data/vision-example.json').read_text());x=np.array(r['demonstrations'][0]['retina']);cols=np.array([n['column'] for n in self.data['nodes']])
  for m in r['models']:
   if m['kind']=='dense_rnn':continue
   h=np.zeros((m['nodes'],m['state_size']));w=np.zeros((m['nodes'],m['nodes']))
   for a,b,v in m['learned_edges']:w[b,a]=v
   nm={k:np.array(v) for k,v in m['node_models'].items()};xm=x
   if 'eye_input_demo' in m:   # eye models: nodes receive the eye's output; recompute it from exported eye parameters
    e=m['eye'];P=np.array(e['projection']);y=x[0].copy();u=[]
    for t in range(10):y=e['adaptation_a']*y+(1-e['adaptation_a'])*x[t];u.append(P@(e['gain']*(x[t]-e['adaptation_k']*y)))
    xm=np.array(m['eye_input_demo'][0]);np.testing.assert_allclose(np.array(u),xm,atol=1e-5)
   for t in range(10):
    h=.5*h+.5*np.tanh(w@h+np.einsum('ni,nij->nj',h,nm['U'])+xm[t,cols,None]*nm['G']+nm['b'])
    np.testing.assert_allclose(h,m['demo_activity'][0][t],atol=2e-5)
if __name__=='__main__':unittest.main()


class EyeLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.data = circuit()

    def test_untrained_eye_is_exactly_the_measured_graph(self):
        x, y, _ = scenes(self.data['columns'], 991, 6)
        base = Net(self.data, 'graph_vector', 401)
        for kind in ['eye_graph', 'eye_dense']:
            eye = Net(self.data, kind, 401)
            for k in base.p: np.testing.assert_array_equal(base.p[k], eye.p[k])
            np.testing.assert_allclose(base.forward(x), eye.forward(x), atol=1e-6)

    def test_eye_gradients_match_finite_differences(self):
        x, y, _ = scenes(self.data['columns'], 992, 3, steps=4); x = x.astype(float); y = y.astype(float)
        for kind in ['eye_graph', 'eye_dense']:
            n = Net(self.data, kind, 401)
            n.p = {k: v.astype(float) for k, v in n.p.items()}
            n.p['E'][:] = [0.3, 0.4, 0.9]                       # away from the identity point
            n.p['P'] += 0.05 * n.pmask * np.random.default_rng(1).normal(size=n.p['P'].shape)
            loss, g = n.loss_grad(x, y)
            off = tuple(np.argwhere(n.pmask - np.eye(len(n.pmask)))[0])
            for key, idx in [('E', (0,)), ('E', (1,)), ('E', (2,)), ('P', (0, 0)), ('P', off), ('W', tuple(np.argwhere(n.mask)[0])), ('G', (0, 0))]:
                old = n.p[key][idx]; eps = 1e-5
                n.p[key][idx] = old + eps; hi = n.loss_grad(x, y)[0]
                n.p[key][idx] = old - eps; lo = n.loss_grad(x, y)[0]
                n.p[key][idx] = old
                self.assertAlmostEqual(g[key][idx], (hi - lo) / (2 * eps), delta=2e-5, msg=f'{kind}/{key}{idx}')

    def test_retinotopic_projection_stays_local(self):
        n = Net(self.data, 'eye_graph', 401)
        self.assertEqual(int(n.pmask.sum()), sum(1 for _ in np.argwhere(n.pmask)))
        self.assertLessEqual(n.pmask.sum(axis=1).max(), 7)       # a column and at most six neighbours
        self.assertEqual(Net(self.data, 'eye_dense', 401).pmask.sum(), 19 * 19)
        x, y, _ = scenes(self.data['columns'], 993, 16)
        for _ in range(4):
            loss, g = n.loss_grad(x, y); n.update(g)
        self.assertEqual(np.count_nonzero(n.p['P'] * (1 - n.pmask)), 0)
        self.assertNotEqual(float(n.p['E'][1]), 0.0)             # the eye actually trains
