"""Generic Hodgkin-Huxley membrane patch with an illustrative dendritic gate.
Classical squid-axon parameters, not fitted to the selected Drosophila neuron.
No morphology-based cable simulation, compartment reconstruction, or SDR training.
"""
import math
import numpy as np
from scipy.integrate import solve_ivp

def vtrap(x,scale):
 return scale if abs(x/scale)<1e-7 else x/math.expm1(x/scale)
def rates(v):
 return (.1*vtrap(-(v+40),10),4*math.exp(-(v+65)/18),.07*math.exp(-(v+65)/20),
         1/(1+math.exp(-(v+35)/10)),.01*vtrap(-(v+55),10),.125*math.exp(-(v+65)/80))
def run_membrane(current=10,inhibition=0,context=0,block_na=False,body_id=10360):
 for k,v,lo,hi in [('current',current,0,25),('inhibition',inhibition,0,20),('context',context,0,8)]:
  if not math.isfinite(float(v)) or not lo<=float(v)<=hi:raise ValueError(f'Invalid {k}')
 current=float(current);inhibition=float(inhibition);context=float(context)
 if not isinstance(block_na,bool):raise ValueError('block_na must be boolean')
 # A hypothetical dendritic coincidence unit: >=4/8 context bits adds a plateau current.
 plateau=4. if context>=4 else 0.
 gn=0. if block_na else 120.
 a,b,c,d,e,f=rates(-65);initial=[-65,a/(a+b),c/(c+d),e/(e+f)]
 def external(t):return (current-inhibition+plateau) if 10<=t<40 else 0.
 def derivative(t,state):
  v,m,h,n=state;am,bm,ah,bh,an,bn=rates(v)
  ina=gn*m**3*h*(v-50);ik=36*n**4*(v+77);il=.3*(v+54.387)
  return [external(t)-ina-ik-il,am*(1-m)-bm*m,ah*(1-h)-bh*h,an*(1-n)-bn*n]
 times=np.arange(0,60.0001,.05)
 sol=solve_ivp(derivative,[0,60],initial,t_eval=times,max_step=.025,rtol=1e-7,atol=1e-9)
 if not sol.success:raise ValueError('Membrane solver did not converge')
 v,m,h,n=sol.y;ina=gn*m**3*h*(v-50);ik=36*n**4*(v+77)
 spikes=np.flatnonzero((v[:-1]<0)&(v[1:]>=0))+1
 return {'bodyId':int(body_id),'parameters':{'current':current,'inhibition':inhibition,'context':context,'block_na':block_na},
         'time_ms':times.tolist(),'voltage_mv':v.tolist(),'sodium_current':ina.tolist(),'potassium_current':ik.tolist(),
         'm':m.tolist(),'h':h.tolist(),'n':n.tolist(),'injected_current':[external(t) for t in times],
         'spike_times_ms':times[spikes].tolist(),'plateau_current':plateau,
         'units':{'currents':'microampere / cm²','conductance':'mS / cm²','capacitance':'1 microfarad / cm²'},
         'note':'Generic HH membrane patch; classical squid-axon parameters, not a fitted fly neuron. The selected ID is contextual only. Context gate is an added illustrative threshold rule, not measured dendritic computation. Network and math models are separate.'}
