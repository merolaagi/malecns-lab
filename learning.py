"""Trial-based associative learning hypothesis constrained by MaleCNS edge support.

Synthetic PN odors, imposed sparsification, reward error, PAM gate, synaptic
plasticity and MBON value readout are engineering assumptions, not recovered biology.
This module does not use the walking model to make choices.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from model import BASE

CONDITIONS=['intact','no_plasticity','silence_kc','no_reward','no_dopamine','shuffled']
DEFAULT=dict(seed=7,train_trials=80,probe_trials=40,learning_rate=.2,temperature=.2,
             retention=.995,delay=30,condition='intact',rewarded_odor='A')

class LearningCircuit:
    def __init__(self,path=None):
        self.data=json.loads(Path(path or BASE/'data/learning-circuit.json').read_text())
        self.groups={r:[n for n in self.data['nodes'] if n['role']==r] for r in ['PN','KC','MBON','PAM']}
        self.maps={r:{n['bodyId']:i for i,n in enumerate(ns)} for r,ns in self.groups.items()}
        self.nk=len(self.groups['KC']);self.np=len(self.groups['PN'])
        pn=[];km=[];gate=np.zeros(self.nk)
        for pre,post,w in self.data['edges']:
            if pre in self.maps['PN'] and post in self.maps['KC']: pn.append([self.maps['PN'][pre],self.maps['KC'][post],w])
            elif pre in self.maps['KC'] and post in self.maps['MBON']: km.append([self.maps['KC'][pre],self.maps['MBON'][post],w,pre,post])
            elif pre in self.maps['PAM'] and post in self.maps['KC']:gate[self.maps['KC'][post]]+=w
        self.pn=np.array(pn);self.km=np.array(km)
        self.gate=np.log1p(gate);self.gate/=max(self.gate.max(),1)
        self.base=np.log1p(self.km[:,2]);self.kidx=self.km[:,0].astype(int)

    def encoder(self,condition,seed):
        pre,post,w=self.pn.T
        if condition=='shuffled':post=np.random.default_rng(seed+55000).permutation(post)
        weights=np.log1p(w);norm=np.bincount(post,weights=weights,minlength=self.nk)
        return csr_matrix((weights/np.maximum(norm[post],1),(post,pre)),shape=(self.nk,self.np))

    def run(self,**options):
        p=DEFAULT|options
        if p['condition'] not in CONDITIONS:raise ValueError('Unknown learning condition')
        if p['rewarded_odor'] not in ['A','B']:raise ValueError('Rewarded odor must be A or B')
        for k,lo,hi in [('seed',0,999999),('train_trials',10,200),('probe_trials',10,100),('delay',0,200)]:
            v=float(p[k])
            if not math.isfinite(v) or v!=int(v) or not lo<=v<=hi:raise ValueError(f'{k} must be an integer in [{lo},{hi}]')
            p[k]=int(v)
        for k,lo,hi in [('learning_rate',0,1),('temperature',.05,2),('retention',0,1)]:
            p[k]=float(p[k])
            if not math.isfinite(p[k]) or not lo<=p[k]<=hi:raise ValueError(f'Invalid {k}')
        rng=np.random.default_rng(p['seed'])
        stimulus_rng=np.random.default_rng(p['seed']+1000)
        w=self.encoder(p['condition'],p['seed'])
        # Two abstract odors each activate a random 20% of measured PN cells.
        patterns=np.zeros((2,self.np))
        for i in range(2):patterns[i,stimulus_rng.choice(self.np,max(1,self.np//5),replace=False)]=1
        prototypes=[]
        # Independent noisy test exemplars; no reward information reaches this encoding.
        features={}
        for phase in ['training','testing']:
            examples=[]
            for sample in range(4):
                odors=[]
                for i in range(2):
                    pn=np.clip(patterns[i]+stimulus_rng.normal(0,.035,self.np),0,1)
                    activity=w.dot(pn)
                    threshold=np.quantile(activity,.95)
                    activity=np.maximum(activity-threshold,0)
                    if p['condition']=='silence_kc':activity[:]=0
                    mass=activity[self.kidx]*self.base
                    idx=np.flatnonzero(mass>0)
                    phi=mass[idx]/max(mass.sum(),1e-12)
                    odors.append((idx,phi))
                    if sample==0 and phase=='training':prototypes.append(set(np.flatnonzero(activity)))
                examples.append(odors)
            features[phase]=examples
        delta=np.zeros(len(self.km))
        trials=[];summaries=[];checkpoints=[]
        def digest():return hashlib.sha256(delta.tobytes()).hexdigest()
        rewarded=0 if p['rewarded_odor']=='A' else 1
        schedule=[('Baseline',p['probe_trials'],False,rewarded),('Acquisition',p['train_trials'],True,rewarded),
                  ('Unrewarded probe',p['probe_trials'],False,rewarded),('Delayed memory',p['probe_trials'],False,rewarded),
                  ('Reversal',p['train_trials'],True,1-rewarded),('Reversal probe',p['probe_trials'],False,1-rewarded)]
        for phase,n,training,target in schedule:
            if phase=='Delayed memory':delta*=p['retention']**p['delay']
            before=digest()
            # Counterbalanced left/right layout in each phase; side is never passed to the value function.
            sides=np.arange(n)%2;rng.shuffle(sides)
            probabilities=[];choices=[]
            for j in range(n):
                variant=j%4
                fs=features['training' if training else 'testing'][variant]
                values=np.array([np.dot(delta[idx],phi) for idx,phi in fs])
                pa=1/(1+math.exp(float(np.clip((values[1]-values[0])/p['temperature'],-60,60))))
                choice=0 if rng.random()<pa else 1
                delivered=float(training and choice==target and p['condition']!='no_reward')
                error=delivered-float(values[choice]) if training else None
                change=0.
                if training and p['condition'] not in ['no_plasticity','no_dopamine']:
                    idx,phi=fs[choice]
                    gate=self.gate[self.kidx[idx]]
                    denom=float(np.sum(phi*phi*gate))
                    if denom>0:
                        old=delta[idx].copy()
                        delta[idx]=np.clip(old+p['learning_rate']*error*phi*gate/denom,-.95,3)
                        change=float(np.linalg.norm(delta[idx]-old))
                probability_target=pa if target==0 else 1-pa
                trials.append({'trial':len(trials)+1,'phase':phase,'phase_trial':j+1,'training':training,
                               'odor_a_side':'left' if sides[j]==0 else 'right','choice':'AB'[choice],
                               'choice_side':('left' if sides[j]==choice else 'right'),
                               'target_odor':'AB'[target], 'reward':delivered,'p_a':pa,'p_target':probability_target,
                               'value_a':float(values[0]),'value_b':float(values[1]),'prediction_error':error,
                               'weight_change_l2':change,'exemplar':variant})
                probabilities.append(probability_target);choices.append(choice==target)
            after=digest()
            successes=int(sum(choices));z=1.96;prop=successes/n;den=1+z*z/n
            center=(prop+z*z/(2*n))/den;half=z*math.sqrt(prop*(1-prop)/n+z*z/(4*n*n))/den
            summaries.append({'phase':phase,'trials':n,'target_odor':'AB'[target],'choice_fraction':prop,
                              'mean_probability':float(np.mean(probabilities)),'ci95':[center-half,center+half],
                              'reward_count':sum(t['reward'] for t in trials[-n:]),'training':training})
            checkpoints.append({'phase':phase,'before_sha256':before,'after_sha256':after,'weights_changed':before!=after})
        top=np.argsort(np.abs(delta))[-20:][::-1]
        return {'parameters':p,'trials':trials,'summary':summaries,'checkpoints':checkpoints,
                'circuit':{'dataset':self.data['dataset'],'nodes':len(self.data['nodes']),'edges':len(self.data['edges']),
                           'groups':{r:len(n) for r,n in self.groups.items()},'plastic_edges':len(delta),
                           'active_kc_overlap':len(prototypes[0]&prototypes[1])/max(1,len(prototypes[0]|prototypes[1]))},
                'weight_change':{'nonzero_edges':int(np.count_nonzero(delta)),'l2':float(np.linalg.norm(delta)),
                                 'min_multiplier':float((1+delta).min()),'max_multiplier':float((1+delta).max()),
                                 'largest_changes':[{'pre':int(self.km[i,3]),'post':int(self.km[i,4]),'multiplier':float(1+delta[i])} for i in top if delta[i]!=0]},
                'provenance':{'source_files':self.data['files'],'selection':self.data['selection']},
                'interpretation':'Constructed associative learner, not an intelligence score or biological validation. Trial-level rate model is separate from walking. Odor encoding, PAM gating, prediction-error learning, decay and action readout are supplied assumptions. Test phases never update weights; delayed memory applies an explicit decay before testing. Shuffled circuits may also learn.'}

    def suite(self,seed=7):
        rows=[]
        for s in range(int(seed),int(seed)+5):
            for condition in CONDITIONS:
                r=self.run(seed=s,condition=condition,rewarded_odor='A' if s%2 else 'B')
                rows.append({'parameters':r['parameters'],'summary':r['summary']})
        return {'runs':rows,'seeds':5,'conditions':CONDITIONS,'interpretation':'Five initialization/stimulus seeds; original reward A/B counterbalanced across seeds. These are model trials, not biological replicates. Defaults only; no claims of connectome superiority.'}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args()
    circuit=LearningCircuit()
    result=circuit.suite(args.seed) if args.benchmark else circuit.run(seed=args.seed)
    target=BASE/'data'/('learning-benchmark.json' if args.benchmark else 'learning-example.json')
    target.write_text(json.dumps(result,indent=2))
    print(target)
