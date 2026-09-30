"""Extra unbiased layer-selection estimate and control-direction diagnostics.
Runs on saved activations without loading the GPU model.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.analysis import direction,layer_cv,auc
OUT=ROOT/'research/results/gemma-chat'
PAIN={'A1','A2','A3','A4','A5'}

def main():
 data=json.loads((ROOT/'data/paper/3.1_pain_and_control_datasets.json').read_text())['datasets']
 custom=json.loads((ROOT/'data/constructs.json').read_text())['datasets']
 with np.load(OUT/'activations.npz') as archive:
  acts={k:archive[k] for k in archive.files if k.endswith('_final')}
 result={'note':'Grouped outer 5-fold evaluation: layer chosen only on the outer training subset by inner grouped 5-fold CV with deepest-layer tie breaks. New-construct groups share authored sentence patterns; this is not independent proof of semantic specificity.', 'nested_cv':{}}
 for axis in ['pain',*custom]:
  keys=['S2_1P','S2_3P'] if axis=='pain' else [axis]
  rows=sum([data[k]['sentences'] if axis=='pain' else custom[k] for k in keys],[])
  x=np.concatenate([acts[k+'_final'] for k in keys]);labels=np.array([r['category'] in PAIN if axis=='pain' else r['category']=='target' for r in rows]);groups=np.array([r['set'] for r in rows])
  folds=np.array_split(np.random.default_rng(43).permutation(np.unique(groups)),5)
  scores=[];layers=[]
  for i,fold in enumerate(folds):
   test=np.isin(groups,fold)
   if axis=='pain':
    n=len(data['S2_1P']['sentences']);person_curves=[]
    for start in [0,n]:
     mask=(~test)[start:start+n]
     _,curve=layer_cv(x[start:start+n][mask],labels[start:start+n][mask],groups[start:start+n][mask],tie='deepest')
     person_curves.append(curve)
    mean_scores=[float(np.mean([curve[l]['auc'] for curve in person_curves])) for l in range(x.shape[1])]
    selected=max(range(x.shape[1]),key=lambda l:(mean_scores[l],l))
    train=(~test)[:n]
    v,_=direction(x[:n,selected][train],labels[:n][train])
    score=float(np.mean([auc(labels[start:start+n][test[start:start+n]],x[start:start+n,selected][test[start:start+n]]@v) for start in [0,n]]))
   else:
    selected,_=layer_cv(x[~test],labels[~test],groups[~test],tie='deepest')
    v,_=direction(x[~test,selected],labels[~test]);score=auc(labels[test],x[test,selected]@v)
   scores.append(score);layers.append(selected)
   print(axis,i,selected,scores[-1],flush=True)
  result['nested_cv'][axis]={'auc':float(np.mean(scores)),'outer_fold_auc':scores,'selected_layers':layers}
 report=json.loads((OUT/'validation.json').read_text());l=report['pain_final']['extraction_layer']
 rows=data['S2_1P']['sentences'];x=acts['S2_1P_final'][:,l];cats=np.array([r['category'] for r in rows])
 vecs={'pain':direction(x,np.isin(cats,list(PAIN)))[0]}
 for name,cat in [('fear','B'),('negative_emotion','C1'),('negative_world','C2'),('body','E')]:
  mask=np.isin(cats,[cat,'D']);vecs[name]=direction(x[mask],cats[mask]==cat)[0]
 keys=list(vecs);unit=np.stack([vecs[k]/np.linalg.norm(vecs[k]) for k in keys])
 result['control_cosines']={'axes':keys,'matrix':(unit@unit.T).tolist(),'layer':l}
 (OUT/'secondary_validation.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
