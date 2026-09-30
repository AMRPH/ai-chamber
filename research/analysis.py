"""Grouped CV, control-only PCA denoising; no validation examples enter fitting."""
import numpy as np

def direction(x, labels):
    x=np.asarray(x,dtype=np.float64); labels=np.asarray(labels,dtype=bool)
    controls=x[~labels]; delta=x[labels].mean(0)-controls.mean(0)
    _,s,basis=np.linalg.svd(controls-controls.mean(0),full_matrices=False)
    variance=s*s
    k=int(np.searchsorted(np.cumsum(variance)/variance.sum(),.5)+1) if variance.sum()>0 else 0
    v=delta-basis[:k].T@(basis[:k]@delta)
    if not np.isfinite(v).all() or np.linalg.norm(v)<1e-10:
        raise ValueError('Degenerate direction')
    return v.astype(np.float32), k

def auc(labels,scores):
    labels=np.asarray(labels,dtype=bool); scores=np.asarray(scores)
    a,b=scores[labels],scores[~labels]
    return float(((a[:,None]>b).sum()+.5*(a[:,None]==b).sum())/(len(a)*len(b)))

def layer_cv(acts,labels,groups,seed=42,tie='earliest'):
    groups=np.asarray(groups); unique=np.unique(groups)
    folds=np.array_split(np.random.default_rng(seed).permutation(unique),5)
    curves=[]
    for layer in range(acts.shape[1]):
        fold_scores=[]; components=[]
        for fold in folds:
            test=np.isin(groups,fold)
            v,k=direction(acts[~test,layer],np.asarray(labels)[~test])
            fold_scores.append(auc(np.asarray(labels)[test],acts[test,layer]@v))
            components.append(k)
        curves.append({'layer':layer,'auc':float(np.mean(fold_scores)),'fold_auc':fold_scores,'removed_components':components})
    best=max(curves,key=lambda row:(row['auc'],row['layer'] if tie=='deepest' else -row['layer']))['layer']
    return best,curves
