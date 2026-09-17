import pandas as pd, numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
r=pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1.csv'); r=r[r.evaluation_role=='development'].copy(); r.exam_case_id=r.exam_case_id.astype(str); d=pd.read_parquet('/root/nailfold/artifacts/experiments/v10_A3_instance_mil/instance_features.parquet'); d.exam_case_id=d.exam_case_id.astype(str); cols=[c for c in d if c not in ('exam_case_id','frame')]; g=d.groupby('exam_case_id')[cols].mean(); ids=sorted(set(r.exam_case_id)&set(g.index)); X=g.loc[ids].to_numpy(); t=r.set_index('exam_case_id').loc[ids]; mp={'不见':0,'可见1排':1,'可见2排':1,'>2排,扩张':1}; y=np.array([mp.get(t.loc[i,'subpapillary_venous_plexus'],-1) for i in ids]); ok=y>=0; X=X[ok]; y=y[ok]; f=t.loc[np.array(ids)[ok],'development_fold'].astype(int).to_numpy(); p=np.zeros(len(y),int)
for k in range(5):
 tr=f!=k; te=f==k; m=ExtraTreesClassifier(n_estimators=120,min_samples_leaf=2,class_weight='balanced',random_state=77+k,n_jobs=1).fit(X[tr],y[tr]); p[te]=m.predict(X[te]); print(k,balanced_accuracy_score(y[te],p[te]))
rng=np.random.default_rng(1); deltas=[]
base=np.bincount(y).max()/len(y)
for _ in range(2000):
 q=rng.integers(0,len(y),len(y)); deltas.append(balanced_accuracy_score(y[q],p[q])-base)
print('n',len(y),'ba',balanced_accuracy_score(y,p),'delta',balanced_accuracy_score(y,p)-base,'ci',np.quantile(deltas,[.025,.975]))
