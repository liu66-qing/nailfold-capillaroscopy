import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
man=pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'); d=man[man.evaluation_role=='development']; idx=pd.read_csv('/root/autodl-tmp/nailfold/artifacts/features/dinov2/index.csv'); F=np.load('/root/autodl-tmp/nailfold/artifacts/features/dinov2/features.npy'); G=np.load('/root/autodl-tmp/nailfold/artifacts/features/geometry_dev/features.npy'); X=pd.DataFrame(np.hstack([F,G])).groupby(idx.exam_case_id).mean(); z=d[d.exam_case_id.isin(X.index)].copy(); y=z.microthrombus.map({'无':0,'1--2':1,'>2':1}).to_numpy(); X=X.loc[z.exam_case_id].to_numpy(); fl=z.development_fold.to_numpy(); p=np.zeros(len(y))
for k in range(5):
 tr=fl!=k; te=fl==k; m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=3000)); m.fit(X[tr],y[tr]); p[te]=m.predict(X[te])
print(len(y),balanced_accuracy_score(y,p),(p==y).mean(),np.bincount(y))
