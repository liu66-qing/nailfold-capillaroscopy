import pandas as pd, numpy as np, json
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score
r=pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1.csv'); r=r[r.evaluation_role=='development'].copy(); r.exam_case_id=r.exam_case_id.astype(str)
d=pd.read_parquet('/root/nailfold/artifacts/experiments/v10_A3_instance_mil/instance_features.parquet'); d.exam_case_id=d.exam_case_id.astype(str); cols=[c for c in d if c not in ('exam_case_id','frame')]; g=d.groupby('exam_case_id')[cols].mean()
x=np.load('/root/nailfold/artifacts/features/dinov2/features.npy'); ix=pd.read_csv('/root/nailfold/artifacts/features/dinov2/index.csv'); ix.exam_case_id=ix.exam_case_id.astype(str); dm={k:x[v].mean(0) for k,v in ix.groupby('exam_case_id').indices.items()}
ids=sorted(set(r.exam_case_id)&set(g.index)&set(dm)); X=np.stack([np.r_[g.loc[i].to_numpy(),dm[i]] for i in ids]); t=r.set_index('exam_case_id').loc[ids]
mp={'clarity':{'清晰':0,'不清':1,'模糊':1},'blood_color':{'浅红':0,'淡红':0,'暗红':1,'暗紫':1},'exudation':{'无':0,'+':1,'++':1,'+++':1},'subpapillary_venous_plexus':{'不见':0,'可见1排':1,'可见2排':1,'>2排,扩张':1},'papilla':{'平坦':0,'浅波纹状':1,'波纹状':1},'microthrombus':{'无':0,'1--2':1,'>2':1},'rbc_aggregation':{'无':0,'轻度':1,'中度':1,'重度':1},'overall_assessment':{'正常':0,'大致正常':0,'轻度异常':1,'中度异常':1,'重度异常':1}}
out={}
for f,mapping in mp.items():
 y=np.array([mapping.get(t.loc[i,f],-1) for i in ids]); ok=y>=0; yy=y[ok]; xx=X[ok]; folds=t.loc[np.array(ids)[ok],'development_fold'].astype(int).to_numpy(); pred=np.zeros(len(yy),int)
 for k in range(5):
  tr=folds!=k; te=folds==k; model=ExtraTreesClassifier(n_estimators=80,min_samples_leaf=2,class_weight='balanced',random_state=100+k,n_jobs=1).fit(xx[tr],yy[tr]); pred[te]=model.predict(xx[te])
 out[f]={'n':len(yy),'balanced_accuracy':float(balanced_accuracy_score(yy,pred))}
print(json.dumps(out,ensure_ascii=False))
