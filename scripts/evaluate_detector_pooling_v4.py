import argparse, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, recall_score, confusion_matrix
FIELDS=("clarity","blood_color","exudation","subpapillary_venous_plexus","papilla")
MAP={"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
def agg(path,index,allowed,mode):
 x=np.load(path,mmap_mode='r'); i=pd.read_csv(index); i.exam_case_id=i.exam_case_id.astype(str); out={}
 for c,pos in i.groupby('exam_case_id',sort=True).indices.items():
  if c not in allowed: continue
  a=np.asarray(x[np.asarray(pos)],np.float32)
  if mode=='median': v=np.median(a,0)
  elif mode=='topk':
   k=max(1,int(np.ceil(len(a)*.3))); score=a[:,1] if a.shape[1]>1 else np.linalg.norm(a,axis=1); v=a[np.argsort(score)[-k:]].mean(0)
  else: v=a.mean(0)
  out[c]=np.r_[v,a.std(0)]
 return out
def main():
 p=argparse.ArgumentParser(); p.add_argument('--detector',type=Path,required=True); p.add_argument('--detector-index',type=Path,required=True); p.add_argument('--dinov2',type=Path,required=True); p.add_argument('--dino-index',type=Path,required=True); p.add_argument('--hulumed',type=Path,required=True); p.add_argument('--hulu-index',type=Path,required=True); p.add_argument('--seg',type=Path,required=True); p.add_argument('--roles-labels',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--mode',choices=['mean','median','topk'],required=True); p.add_argument('--classifier',choices=['extra','catboost','xgb'],default='extra'); a=p.parse_args()
 roles=pd.read_csv(a.roles_labels); roles.exam_case_id=roles.exam_case_id.astype(str); dev=set(roles.loc[roles.evaluation_role=='development','exam_case_id']); locked=set(roles.loc[roles.evaluation_role=='locked_test','exam_case_id']); assert len(dev)==186 and len(locked)==47 and not dev&locked
 d=agg(a.dinov2,a.dino_index,dev,'mean'); h=agg(a.hulumed,a.hulu_index,dev,'mean'); det=agg(a.detector,a.detector_index,dev,a.mode)
 s=pd.read_parquet(a.seg).rename(columns={'case_id':'exam_case_id'}); s.exam_case_id=s.exam_case_id.astype(str); cols=[c for c in s if c.startswith('feature_')]+['frame_count']; sv=s.set_index('exam_case_id')[cols]; ids=sorted(set(d)&set(h)&set(det)&set(sv.index)); X=np.stack([np.r_[d[c],h[c],sv.loc[c].to_numpy(np.float32),det[c]] for c in ids]); tab=roles.set_index('exam_case_id').loc[ids]; recs=[]; results={}
 for field in FIELDS:
  y=np.array([MAP[field].get(tab.loc[c,field],-1) if pd.notna(tab.loc[c,field]) else -1 for c in ids]); ok=y>=0; xx=X[ok]; yy=y[ok]; ci=np.asarray(ids)[ok]; folds=tab.loc[ci,'development_fold'].astype(int).to_numpy(); pred=np.full(len(yy),-1); prob=np.full(len(yy),np.nan); fm=[]
  for fold in range(5):
   tr=folds!=fold; te=folds==fold
   if getattr(a,'classifier','extra')=='catboost':
    from catboost import CatBoostClassifier
    m=CatBoostClassifier(iterations=400,depth=6,learning_rate=.04,verbose=False,random_seed=20260828+fold,loss_function='Logloss',auto_class_weights='Balanced')
   elif getattr(a,'classifier','extra')=='mlp':
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(128,32),max_iter=250,early_stopping=True,random_state=20260828+fold))
   elif getattr(a,'classifier','extra')=='xgb':
    from xgboost import XGBClassifier
    m=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=.03,subsample=.8,colsample_bytree=.7,min_child_weight=2,reg_lambda=2,eval_metric='logloss',random_state=20260828+fold,n_jobs=4)
   else: m=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,class_weight='balanced',random_state=20260828+fold,n_jobs=1)
   m.fit(xx[tr],yy[tr]); pred[te]=m.predict(xx[te]).astype(int).ravel(); prob[te]=m.predict_proba(xx[te])[:,1]; fm.append({'fold':fold,'balanced_accuracy':float(balanced_accuracy_score(yy[te],pred[te]))})
  r=recall_score(yy,pred,labels=[0,1],average=None,zero_division=0); results[field]={'n':len(yy),'balanced_accuracy':float(balanced_accuracy_score(yy,pred)),'accuracy':float(accuracy_score(yy,pred)),'macro_f1':float(f1_score(yy,pred,average='macro')),'class_recall':{'0':float(r[0]),'1':float(r[1])},'confusion_matrix':confusion_matrix(yy,pred,labels=[0,1]).tolist(),'folds':fm}
  recs += [{'exam_case_id':c,'development_fold':int(f),'field':field,'truth':int(t),'prediction':int(q),'positive_probability':float(z)} for c,f,t,q,z in zip(ci,folds,yy,pred,prob)]
 out={'schema_version':'detector-pooling-v4/1.0','mode':a.mode,'locked_cases_seen':0,'feature_dim':int(X.shape[1]),'fields':results}; a.output_dir.mkdir(parents=True,exist_ok=True); pd.DataFrame(recs).to_csv(a.output_dir/'oof_predictions.csv',index=False); (a.output_dir/'metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({f:results[f]['balanced_accuracy'] for f in FIELDS},ensure_ascii=False))
if __name__=='__main__': main()
