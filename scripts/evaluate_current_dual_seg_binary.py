"""Record the fixed current dual_seg binary baseline with strict development OOF."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score

FIELDS=("clarity","blood_color","exudation","subpapillary_venous_plexus","papilla")
MAP={"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
def aggregate(path,index,allowed):
 x=np.load(path,mmap_mode="r"); i=pd.read_csv(index); i.exam_case_id=i.exam_case_id.astype(str); rows={}
 for case,pos in i.groupby("exam_case_id",sort=True).indices.items():
  if case in allowed: rows[case]=np.asarray(x[np.asarray(pos)],dtype=np.float32).mean(0)
 return rows
def digest(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument("--dinov2",type=Path,required=True); p.add_argument("--dino-index",type=Path,required=True); p.add_argument("--hulumed",type=Path,required=True); p.add_argument("--hulu-index",type=Path,required=True); p.add_argument("--seg",type=Path,required=True); p.add_argument("--roles-labels",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
 roles=pd.read_csv(a.roles_labels); roles.exam_case_id=roles.exam_case_id.astype(str); dev=set(roles.loc[roles.evaluation_role.eq("development"),"exam_case_id"]); locked=set(roles.loc[roles.evaluation_role.eq("locked_test"),"exam_case_id"])
 if len(dev)!=186 or len(locked)!=47 or dev&locked: raise ValueError("role boundary failed")
 d=aggregate(a.dinov2,a.dino_index,dev); h=aggregate(a.hulumed,a.hulu_index,dev); s=pd.read_parquet(a.seg).rename(columns={"case_id":"exam_case_id"}); s.exam_case_id=s.exam_case_id.astype(str); sc=[c for c in s if c.startswith("feature_")]+["frame_count"]; sv=s.set_index("exam_case_id")[sc]
 ids=sorted(set(d)&set(h)&set(sv.index)&dev); X=np.stack([np.concatenate([d[c],h[c],sv.loc[c].to_numpy(np.float32)]) for c in ids]); table=roles.set_index("exam_case_id").loc[ids]
 records=[]; results={}
 for field in FIELDS:
  y=np.array([MAP[field].get(table.loc[c,field],-1) if pd.notna(table.loc[c,field]) else -1 for c in ids]); ok=y>=0; xx=X[ok]; yy=y[ok]; ci=np.asarray(ids)[ok]; folds=table.loc[ci,"development_fold"].astype(int).to_numpy(); pred=np.full(len(yy),-1); proba=np.full(len(yy),np.nan); fold_metrics=[]
  for fold in range(5):
   tr=folds!=fold; te=folds==fold; m=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced",random_state=20260828+fold,n_jobs=1); m.fit(xx[tr],yy[tr]); pred[te]=m.predict(xx[te]); proba[te]=m.predict_proba(xx[te])[:,1]
   fold_metrics.append({"fold":fold,"n":int(te.sum()),"balanced_accuracy":float(balanced_accuracy_score(yy[te],pred[te]))})
  labels=[0,1]; rec=recall_score(yy,pred,labels=labels,average=None,zero_division=0); cm=confusion_matrix(yy,pred,labels=labels)
  results[field]={"n":len(yy),"accuracy":float(accuracy_score(yy,pred)),"balanced_accuracy":float(balanced_accuracy_score(yy,pred)),"macro_f1":float(f1_score(yy,pred,average="macro")),"class_recall":{"0":float(rec[0]),"1":float(rec[1])},"confusion_matrix":cm.tolist(),"folds":fold_metrics,"exploratory":field=="papilla"}
  records += [{"exam_case_id":c,"development_fold":int(f),"field":field,"truth":int(t),"prediction":int(q),"positive_probability":float(z)} for c,f,t,q,z in zip(ci,folds,yy,pred,proba)]
 out={"schema_version":"current-dual-seg-binary-oof/1.0","evaluation_role":"development_oof","locked_cases_seen":0,"gpu_used":False,"development_cases":186,"feature_dim":int(X.shape[1]),"model":{"type":"ExtraTreesClassifier","n_estimators":300,"min_samples_leaf":2,"class_weight":"balanced"},"input_sha256":{"dinov2":digest(a.dinov2),"hulumed":digest(a.hulumed),"seg":digest(a.seg),"roles_labels":digest(a.roles_labels)},"fields":results,"delivery_mean_balanced_accuracy":float(np.mean([results[f]["balanced_accuracy"] for f in FIELDS[:-1]]))}
 a.output_dir.mkdir(parents=True,exist_ok=True); pd.DataFrame(records).to_csv(a.output_dir/"oof_predictions.csv",index=False); (a.output_dir/"metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"delivery_mean_ba":out["delivery_mean_balanced_accuracy"],"fields":{f:results[f]["balanced_accuracy"] for f in FIELDS}},ensure_ascii=False))
if __name__=="__main__": main()
