"""Controlled classifier benchmark for development binary weak fields."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,f1_score,confusion_matrix,recall_score
FIELDS=["clarity","blood_color","exudation","subpapillary_venous_plexus","papilla"]; DELIVERY=FIELDS[:-1]
MAP={"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
TH=[.3,.35,.4,.45,.5,.55,.6,.65,.7]
def agg(path,ip,allowed):
 x=np.load(path,mmap_mode="r"); i=pd.read_csv(ip); i.exam_case_id=i.exam_case_id.astype(str); return {c:np.asarray(x[np.asarray(p)],dtype=np.float32).mean(0) for c,p in i.groupby("exam_case_id").indices.items() if c in allowed}
def model(name,p,w,seed):
 if name=="logistic": return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),LogisticRegression(C=p["C"],class_weight=w,max_iter=3000,random_state=seed))
 if name=="extra_trees": return make_pipeline(SimpleImputer(strategy="median"),ExtraTreesClassifier(n_estimators=p["n_estimators"],min_samples_leaf=p["min_samples_leaf"],class_weight=w,random_state=seed,n_jobs=1))
 return make_pipeline(SimpleImputer(strategy="median"),HistGradientBoostingClassifier(**p,class_weight=w,random_state=seed))
def met(y,p): return {"balanced_accuracy":float(balanced_accuracy_score(y,p)),"macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),"class_recall":recall_score(y,p,labels=[0,1],average=None,zero_division=0).tolist(),"confusion_matrix":confusion_matrix(y,p,labels=[0,1]).tolist()}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--dinov2",type=Path,required=True); ap.add_argument("--dino-index",type=Path,required=True); ap.add_argument("--hulumed",type=Path,required=True); ap.add_argument("--hulu-index",type=Path,required=True); ap.add_argument("--seg",type=Path,required=True); ap.add_argument("--roles-labels",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); a=ap.parse_args()
 r=pd.read_csv(a.roles_labels); r.exam_case_id=r.exam_case_id.astype(str); dev=set(r.loc[r.evaluation_role=="development","exam_case_id"]); locked=set(r.loc[r.evaluation_role=="locked_test","exam_case_id"]); assert len(dev)==186 and len(locked)==47 and not dev&locked
 d=agg(a.dinov2,a.dino_index,dev); h=agg(a.hulumed,a.hulu_index,dev); s=pd.read_parquet(a.seg).rename(columns={"case_id":"exam_case_id"}); s.exam_case_id=s.exam_case_id.astype(str); cols=[c for c in s if c.startswith("feature_")]+["frame_count"]; sv=s.set_index("exam_case_id")[cols].to_dict("index"); labels=r.set_index("exam_case_id"); ids=sorted(set(d)&set(h)&set(sv)&dev); X=np.stack([np.concatenate([d[c],h[c],np.asarray(list(sv[c].values()),dtype=np.float32)]) for c in ids])
 out={"schema_version":"controlled-weak-fields-classifier/1.0","evaluation_role":"development_oof","locked_cases_seen":0,"gpu_used":False,"v1_modified":False,"cases":186,"fields":FIELDS,"baseline":{},"optimized":{},"selection_records":{}}
 grid=[("logistic",{"C":c}) for c in [.03,.1,.3,1,3]]+[( "extra_trees",{"n_estimators":n,"min_samples_leaf":l}) for n in [300,600] for l in [1,2,4,8]]+[( "hist_gradient_boosting",{"max_iter":it,"learning_rate":lr,"max_leaf_nodes":ml,"l2_regularization":l2}) for it in [100,200] for lr in [.03,.1] for ml in [7,15] for l2 in [0,1,10]]
 for f in FIELDS:
  y=np.array([MAP[f].get(labels.loc[c,f],-1) if pd.notna(labels.loc[c,f]) else -1 for c in ids],dtype=int); ok=y>=0; xx=X[ok]; yy=y[ok]; folds=labels.loc[np.array(ids)[ok],"development_fold"].astype(int).to_numpy(); base=[]; opt=[]; selections=[]
  for test in range(5):
   va=(folds==(test+1)%5); tr=(folds!=test)&~va; te=folds==test; bm=model("extra_trees",{"n_estimators":300,"min_samples_leaf":2},"balanced",17+test); bm.fit(xx[tr],yy[tr]); base.append({"fold":test,**met(yy[te],bm.predict(xx[te]))}); best=None
   for n,p in grid:
    for w in [None,"balanced"]:
     m=model(n,p,w,17+test); m.fit(xx[tr],yy[tr]); pr=m.predict_proba(xx[va])[:,1]; b0=balanced_accuracy_score(yy[va],pr>=.5); thr=.5; bs=b0
     if np.bincount(yy[va],minlength=2).min()>=3:
      for t in TH:
       z=balanced_accuracy_score(yy[va],pr>=t)
       if z>bs: bs=z; thr=t
      if bs-b0<.02: bs=b0; thr=.5
     rec={"fold":test,"model":n,"params":p,"class_weight":w or "none","threshold":thr,"validation_ba":float(bs)}; selections.append(rec); rank={"logistic":0,"hist_gradient_boosting":1,"extra_trees":2}[n]
     if best is None or (bs,-rank)>(best[0],best[1]): best=(bs,-rank,rec)
   q=best[2]; m=model(q["model"],q["params"],q["class_weight"] if q["class_weight"]!="none" else None,17+test); m.fit(xx[tr|va],yy[tr|va]); pr=m.predict_proba(xx[te])[:,1]; opt.append({"fold":test,"selection":q,**met(yy[te],pr>=q["threshold"])})
  out["baseline"][f]=base; out["optimized"][f]=opt; out["selection_records"][f]=selections
 def mean(z): return {f:float(np.mean([x["balanced_accuracy"] for x in z[f]])) for f in FIELDS}
 b=mean(out["baseline"]); o=mean(out["optimized"]); delta={f:o[f]-b[f] for f in FIELDS}; fd=[np.mean([out["optimized"][f][k]["balanced_accuracy"]-out["baseline"][f][k]["balanced_accuracy"] for f in DELIVERY]) for k in range(5)]; pos=sum(delta[f]>0 for f in DELIVERY); gain=np.mean([delta[f] for f in DELIVERY]); accepted=gain>=.015 and pos>=3 and min(delta[f] for f in DELIVERY)>=-.02 and sum(x>0 for x in fd)>=3; out["summary"]={"baseline":b,"optimized":o}; out["gate"]={"delivery_mean_delta":float(gain),"delta":delta,"positive_fields":pos,"fold_deltas":fd,"decision":"accepted" if accepted else ("local_improvement_only" if gain>0 else "rejected")}
 a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(out["gate"],ensure_ascii=False))
if __name__=="__main__": main()
