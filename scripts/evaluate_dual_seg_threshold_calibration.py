"""Nested development OOF threshold calibration for fixed dual_seg ExtraTrees."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,recall_score
FIELDS=["clarity","blood_color","exudation","subpapillary_venous_plexus","papilla"]
MAP={"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
TH=[.3,.35,.4,.45,.5,.55,.6,.65,.7]
def agg(path,index,allowed):
 x=np.load(path,mmap_mode="r"); i=pd.read_csv(index); i.exam_case_id=i.exam_case_id.astype(str); return {c:np.asarray(x[np.asarray(pos)],dtype=np.float32).mean(0) for c,pos in i.groupby("exam_case_id").indices.items() if c in allowed}
def q(y,p):
 r=recall_score(y,p,labels=[0,1],average=None,zero_division=0); return {"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),"class_recall":r.tolist(),"confusion_matrix":confusion_matrix(y,p,labels=[0,1]).tolist()}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--dinov2",type=Path,required=True); ap.add_argument("--dino-index",type=Path,required=True); ap.add_argument("--hulumed",type=Path,required=True); ap.add_argument("--hulu-index",type=Path,required=True); ap.add_argument("--seg",type=Path,required=True); ap.add_argument("--roles-labels",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); a=ap.parse_args()
 r=pd.read_csv(a.roles_labels); r.exam_case_id=r.exam_case_id.astype(str); dev=r[r.evaluation_role.eq("development")].copy(); locked=set(r.loc[r.evaluation_role.eq("locked_test"),"exam_case_id"]); assert len(dev)==186 and len(locked)==47
 d=agg(a.dinov2,a.dino_index,set(dev.exam_case_id)); h=agg(a.hulumed,a.hulu_index,set(dev.exam_case_id)); s=pd.read_parquet(a.seg).rename(columns={"case_id":"exam_case_id"}); s.exam_case_id=s.exam_case_id.astype(str); cols=[c for c in s if c.startswith("feature_")]+["frame_count"]; sv=s.set_index("exam_case_id")[cols]; ids=sorted(set(d)&set(h)&set(sv.index)); X=np.stack([np.concatenate([d[c],h[c],sv.loc[c].to_numpy(np.float32)]) for c in ids]); tab=dev.set_index("exam_case_id").loc[ids]; out={"schema_version":"dual-seg-threshold-calibration-oof/1.0","evaluation_role":"development_oof","locked_cases_seen":0,"gpu_used":False,"baseline":{},"calibrated":{}}
 for f in FIELDS:
  y=np.array([MAP[f].get(tab.loc[c,f],-1) if pd.notna(tab.loc[c,f]) else -1 for c in ids]); ok=y>=0; xx=X[ok]; yy=y[ok]; ci=np.asarray(ids)[ok]; folds=tab.loc[ci,"development_fold"].astype(int).to_numpy(); bp=np.full(len(yy),-1); cp=np.full(len(yy),-1); chosen=[]
  for fold in range(5):
   tr=(folds!=fold)&(folds!=(fold+1)%5); va=folds==(fold+1)%5; te=folds==fold; m=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced",random_state=20260828+fold,n_jobs=1); m.fit(xx[tr],yy[tr]); pv=m.predict_proba(xx[va])[:,1]; pt=m.predict_proba(xx[te])[:,1]; bp[te]=(pt>=.5); base=balanced_accuracy_score(yy[va],pv>=.5); threshold=.5; best=base
   if np.bincount(yy[va],minlength=2).min()>=3:
    for t in TH:
     z=balanced_accuracy_score(yy[va],pv>=t)
     if z>best: best=z; threshold=t
    if best-base<.02: threshold=.5
   cp[te]=(pt>=threshold); chosen.append({"fold":fold,"threshold":threshold,"validation_ba":float(best),"validation_argmax_ba":float(base)})
  out["baseline"][f]={**q(yy,bp),"n":len(yy)}; out["calibrated"][f]={**q(yy,cp),"n":len(yy),"fold_selection":chosen}
 delivery=FIELDS[:-1]; b=np.mean([out["baseline"][f]["balanced_accuracy"] for f in delivery]); c=np.mean([out["calibrated"][f]["balanced_accuracy"] for f in delivery]); delta={f:out["calibrated"][f]["balanced_accuracy"]-out["baseline"][f]["balanced_accuracy"] for f in FIELDS}; folds=[]
 for k in range(5): folds.append(float(np.mean([out["calibrated"][f]["fold_selection"][k]["validation_ba"]-out["calibrated"][f]["fold_selection"][k]["validation_argmax_ba"] for f in delivery])))
 out["gate"]={"baseline_delivery_ba":float(b),"calibrated_delivery_ba":float(c),"delta":delta,"positive_fields":sum(delta[f]>0 for f in delivery),"decision":"accepted" if c-b>=.01 and sum(delta[f]>0 for f in delivery)>=3 and min(delta[f] for f in delivery)>=-.02 else ("local_improvement_only" if c>b else "rejected")}; a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(out["gate"],ensure_ascii=False))
if __name__=="__main__": main()
