"""Compare formal dual_seg with a compact spatial ROI feature extension."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import balanced_accuracy_score

FIELDS=("clarity","blood_color","exudation","subpapillary_venous_plexus","papilla")
MAP={"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
def agg(path,index):
 x=np.load(path,mmap_mode="r"); i=pd.read_csv(index); i.exam_case_id=i.exam_case_id.astype(str); return {c:np.asarray(x[np.asarray(p)],dtype=np.float32).mean(0) for c,p in i.groupby("exam_case_id",sort=True).indices.items()}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--dinov2",type=Path,required=True); p.add_argument("--dino-index",type=Path,required=True); p.add_argument("--hulumed",type=Path,required=True); p.add_argument("--hulu-index",type=Path,required=True); p.add_argument("--seg",type=Path,required=True); p.add_argument("--roi-dir",type=Path,required=True); p.add_argument("--roles-labels",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--seed-base",type=int,default=20260828); a=p.parse_args()
 roles=pd.read_csv(a.roles_labels); roles.exam_case_id=roles.exam_case_id.astype(str); dev=set(roles.loc[roles.evaluation_role.eq("development"),"exam_case_id"]); locked=set(roles.loc[roles.evaluation_role.eq("locked_test"),"exam_case_id"])
 if len(dev)!=186 or len(locked)!=47: raise ValueError("role boundary")
 d=agg(a.dinov2,a.dino_index); h=agg(a.hulumed,a.hulu_index); r=agg(a.roi_dir/"features.npy",a.roi_dir/"index.csv")
 seg=pd.read_parquet(a.seg).rename(columns={"case_id":"exam_case_id"}); seg.exam_case_id=seg.exam_case_id.astype(str); sc=[c for c in seg if c.startswith("feature_")]+["frame_count"]; sv=seg.set_index("exam_case_id")[sc]
 ids=sorted(set(d)&set(h)&set(r)&set(sv.index)&dev); base=np.stack([np.concatenate([d[c],h[c],sv.loc[c].to_numpy(np.float32)]) for c in ids]); roi=np.stack([r[c] for c in ids]); table=roles.set_index("exam_case_id").loc[ids]
 routes={"dual_seg":base,"dual_seg_roi":np.concatenate([base,roi],axis=1),"roi_only":roi}; out={"schema_version":"dual-seg-roi-binary-oof/1.0","evaluation_role":"development_oof","locked_cases_seen":0,"cases":len(ids),"routes":{}}
 for route,X in routes.items():
  rr={}
  for field in FIELDS:
   y=np.array([MAP[field].get(table.loc[c,field],-1) if pd.notna(table.loc[c,field]) else -1 for c in ids]); ok=y>=0; xx=X[ok]; yy=y[ok]; folds=table.loc[np.asarray(ids)[ok],"development_fold"].astype(int).to_numpy(); pred=np.full(len(yy),-1); fold_scores=[]
   for fold in range(5):
    tr=folds!=fold; te=folds==fold; m=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced",random_state=a.seed_base+fold,n_jobs=1); m.fit(xx[tr],yy[tr]); pred[te]=m.predict(xx[te]); fold_scores.append(float(balanced_accuracy_score(yy[te],pred[te])))
   rr[field]={"n":len(yy),"balanced_accuracy":float(balanced_accuracy_score(yy,pred)),"fold_balanced_accuracy":fold_scores}
  out["routes"][route]=rr
 out["seed_base"]=a.seed_base; base_r=out["routes"]["dual_seg"]; cand=out["routes"]["dual_seg_roi"]; out["delta"]={f:cand[f]["balanced_accuracy"]-base_r[f]["balanced_accuracy"] for f in FIELDS}; out["fold_wins"]={f:int(sum(c>b for c,b in zip(cand[f]["fold_balanced_accuracy"],base_r[f]["fold_balanced_accuracy"]))) for f in FIELDS}; out["gate"]={f:"pass" if out["delta"][f]>=.02 and out["fold_wins"][f]>=3 else "fail" for f in FIELDS}; out["accepted_fields"]=[f for f in FIELDS if out["gate"][f]=="pass" and all(out["delta"][x]>=-.01 for x in FIELDS[:4] if x!=f)]
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"delta":out["delta"],"fold_wins":out["fold_wins"],"accepted_fields":out["accepted_fields"]},ensure_ascii=False))
if __name__=="__main__": main()
