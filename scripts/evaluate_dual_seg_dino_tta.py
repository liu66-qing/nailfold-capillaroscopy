"""Strict OOF probability-averaging TTA against fixed dual_seg baseline."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,recall_score
FIELDS=["clarity","blood_color","exudation","subpapillary_venous_plexus","papilla"]
MAP={"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
def agg(x,idx): return {c:np.asarray(x[np.asarray(p)],dtype=np.float32).mean(0) for c,p in idx.groupby("exam_case_id").indices.items()}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--tta-dir",type=Path,required=True); p.add_argument("--original-dinov2",type=Path,required=True); p.add_argument("--original-index",type=Path,required=True); p.add_argument("--hulumed",type=Path,required=True); p.add_argument("--hulu-index",type=Path,required=True); p.add_argument("--seg",type=Path,required=True); p.add_argument("--roles-labels",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
 z=np.load(a.tta_dir/"features.npz"); di=pd.read_csv(a.tta_dir/"index.csv"); di.exam_case_id=di.exam_case_id.astype(str); oi=pd.read_csv(a.original_index); oi.exam_case_id=oi.exam_case_id.astype(str); hi=pd.read_csv(a.hulu_index); hi.exam_case_id=hi.exam_case_id.astype(str); h=agg(np.load(a.hulumed,mmap_mode="r"),hi); dv={v:agg(z[v],di) for v in z.files if v!="original"}; dv["original"]=agg(np.load(a.original_dinov2,mmap_mode="r"),oi); s=pd.read_parquet(a.seg).rename(columns={"case_id":"exam_case_id"}); s.exam_case_id=s.exam_case_id.astype(str); sc=[c for c in s if c.startswith("feature_")]+["frame_count"]; sv=s.set_index("exam_case_id")[sc]; r=pd.read_csv(a.roles_labels); r.exam_case_id=r.exam_case_id.astype(str); dev=set(r.loc[r.evaluation_role.eq("development"),"exam_case_id"]); locked=set(r.loc[r.evaluation_role.eq("locked_test"),"exam_case_id"]); assert len(dev)==186 and len(locked)==47 and not set(di.exam_case_id)&locked; tab=r.set_index("exam_case_id"); ids=sorted(dev&set(h)&set(sv.index)&set(dv["original"])); xs={v:np.stack([np.concatenate([dv[v][c],h[c],sv.loc[c].to_numpy(np.float32)]) for c in ids]) for v in dv}
 out={"schema_version":"dual-seg-dino-tta-oof/1.0","evaluation_role":"development_oof","locked_cases_seen":0,"variants":list(dv),"baseline":{},"tta":{}}
 for f in FIELDS:
  y=np.array([MAP[f].get(tab.loc[c,f],-1) if pd.notna(tab.loc[c,f]) else -1 for c in ids]); ok=y>=0; yy=y[ok]; folds=tab.loc[np.asarray(ids)[ok],"development_fold"].astype(int).to_numpy(); bp=np.full(len(yy),-1); tp=np.full(len(yy),-1)
  for fold in range(5):
   tr=folds!=fold; te=folds==fold; m=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced",random_state=20260828+fold,n_jobs=1); m.fit(xs["original"][ok][tr],yy[tr]); base_prob=m.predict_proba(xs["original"][ok][te])[:,1]; probs=[base_prob]+[m.predict_proba(xs[v][ok][te])[:,1] for v in ("hflip","center90")]; bp[te]=(base_prob>=.5); tp[te]=(np.mean(probs,axis=0)>=.5)
  def q(p):
   rec=recall_score(yy,p,labels=[0,1],average=None,zero_division=0); return {"accuracy":float(accuracy_score(yy,p)),"balanced_accuracy":float(balanced_accuracy_score(yy,p)),"macro_f1":float(f1_score(yy,p,average="macro")),"class_recall":rec.tolist(),"confusion_matrix":confusion_matrix(yy,p,labels=[0,1]).tolist()}
  out["baseline"][f]=q(bp); out["tta"][f]=q(tp)
 delivery=FIELDS[:-1]; b=np.mean([out["baseline"][f]["balanced_accuracy"] for f in delivery]); t=np.mean([out["tta"][f]["balanced_accuracy"] for f in delivery]); delta={f:out["tta"][f]["balanced_accuracy"]-out["baseline"][f]["balanced_accuracy"] for f in FIELDS}; accepted=t-b>=.01 and sum(delta[f]>0 for f in delivery)>=3 and min(delta[f] for f in delivery)>=-.02; out["gate"]={"baseline_delivery_ba":float(b),"tta_delivery_ba":float(t),"delta":delta,"decision":"accepted" if accepted else "rejected"}; a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(out["gate"],ensure_ascii=False))
if __name__=="__main__": main()
