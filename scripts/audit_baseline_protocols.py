"""Recompute and reconcile dual_seg baseline protocol variants."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score
FIELDS=["clarity","blood_color","exudation","subpapillary_venous_plexus","papilla"]
MAP={"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
def agg(path,index,allowed):
 x=np.load(path,mmap_mode="r"); i=pd.read_csv(index); i.exam_case_id=i.exam_case_id.astype(str); return {c:np.asarray(x[np.asarray(p)],dtype=np.float32).mean(0) for c,p in i.groupby("exam_case_id").indices.items() if c in allowed}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--dinov2",type=Path,required=True); p.add_argument("--dino-index",type=Path,required=True); p.add_argument("--hulumed",type=Path,required=True); p.add_argument("--hulu-index",type=Path,required=True); p.add_argument("--seg",type=Path,required=True); p.add_argument("--roles-labels",type=Path,required=True); p.add_argument("--reference-predictions",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
 r=pd.read_csv(a.roles_labels); r.exam_case_id=r.exam_case_id.astype(str); dev=r[r.evaluation_role.eq("development")].copy(); locked=set(r.loc[r.evaluation_role.eq("locked_test"),"exam_case_id"]); assert len(dev)==186 and len(locked)==47
 d=agg(a.dinov2,a.dino_index,set(dev.exam_case_id)); h=agg(a.hulumed,a.hulu_index,set(dev.exam_case_id)); s=pd.read_parquet(a.seg).rename(columns={"case_id":"exam_case_id"}); s.exam_case_id=s.exam_case_id.astype(str); cols=[c for c in s if c.startswith("feature_")]+["frame_count"]; sv=s.set_index("exam_case_id")[cols]; ids=sorted(set(d)&set(h)&set(sv.index)); X=np.stack([np.concatenate([d[c],h[c],sv.loc[c].to_numpy(np.float32)]) for c in ids]); tab=dev.set_index("exam_case_id").loc[ids]; variants={"deployment_all_non_test":{"exclude_validation":False,"seed_base":20260828},"nested_train_only_current_seed":{"exclude_validation":True,"seed_base":20260828},"nested_train_only_controlled_seed":{"exclude_validation":True,"seed_base":17}}; out={"schema_version":"dual-seg-baseline-protocol-audit/1.0","evaluation_role":"development_oof","locked_cases_seen":0,"gpu_used":False,"feature_dim":int(X.shape[1]),"variants":{},"reference_match":{}}; rows=[]
 for f in FIELDS:
  y=np.array([MAP[f].get(tab.loc[c,f],-1) if pd.notna(tab.loc[c,f]) else -1 for c in ids]); ok=y>=0; xx=X[ok]; yy=y[ok]; ci=np.asarray(ids)[ok]; folds=tab.loc[ci,"development_fold"].astype(int).to_numpy()
  for name,cfg in variants.items():
   pred=np.full(len(yy),-1); fba=[]
   for fold in range(5):
    tr=folds!=fold
    if cfg["exclude_validation"]: tr &= folds!=(fold+1)%5
    te=folds==fold; m=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced",random_state=cfg["seed_base"]+fold,n_jobs=1); m.fit(xx[tr],yy[tr]); pred[te]=m.predict(xx[te]); fba.append(float(balanced_accuracy_score(yy[te],pred[te])))
   z={"n":len(yy),"accuracy":float(accuracy_score(yy,pred)),"global_oof_ba":float(balanced_accuracy_score(yy,pred)),"unweighted_fold_mean_ba":float(np.mean(fba)),"fold_ba":fba}; out["variants"].setdefault(name,{})[f]=z; rows += [{"exam_case_id":c,"field":f,"variant":name,"truth":int(t),"prediction":int(q)} for c,t,q in zip(ci,yy,pred)]
 ref=pd.read_csv(a.reference_predictions); audit=pd.DataFrame(rows); dep=audit[audit.variant.eq("deployment_all_non_test")].merge(ref[["exam_case_id","field","prediction"]],on=["exam_case_id","field"],suffixes=("_audit","_reference"),validate="one_to_one"); out["reference_match"]={"rows":len(dep),"prediction_disagreements":int((dep.prediction_audit!=dep.prediction_reference).sum())}; out["delivery_means"]={name:{"global_oof_ba":float(np.mean([out["variants"][name][f]["global_oof_ba"] for f in FIELDS[:-1]])),"unweighted_fold_mean_ba":float(np.mean([out["variants"][name][f]["unweighted_fold_mean_ba"] for f in FIELDS[:-1]]))} for name in variants}; a.output_dir.mkdir(parents=True,exist_ok=True); audit.to_csv(a.output_dir/"predictions.csv",index=False); (a.output_dir/"metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n"); print(json.dumps({"delivery_means":out["delivery_means"],"reference_match":out["reference_match"]},ensure_ascii=False))
if __name__=="__main__": main()
