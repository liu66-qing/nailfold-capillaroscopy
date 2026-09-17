"""Development-only OOF retraining with binary delivery labels."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline

FIELDS = ("clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla")
def canon(field, value):
    if pd.isna(value): return None
    v=str(value).strip()
    if field == "clarity": return "poor" if v in {"不清","模糊"} else "clear"
    if field == "blood_color": return "dark" if v in {"暗红","暗紫"} else "light"
    if field == "exudation": return "absent" if v == "无" else "present"
    if field == "subpapillary_venous_plexus": return "absent" if v == "不见" else "present"
    if field == "papilla": return "flat" if v == "平坦" else "wavy"
    raise ValueError(field)

def aggregate(matrix_path, index_path, allowed):
    matrix=np.load(matrix_path,mmap_mode="r"); idx=pd.read_csv(index_path); rows=[]
    for case, pos in idx.groupby("exam_case_id",sort=True).indices.items():
        if str(case) not in allowed: continue
        x=np.asarray(matrix[np.asarray(pos)],dtype=np.float32)
        rows.append((str(case),np.concatenate((x.mean(0),np.median(x,0),x.std(0)))))
    return pd.DataFrame({"exam_case_id":[x[0] for x in rows],"vector":[x[1] for x in rows]})

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--features",type=Path,required=True); ap.add_argument("--feature-index",type=Path,required=True); ap.add_argument("--roles-labels",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--seed",type=int,default=20260828); a=ap.parse_args()
    roles=pd.read_csv(a.roles_labels); dev=roles[roles.evaluation_role.eq("development")].copy(); allowed=set(dev.exam_case_id.astype(str)); locked=set(roles.loc[roles.evaluation_role.eq("locked_test"),"exam_case_id"].astype(str))
    labels=dev[["exam_case_id","development_fold",*FIELDS]].copy(); feat=aggregate(a.features,a.feature_index,allowed); frame=feat.merge(labels,on="exam_case_id",validate="one_to_one")
    out=[]; reports={}
    for field in FIELDS:
        frame["target"]=frame[field].map(lambda x, f=field: canon(f,x)); use=frame[frame.target.notna()].copy(); ids=use.exam_case_id.to_numpy(); y=use.target.to_numpy(); folds=use.development_fold.astype(int).to_numpy(); vectors=dict(zip(frame.exam_case_id,frame.vector)); pred=np.full(len(use),None,dtype=object)
        for fold in range(5):
            tr=folds!=fold; te=folds==fold
            model=make_pipeline(SimpleImputer(strategy="median"),ExtraTreesClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced",max_features="sqrt",random_state=a.seed+fold,n_jobs=1)); model.fit(np.stack([vectors[x] for x in ids[tr]]),y[tr]); pred[te]=model.predict(np.stack([vectors[x] for x in ids[te]]))
        labels_sorted=sorted(set(y)); ba=float(balanced_accuracy_score(y,pred)); rec=recall_score(y,pred,labels=labels_sorted,average=None,zero_division=0)
        reports[field]={"cases":len(y),"balanced_accuracy":ba,"class_support":{x:int((y==x).sum()) for x in labels_sorted},"class_recall":{x:float(r) for x,r in zip(labels_sorted,rec)},"confusion_labels":labels_sorted,"confusion_matrix":confusion_matrix(y,pred,labels=labels_sorted).tolist()}
        out.extend({"exam_case_id":i,"field":field,"truth":t,"prediction":p,"correct":bool(t==p)} for i,t,p in zip(ids,y,pred))
    a.output_dir.mkdir(parents=True,exist_ok=True); pd.DataFrame(out).to_csv(a.output_dir/"oof_predictions.csv",index=False)
    report={"schema_version":"weak-fields-binary-gbt-development-oof/1.0","evaluation_role":"development_oof","locked_cases_seen":0,"locked_cases_in_source_manifest":len(locked),"development_cases":len(dev),"gpu_used":False,"v1_modified":False,"delivery_labels":"field-specific binary mappings in canon()","fields":reports}
    (a.output_dir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"cases":len(dev),"locked_cases_seen":0},ensure_ascii=False))
if __name__=="__main__": main()
