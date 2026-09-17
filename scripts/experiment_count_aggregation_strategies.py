"""CPU development-only comparison of count aggregation and threshold choices."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import balanced_accuracy_score,confusion_matrix

def label_mid(x): return {"3--4":3.5,"5--6":5.5,">=7":7.5}.get(str(x),np.nan)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--count-oof",type=Path,required=True); ap.add_argument("--roles",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); a=ap.parse_args()
    roles=pd.read_csv(a.roles); dev=set(roles.loc[roles.evaluation_role.eq("development"),"exam_case_id"].astype(str)); locked=set(roles.loc[roles.evaluation_role.eq("locked_test"),"exam_case_id"].astype(str)); d=pd.read_csv(a.count_oof); d=d[d.case_id.astype(str).isin(dev)].copy(); d["case_id"]=d.case_id.astype(str); d["truth_mid"]=d.target.map(label_mid)
    methods={};
    for c in d.columns:
        m=re.match(r"count_t([0-9p]+)_(mean|median|q25|q75|max|min)$",c)
        if m: methods[f"t{m.group(1)}_{m.group(2)}"]=c
    results=[]
    for name,col in methods.items():
        x=pd.to_numeric(d[col],errors="coerce"); valid=x.notna()&d.truth_mid.notna(); xv=x[valid].to_numpy(); y=d.truth_mid[valid].to_numpy();
        # Fixed ordinal bins: choose train-fold cut points only, then evaluate OOF.
        pred=np.full(len(y),np.nan); folds=d.loc[valid,"development_fold"].astype(int).to_numpy()
        for fold in range(5):
            tr=folds!=fold; te=folds==fold; cuts=[np.nanmedian(xv[tr][y[tr]<=3.5]),np.nanmedian(xv[tr][y[tr]<=5.5])]; pred[te]=np.select([xv[te]<=cuts[0],xv[te]<=cuts[1]],[3.5,5.5],default=7.5)
        ycat=np.asarray([str(v) for v in y]); pcat=np.asarray([str(v) for v in pred]); ba=float(balanced_accuracy_score(ycat,pcat)); results.append({"strategy":name,"source_column":col,"cases":len(y),"balanced_accuracy":ba,"confusion_matrix":confusion_matrix(ycat,pcat,labels=["3.5","5.5","7.5"]).tolist()})
    summary=pd.DataFrame(results).sort_values("balanced_accuracy",ascending=False); a.output_dir.mkdir(parents=True,exist_ok=True); summary.to_csv(a.output_dir/"strategy_metrics.csv",index=False)
    report={"schema_version":"count-aggregation-development-oof/1.0","evaluation_role":"development_oof","locked_cases_seen":0,"locked_cases_in_source_manifest":len(locked),"development_cases":len(dev),"cases_in_input":len(d),"gpu_used":False,"v1_modified":False,"note":"q25/q75/max are existing frame-summary aggregates; no raw frame masks were modified","best_strategy":summary.iloc[0].to_dict() if len(summary) else None,"strategies":results,"limitations":["This is a count-feature threshold proxy, not a new segmentation postprocessor.","No instance-level masks or per-frame raw predictions were available locally."]}
    (a.output_dir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,default=float)+"\n",encoding="utf-8"); print(json.dumps({"cases":len(d),"locked_cases_seen":0},ensure_ascii=False))
if __name__=="__main__": main()
