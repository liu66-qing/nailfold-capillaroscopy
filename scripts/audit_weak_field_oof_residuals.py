"""Audit baseline OOF residuals by confidence, fold, class, and co-occurrence."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
FIELDS=["clarity","blood_color","exudation","subpapillary_venous_plexus","papilla"]
def main():
 p=argparse.ArgumentParser(); p.add_argument("--predictions",type=Path,required=True); p.add_argument("--labels",type=Path,required=True); p.add_argument("--roles",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
 pred=pd.read_csv(a.predictions); lab=pd.read_csv(a.labels); lab.exam_case_id=lab.exam_case_id.astype(str); roles=pd.read_csv(a.roles); roles.exam_case_id=roles.exam_case_id.astype(str); pred.exam_case_id=pred.exam_case_id.astype(str)
 if set(pred.exam_case_id)&set(roles.loc[roles.evaluation_role.eq("locked_test"),"exam_case_id"]): raise ValueError("locked case in predictions")
 out={"schema_version":"weak-field-oof-residual-audit/1.0","evaluation_role":"development_oof","locked_cases_seen":0,"gpu_used":False,"fields":{},"error_cooccurrence":{}}
 wide=[]
 for f in FIELDS:
  d=pred[pred.field.eq(f)].merge(lab[["exam_case_id",f,f+"__confidence",f+"__status"]],on="exam_case_id",validate="one_to_one"); d["correct"]=d.truth.eq(d.prediction); d["confidence_bin"]=pd.cut(d[f+"__confidence"],[-.01,.8,.95,1.01],labels=["<0.8","0.8-0.95",">=0.95"])
  groups={"overall":d.correct.mean()}
  for col in ["confidence_bin",f+"__status","development_fold","truth"]:
   groups[col]={str(k):float(v.correct.mean()) for k,v in d.groupby(col,dropna=False)}
  out["fields"][f]={"n":len(d),"accuracy":float(d.correct.mean()),"groups":groups,"errors":d.loc[~d.correct,["exam_case_id","development_fold","truth","prediction",f+"__confidence",f+"__status"]].to_dict("records")}; wide.append(d[["exam_case_id","correct"]].rename(columns={"correct":f}))
 w=wide[0]
 for x in wide[1:]: w=w.merge(x,on="exam_case_id",how="outer")
 out["error_cooccurrence"]={f:{g:int(((~w[f].fillna(True))&(~w[g].fillna(True))).sum()) for g in FIELDS} for f in FIELDS}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps({f:out["fields"][f]["accuracy"] for f in FIELDS},ensure_ascii=False))
if __name__=="__main__": main()
