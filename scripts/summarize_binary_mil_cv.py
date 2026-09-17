"""Summarize completed binary MIL fold metrics without touching locked evaluation."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

FIELDS = ["clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla"]
DELIVERY = FIELDS[:-1]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input-dir", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    rows=[]; reports=[]
    for fold in range(5):
        p=args.input_dir/f"fold{fold}"/"metrics.json"
        if not p.exists(): raise FileNotFoundError(p)
        report=json.loads(p.read_text(encoding="utf-8")); reports.append(report)
        if report["locked_cases_seen"] != 0 or report["gpu_used"] or report["cases"]["train"] + report["cases"]["val"] + report["cases"]["test"] != 186: raise ValueError(f"audit failed fold {fold}")
        for seed, result in report["seeds"].items():
            for field in FIELDS:
                m=result["test"][field]; rows.append({"fold":fold,"seed":int(seed),"field":field,"n":m["n"],"balanced_accuracy":m["balanced_accuracy"],"macro_f1":m["macro_f1"],"exploratory":m["exploratory"]})
    df=pd.DataFrame(rows)
    summary=[]
    for field,g in df.groupby("field"):
        summary.append({"field":field,"exploratory":bool(g.exploratory.iloc[0]),"n_oof":int(g.n.min()),"seed_fold_mean_ba":float(g.balanced_accuracy.mean()),"seed_fold_std_ba":float(g.balanced_accuracy.std(ddof=1)),"seed_fold_mean_macro_f1":float(g.macro_f1.mean())})
    delivery=df[df.field.isin(DELIVERY)].groupby(["fold","seed"]).balanced_accuracy.mean()
    out={"schema_version":"binary-attention-mil-cv-summary/1.0","method":"bag-level attention MIL","evaluation_role":"development_oof","locked_cases_seen":0,"gpu_used":False,"folds":5,"seeds":sorted(df.seed.unique().tolist()),"field_summary":summary,"delivery_field_mean_ba":float(delivery.mean()),"delivery_field_std_ba":float(delivery.std(ddof=1)),"raw_test_metrics":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"delivery_field_mean_ba":out["delivery_field_mean_ba"],"fields":{x["field"]:x["seed_fold_mean_ba"] for x in summary}},ensure_ascii=False))
if __name__ == "__main__": main()
