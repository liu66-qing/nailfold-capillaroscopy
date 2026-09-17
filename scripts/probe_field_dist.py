import pandas as pd, json, numpy as np
M='/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
R='/root/nailfold/artifacts/labels/score_rules_v3.json'
df=pd.read_csv(M,dtype={'exam_case_id':str})
dev=df[df.evaluation_role=='development']
rules=json.load(open(R,encoding='utf-8'))
groups=rules.get('groups',{})
fields=[]
for g,fl in groups.items():
    fields.extend(fl)
fields=list(dict.fromkeys(fields))
print("n_dev =",len(dev))
print("n_fields =",len(fields))
print()
rows=[]
for f in fields:
    if f not in dev.columns:
        rows.append((f,0,0,'MISSING_COL','',0.0)); continue
    s=dev[f]
    nn=s.notna().sum()
    vc=s.dropna().astype(str).value_counts()
    cov=nn/len(dev)
    top=f"{vc.index[0]}={vc.iloc[0]}" if len(vc) else '-'
    rows.append((f,len(vc),nn,top,'|'.join(f"{k}:{v}" for k,v in vc.items()),cov))
rows.sort(key=lambda r:(r[1],-r[5]))
print(f"{'field':38s} {'nuniq':>5s} {'ncov':>5s} {'cov':>6s}  top")
for f,nu,nn,top,full,cov in rows:
    print(f"{f:38s} {nu:5d} {nn:5d} {cov:6.2f}  {top}")
print()
print("=== full distributions for fields with <=6 distinct values ===")
for f,nu,nn,top,full,cov in rows:
    if nu<=6:
        print(f"{f}: {full}")
