import pandas as pd, json, numpy as np
M='/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
R='/root/nailfold/artifacts/labels/score_rules_v3.json'
df=pd.read_csv(M,dtype={'exam_case_id':str})
dev=df[df.evaluation_role=='development'].copy()
rules=json.load(open(R,encoding='utf-8'))
FR=rules['field_rules']
fields=[]
for g,fl in rules['groups'].items(): fields.extend(fl)
fields=list(dict.fromkeys(fields))

def score_of(f,v):
    r=FR.get(f)
    if r is None: return None
    if r['type']=='categorical_lookup':
        return r['mapping'].get(str(v))
    if r['type']=='numeric_tree':
        try: x=float(v)
        except: return None
        node=r['tree']
        while 'threshold' in node:
            node=node['left'] if x<=node['threshold'] else node['right']
        return node.get('value')
    return None

sc={}
for f in fields:
    if f not in dev.columns: continue
    s=pd.to_numeric(dev[f].astype(str).map(lambda v: score_of(f,v)),errors='coerce')
    if s.notna().sum()>0: sc[f]=s
S=pd.DataFrame(sc)
ts=pd.to_numeric(dev['total_score'],errors='coerce')
print("fields scored: %d of %d   total_score sd=%.3f var=%.3f"%(S.shape[1],len(fields),ts.std(),ts.var()))

rows=[]
for f in S.columns:
    v=S[f].dropna()
    if len(v)==0: continue
    idx=S[f].notna()&ts.notna()
    r=np.corrcoef(S[f][idx],ts[idx])[0,1] if idx.sum()>3 else np.nan
    rows.append((f,v.max()-v.min(),v.std(),v.value_counts(normalize=True).iloc[0],r,v.notna().sum()))
rows.sort(key=lambda x:-x[2])
tot_var=sum(x[2]**2 for x in rows)
print(f"\n{'field':34s} {'range':>6s} {'SD':>6s} {'var%':>6s} {'modeShr':>8s} {'r_tot':>6s}")
for f,rng,sd,ms,r,n in rows:
    print(f"{f:34s} {rng:6.2f} {sd:6.3f} {100*sd**2/tot_var:6.2f} {ms:8.2f} {r:6.3f}")
print("\nsum per-field var=%.3f (sd equiv %.3f) vs total_score var=%.3f"%(tot_var,tot_var**0.5,ts.var()))

# cumulative
print("\n=== cumulative variance share ===")
c=0
for f,rng,sd,ms,r,n in rows:
    c+=100*sd**2/tot_var
    print(f"{f:34s} cum={c:6.2f}%")

# how much of total_score sd is explained by top-k fields
print("\n=== MAE if we predict top-k fields perfectly, rest = mode ===")
mode_s={f:S[f].dropna().mode().iloc[0] for f in S.columns}
base=pd.Series(0.0,index=dev.index)
for f in S.columns: base+=S[f].fillna(mode_s[f]).where(S[f].notna(),mode_s[f])
recon=pd.Series(0.0,index=dev.index)
order=[r[0] for r in rows]
for k in [0,1,2,3,4,5,7,10,21]:
    pred=pd.Series(0.0,index=dev.index)
    for f in S.columns:
        if f in order[:k]: pred+=S[f].fillna(mode_s[f])
        else: pred+=mode_s[f]
    ok=ts.notna()
    print("  top%2d perfect -> MAE=%.3f  rho=%.3f"%(k,(pred[ok]-ts[ok]).abs().mean(),pd.Series(pred[ok]).corr(ts[ok],method='spearman')))
