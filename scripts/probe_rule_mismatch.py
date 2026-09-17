import pandas as pd, json, numpy as np
M='/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
R='/root/nailfold/artifacts/labels/score_rules_v3.json'
df=pd.read_csv(M,dtype={'exam_case_id':str})
dev=df[df.evaluation_role=='development'].copy()
J=json.load(open(R,encoding='utf-8')); FR=J['field_rules']
fields=[]
for g,fl in J['groups'].items(): fields.extend(fl)
fields=list(dict.fromkeys(fields))
def sc(f,v):
    r=FR.get(f)
    if r is None or r['type']=='missing': return None
    if r['type']=='categorical_lookup': return r['mapping'].get(str(v))
    if 'tree' not in r: return None
    try: x=float(v)
    except: return None
    n=r['tree']
    while 'threshold' in n: n=n['left'] if x<=n['threshold'] else n['right']
    return n.get('value')
S=pd.DataFrame({f:pd.to_numeric(dev[f].astype(str).map(lambda v: sc(f,v)),errors='coerce') for f in fields if f in dev.columns})
ts=pd.to_numeric(dev['total_score'],errors='coerce')
d=S.sum(axis=1,min_count=1)-ts
ok=d.notna()

print("=== LEAVE-ONE-FIELD-OUT: which field's removal most reduces the gap? ===")
base=d[ok].abs().mean()
print("  baseline MAE(all fields) = %.3f"%base)
rows=[]
for f in S.columns:
    d2=(S.drop(columns=[f]).sum(axis=1,min_count=1)-ts)
    rows.append((f,d2[ok].abs().mean()))
rows.sort(key=lambda x:x[1])
for f,m in rows[:10]:
    print(f"    drop {f:30s} -> MAE={m:.3f}  (delta {m-base:+.3f})")

print("\n=== group-level check: do the 3 group subtotals exist as columns? ===")
gcols=[c for c in dev.columns if 'score' in c.lower()]
print("  score-ish columns:",gcols)
for g,fl in J['groups'].items():
    if g in dev.columns:
        rep=pd.to_numeric(dev[g],errors='coerce')
        mine=S[[f for f in fl if f in S.columns]].sum(axis=1,min_count=1)
        m=rep.notna()&mine.notna()
        print(f"  {g:20s} n={m.sum():3d} MAE={ (mine[m]-rep[m]).abs().mean():.3f} bias={(mine[m]-rep[m]).mean():+.3f} within0.05={100*((mine[m]-rep[m]).abs()<0.05).mean():.0f}%")

print("\n=== is total_score just the sum of the 3 group scores? ===")
gs=[g for g in J['groups'] if g in dev.columns]
if len(gs)==3:
    gsum=sum(pd.to_numeric(dev[g],errors='coerce') for g in gs)
    m=gsum.notna()&ts.notna()
    e=(gsum[m]-ts[m])
    print("  n=%d MAE=%.4f bias=%+.4f within0.001=%.1f%% max=%.3f"%(m.sum(),e.abs().mean(),e.mean(),100*(e.abs()<0.001).mean(),e.abs().max()))

print("\n=== case archive1/37 detail (ts=2.00, recon=8.80) ===")
i=dev.index[dev.exam_case_id=='recovered_archive1/37']
if len(i):
    i=i[0]
    print(f"  total_score={ts[i]}  overall={dev.loc[i,'overall_assessment']}")
    for g in gs: print(f"  {g} = {dev.loc[i,g]}")
    for f in sorted(S.columns,key=lambda x:-(S[x][i] if S[x][i]==S[x][i] else -1)):
        v=S[f][i]
        if v==v and v>0: print(f"    {f:30s} raw={str(dev.loc[i,f])[:22]:24s} score={v}")
