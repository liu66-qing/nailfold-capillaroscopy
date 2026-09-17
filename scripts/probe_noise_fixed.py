import pandas as pd, json, numpy as np, math
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
    if v is None: return None
    s=str(v).strip()
    if s in ('','nan','NaN','None','<NA>'): return None      # <-- FIX
    if r['type']=='categorical_lookup': return r['mapping'].get(s)
    if 'tree' not in r: return None
    try: x=float(s)
    except: return None
    if math.isnan(x): return None                             # <-- FIX
    n=r['tree']
    while 'threshold' in n: n=n['left'] if x<=n['threshold'] else n['right']
    return n.get('value')

S=pd.DataFrame({f:pd.to_numeric(dev[f].map(lambda v: sc(f,v)),errors='coerce') for f in fields if f in dev.columns})
ts=pd.to_numeric(dev['total_score'],errors='coerce')
recon=S.sum(axis=1,min_count=1)
d=recon-ts; ok=d.notna()
print("=== FIXED: rule-reconstructed vs OCR total_score ===")
print("  n=%d MAE=%.3f bias=%+.3f sd=%.3f"%(ok.sum(),d[ok].abs().mean(),d[ok].mean(),d[ok].std()))
print("  within 0.05: %.1f%%  |d|>1.0: %d  max|d|=%.2f"%(100*(d[ok].abs()<0.05).mean(),int((d[ok].abs()>1).sum()),d[ok].abs().max()))

print("\n=== group subtotals (fixed) ===")
for g,fl in J['groups'].items():
    if g in dev.columns:
        rep=pd.to_numeric(dev[g],errors='coerce')
        mine=S[[f for f in fl if f in S.columns]].sum(axis=1,min_count=1)
        m=rep.notna()&mine.notna(); e=mine[m]-rep[m]
        print(f"  {g:18s} n={m.sum():3d} MAE={e.abs().mean():.3f} bias={e.mean():+.3f} within0.05={100*(e.abs()<0.05).mean():.0f}%  >1.0={int((e.abs()>1).sum())}")

print("\n=== per-field variance (fixed) ===")
rows=[]
for f in S.columns:
    v=S[f].dropna()
    if len(v)==0: continue
    rows.append((f,v.max()-v.min(),v.std(),v.value_counts(normalize=True).iloc[0],int(S[f].isna().sum())))
tv=sum(r[2]**2 for r in rows); rows.sort(key=lambda x:-x[2])
print(f"  {'field':30s} {'range':>6s} {'SD':>6s} {'var%':>6s} {'modeShr':>7s} {'unscored':>8s}")
for f,rng,sd,ms,nm in rows:
    print(f"  {f:30s} {rng:6.2f} {sd:6.3f} {100*sd**2/tv:6.2f} {ms:7.2f} {nm:8d}")
print("\n  sum field var=%.3f (sd %.3f) | total_score var=%.3f (sd %.3f)"%(tv,tv**0.5,ts.var(),ts.std()))

print("\n=== ceiling: top-k fields perfect, rest = mode ===")
mode_s={f:S[f].dropna().mode().iloc[0] for f in S.columns}
order=[r[0] for r in rows]
for k in [0,1,2,3,4,5,7,10,len(order)]:
    pred=pd.Series(0.0,index=dev.index)
    for f in S.columns:
        pred+= S[f].fillna(mode_s[f]) if f in order[:k] else mode_s[f]
    m=ts.notna()
    print("  top%2d -> MAE=%.3f rho=%.3f"%(k,(pred[m]-ts[m]).abs().mean(),pd.Series(pred[m]).corr(ts[m],method='spearman')))

print("\n=== worst 12 (fixed) ===")
w=d[ok].abs().sort_values(ascending=False).head(12)
for i in w.index:
    nm=int(S.loc[i].isna().sum())
    print(f"  {str(dev.loc[i,'exam_case_id']):24s} ts={ts[i]:6.2f} recon={recon[i]:7.2f} d={d[i]:+6.2f} unscored={nm}")
