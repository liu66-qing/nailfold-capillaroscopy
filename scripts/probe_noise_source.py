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

S=pd.DataFrame({f:pd.to_numeric(dev[f].astype(str).map(lambda v: sc(f,v)),errors='coerce')
                for f in fields if f in dev.columns})
ts=pd.to_numeric(dev['total_score'],errors='coerce')
recon=S.sum(axis=1,min_count=1)
d=(recon-ts)
ok=d.notna()
print("=== rule-reconstructed total vs OCR total_score ===")
print("  n=%d  MAE=%.3f  bias(mean recon-ts)=%.3f  sd=%.3f"%(ok.sum(),d[ok].abs().mean(),d[ok].mean(),d[ok].std()))
print("  within 0.05: %.1f%%   |d|>1.0: %d   max|d|=%.2f"%(100*(d[ok].abs()<0.05).mean(),int((d[ok].abs()>1).sum()),d[ok].abs().max()))
print("  recon mean=%.3f  ts mean=%.3f"%(recon[ok].mean(),ts[ok].mean()))

print("\n=== how many fields fail to score per case ===")
nmiss=S.isna().sum(axis=1)
print("  ",dict(nmiss.value_counts().sort_index()))
print("  fields that NEVER score:",[c for c in S.columns if S[c].notna().sum()==0])
print("  per-field unscored counts (top):")
for f,n in S.isna().sum().sort_values(ascending=False).head(8).items():
    print(f"    {f:32s} unscored={n}")

print("\n=== does the gap correlate with n_unscored_fields? ===")
m=ok&nmiss.notna()
print("  corr(|d|, n_unscored) = %.3f"%np.corrcoef(d[m].abs(),nmiss[m])[0,1])
for k in sorted(nmiss.unique()):
    s=d[ok&(nmiss==k)]
    if len(s): print(f"    n_unscored={k}: cases={len(s):3d}  MAE={s.abs().mean():.3f}  bias={s.mean():+.3f}")

print("\n=== impact of physiologically implausible values ===")
lim={'afferent_diameter':(4,30),'apex_diameter':(6,40),'efferent_diameter':(6,35),'loop_length':(80,500)}
bad=pd.Series(False,index=dev.index)
for f,(lo,hi) in lim.items():
    v=pd.to_numeric(dev[f],errors='coerce'); bad|=((v<lo)|(v>hi)).fillna(False)
print("  cases with >=1 implausible value: %d/%d"%(int(bad.sum()),len(dev)))
print("  MAE among implausible cases   = %.3f (n=%d)"%(d[ok&bad].abs().mean(),int((ok&bad).sum())))
print("  MAE among plausible cases     = %.3f (n=%d)"%(d[ok&~bad].abs().mean(),int((ok&~bad).sum())))

print("\n=== worst 15 cases by |recon - ts| ===")
w=d[ok].abs().sort_values(ascending=False).head(15)
print(f"  {'case':26s} {'ts':>6s} {'recon':>7s} {'diff':>7s} {'nmiss':>5s}")
for i in w.index:
    print(f"  {str(dev.loc[i,'exam_case_id']):26s} {ts[i]:6.2f} {recon[i]:7.2f} {d[i]:+7.2f} {nmiss[i]:5d}")
