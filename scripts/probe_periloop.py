import pandas as pd, json, numpy as np, math
M='/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
R='/root/nailfold/artifacts/labels/score_rules_v3.json'
df=pd.read_csv(M,dtype={'exam_case_id':str})
dev=df[df.evaluation_role=='development'].copy()
J=json.load(open(R,encoding='utf-8')); FR=J['field_rules']
G=J['groups']
print("groups:")
for g,fl in G.items(): print(f"  {g}: {fl}")

def sc(f,v):
    r=FR.get(f)
    if r is None or r['type']=='missing': return None
    if v is None: return None
    s=str(v).strip()
    if s in ('','nan','NaN','None','<NA>'): return None
    if r['type']=='categorical_lookup': return r['mapping'].get(s)
    if 'tree' not in r: return None
    try: x=float(s)
    except: return None
    if math.isnan(x): return None
    n=r['tree']
    while 'threshold' in n: n=n['left'] if x<=n['threshold'] else n['right']
    return n.get('value')

pl=G['periloop_score']
S=pd.DataFrame({f:pd.to_numeric(dev[f].map(lambda v: sc(f,v)),errors='coerce') for f in pl if f in dev.columns})
rep=pd.to_numeric(dev['periloop_score'],errors='coerce')
mine=S.sum(axis=1,min_count=1)
e=mine-rep; ok=e.notna()
print(f"\nperiloop fields scored: {list(S.columns)}")
print("MAE=%.3f bias=%+.3f  >1.0=%d"%(e[ok].abs().mean(),e[ok].mean(),int((e[ok].abs()>1).sum())))

print("\n=== leave-one-out within periloop ===")
base=e[ok].abs().mean()
for f in S.columns:
    e2=S.drop(columns=[f]).sum(axis=1,min_count=1)-rep
    print(f"  drop {f:30s} MAE={e2[ok].abs().mean():.3f} ({e2[ok].abs().mean()-base:+.3f})")

print("\n=== reported periloop_score distinct values ===")
print(dict(rep.value_counts().sort_index().head(20)))
print("\n=== is periloop just ONE field? test each single field ===")
for f in S.columns:
    m=S[f].notna()&rep.notna()
    if m.sum()<20: continue
    d1=(S[f][m]-rep[m])
    print(f"  {f:30s} alone: MAE={d1.abs().mean():.3f} within0.05={100*(d1.abs()<0.05).mean():.0f}%  corr={S[f][m].corr(rep[m]):.3f}")

print("\n=== worst periloop cases ===")
w=e[ok].abs().sort_values(ascending=False).head(10)
for i in w.index:
    print(f"  {str(dev.loc[i,'exam_case_id']):22s} rep={rep[i]:5.2f} mine={mine[i]:6.2f} d={e[i]:+6.2f}  "+" ".join(f"{f.split('_')[0]}={S[f][i]}" for f in S.columns if S[f][i]==S[f][i]))
