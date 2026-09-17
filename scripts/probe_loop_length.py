import pandas as pd, json, numpy as np
M='/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
R='/root/nailfold/artifacts/labels/score_rules_v3.json'
df=pd.read_csv(M,dtype={'exam_case_id':str})
dev=df[df.evaluation_role=='development'].copy()
FR=json.load(open(R,encoding='utf-8'))['field_rules']

def sc(f,v):
    r=FR.get(f)
    if r is None: return None
    if r['type']=='categorical_lookup': return r['mapping'].get(str(v))
    try: x=float(v)
    except: return None
    n=r['tree']
    while 'threshold' in n: n=n['left'] if x<=n['threshold'] else n['right']
    return n.get('value')

for f in ['loop_length','afferent_diameter','apex_diameter','efferent_diameter','output_input_ratio']:
    raw=pd.to_numeric(dev[f],errors='coerce')
    s=pd.to_numeric(dev[f].astype(str).map(lambda v: sc(f,v)),errors='coerce')
    print("="*70)
    print(f"{f}: n={raw.notna().sum()} min={raw.min()} max={raw.max()} median={raw.median()}")
    print("  rule:",json.dumps(FR[f],ensure_ascii=False)[:220] if f in FR else "NO RULE")
    if s.notna().sum():
        vc=s.value_counts().sort_index()
        print("  score dist:",dict(vc))
        print("  score sd=%.3f range=%.2f"%(s.std(),s.max()-s.min()))
    # implausible values
    print("  raw quantiles:",[round(float(raw.quantile(q)),1) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]])

print("="*70)
print("loop_length: physiological plausibility check")
ll=pd.to_numeric(dev['loop_length'],errors='coerce')
print("  <100 um:",int((ll<100).sum()),"  >500 um:",int((ll>500).sum()),"  >1000:",int((ll>1000).sum()))
print("  values >500:",sorted(ll[ll>500].dropna().tolist())[:20])
print("  values <100:",sorted(ll[ll<100].dropna().tolist())[:20])
st=dev['__status'] if '__status' in dev.columns else None
if st is not None:
    print("\n  __status among loop_length outliers (>500 or <100):")
    m=(ll>500)|(ll<100)
    print("   ",dict(dev.loc[m,'__status'].astype(str).value_counts()))
    print("  __status overall:")
    print("   ",dict(dev['__status'].astype(str).value_counts()))
