import pandas as pd, json, numpy as np
M='/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
R='/root/nailfold/artifacts/labels/score_rules_v3.json'
df=pd.read_csv(M,dtype={'exam_case_id':str})
dev=df[df.evaluation_role=='development'].copy()
FR=json.load(open(R,encoding='utf-8'))['field_rules']
print("output_input_ratio rule:")
print(json.dumps(FR['output_input_ratio'],ensure_ascii=False)[:600])
print("\nfields whose rule lacks tree/mapping:")
for k,v in FR.items():
    if v['type']=='numeric_tree' and 'tree' not in v: print("  NO TREE:",k,list(v.keys()))
    if v['type']=='categorical_lookup' and 'mapping' not in v: print("  NO MAPPING:",k,list(v.keys()))

print("\n=== physiologically implausible values ===")
# nailfold capillary diameters are ~8-20 um; loop length ~150-400 um
lim={'afferent_diameter':(4,30),'apex_diameter':(6,40),'efferent_diameter':(6,35),'loop_length':(80,500)}
st=dev['__status'].astype(str) if '__status' in dev.columns else None
for f,(lo,hi) in lim.items():
    v=pd.to_numeric(dev[f],errors='coerce')
    bad=(v<lo)|(v>hi)
    print(f"\n{f}: plausible {lo}-{hi} um | out-of-range {int(bad.sum())}/{int(v.notna().sum())} ({100*bad.sum()/max(v.notna().sum(),1):.0f}%)")
    print("  worst:",sorted(v[bad].dropna().tolist(),reverse=True)[:12])
    if st is not None and bad.sum():
        print("  status of out-of-range:",dict(st[bad].value_counts().head(5)))

print("\n=== __status overall ===")
if st is not None:
    for k,c in st.value_counts().items(): print(f"  {c:4d}  {k[:110]}")
