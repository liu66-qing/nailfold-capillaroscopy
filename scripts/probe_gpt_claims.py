import pandas as pd, os, collections
IDX='/root/nailfold/artifacts/features/image_index.csv'
ix=pd.read_csv(IDX,dtype=str)
print("image_index columns:",list(ix.columns),"rows:",len(ix))
print(ix.head(3).to_string())
pathcol=[c for c in ix.columns if 'path' in c.lower() or 'file' in c.lower() or 'image' in c.lower()]
pc=pathcol[0] if pathcol else ix.columns[-1]
print("\nusing path column:",pc)
base=ix[pc].map(lambda p: os.path.basename(str(p)))
pref=base.map(lambda b: ''.join(ch for ch in b.split('.')[0] if not ch.isdigit()))
print("\n=== frame filename prefixes (is 报告截图 rep*.jpg included?) ===")
for k,v in collections.Counter(pref).most_common(15):
    print(f"  {k:24s} {v}")
print("\n  frames whose name starts with 'rep':",int(base.str.lower().str.startswith('rep').sum()))
print("  frames whose name starts with 'CAPorg':",int(base.str.startswith('CAPorg').sum()))

M='/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
df=pd.read_csv(M,dtype={'exam_case_id':str})
dev=df[df.evaluation_role=='development']
print("\n=== UNITS: are the top-variance fields temporal (per min/sec)? ===")
for f in ['microthrombus','wbc_count','vasomotion','rbc_aggregation','exudation','capillary_count','flow_state','papilla']:
    vals=dev[f].astype(str).value_counts()
    print(f"\n  {f}:")
    for k,v in vals.items(): print(f"      {k!r:28s} {v}")
