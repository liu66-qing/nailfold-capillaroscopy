import pandas as pd, json, numpy as np
M='/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
R='/root/nailfold/artifacts/labels/score_rules_v3.json'
df=pd.read_csv(M,dtype={'exam_case_id':str})
dev=df[df.evaluation_role=='development'].copy()
FR=json.load(open(R,encoding='utf-8'))['field_rules']
groups=json.load(open(R,encoding='utf-8'))['groups']
fields=[]
for g,fl in groups.items(): fields.extend(fl)
fields=list(dict.fromkeys(fields))

VAR={'microthrombus':41.24,'exudation':13.12,'rbc_aggregation':7.85,'loop_length':7.31,
 'capillary_count':6.91,'flow_state':6.36,'papilla':4.93,'afferent_diameter':2.70,
 'malformation_ratio':2.39,'apex_diameter':2.26,'subpapillary_venous_plexus':1.60,
 'efferent_diameter':1.27,'crossing_ratio':0.66,'hemorrhage':0.53,'clarity':0.51,
 'blood_color':0.35,'wbc_count':0.01,'vasomotion':0.00,'sweat_duct':0.00}

print(f"{'field':30s} {'var%':>6s} {'conf_mean':>9s} {'conf_lo':>7s} {'status distribution'}")
rows=[]
for f in fields:
    cc,sc_=f+'__confidence',f+'__status'
    if cc not in dev.columns: continue
    conf=pd.to_numeric(dev[cc],errors='coerce')
    st=dev[sc_].astype(str) if sc_ in dev.columns else pd.Series(['-']*len(dev),index=dev.index)
    top=st.value_counts()
    tops='  '.join(f"{k[:26]}={v}" for k,v in top.head(3).items())
    rows.append((f,VAR.get(f,np.nan),conf.mean(),(conf<0.8).sum(),tops))
rows.sort(key=lambda r:-(r[1] if r[1]==r[1] else -1))
for f,v,cm,clo,tops in rows:
    print(f"{f:30s} {v:6.2f} {cm:9.3f} {clo:7d} {tops}")

print("\n=== does low confidence concentrate in high-variance fields? ===")
v=np.array([r[1] for r in rows if r[1]==r[1]])
c=np.array([r[2] for r in rows if r[1]==r[1]])
ok=~np.isnan(c)
if ok.sum()>3:
    print("  corr(var%%, mean_confidence) = %.3f"%np.corrcoef(v[ok],c[ok])[0,1])

print("\n=== all distinct status values across fields ===")
allst={}
for f in fields:
    s=f+'__status'
    if s in dev.columns:
        for k,n in dev[s].astype(str).value_counts().items():
            allst[k]=allst.get(k,0)+n
for k,n in sorted(allst.items(),key=lambda x:-x[1]):
    print(f"  {n:5d}  {k[:120]}")
