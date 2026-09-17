import pandas as pd, json, numpy as np, math, glob, re, os
BS=chr(92)
def decode_rtf(p):
    txt=open(p,'rb').read().decode('latin-1')
    out=[];i=0;n=len(txt)
    while i<n:
        c=txt[i]
        if c==BS:
            if i+1<n and txt[i+1]=="'":
                try: out.append(bytes([int(txt[i+2:i+4],16)])); i+=4; continue
                except: pass
            m=re.match(r'[a-zA-Z]+(-?[0-9]+)?[ ]?',txt[i+1:])
            if m:
                if m.group(0).strip().startswith(('par','line')): out.append(b'\n')
                i+=1+len(m.group(0)); continue
            i+=2; continue
        if c in '{}': i+=1; continue
        out.append(c.encode('latin-1')); i+=1
    b=b''.join(out)
    for e in ('gbk','gb18030','utf-8'):
        try: return b.decode(e)
        except: pass
    return b.decode('latin-1',errors='replace')

M='/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
R='/root/nailfold/artifacts/labels/score_rules_v3.json'
df=pd.read_csv(M,dtype={'exam_case_id':str})
dev=df[df.evaluation_role=='development'].copy()
J=json.load(open(R,encoding='utf-8')); FR=J['field_rules']
def sc(f,v):
    r=FR.get(f)
    if r is None or r['type']=='missing': return None
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
PL=J['groups']['periloop_score']
S=pd.DataFrame({f:pd.to_numeric(dev[f].map(lambda v: sc(f,v)),errors='coerce') for f in PL})
rep=pd.to_numeric(dev['periloop_score'],errors='coerce')
e=S.sum(axis=1,min_count=1)-rep

KEY='渗出'
rows=[]
for i in dev.index:
    case=dev.loc[i,'exam_case_id']
    d=f"/root/nailfold/data/{case}"
    txt=''
    for f in sorted(glob.glob(d+"/*.rtf"))+sorted(glob.glob(d+"/*.RTF")):
        try: txt+=decode_rtf(f)
        except: pass
    has_exu = KEY in txt
    strong = ('明显' in txt and KEY in txt)
    rows.append((case,has_exu,strong,str(dev.loc[i,'exudation']),S['exudation'][i],rep[i],e[i],len(txt)))
T=pd.DataFrame(rows,columns=['case','rtf_has_exudation','rtf_strong','exu_val','exu_score','periloop_rep','gap','rtf_len'])
print("cases with any RTF text: %d/%d"%((T.rtf_len>0).sum(),len(T)))
print("RTF mentions 渗出: %d   with 明显: %d"%(T.rtf_has_exudation.sum(),T.rtf_strong.sum()))

print("\n=== KEY TEST: RTF says 渗出 but extracted exudation == 无 ===")
bad=T[T.rtf_has_exudation&(T.exu_val=='无')]
print("  count: %d"%len(bad))
print("  their mean gap (recon-report periloop): %+.3f"%bad.gap.mean())
print("  mean |gap|: %.3f"%bad.gap.abs().mean())
ok2=T[~(T.rtf_has_exudation&(T.exu_val=='无'))]
print("  all others mean |gap|: %.3f (n=%d)"%(ok2.gap.abs().mean(),len(ok2)))

print("\n=== agreement table: RTF mention vs extracted value ===")
print(pd.crosstab(T.rtf_has_exudation,T.exu_val))

print("\n=== among the 51 big-gap cases (|gap|>1.0), how many have this contradiction? ===")
big=T[T.gap.abs()>1.0]
print("  big-gap n=%d ; of those RTF-says-渗出-but-value-无: %d (%.0f%%)"%(len(big),int((big.rtf_has_exudation&(big.exu_val=='无')).sum()),100*(big.rtf_has_exudation&(big.exu_val=='无')).mean()))
print("  of those RTF mentions 渗出 at all: %d"%int(big.rtf_has_exudation.sum()))
print("\n  sample of big-gap contradictions:")
for _,r in big[big.rtf_has_exudation&(big.exu_val=='无')].head(10).iterrows():
    print(f"    {r.case:24s} exu_val={r.exu_val:4s} periloop_rep={r.periloop_rep:5.2f} gap={r.gap:+6.2f}")
T.to_csv('/root/nailfold/artifacts/experiments/rtf_vs_field.csv',index=False)
print("\nsaved -> /root/nailfold/artifacts/experiments/rtf_vs_field.csv")
