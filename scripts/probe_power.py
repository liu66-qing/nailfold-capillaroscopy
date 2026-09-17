import pandas as pd, numpy as np, json, glob
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
rng=np.random.default_rng(17)
M='/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv'
df=pd.read_csv(M,dtype={'exam_case_id':str})
dev=df[df.evaluation_role=='development']
LV=["正常","大致正常","轻度异常","中度异常","重度异常"]
L2I={l:i for i,l in enumerate(LV)}
f=glob.glob('/root/nailfold/artifacts/experiments/exp_c_v2_g0/B_reg_only_znorm_oof.csv')
oof=pd.read_csv(f[0],dtype={'case':str})
print("oof cols:",list(oof.columns),"n=",len(oof))
m=oof.merge(dev[['exam_case_id','overall_assessment','total_score']],left_on='case',right_on='exam_case_id')
print("merged n=",len(m))
TH=[1.0,2.0,4.0,8.0]
def cut(s):
    for t,l in zip(TH,LV):
        if s<=t: return l
    return LV[4]
yt=m.overall_assessment.map(L2I).values
yp=m.pred_ts.map(cut).map(L2I).values
ts=pd.to_numeric(m.total_score,errors='coerce').values
ps=m.pred_ts.values
n=len(yt)

def metrics(idx):
    a,b=yt[idx],yp[idx]
    o={}
    o['mod_BA']=balanced_accuracy_score((a>=3).astype(int),(b>=3).astype(int))
    o['sev_BA']=balanced_accuracy_score((a>=4).astype(int),(b>=4).astype(int))
    o['exact']=(a==b).mean()
    o['w1']=(np.abs(a-b)<=1).mean()
    try: o['QWK']=cohen_kappa_score(a,b,weights='quadratic')
    except: o['QWK']=np.nan
    o['MAE']=np.abs(ps[idx]-ts[idx]).mean()
    # c-index
    t,p=ts[idx],ps[idx]; c=d=0
    for i in range(len(t)):
        dt=t[i+1:]-t[i]; dp=p[i+1:]-p[i]
        v=dt!=0; c+=np.sum((dt[v]*dp[v])>0); d+=np.sum(v)
    o['Cidx']=c/d if d else np.nan
    return o

base=metrics(np.arange(n))
B=1000
boot={k:[] for k in base}
for _ in range(B):
    idx=rng.integers(0,n,n)
    if len(np.unique(yt[idx]))<2: continue
    r=metrics(idx)
    for k,v in r.items(): boot[k].append(v)

print("\n=== arm B, n=%d, bootstrap %d resamples ==="%(n,B))
print(f"{'metric':10s} {'point':>7s} {'95% CI':>18s} {'CI width':>9s}  baseline")
BL={'mod_BA':0.500,'sev_BA':0.500,'exact':0.415,'w1':0.863,'QWK':0.000,'MAE':2.940,'Cidx':0.500}
for k in ['mod_BA','sev_BA','exact','w1','QWK','Cidx','MAE']:
    a=np.array(boot[k]); lo,hi=np.percentile(a,[2.5,97.5])
    excl = "EXCLUDES baseline" if (lo>BL[k] if k!='MAE' else hi<BL[k]) else "overlaps baseline"
    print(f"{k:10s} {base[k]:7.3f} [{lo:6.3f},{hi:6.3f}] {hi-lo:9.3f}  {BL[k]:.3f} {excl}")

print("\n=== how big must a TRUE improvement be to be detectable at n=%d? ==="%n)
print("  (half-width of the 95%% CI = minimum detectable difference, roughly)")
for k in ['mod_BA','sev_BA','exact','QWK','Cidx']:
    a=np.array(boot[k]); lo,hi=np.percentile(a,[2.5,97.5])
    print(f"  {k:10s} needs > {(hi-lo)/2:.3f} improvement to be distinguishable")

print("\n=== n needed to resolve a +0.05 gain in moderate+ BA (rough scaling) ===")
a=np.array(boot['mod_BA']); hw=(np.percentile(a,97.5)-np.percentile(a,2.5))/2
print("  current half-width %.3f at n=%d"%(hw,n))
for target in [0.05,0.10]:
    need=n*(hw/target)**2
    print("  to detect +%.2f  ->  need n ~= %.0f  (%.1fx current)"%(target,need,need/n))
