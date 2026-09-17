import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, recall_score

B=Path('/root/autodl-tmp/nailfold')
idx=pd.read_csv(B/'artifacts/experiments/model-v3-20260901/detector_features_v3.csv')
X=np.load(B/'artifacts/experiments/model-v3-20260901/detector_features_v3.npy')
lab=pd.read_csv(B/'artifacts/labels/multisource_confidence_v3.csv',encoding='utf-8-sig')
fold=pd.read_csv(B/'artifacts/manifest/stratified_folds_v3.csv')[['exam_case_id','fold']]
rows=[]
for c,g in idx.groupby('exam_case_id'):
 a=X[g.index]
 # detector layout: each of 3 classes has 9 stats; first stat is count
 cnt=a[:,[0,9,18]].mean(0); total=cnt.sum()
 rows.append({'exam_case_id':c,'vessel':cnt[0],'malformed':cnt[1],'cross':cnt[2],
              'cross_ratio':cnt[2]/max(total,1e-6),'mal_ratio':cnt[1]/max(total,1e-6),'total':total})
d=pd.DataFrame(rows).merge(lab,on='exam_case_id').merge(fold,on='exam_case_id')
out=B/'artifacts/experiments/model-v6-20260901/lagging_experts';out.mkdir(parents=True,exist_ok=True)
res={}
for field,score,classes in [('crossing_ratio','cross_ratio',['<=30%','30--60%','60--80%','>80%']),('malformation_ratio','mal_ratio',['<=10%','10--30%','30--60%','>60%'])]:
 z=d.dropna(subset=[field]).copy(); z=z[z[field].isin(classes)].reset_index(drop=True); y=z[field].map({c:i for i,c in enumerate(classes)}).to_numpy(); pred=np.zeros(len(z),int); fold_ba=[]
 for k in sorted(z.fold.astype(int).unique()):
  tr=z.fold.to_numpy()!=k; va=~tr; vals=score and z[score].to_numpy()
  # optimize monotonic thresholds on training grid
  grid=np.linspace(0,1,21); best=(-1,None)
  for t1 in grid:
   for t2 in grid[grid>=t1]:
    for t3 in grid[grid>=t2]:
     pp=np.digitize(vals[tr],[t1,t2,t3]); ba=balanced_accuracy_score(y[tr],pp)
     if ba>best[0]: best=(ba,(t1,t2,t3))
  pred[va]=np.digitize(vals[va],best[1]); fold_ba.append(float(balanced_accuracy_score(y[va],pred[va])))
 res[field]={'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'fold_ba':fold_ba,
  'minority_recall':recall_score(y,pred,average=None,zero_division=0).tolist(),
  'spearman':float(spearmanr(z[score],y).statistic),'n':len(z),'locked_cases_seen':0}
 z['prediction']=[classes[i] for i in pred]; z.to_csv(out/(field+'_proxy_oof.csv'),index=False)
res['diagnostic']={'cross_ratio_spearman':float(spearmanr(d.dropna(subset=['crossing_ratio']).cross_ratio,d.dropna(subset=['crossing_ratio']).crossing_ratio.map({c:i for i,c in enumerate(classes)})).statistic)}
(out/'proxy_metrics.json').write_text(json.dumps(res,indent=2))
print(json.dumps(res,indent=2))
