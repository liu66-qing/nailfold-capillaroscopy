"""Development-only 5-fold OOF GBT for dual_seg_geo binary fields."""
from pathlib import Path
import argparse, json
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
FIELDS=['clarity','blood_color','exudation','subpapillary_venous_plexus','papilla']; DELIVERY=FIELDS[:-1]
MAP={'clarity':{'清晰':'clear','不清':'poor','模糊':'poor'},'blood_color':{'暗红':'dark','暗紫':'dark','浅红':'light','淡红':'light'},'exudation':{'无':'absent','+':'present','++':'present','+++':'present'},'subpapillary_venous_plexus':{'不见':'absent','可见1排':'present','可见2排':'present','>2排,扩张':'present'},'papilla':{'平坦':'flat','浅波纹状':'wavy','波纹状':'wavy'}}
def aggregate(x,idx):
 d=pd.DataFrame(x); d['exam_case_id']=idx.exam_case_id.values; return d.groupby('exam_case_id').mean(numeric_only=True)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--dinov2',type=Path,required=True); ap.add_argument('--dino-index',type=Path,required=True); ap.add_argument('--hulumed',type=Path,required=True); ap.add_argument('--hulu-index',type=Path,required=True); ap.add_argument('--seg',type=Path,required=True); ap.add_argument('--geometry',type=Path,required=True); ap.add_argument('--labels',type=Path,required=True); ap.add_argument('--roles',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
 roles=pd.read_csv(args.roles,usecols=['exam_case_id','evaluation_role','development_fold']); roles.exam_case_id=roles.exam_case_id.astype(str); dev=set(roles.loc[roles.evaluation_role.eq('development'),'exam_case_id']); locked=set(roles.loc[roles.evaluation_role.eq('locked_test'),'exam_case_id'])
 if len(dev)!=186 or len(locked)!=47: raise ValueError('invalid role boundary')
 def load(p,ip):
  ix=pd.read_csv(ip); ix.exam_case_id=ix.exam_case_id.astype(str); return np.load(p,mmap_mode='r'),ix
 d,di=load(args.dinov2,args.dino_index); h,hi=load(args.hulumed,args.hulu_index); g,gi=load(args.geometry/'features.npy',args.geometry/'index.csv')
 if not di[['exam_case_id','image_path']].equals(hi[['exam_case_id','image_path']]): raise ValueError('dino/hulu index mismatch')
 if not di[['exam_case_id','image_path']].equals(gi[['exam_case_id','image_path']]): raise ValueError('geometry index mismatch')
 if set(di.exam_case_id)&locked or set(gi.exam_case_id)&locked: raise ValueError('locked feature row')
 seg=pd.read_parquet(args.seg).rename(columns={'case_id':'exam_case_id'}); seg.exam_case_id=seg.exam_case_id.astype(str); sc=[c for c in seg if c.startswith('feature_')]+['frame_count']; segv=seg.set_index('exam_case_id')[sc]
 vectors={'dual_seg_geo':aggregate(np.concatenate([d,h],1),di).join(segv).join(aggregate(g,gi),rsuffix='_geo')}
 vectors['dual_seg']=vectors['dual_seg_geo'].iloc[:,:d.shape[1]+h.shape[1]+len(sc)]
 labels=pd.read_csv(args.labels); labels.exam_case_id=labels.exam_case_id.astype(str); lab=labels.set_index('exam_case_id'); out={'schema_version':'dual-seg-geo-binary-oof/1.0','evaluation_role':'development_oof','locked_cases_seen':0,'cases':186,'routes':{},'fields':FIELDS}
 for route,Xdf in vectors.items():
  frame=Xdf.join(roles.set_index('exam_case_id')[['development_fold']]).join(lab[FIELDS]); route_out={}
  for field in FIELDS:
   y=frame[field].map(MAP[field]); valid=y.notna(); X=frame.drop(columns=FIELDS+['development_fold']).loc[valid].to_numpy(float); yy=y[valid].to_numpy(); folds=frame.loc[valid,'development_fold'].astype(int).to_numpy(); preds=np.full(len(yy),'',dtype=object)
   for test in range(5):
    val=(folds==(test+1)%5); train=(folds!=test)&(folds!=(test+1)%5); clf=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=2,class_weight='balanced',random_state=17,n_jobs=1) if field in DELIVERY else LogisticRegression(max_iter=2000,class_weight='balanced')
    clf.fit(X[train],yy[train]); preds[folds==test]=clf.predict(X[folds==test])
   route_out[field]={'n':int(len(yy)),'balanced_accuracy':float(balanced_accuracy_score(yy,preds)),'macro_f1':float(f1_score(yy,preds,average='macro')),'exploratory':field=='papilla'}
  out['routes'][route]=route_out
 out['delivery_mean_ba']={r:float(np.mean([m[f]['balanced_accuracy'] for f in FIELDS if f in DELIVERY])) for r,m in out['routes'].items()}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(out['delivery_mean_ba']))
if __name__=='__main__': main()
