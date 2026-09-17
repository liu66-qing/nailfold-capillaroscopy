"""Stages 3/4 using each outer model's own validation/test frame logits only."""
import argparse, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

FIELDS=["clarity","blood_color","exudation","subpapillary_venous_plexus","papilla"]
MAP={"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
SEEDS=[17,42,123,456,789]
BASE=Path('/root/nailfold/artifacts/experiments/unified_protocol_20260908')
OUT=Path('/root/nailfold/artifacts/experiments/downstream_20260909')

class FieldAttention(nn.Module):
    def __init__(self,dim=10,hidden=16):
        super().__init__(); self.att=nn.Sequential(nn.Linear(dim,hidden),nn.Tanh(),nn.Linear(hidden,1)); self.cls=nn.Linear(dim,2)
    def forward(self,x):
        a=torch.softmax(self.att(x).squeeze(-1),0); z=(a[:,None]*x).sum(0); return self.cls(z),a

def case_features(df):
    # All five heads' two logits form a 10D per-frame representation.
    x=df.pivot_table(index=['exam_case_id','image_path'],columns='field',values=['logit_0','logit_1'])
    wanted=pd.MultiIndex.from_product([['logit_0','logit_1'],FIELDS]); x=x.reindex(columns=wanted).dropna().reset_index()
    return {c:g.iloc[:,2:].to_numpy(np.float32) for c,g in x.groupby('exam_case_id')}

def labels():
    d=pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv',dtype={'exam_case_id':str})
    d=d[d.evaluation_role.eq('development')].copy(); d.development_fold=d.development_fold.astype(int); return d.set_index('exam_case_id')

def train_attention(features,ids,y,seed,loss_mode):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); m=FieldAttention(); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=.01)
    order=list(ids)
    for _ in range(150):
        random.shuffle(order)
        for c in order:
            if c not in features or c not in y: continue
            logit,_=m(torch.tensor(features[c])); target=torch.tensor([y[c]])
            loss=nn.functional.cross_entropy(logit[None],target); opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()

def predict_attention(m,features,ids):
    out={}; weights={}
    with torch.inference_mode():
        for c in ids:
            if c not in features: continue
            z,a=m(torch.tensor(features[c])); out[c]=float(torch.softmax(z,0)[1]); weights[c]=a.numpy().tolist()
    return out,weights

def metric(truth,prob):
    ids=sorted(set(truth)&set(prob)); y=np.array([truth[c] for c in ids]); p=np.array([prob[c] for c in ids])
    return {'n':len(ids),'ba':float(balanced_accuracy_score(y,p>=.5)),'auc':float(roc_auc_score(y,p))}

def run_attention(config):
    lab=labels(); rows=[]; weight_rows=[]; report={}
    for seed in SEEDS:
        va=pd.read_csv(BASE/f'{config}_validation_frames.csv.gz'); te=pd.read_csv(BASE/f'{config}_test_frames.csv.gz')
        va=va[va.seed.eq(seed)]; te=te[te.seed.eq(seed)]
        for field in FIELDS:
            truth={c:MAP[field][v] for c,v in lab[field].dropna().items()}
            for fold in range(5):
                vf=va[va.test_fold.eq(fold)]; tf=te[te.test_fold.eq(fold)]; vx=case_features(vf); tx=case_features(tf)
                # The outer model's validation fold is the only data used to fit the attention aggregator.
                m=train_attention(vx,list(vx),truth,seed+fold,'case_only'); pred,w=predict_attention(m,tx,list(tx))
                rows.extend(dict(config=config,seed=seed,test_fold=fold,field=field,exam_case_id=c,truth=truth[c],probability=p,prediction=int(p>=.5)) for c,p in pred.items() if c in truth)
                weight_rows.extend(dict(config=config,seed=seed,test_fold=fold,field=field,exam_case_id=c,attention=json.dumps(a)) for c,a in w.items())
    d=pd.DataFrame(rows); d.to_csv(OUT/'frame_attention_predictions.csv.gz',index=False,compression='gzip'); pd.DataFrame(weight_rows).to_csv(OUT/'frame_attention_weights.csv.gz',index=False,compression='gzip')
    for f in FIELDS:
        z=d[d.field.eq(f)]; ens=z.groupby('exam_case_id').agg(truth=('truth','first'),probability=('probability','mean'))
        report[f]=metric(ens.truth.to_dict(),ens.probability.to_dict())
    report['mean_ba']=float(np.mean([report[f]['ba'] for f in FIELDS])); return report

def papilla_posthoc(config):
    lab=labels(); truth={c:MAP['papilla'][v] for c,v in lab.papilla.dropna().items()}; rows=[]
    thresholds=np.round(np.arange(.2,.901,.01),2); margins=np.round(np.arange(0,.301,.02),2)
    va=pd.read_csv(BASE/f'{config}_validation_frames.csv.gz'); te=pd.read_csv(BASE/f'{config}_test_frames.csv.gz')
    for seed in SEEDS:
        for fold in range(5):
            def cp(d):
                d=d[(d.seed==seed)&(d.test_fold==fold)&(d.field=='papilla')]; g=d.groupby('exam_case_id')
                return g.apply(lambda x:1/(1+np.exp(-((x.logit_1-x.logit_0).mean()))),include_groups=False)
            vp,tp=cp(va),cp(te); vids=[c for c in vp.index if c in truth]; vy=np.array([truth[c] for c in vids]); vv=vp.loc[vids].values
            scores=[balanced_accuracy_score(vy,vv>=t) for t in thresholds]; th=float(thresholds[int(np.argmax(scores))])
            tids=[c for c in tp.index if c in truth]
            for c in tids: rows.append(dict(seed=seed,test_fold=fold,exam_case_id=c,truth=truth[c],probability=float(tp[c]),threshold=th,prediction=int(tp[c]>=th)))
    d=pd.DataFrame(rows); d.to_csv(OUT/'papilla_nested_threshold_predictions.csv.gz',index=False,compression='gzip')
    ens=d.groupby('exam_case_id').agg(truth=('truth','first'),probability=('probability','mean'),prediction=('prediction',lambda x:int(x.mean()>=.5)))
    report={'nested_threshold_ba':float(balanced_accuracy_score(ens.truth,ens.prediction)),'auc':float(roc_auc_score(ens.truth,ens.probability)),'selective':{}}
    # Abstention is evaluated only as coverage/BA curves, never counted as full-sample improvement.
    for margin in margins:
        keep=np.abs(ens.probability-.5)>=margin
        if keep.sum() and ens.loc[keep,'truth'].nunique()==2:
            report['selective'][str(float(margin))]={'coverage':float(keep.mean()),'ba':float(balanced_accuracy_score(ens.loc[keep,'truth'],ens.loc[keep,'probability']>=.5))}
    return report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    report={'selected_config':args.config,'stage3_frame_attention':run_attention(args.config),'stage4_papilla':papilla_posthoc(args.config),
            'limitations':['attention is fitted only on each outer model validation fold','papilla class-weighted image training is run separately by the continuation pipeline']}
    (OUT/'downstream_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); (OUT/'DOWNSTREAM_DONE').write_text('ok\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
