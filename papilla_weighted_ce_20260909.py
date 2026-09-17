"""Nested-development papilla ordinary-vs-weighted CE retraining."""
import argparse, copy, json, random
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.insert(0,'/tmp')
import unified_protocol_20260908 as u

def run(mode, name, rank, lr, epochs=20):
    out=Path('/root/nailfold/artifacts/experiments/downstream_20260909'); out.mkdir(parents=True,exist_ok=True)
    labels=pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv',dtype={'exam_case_id':str})
    dev=labels[labels.evaluation_role.eq('development')].copy(); dev.development_fold=dev.development_fold.astype(int)
    frames=pd.read_csv('/root/nailfold/artifacts/features/image_index.csv',dtype={'exam_case_id':str}); frames=frames[frames.exam_case_id.isin(set(dev.exam_case_id))]
    proto=u.timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=False,num_classes=0); dc=u.timm.data.resolve_model_data_config(proto); del proto
    tr=u.timm.data.create_transform(**dc,is_training=True,color_jitter=0,hflip=.5,vflip=0); ev=u.timm.data.create_transform(**dc,is_training=False)
    idx=dev.set_index('exam_case_id'); device='cuda:0'; rows=[]
    for seed in u.SEEDS:
        u.seed_all(seed)
        for fold in range(5):
            valfold=(fold+1)%5; trainids=dev.loc[~dev.development_fold.isin([fold,valfold]),'exam_case_id'].tolist(); valids=dev.loc[dev.development_fold.eq(valfold),'exam_case_id'].tolist(); testids=dev.loc[dev.development_fold.eq(fold),'exam_case_id'].tolist()
            train=DataLoader(u.Cases(trainids,frames,dev,'/root/nailfold/data',tr),batch_size=4,shuffle=True,collate_fn=u.collate,num_workers=2,pin_memory=True)
            val=DataLoader(u.Cases(valids,frames,dev,'/root/nailfold/data',ev),batch_size=4,collate_fn=u.collate,num_workers=2)
            test=DataLoader(u.Cases(testids,frames,dev,'/root/nailfold/data',ev),batch_size=4,collate_fn=u.collate,num_workers=2)
            cfg={'kind':'cls','rank':rank,'lr':lr}; model=u.build(cfg,'/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth',device); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=.05); sch=CosineAnnealingLR(opt,T_max=epochs)
            weights=None
            if mode=='weighted':
                y=dev.loc[dev.exam_case_id.isin(trainids),'papilla'].map(u.MAP['papilla']).dropna().astype(int); counts=np.bincount(y,minlength=2); weights={'papilla':torch.tensor(len(y)/(2*np.maximum(counts,1)),dtype=torch.float32)}
            best=(-1,None,0)
            for ep in range(1,epochs+1):
                u.train_epoch(model,train,opt,device,case_only=False,papilla_weighted=(mode=='weighted'),class_weights=weights); sch.step(); vp,_=u.infer(model,val,device,'validation',seed,fold,name)
                score=u.objective(vp,valids,idx)
                if score>best[0]: best=(score,{k:v.cpu().clone() for k,v in model.state_dict().items()},ep)
            model.load_state_dict(best[1]); _,rr=u.infer(model,test,device,'test',seed,fold,name,True); rows.extend(rr); print(mode,seed,fold,best[2],best[0],flush=True); del model,opt,sch; torch.cuda.empty_cache()
    d=pd.DataFrame(rows); d.to_csv(out/f'papilla_{mode}_frame_logits.csv.gz',index=False,compression='gzip')
    z=d[d.field.eq('papilla')].groupby('exam_case_id').apply(lambda x:1/(1+np.exp(-((x.logit_1-x.logit_0).mean()))),include_groups=False); truth=dev.set_index('exam_case_id').papilla.map(u.MAP['papilla']); ids=[c for c in z.index if pd.notna(truth.get(c))]; y=np.array([truth[c] for c in ids]); p=z.loc[ids].values
    return {'mode':mode,'n':len(ids),'ba':float(u.balanced_accuracy_score(y,p>=.5)),'auc':float(u.roc_auc_score(y,p))}

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--rank',type=int,default=8); ap.add_argument('--lr',type=float,default=2e-4); a=ap.parse_args(); results=[run('ordinary','papilla_ordinary',a.rank,a.lr),run('weighted','papilla_weighted',a.rank,a.lr)]; Path('/root/nailfold/artifacts/experiments/downstream_20260909/papilla_ce_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n'); print(results)
