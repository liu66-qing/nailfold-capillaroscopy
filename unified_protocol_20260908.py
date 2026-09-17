"""Leakage-safe nested development OOF comparison for nailfold classification."""
import argparse, copy, gzip, json, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

FIELDS = ["clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla"]
MAP = {"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
SEEDS = [17, 42, 123, 456, 789]
CONFIGS = {
    "frozen_cls": dict(kind="cls", rank=0, lr=1e-3),
    "frozen_cls_patch": dict(kind="cls_patch", rank=0, lr=1e-3),
    "lora_r4": dict(kind="cls", rank=4, lr=1e-4),
    "lora_r8": dict(kind="cls", rank=8, lr=2e-4),
    "lora_r16": dict(kind="cls", rank=16, lr=2e-4),
}

class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha=8):
        super().__init__(); self.base=base; self.scale=alpha/rank
        self.a=nn.Linear(base.in_features,rank,bias=False); self.b=nn.Linear(rank,base.out_features,bias=False)
        nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5)); nn.init.zeros_(self.b.weight)
        for p in base.parameters(): p.requires_grad=False
    def forward(self,x): return self.base(x)+self.b(self.a(x))*self.scale

class GatedPool(nn.Module):
    def __init__(self, dim=768, hidden=256):
        super().__init__(); self.v=nn.Linear(dim,hidden); self.u=nn.Linear(dim,hidden); self.w=nn.Linear(hidden,1)
    def forward(self,x):
        a=torch.softmax(self.w(torch.tanh(self.v(x))*torch.sigmoid(self.u(x))),dim=1)
        return (a*x).sum(1)

class Model(nn.Module):
    def __init__(self, backbone, kind):
        super().__init__(); self.b=backbone; self.kind=kind; self.drop=nn.Dropout(.1)
        if kind == "cls_patch": self.patch_pool=GatedPool()
        dim=1536 if kind == "cls_patch" else 768
        self.heads=nn.ModuleDict({f:nn.Linear(dim,2) for f in FIELDS})
    def forward(self,x):
        if self.kind == "cls_patch":
            z=self.b.forward_features(x); z=torch.cat([z[:,0],self.patch_pool(z[:,1:])],1)
        else: z=self.b(x)
        z=self.drop(z); return {f:h(z) for f,h in self.heads.items()}

class Cases(Dataset):
    def __init__(self, ids, frames, labels, root, transform):
        self.ids=list(ids); self.groups={c:g for c,g in frames[frames.exam_case_id.isin(ids)].groupby('exam_case_id')}
        self.labels=labels.set_index('exam_case_id'); self.root=Path(root); self.transform=transform
    def __len__(self): return len(self.ids)
    def __getitem__(self,i):
        cid=self.ids[i]; rows=self.groups[cid]
        imgs=torch.stack([self.transform(Image.open(self.root/r.image_path).convert('RGB')) for _,r in rows.iterrows()])
        paths=rows.image_path.tolist(); row=self.labels.loc[cid]
        y=torch.tensor([MAP[f].get(row[f],-1) if pd.notna(row[f]) else -1 for f in FIELDS])
        return imgs,y,cid,paths

def collate(batch):
    return [x[0] for x in batch],torch.stack([x[1] for x in batch]),[x[2] for x in batch],[x[3] for x in batch]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build(cfg, weights, device):
    b=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=False,num_classes=0)
    b.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=False)
    for p in b.parameters(): p.requires_grad=False
    if cfg['rank']:
        for block in b.blocks[-4:]:
            block.attn.qkv=LoRALinear(block.attn.qkv,cfg['rank']); block.attn.proj=LoRALinear(block.attn.proj,cfg['rank'])
    return Model(b,cfg['kind']).to(device)

def train_epoch(model,loader,opt,device,case_only=False,papilla_weighted=False,class_weights=None):
    model.train()
    for imgs_list,ys,_,_ in loader:
        loss=0.; ncases=0
        for imgs,y in zip(imgs_list,ys):
            imgs=imgs.to(device); y=y.to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16): out=model(imgs)
            one=0.; nt=0
            for i,f in enumerate(FIELDS):
                if y[i]<0: continue
                weight=class_weights[f].to(device) if papilla_weighted and f=='papilla' else None
                case_ce=nn.functional.cross_entropy(out[f].mean(0,keepdim=True),y[i:i+1],weight=weight)
                if case_only: term=case_ce
                else:
                    target=y[i].expand(len(imgs)); frame_ce=nn.functional.cross_entropy(out[f],target,weight=weight)
                    term=.75*case_ce+.25*frame_ce
                one += term; nt += 1
            if nt: loss += one/nt; ncases += 1
        loss /= max(ncases,1); opt.zero_grad(); loss.backward(); opt.step()

def infer(model,loader,device,split,seed,fold,config,save_frames=False):
    cases={f:{} for f in FIELDS}; rows=[]; model.eval()
    with torch.inference_mode():
        for imgs_list,_,ids,paths_list in loader:
            for imgs,cid,paths in zip(imgs_list,ids,paths_list):
                with torch.autocast('cuda',dtype=torch.bfloat16): out=model(imgs.to(device))
                for f in FIELDS:
                    logits=out[f].float().cpu(); prob=torch.softmax(logits,1)[:,1].numpy()
                    cases[f][cid]=float(torch.softmax(logits.mean(0),0)[1])
                    if save_frames:
                        rows.extend(dict(config=config,seed=seed,test_fold=fold,split=split,exam_case_id=cid,image_path=p,
                                         field=f,logit_0=float(z[0]),logit_1=float(z[1]),probability=float(q))
                                    for p,z,q in zip(paths,logits.numpy(),prob))
    return cases,rows

def objective(pred,ids,label_index):
    vals=[]
    for f in FIELDS:
        use=[c for c in ids if pd.notna(label_index.at[c,f])]
        y=np.array([MAP[f][label_index.at[c,f]] for c in use]); p=np.array([pred[f][c]>=.5 for c in use])
        vals.append(balanced_accuracy_score(y,p))
    return float(np.mean(vals))

def evaluate(config, all_test, all_val, labels, outdir):
    idx=labels.set_index('exam_case_id'); thresholds=np.round(np.arange(.2,.801,.01),2)
    methods=['mean_logits','mean_prob','median_prob','max_prob','top3_prob']
    report={'config':config,'seeds':{},'ensemble':{},'selection':{}}
    testdf=pd.DataFrame(all_test); valdf=pd.DataFrame(all_val)
    for seed in SEEDS:
        report['seeds'][str(seed)]={}
        for f in FIELDS:
            d=testdf[(testdf.seed==seed)&(testdf.field==f)]
            g=d.groupby('exam_case_id'); cp=g.apply(lambda x:1/(1+np.exp(-((x.logit_1-x.logit_0).mean()))),include_groups=False)
            ids=[c for c in cp.index if pd.notna(idx.at[c,f])]; y=np.array([MAP[f][idx.at[c,f]] for c in ids]); p=cp.loc[ids].values
            report['seeds'][str(seed)][f]={'ba':float(balanced_accuracy_score(y,p>=.5)),'auc':float(roc_auc_score(y,p))}
    # Ensemble raw mean-logit probability, plus strictly nested threshold/aggregation selection.
    raw_by_seed={}
    nested_rows=[]
    for seed in SEEDS:
        raw_by_seed[seed]={}
        for f in FIELDS:
            d=testdf[(testdf.seed==seed)&(testdf.field==f)]; g=d.groupby('exam_case_id')
            raw_by_seed[seed][f]=g.apply(lambda x:1/(1+np.exp(-((x.logit_1-x.logit_0).mean()))),include_groups=False).to_dict()
            for fold in range(5):
                vd=valdf[(valdf.seed==seed)&(valdf.test_fold==fold)&(valdf.field==f)]
                td=testdf[(testdf.seed==seed)&(testdf.test_fold==fold)&(testdf.field==f)]
                def agg(d):
                    g=d.groupby('exam_case_id'); z=pd.DataFrame(index=g.size().index)
                    z['mean_logits']=g.apply(lambda x:1/(1+np.exp(-((x.logit_1-x.logit_0).mean()))),include_groups=False)
                    z['mean_prob']=g.probability.mean(); z['median_prob']=g.probability.median(); z['max_prob']=g.probability.max()
                    z['top3_prob']=g.probability.apply(lambda x:x.nlargest(min(3,len(x))).mean()); return z
                va,ta=agg(vd),agg(td); best=None
                for m in methods:
                    vids=[c for c in va.index if pd.notna(idx.at[c,f])]; vy=np.array([MAP[f][idx.at[c,f]] for c in vids])
                    for t in thresholds:
                        ba=balanced_accuracy_score(vy,va.loc[vids,m].values>=t)
                        key=(ba,-abs(t-.5),m=='mean_logits')
                        if best is None or key>best[0]: best=(key,m,float(t),float(ba))
                tids=[c for c in ta.index if pd.notna(idx.at[c,f])]
                for c in tids: nested_rows.append(dict(seed=seed,field=f,exam_case_id=c,truth=MAP[f][idx.at[c,f]],probability=float(ta.at[c,best[1]]),prediction=int(ta.at[c,best[1]]>=best[2]),method=best[1],threshold=best[2],test_fold=fold))
                report['selection'].setdefault(str(seed),{}).setdefault(f,[]).append({'test_fold':fold,'method':best[1],'threshold':best[2],'validation_ba':best[3]})
    nested=pd.DataFrame(nested_rows); nested.to_csv(outdir/f'{config}_nested_predictions.csv.gz',index=False,compression='gzip')
    for f in FIELDS:
        common=sorted(set.intersection(*[set(raw_by_seed[s][f]) for s in SEEDS])); y=np.array([MAP[f][idx.at[c,f]] for c in common if pd.notna(idx.at[c,f])]); common=[c for c in common if pd.notna(idx.at[c,f])]
        p=np.array([np.mean([raw_by_seed[s][f][c] for s in SEEDS]) for c in common])
        nd=nested[nested.field==f]; npred=nd.groupby('exam_case_id').prediction.mean(); nids=[c for c in npred.index if pd.notna(idx.at[c,f])]; ny=np.array([MAP[f][idx.at[c,f]] for c in nids])
        report['ensemble'][f]={'ba_0_5':float(balanced_accuracy_score(y,p>=.5)),'auc':float(roc_auc_score(y,p)),
                               'nested_selected_ba':float(balanced_accuracy_score(ny,npred.loc[nids]>=.5))}
    report['mean_ba_0_5']=float(np.mean([v['ba_0_5'] for v in report['ensemble'].values()]))
    report['mean_nested_selected_ba']=float(np.mean([v['nested_selected_ba'] for v in report['ensemble'].values()]))
    (outdir/f'{config}_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--configs',nargs='+',required=True); ap.add_argument('--gpu',type=int,required=True); ap.add_argument('--epochs',type=int,default=20); args=ap.parse_args()
    out=Path('/root/nailfold/artifacts/experiments/unified_protocol_20260908'); out.mkdir(parents=True,exist_ok=True)
    labels=pd.read_csv('/root/nailfold/artifacts/manifest/locked_evaluation_v1_reviewed.csv',dtype={'exam_case_id':str})
    assert set(labels.evaluation_role)=={'development','locked_test'}
    dev=labels[labels.evaluation_role.eq('development')].copy(); dev.development_fold=dev.development_fold.astype(int)
    frames=pd.read_csv('/root/nailfold/artifacts/features/image_index.csv',dtype={'exam_case_id':str}); frames=frames[frames.exam_case_id.isin(set(dev.exam_case_id))]
    assert not set(frames.exam_case_id)&set(labels.loc[labels.evaluation_role.eq('locked_test'),'exam_case_id'])
    proto=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=False,num_classes=0); dc=timm.data.resolve_model_data_config(proto); del proto
    tr=timm.data.create_transform(**dc,is_training=True,color_jitter=0,hflip=.5,vflip=0); ev=timm.data.create_transform(**dc,is_training=False)
    idx=dev.set_index('exam_case_id'); device=f'cuda:{args.gpu}'
    for name in args.configs:
        cfg=CONFIGS[name]; all_val=[]; all_test=[]; selections=[]
        print('CONFIG',name,flush=True)
        for seed in SEEDS:
            seed_all(seed)
            for fold in range(5):
                valfold=(fold+1)%5; trainids=dev.loc[~dev.development_fold.isin([fold,valfold]),'exam_case_id'].tolist(); valids=dev.loc[dev.development_fold.eq(valfold),'exam_case_id'].tolist(); testids=dev.loc[dev.development_fold.eq(fold),'exam_case_id'].tolist()
                train=DataLoader(Cases(trainids,frames,dev,'/root/nailfold/data',tr),batch_size=4,shuffle=True,collate_fn=collate,num_workers=2,pin_memory=True)
                val=DataLoader(Cases(valids,frames,dev,'/root/nailfold/data',ev),batch_size=4,collate_fn=collate,num_workers=2)
                test=DataLoader(Cases(testids,frames,dev,'/root/nailfold/data',ev),batch_size=4,collate_fn=collate,num_workers=2)
                model=build(cfg,'/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth',device); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],weight_decay=.05); sched=CosineAnnealingLR(opt,T_max=args.epochs)
                best=(-1,None,0)
                for epoch in range(1,args.epochs+1):
                    train_epoch(model,train,opt,device); sched.step(); vp,_=infer(model,val,device,'validation',seed,fold,name)
                    obj=objective(vp,valids,idx)
                    if obj>best[0]: best=(obj,{k:v.cpu().clone() for k,v in model.state_dict().items()},epoch)
                model.load_state_dict(best[1]); _,vr=infer(model,val,device,'validation',seed,fold,name,True); _,te=infer(model,test,device,'test',seed,fold,name,True)
                all_val.extend(vr); all_test.extend(te); selections.append({'seed':seed,'test_fold':fold,'validation_fold':valfold,'epoch':best[2],'validation_mean_ba':best[0]})
                print(name,seed,fold,'epoch',best[2],'val',round(best[0],4),flush=True); del model,opt,sched,best; torch.cuda.empty_cache()
        pd.DataFrame(all_val).to_csv(out/f'{name}_validation_frames.csv.gz',index=False,compression='gzip'); pd.DataFrame(all_test).to_csv(out/f'{name}_test_frames.csv.gz',index=False,compression='gzip')
        report=evaluate(name,all_test,all_val,dev,out); report['epoch_selections']=selections; (out/f'{name}_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print('DONE',name,report['mean_ba_0_5'],report['mean_nested_selected_ba'],flush=True)

if __name__=='__main__': main()
