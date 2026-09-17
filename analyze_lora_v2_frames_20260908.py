import json, math
from pathlib import Path

import numpy as np
import pandas as pd
import torch, timm
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path("/root/nailfold")
ART = ROOT / "artifacts"
FIELDS = ["clarity", "blood_color", "exudation", "subpapillary_venous_plexus", "papilla"]
MAP = {"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
ROUTE = {
    "clarity": ("rank8", 8), "blood_color": ("rank8", 8),
    "exudation": ("rank4_lr1e4", 4), "subpapillary_venous_plexus": ("rank4_lr1e4", 4),
    "papilla": ("rank16", 16),
}

class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha=8):
        super().__init__(); self.base=base
        self.a=nn.Linear(base.in_features,rank,bias=False); self.b=nn.Linear(rank,base.out_features,bias=False)
        self.scale=alpha/rank
        for p in base.parameters(): p.requires_grad=False
    def forward(self,x): return self.base(x)+self.b(self.a(x))*self.scale

class Model(nn.Module):
    def __init__(self,b):
        super().__init__(); self.b=b; self.heads=nn.ModuleDict({f:nn.Linear(768,2) for f in FIELDS})
    def forward(self,x):
        z=self.b(x); return {f:h(z) for f,h in self.heads.items()}

class Frames(Dataset):
    def __init__(self, rows, transform): self.rows=rows.reset_index(drop=True); self.transform=transform
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows.iloc[i]
        x=self.transform(Image.open(ROOT/'data'/r.image_path).convert('RGB'))
        return x, r.exam_case_id, r.image_path

def build(rank, checkpoint):
    b=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=False,num_classes=0)
    b.load_state_dict(torch.load('/root/autodl-tmp/nailfold/Model/DINOv2-base/dinov2_vitb14_pretrain.pth',map_location='cpu',weights_only=True),strict=False)
    for p in b.parameters(): p.requires_grad=False
    for block in b.blocks[-4:]:
        block.attn.qkv=LoRALinear(block.attn.qkv,rank); block.attn.proj=LoRALinear(block.attn.proj,rank)
    m=Model(b)
    m.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'])
    return m.cuda().eval(), timm.data.create_transform(**timm.data.resolve_model_data_config(b),is_training=False)

def score(y,p):
    return {"ba":float(balanced_accuracy_score(y,p>=.5)),"auc":float(roc_auc_score(y,p))}

def main():
    labels=pd.read_csv(ART/'manifest/locked_evaluation_v1_reviewed.csv',dtype=str)
    labels=labels[labels.evaluation_role.eq('development')].copy(); labels.development_fold=labels.development_fold.astype(float).astype(int)
    frames=pd.read_csv(ART/'features/image_index.csv',dtype=str).merge(labels[['exam_case_id','development_fold']],on='exam_case_id')
    records=[]
    for fold in range(5):
        fold_rows=frames[frames.development_fold.eq(fold)]
        for config,rank in sorted(set(ROUTE.values())):
            model,tf=build(rank,ART/'experiments/retrain_reviewed'/config/f'fold{fold}.pt')
            # Match the original baseline evaluation batch size; bf16 kernels can
            # otherwise move borderline cases across the decision boundary.
            loader=DataLoader(Frames(fold_rows,tf),batch_size=16,num_workers=4,pin_memory=True)
            wanted=[f for f,v in ROUTE.items() if v==(config,rank)]
            with torch.inference_mode():
                for x,cases,paths in loader:
                    with torch.autocast('cuda',dtype=torch.bfloat16): out=model(x.cuda(non_blocking=True))
                    for f in wanted:
                        logits=out[f].float().cpu(); probs=torch.softmax(logits,1)[:,1]
                        for c,pth,lo,pr in zip(cases,paths,logits.numpy(),probs.numpy()):
                            records.append({'exam_case_id':c,'development_fold':fold,'image_path':pth,'field':f,
                                            'logit_0':float(lo[0]),'logit_1':float(lo[1]),'probability':float(pr)})
            del model; torch.cuda.empty_cache()
    pred=pd.DataFrame(records)
    outdir=ART/'experiments/audit_20260908'; outdir.mkdir(parents=True,exist_ok=True)
    pred.to_csv(outdir/'lora_v2_frame_predictions.csv',index=False)

    report={}
    for f in FIELDS:
        d=pred[pred.field.eq(f)].copy(); groups=d.groupby('exam_case_id')
        case=pd.DataFrame({
            'mean_logit_0':groups.logit_0.mean(),'mean_logit_1':groups.logit_1.mean(),
            'mean_prob':groups.probability.mean(),'median_prob':groups.probability.median(),
            'max_prob':groups.probability.max(),'top3_prob':groups.probability.apply(lambda x:x.nlargest(min(3,len(x))).mean()),
            'std_prob':groups.probability.std().fillna(0),
            'range_prob':groups.probability.max()-groups.probability.min(),
            'crosses_05':groups.probability.apply(lambda x:bool((x<.5).any() and (x>=.5).any())),
            'n_frames':groups.size(),
        })
        case['mean_logits_prob']=1/(1+np.exp(-(case.mean_logit_1-case.mean_logit_0)))
        ymap=labels.set_index('exam_case_id')[f].map(MAP[f]); case['truth']=ymap; case=case.dropna(subset=['truth']); case.truth=case.truth.astype(int)
        methods=['mean_logits_prob','mean_prob','median_prob','max_prob','top3_prob']
        metrics={m:score(case.truth.values,case[m].values) for m in methods}
        selected=[]; yp=[]; yt=[]
        for test_fold in range(5):
            val_fold=(test_fold+1)%5
            val_ids=labels.loc[labels.development_fold.eq(val_fold),'exam_case_id']; val=case.loc[case.index.intersection(val_ids)]
            bas={m:balanced_accuracy_score(val.truth,val[m]>=.5) for m in methods}
            chosen=max(methods,key=lambda m:(bas[m],m=='mean_logits_prob'))
            test_ids=labels.loc[labels.development_fold.eq(test_fold),'exam_case_id']; test=case.loc[case.index.intersection(test_ids)]
            yp.extend((test[chosen]>=.5).astype(int)); yt.extend(test.truth.astype(int))
            selected.append({'test_fold':test_fold,'validation_fold':val_fold,'method':chosen,'validation_ba':float(bas[chosen])})
        report[f]={
            'metrics_fixed_0_5':metrics,
            'cross_fitted_aggregation_ba':float(balanced_accuracy_score(yt,yp)),
            'aggregation_selections':selected,
            'frame_disagreement':{
                'median_std_probability':float(case.std_prob.median()),
                'median_probability_range':float(case.range_prob.median()),
                'cases_crossing_0_5':int(case.crosses_05.sum()),
                'fraction_cases_crossing_0_5':float(case.crosses_05.mean()),
                'median_frames_per_case':float(case.n_frames.median()),
            }
        }
    (outdir/'lora_v2_frame_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
