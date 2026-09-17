"""Strict nested-development OOF pilot for supervised DINOv2 LoRA binary heads."""
import argparse,copy,json,math
from pathlib import Path
import numpy as np,pandas as pd,torch,timm
from PIL import Image
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,confusion_matrix,recall_score
from torch import nn
from torch.utils.data import Dataset,DataLoader
FIELDS=["clarity","blood_color","exudation","subpapillary_venous_plexus","papilla"]
MAP={"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
class LoRALinear(nn.Module):
 def __init__(self,base,rank=4,alpha=8):
  super().__init__(); self.base=base; self.a=nn.Linear(base.in_features,rank,bias=False); self.b=nn.Linear(rank,base.out_features,bias=False); self.scale=alpha/rank; nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5)); nn.init.zeros_(self.b.weight)
  for p in base.parameters(): p.requires_grad=False
 def forward(self,x): return self.base(x)+self.b(self.a(x))*self.scale
class Frames(Dataset):
 def __init__(self,f,root,tf): self.f=f.reset_index(drop=True); self.root=root; self.tf=tf
 def __len__(self): return len(self.f)
 def __getitem__(self,i):
  r=self.f.iloc[i]; im=self.tf(Image.open(self.root/r.image_path).convert("RGB")); y=torch.tensor([MAP[x].get(r[x],-1) if pd.notna(r[x]) else -1 for x in FIELDS]); return im,y,r.exam_case_id
class Model(nn.Module):
 def __init__(self,b): super().__init__(); self.b=b; self.heads=nn.ModuleDict({f:nn.Linear(768,2) for f in FIELDS})
 def forward(self,x): z=self.b(x); return {f:h(z) for f,h in self.heads.items()}
def predict(m,loader):
 vals={f:{} for f in FIELDS}; m.eval()
 with torch.inference_mode():
  for x,_,ids in loader:
   with torch.autocast("cuda",dtype=torch.bfloat16): o=m(x.cuda(non_blocking=True))
   for f in FIELDS:
    for c,v in zip(ids,o[f].float().cpu().numpy()): vals[f].setdefault(c,[]).append(v)
 return {f:{c:int(np.mean(v,0).argmax()) for c,v in z.items()} for f,z in vals.items()}
def score(pred,cases,fold):
 z=cases[cases.development_fold.astype(int).eq(fold)]; out={}
 for f in FIELDS:
  y=np.array([MAP[f].get(v,-1) if pd.notna(v) else -1 for v in z[f]]); ok=y>=0; ids=z.exam_case_id.to_numpy()[ok]; p=np.array([pred[f][c] for c in ids]); out[f]=float(balanced_accuracy_score(y[ok],p))
 return out
def main():
 p=argparse.ArgumentParser(); p.add_argument("--index",type=Path,required=True); p.add_argument("--roles-labels",type=Path,required=True); p.add_argument("--image-root",type=Path,required=True); p.add_argument("--weights",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--epochs",type=int,default=5); p.add_argument("--batch-size",type=int,default=32); a=p.parse_args(); torch.manual_seed(17); np.random.seed(17)
 roles=pd.read_csv(a.roles_labels); roles.exam_case_id=roles.exam_case_id.astype(str); dev=roles[roles.evaluation_role.eq("development")].copy(); locked=set(roles.loc[roles.evaluation_role.eq("locked_test"),"exam_case_id"]); idx=pd.read_csv(a.index); idx.exam_case_id=idx.exam_case_id.astype(str); frame=idx.merge(dev[["exam_case_id","development_fold",*FIELDS]],on="exam_case_id",validate="many_to_one"); assert len(dev)==186 and len(locked)==47 and not set(frame.exam_case_id)&locked
 allpred={f:{} for f in FIELDS}; selections=[]; peak=0
 for test in range(5):
  val=(test+1)%5; b=timm.create_model("vit_base_patch14_dinov2.lvd142m",pretrained=False,num_classes=0); b.load_state_dict(torch.load(a.weights,map_location="cpu",weights_only=True),strict=False)
  for q in b.parameters(): q.requires_grad=False
  for block in b.blocks[-4:]: block.attn.qkv=LoRALinear(block.attn.qkv); block.attn.proj=LoRALinear(block.attn.proj)
  m=Model(b).cuda(); cfg=timm.data.resolve_model_data_config(b); trtf=timm.data.create_transform(**cfg,is_training=True,color_jitter=0,hflip=.5,vflip=0); evtf=timm.data.create_transform(**cfg,is_training=False); trf=frame[~frame.development_fold.astype(int).isin([test,val])]; vf=frame[frame.development_fold.astype(int).eq(val)]; tf=frame[frame.development_fold.astype(int).eq(test)]; tl=DataLoader(Frames(trf,a.image_root,trtf),batch_size=a.batch_size,shuffle=True,num_workers=4,pin_memory=True); vl=DataLoader(Frames(vf,a.image_root,evtf),batch_size=a.batch_size,num_workers=4); testl=DataLoader(Frames(tf,a.image_root,evtf),batch_size=a.batch_size,num_workers=4); opt=torch.optim.AdamW([q for q in m.parameters() if q.requires_grad],lr=2e-4,weight_decay=.05); best=(-1,None,0,{})
  for epoch in range(1,a.epochs+1):
   m.train()
   for x,y,_ in tl:
    x=x.cuda(non_blocking=True); y=y.cuda(); opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda",dtype=torch.bfloat16):
     o=m(x); losses=[nn.functional.cross_entropy(o[f][y[:,i]>=0],y[y[:,i]>=0,i]) for i,f in enumerate(FIELDS) if (y[:,i]>=0).any()]; loss=torch.stack(losses).mean()
    loss.backward(); nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad],1); opt.step(); peak=max(peak,torch.cuda.max_memory_allocated())
   vs=score(predict(m,vl),dev,val); obj=np.mean([vs[f] for f in FIELDS[:-1]]); print(test,epoch,obj,flush=True)
   if obj>best[0]: best=(obj,copy.deepcopy(m.state_dict()),epoch,vs)
  m.load_state_dict(best[1]); pr=predict(m,testl)
  for f in FIELDS: allpred[f].update(pr[f])
  selections.append({"fold":test,"epoch":best[2],"validation_delivery_ba":best[0],"validation_fields":best[3]}); a.output_dir.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":best[1],"rank":4},a.output_dir/f"fold{test}.pt"); del m,b,opt; torch.cuda.empty_cache()
 results={}
 for f in FIELDS:
  y=np.array([MAP[f].get(v,-1) if pd.notna(v) else -1 for v in dev[f]]); ok=y>=0; ids=dev.exam_case_id.to_numpy()[ok]; pr=np.array([allpred[f][c] for c in ids]); rec=recall_score(y[ok],pr,labels=[0,1],average=None,zero_division=0); results[f]={"n":int(ok.sum()),"accuracy":float(accuracy_score(y[ok],pr)),"balanced_accuracy":float(balanced_accuracy_score(y[ok],pr)),"macro_f1":float(f1_score(y[ok],pr,average="macro")),"class_recall":rec.tolist(),"confusion_matrix":confusion_matrix(y[ok],pr,labels=[0,1]).tolist(),"exploratory":f=="papilla"}
 out={"schema_version":"dinov2-lora-binary-development-oof/1.0","evaluation_role":"development_oof","locked_cases_seen":0,"gpu_used":True,"rank":4,"supervision":"case labels replicated to frames","selections":selections,"peak_vram_gib":peak/2**30,"fields":results,"delivery_mean_ba":float(np.mean([results[f]["balanced_accuracy"] for f in FIELDS[:-1]]))}; (a.output_dir/"metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n"); print(json.dumps({"delivery_mean_ba":out["delivery_mean_ba"]},ensure_ascii=False))
if __name__=="__main__": main()
