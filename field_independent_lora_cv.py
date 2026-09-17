import argparse,copy,json,math
from pathlib import Path
import numpy as np,pandas as pd,torch,timm
from PIL import Image
from sklearn.metrics import balanced_accuracy_score,accuracy_score,f1_score,confusion_matrix,recall_score
from torch import nn
from torch.utils.data import Dataset,DataLoader

MAP={"clarity":{"清晰":0,"不清":1,"模糊":1},"blood_color":{"暗红":0,"暗紫":0,"浅红":1,"淡红":1},"exudation":{"无":0,"+":1,"++":1,"+++":1},"subpapillary_venous_plexus":{"不见":0,"可见1排":1,"可见2排":1,">2排,扩张":1},"papilla":{"平坦":0,"浅波纹状":1,"波纹状":1}}
class LoRALinear(nn.Module):
 def __init__(self,base,rank):
  super().__init__(); self.base=base; self.a=nn.Linear(base.in_features,rank,bias=False); self.b=nn.Linear(rank,base.out_features,bias=False); self.scale=8/rank; nn.init.kaiming_uniform_(self.a.weight,a=math.sqrt(5)); nn.init.zeros_(self.b.weight)
  for p in base.parameters(): p.requires_grad=False
 def forward(self,x): return self.base(x)+self.b(self.a(x))*self.scale
class Frames(Dataset):
 def __init__(self,f,root,tf,field): self.f=f.reset_index(drop=True); self.root=root; self.tf=tf; self.field=field
 def __len__(self): return len(self.f)
 def __getitem__(self,i):
  r=self.f.iloc[i]; return self.tf(Image.open(self.root/r.image_path).convert("RGB")), MAP[self.field].get(r[self.field],-1) if pd.notna(r[self.field]) else -1, str(r.exam_case_id)
class Model(nn.Module):
 def __init__(self,b,field): super().__init__(); self.b=b; self.head=nn.Linear(768,2); self.field=field
 def forward(self,x): return self.head(self.b(x))
def predict(m,loader):
 z={}
 m.eval()
 with torch.inference_mode():
  for x,_,ids in loader:
   with torch.autocast("cuda",dtype=torch.bfloat16): o=m(x.cuda(non_blocking=True))
   for c,v in zip(ids,o.float().cpu().numpy()): z.setdefault(c,[]).append(v)
 return {c:float(torch.softmax(torch.tensor(np.mean(v,0)),0)[1]) for c,v in z.items()}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--field",required=True,choices=list(MAP)); p.add_argument("--rank",type=int,required=True); p.add_argument("--lr",type=float,required=True); p.add_argument("--index",type=Path,required=True); p.add_argument("--roles-labels",type=Path,required=True); p.add_argument("--image-root",type=Path,required=True); p.add_argument("--weights",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--seeds",type=int,nargs="+",default=[17,29,43,71,101]); a=p.parse_args()
 roles=pd.read_csv(a.roles_labels); roles.exam_case_id=roles.exam_case_id.astype(str); dev=roles[roles.evaluation_role.eq("development")].copy(); locked=set(roles.loc[roles.evaluation_role.eq("locked_test"),"exam_case_id"]); idx=pd.read_csv(a.index); idx.exam_case_id=idx.exam_case_id.astype(str); frame=idx.merge(dev[["exam_case_id","development_fold",a.field]],on="exam_case_id",validate="many_to_one"); assert len(dev)==186 and len(locked)==47 and not set(frame.exam_case_id)&locked
 cfg=None; allpred={}; selections=[]
 for seed in a.seeds:
  torch.manual_seed(seed); np.random.seed(seed); allpred[seed]={}
  for test in range(5):
   val=(test+1)%5; b=timm.create_model("vit_base_patch14_dinov2.lvd142m",pretrained=False,num_classes=0); b.load_state_dict(torch.load(a.weights,map_location="cpu",weights_only=True),strict=False)
   for q in b.parameters(): q.requires_grad=False
   for block in b.blocks[-4:]: block.attn.qkv=LoRALinear(block.attn.qkv,a.rank); block.attn.proj=LoRALinear(block.attn.proj,a.rank)
   m=Model(b,a.field).cuda(); cfg=cfg or timm.data.resolve_model_data_config(b); trtf=timm.data.create_transform(**cfg,is_training=True,color_jitter=0,hflip=.5,vflip=0); evtf=timm.data.create_transform(**cfg,is_training=False)
   trf=frame[~frame.development_fold.astype(int).isin([test,val])]; vf=frame[frame.development_fold.astype(int).eq(val)]; tf=frame[frame.development_fold.astype(int).eq(test)]
   tl=DataLoader(Frames(trf,a.image_root,trtf,a.field),batch_size=a.batch_size,shuffle=True,num_workers=4,pin_memory=True); vl=DataLoader(Frames(vf,a.image_root,evtf,a.field),batch_size=a.batch_size,num_workers=4); testl=DataLoader(Frames(tf,a.image_root,evtf,a.field),batch_size=a.batch_size,num_workers=4); opt=torch.optim.AdamW([q for q in m.parameters() if q.requires_grad],lr=a.lr,weight_decay=.05); best=(-1,None,0)
   for epoch in range(1,a.epochs+1):
    m.train()
    for x,y,_ in tl:
     ok=y>=0
     if not ok.any(): continue
     x=x.cuda(non_blocking=True); y=y.cuda(); opt.zero_grad(set_to_none=True)
     with torch.autocast("cuda",dtype=torch.bfloat16): loss=nn.functional.cross_entropy(m(x)[ok],y[ok])
     loss.backward(); nn.utils.clip_grad_norm_([q for q in m.parameters() if q.requires_grad],1); opt.step()
    vp=predict(m,vl); vy=vf[a.field].map(MAP[a.field]).to_numpy(); ids=vf.exam_case_id.to_numpy(); ok=~pd.isna(vy); y=vy[ok].astype(int); pr=np.array([vp[c]>=.5 for c in ids[ok]],dtype=int); score=balanced_accuracy_score(y,pr)
    if score>best[0]: best=(score,copy.deepcopy(m.state_dict()),epoch)
   m.load_state_dict(best[1]); allpred[seed].update(predict(m,testl)); selections.append({"seed":seed,"test_fold":test,"validation_fold":val,"best_epoch":best[2],"validation_ba":best[0]}); a.output_dir.mkdir(parents=True,exist_ok=True); del m,b
 results=[]
 for seed,z in allpred.items():
  y=dev[a.field].map(MAP[a.field]).to_numpy(); ids=dev.exam_case_id.to_numpy(); ok=~pd.isna(y); pr=np.array([z[c]>=.5 for c in ids[ok]],dtype=int); results.append({"seed":seed,"ba":float(balanced_accuracy_score(y[ok].astype(int),pr)),"n":int(ok.sum())})
 out={"schema_version":"field-independent-lora-cv/1.0","field":a.field,"rank":a.rank,"learning_rate":a.lr,"evaluation_role":"development_oof","locked_cases_seen":0,"gpu_used":True,"selections":selections,"seed_results":results}; a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"metrics.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n"); (a.output_dir/"oof_soft_predictions.json").write_text(json.dumps({str(s):z for s,z in allpred.items()},ensure_ascii=False)+"\n")
if __name__=="__main__": main()
