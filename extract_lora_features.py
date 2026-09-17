import argparse, numpy as np, pandas as pd, torch, timm
from pathlib import Path
from PIL import Image
from torch import nn
from torch.utils.data import Dataset,DataLoader
class L(nn.Module):
 def __init__(self,b,r):
  super().__init__(); self.base=b; self.a=nn.Linear(b.in_features,r,bias=False); self.b=nn.Linear(r,b.out_features,bias=False); self.scale=8/r
  for p in b.parameters(): p.requires_grad=False
 def forward(self,x): return self.base(x)+self.b(self.a(x))*self.scale
class D(Dataset):
 def __init__(self,i,root,t): self.i=i.reset_index(drop=True); self.root=root; self.t=t
 def __len__(self): return len(self.i)
 def __getitem__(self,n):
  r=self.i.iloc[n]; return self.t(Image.open(self.root/r.image_path).convert('RGB')),n
def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint-dir',type=Path,required=True); p.add_argument('--rank',type=int,required=True); p.add_argument('--index',type=Path,required=True); p.add_argument('--image-root',type=Path,required=True); p.add_argument('--weights',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--batch-size',type=int,default=32); a=p.parse_args()
 i=pd.read_csv(a.index); b=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=False,num_classes=0); b.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True),strict=False)
 for q in b.blocks[-4:]: q.attn.qkv=L(q.attn.qkv,a.rank); q.attn.proj=L(q.attn.proj,a.rank)
 class M(nn.Module):
  def __init__(s,b): super().__init__(); s.b=b; s.heads=nn.ModuleDict({f:nn.Linear(768,2) for f in ['clarity','blood_color','exudation','subpapillary_venous_plexus','papilla']})
 cfg=timm.data.resolve_model_data_config(b); dl=DataLoader(D(i,a.image_root,timm.data.create_transform(**cfg,is_training=False)),batch_size=a.batch_size,num_workers=4); a.output.parent.mkdir(parents=True,exist_ok=True)
 for k in range(5):
  m=M(b); m.load_state_dict(torch.load(a.checkpoint_dir/f'fold{k}.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True); m.cuda().eval(); z=np.zeros((len(i),768),np.float16)
  with torch.inference_mode():
   for x,n in dl:
    with torch.autocast('cuda',dtype=torch.bfloat16): y=m.b(x.cuda())
    z[np.asarray(n)]=y.float().cpu().numpy()
  np.save(a.output.parent/f'lora_features_fold{k}.npy',z); del m; torch.cuda.empty_cache(); print(k,flush=True)
if __name__=='__main__': main()
