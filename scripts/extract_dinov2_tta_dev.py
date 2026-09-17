"""Extract label-free DINOv2 TTA views for development frames only."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch,timm
from PIL import Image
from torch.utils.data import DataLoader,Dataset
class Images(Dataset):
 def __init__(self,frame,root,transform,variant): self.frame=frame; self.root=root; self.transform=transform; self.variant=variant
 def __len__(self): return len(self.frame)
 def __getitem__(self,i):
  im=Image.open(self.root/self.frame.iloc[i].image_path).convert("RGB")
  if self.variant=="hflip": im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
  if self.variant=="center90":
   w,h=im.size; dx,dy=int(w*.05),int(h*.05); im=im.crop((dx,dy,w-dx,h-dy))
  return self.transform(im)
def main():
 p=argparse.ArgumentParser(); p.add_argument("--index",type=Path,required=True); p.add_argument("--roles",type=Path,required=True); p.add_argument("--image-root",type=Path,required=True); p.add_argument("--weights",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--batch-size",type=int,default=32); a=p.parse_args()
 idx=pd.read_csv(a.index); idx.exam_case_id=idx.exam_case_id.astype(str); roles=pd.read_csv(a.roles,usecols=["exam_case_id","evaluation_role"]); roles.exam_case_id=roles.exam_case_id.astype(str); dev=set(roles.loc[roles.evaluation_role.eq("development"),"exam_case_id"]); locked=set(roles.loc[roles.evaluation_role.eq("locked_test"),"exam_case_id"]); idx=idx[idx.exam_case_id.isin(dev)].reset_index(drop=True)
 if len(dev)!=186 or len(locked)!=47 or set(idx.exam_case_id)&locked or idx.exam_case_id.nunique()!=186: raise ValueError("role boundary failed")
 m=timm.create_model("vit_base_patch14_dinov2.lvd142m",pretrained=False,num_classes=0); m.load_state_dict(torch.load(a.weights,map_location="cpu",weights_only=True),strict=False); m=m.cuda().eval(); cfg=timm.data.resolve_model_data_config(m); tf=timm.data.create_transform(**cfg,is_training=False); out={}
 for v in ["original","hflip","center90"]:
  vals=[]; loader=DataLoader(Images(idx,a.image_root,tf,v),batch_size=a.batch_size,num_workers=4,pin_memory=True)
  with torch.inference_mode():
   for x in loader:
    with torch.autocast("cuda",dtype=torch.bfloat16): vals.append(m(x.cuda(non_blocking=True)).float().cpu().numpy())
  out[v]=np.concatenate(vals).astype(np.float16); print(v,out[v].shape,flush=True)
 a.output_dir.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output_dir/"features.npz",**out); idx.to_csv(a.output_dir/"index.csv",index=False); (a.output_dir/"metadata.json").write_text(json.dumps({"evaluation_role":"development","locked_cases_seen":0,"gpu_used":True,"cases":186,"frames":len(idx),"variants":list(out),"shape":list(out["original"].shape)},indent=2)+"\n")
if __name__=="__main__": main()
