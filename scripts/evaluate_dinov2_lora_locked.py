import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch,timm
from torch.utils.data import DataLoader
from finetune_dinov2_lora_binary_cv import LoRALinear,Model,Frames,FIELDS,MAP,predict
def main():
 p=argparse.ArgumentParser(); p.add_argument('--index',type=Path,required=True); p.add_argument('--roles-labels',type=Path,required=True); p.add_argument('--image-root',type=Path,required=True); p.add_argument('--weights',type=Path,required=True); p.add_argument('--checkpoint-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 roles=pd.read_csv(a.roles_labels); roles.exam_case_id=roles.exam_case_id.astype(str); lock=roles[roles.evaluation_role.eq('locked_test')].copy(); idx=pd.read_csv(a.index); idx.exam_case_id=idx.exam_case_id.astype(str); frame=idx.merge(lock[['exam_case_id',*FIELDS]],on='exam_case_id',validate='many_to_one'); assert len(lock)==47
 votes={f:{} for f in FIELDS}
 for fold in range(5):
  b=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=False,num_classes=0); b.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True),strict=False)
  for q in b.parameters(): q.requires_grad=False
  for block in b.blocks[-4:]: block.attn.qkv=LoRALinear(block.attn.qkv); block.attn.proj=LoRALinear(block.attn.proj)
  m=Model(b).cuda(); tf=timm.data.create_transform(**timm.data.resolve_model_data_config(b),is_training=False); dl=DataLoader(Frames(frame,a.image_root,tf),batch_size=16,num_workers=4); ck=torch.load(a.checkpoint_dir/f'fold{fold}.pt',map_location='cpu',weights_only=True); m.load_state_dict(ck['state_dict']); pr=predict(m,dl)
  for f in FIELDS:
   for c,v in pr[f].items(): votes[f].setdefault(c,[]).append(v)
  del m,b; torch.cuda.empty_cache()
 out={'schema_version':'dinov2-lora-locked-evaluation/1.0','evaluation_role':'locked_test','locked_cases_seen':47,'fields':{}}
 rows=[]
 for f in FIELDS:
  truth=[]; pred=[]; ids=[]
  for c,v in votes[f].items():
   raw=lock.loc[lock.exam_case_id.eq(c),f].iloc[0]; y=MAP[f].get(raw,-1) if pd.notna(raw) else -1
   if y>=0: truth.append(y); pred.append(int(round(np.mean(v)))); ids.append(c)
  from sklearn.metrics import balanced_accuracy_score,recall_score
  out['fields'][f]={'n':len(truth),'balanced_accuracy':float(balanced_accuracy_score(truth,pred)) if truth else None,'class_recall':recall_score(truth,pred,labels=[0,1],average=None,zero_division=0).tolist() if truth else []}
  rows += [{'exam_case_id':c,'field':f,'truth':int(y),'prediction':int(q)} for c,y,q in zip(ids,truth,pred)]
 a.output_dir.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output_dir/'locked_predictions.csv',index=False); (a.output_dir/'metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__': main()
