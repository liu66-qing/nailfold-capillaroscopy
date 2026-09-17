import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch,timm
from torch.utils.data import DataLoader
from finetune_dinov2_lora_binary_cv import LoRALinear,Model,Frames,FIELDS,MAP,predict
from sklearn.metrics import balanced_accuracy_score,recall_score,confusion_matrix
def main():
 p=argparse.ArgumentParser(); p.add_argument('--index',type=Path,required=True); p.add_argument('--roles-labels',type=Path,required=True); p.add_argument('--image-root',type=Path,required=True); p.add_argument('--weights',type=Path,required=True); p.add_argument('--checkpoint-dir',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
 roles=pd.read_csv(a.roles_labels); roles.exam_case_id=roles.exam_case_id.astype(str); dev=roles[roles.evaluation_role.eq('development')].copy(); locked=set(roles.loc[roles.evaluation_role.eq('locked_test'),'exam_case_id']); idx=pd.read_csv(a.index); idx.exam_case_id=idx.exam_case_id.astype(str); frame=idx.merge(dev[['exam_case_id','development_fold',*FIELDS]],on='exam_case_id',validate='many_to_one'); assert len(dev)==186 and len(locked)==47 and not set(frame.exam_case_id)&locked
 allpred={f:{} for f in FIELDS}; rows=[]; cfg=None; results={}
 for fold in range(5):
  b=timm.create_model('vit_base_patch14_dinov2.lvd142m',pretrained=False,num_classes=0); b.load_state_dict(torch.load(a.weights,map_location='cpu',weights_only=True),strict=False)
  for q in b.parameters(): q.requires_grad=False
  for block in b.blocks[-4:]: block.attn.qkv=LoRALinear(block.attn.qkv); block.attn.proj=LoRALinear(block.attn.proj)
  m=Model(b).cuda(); cfg=timm.data.resolve_model_data_config(b); tf=timm.data.create_transform(**cfg,is_training=False); testf=frame[frame.development_fold.astype(int).eq(fold)]; dl=DataLoader(Frames(testf,a.image_root,tf),batch_size=16,num_workers=4); ck=torch.load(a.checkpoint_dir/f'fold{fold}.pt',map_location='cpu',weights_only=True); m.load_state_dict(ck['state_dict']); pr=predict(m,dl)
  for f in FIELDS:
   allpred[f].update(pr[f])
  del m,b; torch.cuda.empty_cache()
 for f in FIELDS:
  truth=[]; pred=[]; folds=[]; ids=[]
  for _,r in dev.iterrows():
   y=MAP[f].get(r[f],-1) if pd.notna(r[f]) else -1
   if y>=0 and r.exam_case_id in allpred[f]: truth.append(y); pred.append(allpred[f][r.exam_case_id]); folds.append(int(r.development_fold)); ids.append(r.exam_case_id)
  truth=np.array(truth); pred=np.array(pred); rec=recall_score(truth,pred,labels=[0,1],average=None,zero_division=0); fm=[]
  for k in range(5):
   z=np.array(folds)==k; fm.append({'fold':k,'n':int(z.sum()),'balanced_accuracy':float(balanced_accuracy_score(truth[z],pred[z]))})
  for c,k,y,q in zip(ids,folds,truth,pred): rows.append({'exam_case_id':c,'development_fold':k,'field':f,'truth':int(y),'prediction':int(q)})
  print(f,balanced_accuracy_score(truth,pred),rec)
  results[f]={'n':len(truth),'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),'class_recall':rec.tolist(),'confusion_matrix':confusion_matrix(truth,pred,labels=[0,1]).tolist(),'folds':fm}
 a.output_dir.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.output_dir/'oof_predictions.csv',index=False); (a.output_dir/'metrics.json').write_text(json.dumps({'schema_version':'dinov2-lora-oof/1.0','locked_cases_seen':0,'fields':results},indent=2)+'\n')
if __name__=='__main__': main()
