import csv,json,hashlib,random
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MAP=ROOT/'artifacts/vascular_case_mapping_exact_20260831.csv'
DATA=ROOT/'vascular_train_ready'
OUT=ROOT/'artifacts/vascular_segmentation_qc_20260831.json'
rows=list(csv.DictReader(MAP.open(encoding='utf-8-sig')))
mapped=[r for r in rows if r['status']=='MAPPED_SOURCE_CASE_GROUP']
issues=[]; stats={'mapped_rows':len(mapped),'checked':0,'dimension_mismatch':0,'empty_masks':0,'annotation_malformed':0,'polygon_mask_iou_sample':[]}
for r in mapped:
 sid=r['sample_id']; ip=DATA/'train_ready/segmentation/images'/f'{sid}.jpg'; mp=DATA/'train_ready/segmentation/masks'/f'{sid}.png'; jp=DATA/'train_ready/segmentation/annotations'/f'{sid}.json'
 try:
  im=Image.open(ip); mask=Image.open(mp).convert('L'); stats['checked']+=1
  if im.size!=mask.size: stats['dimension_mismatch']+=1; issues.append((sid,'dimension_mismatch'))
  a=np.array(mask); fg=int((a>0).sum())
  if fg==0: stats['empty_masks']+=1; issues.append((sid,'empty_mask'))
  try:
   d=json.loads(jp.read_text(encoding='utf-8')); shapes=d.get('shapes',[])
   if not isinstance(shapes,list): raise ValueError('shapes')
   # sampled polygon/mask agreement
   if len(stats['polygon_mask_iou_sample'])<100 and shapes:
    pm=Image.new('1',im.size); dr=ImageDraw.Draw(pm)
    for s in shapes:
     pts=[tuple(map(float,p)) for p in s.get('points',[])]
     if len(pts)>=3: dr.polygon(pts,fill=1)
    p=np.array(pm,dtype=bool); m=a>0; u=(p|m).sum(); inter=(p&m).sum()
    stats['polygon_mask_iou_sample'].append({'sample_id':sid,'iou':float(inter/u) if u else 0.0,'polygons':len(shapes)})
  except Exception as e: stats['annotation_malformed']+=1; issues.append((sid,'annotation_malformed'))
 except Exception as e: issues.append((sid,'missing_or_unreadable'))
stats['issue_count']=len(issues); stats['status']='PASS_AUXILIARY' if not issues else 'HOLD'
OUT.write_text(json.dumps({'schema_version':'vascular-seg-qc/1.0','stats':stats,'issues':issues[:500]},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False))
