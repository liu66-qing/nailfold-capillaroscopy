import csv,hashlib,json
from pathlib import Path
from PIL import Image,ImageFilter
import numpy as np
root=Path('data/血管数据集/分类数据集/扩充之前/images'); out=Path('artifacts/annotations'); out.mkdir(parents=True,exist_ok=True)
rows=[]
for p in sorted(root.glob('*')):
 try:
  b=p.read_bytes(); im=Image.open(p).convert('L'); a=np.asarray(im,dtype=np.float32)
  edge=float(np.asarray(im.filter(ImageFilter.FIND_EDGES),dtype=np.float32).var()); c=float(a.std())
  q='poor' if min(im.size)<300 or c<12 else ('uncertain' if edge<180 else 'usable')
  rows.append({'image_path':str(p),'image_sha256':hashlib.sha256(b).hexdigest(),'width':im.width,'height':im.height,'contrast_std':round(c,3),'edge_variance':round(edge,3),'quality_label':q,'measurement_visibility':'unknown','loop_boundary':'unknown','annotation_type':'auxiliary_ai_annotation','rule_version':'quality_triage_v1','unknown_reason':'heuristic quality only; no clinical review'})
 except Exception as e: rows.append({'image_path':str(p),'annotation_type':'auxiliary_ai_annotation','quality_label':'unknown','unknown_reason':str(e)})
f=out/'image_quality_auxiliary.csv'
with f.open('w',newline='',encoding='utf-8') as h:
 w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
meta={'n_images':len(rows),'source':str(root),'clinical_gold_standard':False,'rule_version':'quality_triage_v1','sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
(out/'image_quality_auxiliary_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');print(meta)
