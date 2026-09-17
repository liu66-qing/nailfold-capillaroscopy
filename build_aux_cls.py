import json,glob,os,csv,hashlib
from pathlib import Path
j=json.load(open('/root/autodl-tmp/nailfold/artifacts/pseudo_labels/round1_multiclass_domain_adapted.json'))
out='/root/autodl-tmp/nailfold/artifacts/experiments/model-upgrade-20260831/instance_cls_auxiliary.csv'
rows=[]
for k,v in j.items():
 stem=k.replace('/','__').replace('.jpg','')
 fs=[i for i in v.get('instances',[])]
 c={}
 for i in fs:c[i.get('class_name','unknown')]=c.get(i.get('class_name','unknown'),0)+1
 rows.append({'case_key':k,'npz_basename':stem+'.npz','instance_count':len(fs),'normal_count':c.get('normal',0),'hemo_count':c.get('hemo',0),'aggregation_count':c.get('aggregation',0),'blur_count':c.get('blur',0),'abnormal_count':c.get('abnormal',0),'annotation_type':'auxiliary_pseudo_label','source':'round1_multiclass_domain_adapted.json'})
with open(out,'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps({'rows':len(rows),'sha256':hashlib.sha256(open(out,'rb').read()).hexdigest(),'path':out}))
