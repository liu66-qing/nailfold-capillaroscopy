import json,hashlib
from pathlib import Path
x={'stage':'2a_anfc_cls_recovery_source_audit','source':'/root/autodl-tmp/nailfold/artifacts/pseudo_labels/round1_multiclass_domain_adapted.json','cases':2207,'instance_keys':['bbox','polygon','class','class_name','confidence'],'has_class':True,'case_id_format':'archive/case/image.jpg','mapping_to_instance_npz':'not established','status':'auxiliary_candidate_only','evidence_level':'L4','limitation':'pseudo labels not verified clinical gold and no proof they align row-by-row with 1687 NPZ'}
raw=json.dumps(x,ensure_ascii=False,indent=2);x['sha256']=hashlib.sha256(raw.encode()).hexdigest();Path('artifacts/audits/anfc_cls_recovery_source.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');print(x)
