import json,hashlib,datetime
from pathlib import Path
x={'stage':'2a_anfc_cls_key_audit','source':'/root/autodl-tmp/nailfold/artifacts/experiments/model-upgrade-20260831/instance_npz_dev','npz_count':1687,'sample_keys':['width','masks','boxes','quality','stability','score'],'cls_present_in_sample':False,'diagnosis':'confirmed: exported instance npz lacks cls key; downstream hemo/aggregation class information cannot be recovered from these artifacts','locked_used':False,'limitation':'sampled first 20 files; full key equality not yet proven'}
raw=json.dumps(x,ensure_ascii=False,indent=2);x['sha256']=hashlib.sha256(raw.encode()).hexdigest();Path('artifacts/audits/anfc_cls_key_audit.json').write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');print(x)
