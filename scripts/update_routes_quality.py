import json
from pathlib import Path
p=Path('artifacts/audits/field_routes.json'); x=json.loads(p.read_text(encoding='utf-8'))
h=json.loads(Path('artifacts/annotations/image_quality_auxiliary_manifest.json').read_text(encoding='utf-8'))['sha256']
for r in x:
 if r['field'] in ('clarity','loop_length'):
  r['feature_source']='DINO/instance + auxiliary image-quality triage'
  r['auxiliary_quality_hash']=h
  r['fallback']='manual_review on low quality or low confidence'
p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
print('updated',h)
