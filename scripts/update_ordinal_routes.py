import json
from pathlib import Path
p=Path('artifacts/audits/field_routes.json');x=json.loads(p.read_text())
for r in x:
 if r['field'] in ('crossing_ratio','malformation_ratio'):
  r.update({'endpoint':'ordinal interval categories','status':'candidate','fallback':'manual_review for unknown or low confidence','evidence_level':'L1','training_protocol':'not yet trained; endpoint audit only'})
p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
