import csv,json,hashlib,random
from pathlib import Path
labels=Path('artifacts/audits/canonical_labels.csv')
rows=__import__('pandas').read_csv(labels).fillna('').to_dict('records')
# locked IDs are sourced only from existing evaluation manifest if present
locked=set()
for p in Path('artifacts').rglob('*.csv'):
 if 'locked' in p.name.lower() or 'evaluation_case' in str(p).lower():
  try:
   for r in csv.DictReader(p.open(encoding='utf-8-sig')):
    if str(r.get('evaluation_role','')).lower()=='locked_test': locked.add(str(r.get('exam_case_id')))
  except Exception: pass
# no image-path assumption: emit case-level allowlist for later join
candidates=[r['exam_case_id'] for r in rows if r['exam_case_id'] not in locked]
random.Random(20260916).shuffle(candidates)
selected=candidates[:300]
out=Path('artifacts/audits/gpt_pilot_case_manifest.csv')
with out.open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=['exam_case_id','evaluation_role','external_send_allowed','selection_seed']);w.writeheader()
 for cid in selected:w.writerow({'exam_case_id':cid,'evaluation_role':'development','external_send_allowed':'true','selection_seed':'20260916'})
meta={'selected_cases':len(selected),'locked_excluded':len(set(selected)&locked),'locked_ids_found':len(locked),'seed':20260916,'external_api_called':False,'requirement':'case-level filter before any image transmission','sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
Path('artifacts/audits/gpt_pilot_case_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(meta)

