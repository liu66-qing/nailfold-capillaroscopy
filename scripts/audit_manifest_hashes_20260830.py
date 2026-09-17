from pathlib import Path
import hashlib, json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
manifest=ROOT/'artifacts/manifest/files.csv'; splits=ROOT/'artifacts/manifest/splits.csv'
out=ROOT/'artifacts/audits/model_upgrade_20260830/hash_audit_20260830'; out.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(manifest,dtype=str).fillna(''); sp=pd.read_csv(splits,dtype=str).fillna('')
role=sp.set_index('exam_case_id')['split'].to_dict()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
rows=[]
for _,r in df.iterrows():
 p=ROOT/str(r['path']);
 if not p.exists(): p=ROOT/'.transfer/images512'/str(r['path'])
 old=str(r.get('sha256','')); new=old; exists=p.exists()
 if exists and not old: new=sha(p)
 rows.append({**r.to_dict(),'sha256_audited':new,'hash_was_missing':not bool(old),'file_exists':exists,'split':role.get(str(r['exam_case_id']),'')})
aud=pd.DataFrame(rows); aud.to_csv(out/'files_hash_audited.csv',index=False)
non=aud[aud.sha256_audited.ne('') & aud.file_exists.eq(True)]
dups=[]; cross=[]
for h,g in non.groupby('sha256_audited'):
 if len(g)>1:
  rec={'sha256':h,'rows':len(g),'cases':';'.join(sorted(g.exam_case_id.unique())),'splits':';'.join(sorted(g.split.unique())),'paths':';'.join(g.path.tolist())}; dups.append(rec)
  assigned=g[g.split.ne('')]
  if assigned.split.nunique()>1: cross.append(rec)
pd.DataFrame(dups).to_csv(out/'full_hash_duplicate_groups.csv',index=False)
pd.DataFrame(cross).to_csv(out/'cross_fold_hash_leaks.csv',index=False)
summary={'manifest_rows':len(aud),'missing_hash_before':int(aud.hash_was_missing.sum()),'missing_hash_after':int((aud.sha256_audited=='').sum()),'missing_files':int((~aud.file_exists).sum()),'duplicate_hash_groups':len(dups),'cross_fold_hash_groups':len(cross),'gpu_used':False}
(out/'hash_audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(summary,ensure_ascii=False))
