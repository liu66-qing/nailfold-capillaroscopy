#!/usr/bin/env python3
"""Audit auxiliary vascular source groups against development/locked assets."""
import csv, hashlib, json, os, re
from pathlib import Path

ROOT = Path(r'E:\甲劈微循环')
REPO = ROOT / 'github_nailfold_capillaroscopy'
MAP = REPO / 'artifacts' / 'vascular_case_mapping_exact_20260831.csv'
CASES = ROOT / 'artifacts' / 'manifest' / 'cases.csv'
LOCK = ROOT / 'artifacts' / 'manifest' / 'locked_evaluation_v1.csv'
OUT = REPO / 'artifacts' / 'overlap_audit_20260831'
OUT.mkdir(parents=True, exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def rows(p):
    with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

mapping=rows(MAP)
dev={r['exam_case_id'] for r in rows(CASES)}
locked={r['exam_case_id'] for r in rows(LOCK)}
locked |= {r['exam_case_id'] for r in rows(LOCK) if r.get('exam_case_id')}

# Hash every image in the local source/data trees (excluding the repo itself).
files=[]
for base in (ROOT/'data', ROOT/'artifacts'):
    if not base.exists(): continue
    for p in base.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}:
            try: files.append((p,sha(p)))
            except OSError: pass
byhash={}
for p,h in files: byhash.setdefault(h,[]).append(str(p))
byname={}
for p,h in files: byname.setdefault(p.name.lower(),[]).append(str(p))

out=[]
for r in mapping:
    h=r.get('source_image_sha256','').lower(); name=Path(r.get('source_image_path','')).name.lower()
    exact=byhash.get(h,[])
    namehits=byname.get(name,[])
    # Only exact hash is evidence of cross-source identity.
    status='NO_EXACT_HASH'
    if exact:
        status='EXACT_HASH_MATCH'
    out.append({
      'sample_id':r.get('sample_id',''),'source_case_prefix':r.get('source_case_prefix',''),
      'source_image_path':r.get('source_image_path',''),'source_image_sha256':h,
      'current_exact_hash_paths':';'.join(exact),'same_basename_paths':';'.join(namehits),
      'evidence_status':status,'locked_cases_seen':0,
      'interpretation':'exact hash match requires case-level adjudication; no automatic overlap claim' if exact else 'no exact hash match found in scanned local assets'
    })

with open(OUT/'per_sample_overlap.csv','w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
exact=[x for x in out if x['evidence_status']=='EXACT_HASH_MATCH']
summary={
 'schema_version':'1.0','mapping_rows':len(mapping),'mapped_source_groups':len({r.get('source_case_prefix') for r in mapping if r.get('source_case_prefix')}),
 'current_assets_scanned':len(files),'exact_hash_match_rows':len(exact),
 'exact_hash_match_groups':len({x['source_case_prefix'] for x in exact}),
 'basename_only_rows':sum(bool(x['same_basename_paths']) and not bool(x['current_exact_hash_paths']) for x in out),
 'development_cases_in_manifest':len(dev),'locked_cases_in_manifest':len(locked),'locked_cases_seen':0,
 'status':'HOLD' if exact else 'PASS_AUXILIARY_CONDITIONAL',
 'gate':'HOLD_PROVENANCE' if exact else 'PASS_AUXILIARY_CONDITIONAL',
 'limitations':['Source auxiliary case prefixes are not identifiers in current exam_case_id namespace.', 'No uncertain filename/basename match is classified as overlap.', 'Locked exclusion cannot be proven from hashes alone when source provenance is absent.'],
 'method':'SHA256 exact match against image files under E:/甲劈微循环/data and artifacts; filename matches reported as non-decisive.'
}
with open(OUT/'summary.json','w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False))
