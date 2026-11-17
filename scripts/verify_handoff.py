"""Check the handoff using stdlib; run at repository root. No model fitting."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def check(check_models=False):
    errors=[]; passed=[]
    def require(ok, message):
        if not ok:errors.append(message)
    def load(rel):return json.loads((ROOT/rel).read_text(encoding='utf-8'))
    manifest=load('artifacts/models/expert_router_v1/manifest.json')
    metrics=load('evaluation/metrics.json')
    subset=load('evaluation/subset_47/metrics.json')
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    for rel in tracked:
        p=ROOT/rel
        if p.suffix!='.md' or not p.exists():continue
        for dest in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            dest=dest.strip('<>').split('#')[0]
            if not dest or '://' in dest or dest.startswith('mailto:'):continue
            require((p.parent/dest).exists(),f'broken local link: {rel} -> {dest}')
    passed.append('tracked Markdown file links checked (anchors/external URLs excluded)')
    for rel,want in manifest['release_files'].items():
        require((ROOT/rel).is_file(),f'missing resource: {rel}')
        if (ROOT/rel).is_file():require(sha(ROOT/rel)==want,f'resource hash mismatch: {rel}')
    for rel,want in metrics['source_sha256'].items():
        require((ROOT/rel).is_file(),f'missing evaluation source: {rel}')
        if (ROOT/rel).is_file():require(sha(ROOT/rel)==want,f'evaluation source hash mismatch: {rel}')
    passed.append('resource and evaluation-source hashes checked')
    for doc in (metrics,subset):
        require(len(doc['fields'])==20,'expected 20 output fields')
        for f in doc['fields']:
            if 'accuracy' not in f:continue
            cm=f['confusion_matrix']; n=sum(map(sum,cm)); diag=sum(cm[i][i] for i in range(len(cm)))
            require(n==f['n'],f"{f['field']}: confusion matrix n mismatch")
            require(abs(diag/n-f['accuracy'])<1e-10,f"{f['field']}: accuracy mismatch")
            recalls=[cm[i][i]/sum(row) if sum(row) else 0 for i,row in enumerate(cm)]
            require(all(abs(a-b)<1e-10 for a,b in zip(recalls,f['recall'])),f"{f['field']}: recall mismatch")
            present=[r for r,row in zip(recalls,cm) if sum(row)]
            require(abs(sum(present)/len(present)-f['balanced_accuracy'])<1e-10,f"{f['field']}: BA mismatch")
    require(metrics['release_id']==subset['release_id']==manifest['release_id'],'release ID mismatch')
    require(subset['subset_exams']==47,'subset size must be 47')
    by_field={f['field']:f for f in subset['fields']}
    for f in metrics['fields']:
        if 'subset47_oof_accuracy' in f:
            require(abs(f['subset47_oof_accuracy']-by_field[f['field']]['accuracy'])<=0.000051,f"{f['field']}: subset accuracy mismatch")
    # Recompute delivered OOF decision accuracy/BA directly from saved predictions.
    with (ROOT/'artifacts/experiments/expert_routes_20260928/calibrated_oof.csv').open(encoding='utf-8') as f:allrows=list(csv.DictReader(f))
    import runpy
    fields=runpy.run_path(str(ROOT/'src/nailfold_report/final_registry.py'))['FIELDS']
    lookup={x[0]:x for x in fields}
    for f in metrics['fields']:
        if 'accuracy' not in f:continue
        spec=lookup[f['field']][3]
        if f['kind']=='binary':
            rr=[r for r in allrows if r['target']==spec[0]]
            pairs=[(int(r['y']),int(float(r['q'])>0.5)) for r in rr]
        else:
            key=spec[0].split('_')[0]
            with (ROOT/f'artifacts/experiments/expert_routes_20260928/calibrated_3band_{key}.csv').open(encoding='utf-8') as fh:rr=list(csv.DictReader(fh))
            pairs=[(int(r['y']),int(r['pred'])) for r in rr]
        require(len({r['exam_case_id'] for r in rr})==len(rr),f"{f['field']}: duplicate case predictions")
        cm=[[sum(y==i and h==j for y,h in pairs) for j in range(len(f['class_labels']))] for i in range(len(f['class_labels']))]
        require(cm==f['confusion_matrix'],f"{f['field']}: saved OOF decisions mismatch")
    passed.append('20 field definitions, confusion matrices and OOF decisions checked; 47-exam subset consistency checked')
    if check_models:
        for rel,want in manifest['model_assets'].items():
            require((ROOT/rel).is_file(),f'missing model: {rel}')
            if (ROOT/rel).is_file():require(sha(ROOT/rel)==want,f'model hash mismatch: {rel}')
        passed.append('all 5 local model assets checked')
    for msg in passed:print('CHECK:',msg)
    for msg in errors:print('ERROR:',msg)
    if errors:raise SystemExit(1)
    print('PASS: handoff consistency checks; this is not an accuracy or clinical validation test')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-models',action='store_true')
    check(p.parse_args().check_models)
