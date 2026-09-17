import json,csv,re
from sklearn.metrics import balanced_accuracy_score
fields=['clarity','capillary_count','crossing_ratio','malformation_ratio','blood_color','exudation','hemorrhage','subpapillary_venous_plexus','papilla','sweat_duct']
continuous={'capillary_count':1.0,'crossing_ratio':10.0,'malformation_ratio':10.0,'sweat_duct':1.0}
def num(s):
    if s is None:return None
    vals=[float(x) for x in re.findall(r'\d+(?:\.\d+)?',str(s))]
    return sum(vals[:2])/min(2,len(vals)) if vals else None
rows=[]
for model,method,path in [('Qwen3-VL-8B','zero_shot','qwen_zero_pred.jsonl'),('Qwen3-VL-8B','few_shot','qwen_few_pred.jsonl'),('MedGemma-4B','zero_shot','med_zero_pred.jsonl'),('MedGemma-4B','few_shot','med_few_pred.jsonl'),('InternVL3-8B','zero_shot','intern_zero_pred.jsonl'),('InternVL3-8B','few_shot','intern_few_pred.jsonl')]:
    for line in open(path,encoding='utf-8'):
        rows.append((model,method,json.loads(line)))
out=[]
for model,method in sorted(set((x[0],x[1]) for x in rows)):
    subset=[r for m,me,r in rows if m==model and me==method]
    for f in fields:
        if f in continuous:
            pairs=[(num(r['target'].get(f)),num(r['prediction'].get(f))) for r in subset]
            pairs=[p for p in pairs if p[0] is not None]; errs=[abs(a-b) for a,b in pairs if b is not None]
            tol=continuous[f]
            out += [dict(model=model,method=method,field=f,metric='MAE',value=sum(errs)/len(errs) if errs else None,n=len(pairs)),dict(model=model,method=method,field=f,metric='clinical_tolerance_hit',value=sum(e<=tol for e in errs)/len(pairs) if pairs else 0,n=len(pairs))]
        else:
            pairs=[(r['target'].get(f),r['prediction'].get(f) or '__INVALID__') for r in subset if r['target'].get(f) is not None]
            y=[a for a,b in pairs];p=[b for a,b in pairs]
            out.append(dict(model=model,method=method,field=f,metric='balanced_accuracy',value=balanced_accuracy_score(y,p),n=len(y)))
with open('new_metrics.csv','w',newline='',encoding='utf-8-sig') as h:
    w=csv.DictWriter(h,fieldnames=['model','method','field','metric','value','n']);w.writeheader();w.writerows(out)
for r in out:
    if r['metric']=='balanced_accuracy': r['normalized_score']=r['value']
    elif r['metric']=='clinical_tolerance_hit': r['normalized_score']=r['value']
    else: r['normalized_score']=max(0.0,1.0-r['value']/(10.0 if r['field'] in ('crossing_ratio','malformation_ratio') else 2.0))
with open('normalized_summary.csv','w',newline='',encoding='utf-8-sig') as h:
    w=csv.DictWriter(h,fieldnames=['model','method','field','normalized_score']);w.writeheader()
    for r in out:
        if r['metric']!='MAE': w.writerow({k:r[k] for k in ['model','method','field','normalized_score']})
groups={}
for r in out:
    if r['metric']!='MAE': groups.setdefault((r['model'],r['method']),[]).append(r['normalized_score'])
with open('model_normalized_overall.csv','w',newline='',encoding='utf-8-sig') as h:
    w=csv.writer(h);w.writerow(['model','method','normalized_overall_score'])
    for (m,me),v in groups.items(): w.writerow([m,me,sum(v)/len(v)])
