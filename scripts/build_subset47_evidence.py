"""Summarise saved internal OOF predictions for the designated 47-exam subset."""
import csv,json,sys,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,recall_score,confusion_matrix,roc_auc_score
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from nailfold_report.final_registry import FIELDS,RELEASE_ID
E=R/'artifacts/experiments/expert_routes_20260928'
I=R/'artifacts/experiments/locked_consumed_20260924/features_locked/index.csv'
D=R/'evaluation/subset_47';D.mkdir(parents=True,exist_ok=True)
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
ids={r['exam_case_id'] for r in rows(I)}
assert len(ids)==47,len(ids)
allrows=rows(E/'calibrated_oof.csv')
source=json.loads((E/'calibrated_router.json').read_text(encoding='utf-8'))
out=[]
for field,name,kind,spec,_ in FIELDS:
    a=dict(field=field,name=name,kind=kind)
    if kind in ('fixed','derived'):
        a.update(status='not_evaluated',n=None);out.append(a);continue
    if kind=='binary':
        rr=[r for r in allrows if r['target']==spec[0] and r['exam_case_id'] in ids]
        y=np.array([int(r['y']) for r in rr]);p=np.array([float(r['q']) for r in rr]);h=(p>0.5).astype(int)
        labels=list(spec[1]);s=source['binary'][spec[0]]['acc_rule'];auc=float(roc_auc_score(y,p)) if len(set(y))==2 else None
    else:
        key=spec[0].split('_')[0];rr=[r for r in rows(E/('calibrated_3band_'+key+'.csv')) if r['exam_case_id'] in ids]
        y=np.array([int(r['y']) for r in rr]);h=np.array([int(r['pred']) for r in rr]);labels=list(spec[2]);s=source['three_band'][key];auc=None
    assert len({r['exam_case_id'] for r in rr})==len(rr)
    acc=float(accuracy_score(y,h));assert abs(acc-s['locked_acc'])<0.00006,field
    a.update(status='evaluated',n=len(y),missing_labels=47-len(y),accuracy=acc,balanced_accuracy=float(balanced_accuracy_score(y,h)),auroc=auc,class_labels=labels,recall=recall_score(y,h,labels=list(range(len(labels))),average=None,zero_division=0).tolist(),confusion_matrix=confusion_matrix(y,h,labels=list(range(len(labels)))).tolist())
    out.append(a)
obj=dict(title='47例评估子集',release_id=RELEASE_ID,evaluation_type='internal_cross_validation_out_of_fold_subset',subset_exams=47,source='artifacts/experiments/expert_routes_20260928',subset_index_sha256=hashlib.sha256(I.read_bytes()).hexdigest(),fields=out)
(D/'metrics.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
s=['# 47例评估子集','','版本：`expert-router-final`。评估类型：**内部交叉验证折外结果**。各字段按有效标签数统计。','', '|字段|有效n|ACC|BA|AUROC|各类召回|','|---|---:|---:|---:|---:|---|']
for a in out:
    if a['n'] is None:s.append(f"|{a['name']}|—|—|—|—|固定/派生项，不适用|");continue
    auc='—' if a['auroc'] is None else f"{a['auroc']:.4f}"
    s.append(f"|{a['name']}|{a['n']}|{a['accuracy']:.4f}|{a['balanced_accuracy']:.4f}|{auc}|"+'/'.join(f'{v:.3f}' for v in a['recall'])+'|')
s+=['','完整类别顺序和混淆矩阵见 [metrics.json](metrics.json)。三分类AUROC未计算。','', '复现：`python scripts/build_subset47_evidence.py`。仅汇总已保存预测，不重新运行模型；15个预测字段的ACC均与既有子集统计核对一致。']
s += ['', '原图与逐病例标签：[下载47例数据包](https://github.com/liu66-qing/nailfold-capillaroscopy/releases/download/expert-router-final/subset47_public_v1.zip)。47例、402张原图、21字段；校验值见 [data_download.json](data_download.json)。']
(D/'README.md').write_text('\n'.join(s)+'\n',encoding='utf-8')
print('Verified 47 exam IDs and 15 field accuracies; exported aggregate metrics only.')
