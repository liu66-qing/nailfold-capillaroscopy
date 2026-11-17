import csv, json, hashlib, sys, subprocess
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'src'))
from nailfold_report.final_registry import FIELDS, BINARY_TARGETS
D=R/'release/final_v1/evidence'; D.mkdir(exist_ok=True)
E=R/'artifacts/experiments/expert_routes_20260928'
read=lambda p: json.loads(p.read_text(encoding='utf-8'))
cal=read(E/'calibrated_router.json')
with (E/'calibrated_oof.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
result=[]
for field,name,kind,spec,ref in FIELDS:
    o=dict(field=field,name=name,kind=kind,independent_holdout=None,release_matched_loao=None)
    if kind in ('fixed','derived'):
        o.update(status='not_image_prediction' if kind=='fixed' else 'derived_not_independently_evaluated',value=spec)
        result.append(o); continue
    if kind=='binary':
        rr=[x for x in rows if x['target']==spec[0]]
        y=np.array([int(x['y']) for x in rr]); q=np.array([float(x['q']) for x in rr]); h=(q>0.5).astype(int)
        s=cal['binary'][spec[0]]['acc_rule']; src='calibrated_oof.csv'; labels=list(spec[1])
        auc=float(roc_auc_score(y,q))
    else:
        key=spec[0].split('_')[0]; src='calibrated_3band_'+key+'.csv'
        with (E/src).open(encoding='utf-8') as f: rr=list(csv.DictReader(f))
        y=np.array([int(x['y']) for x in rr]); h=np.array([int(x['pred']) for x in rr]); s=cal['three_band'][key]; labels=list(spec[2]); auc=None
    acc=float(accuracy_score(y,h)); ba=float(balanced_accuracy_score(y,h))
    assert abs(acc-s['acc'])<0.00006 and abs(ba-s['ba'])<0.00006, field
    o.update(status='exploratory_internal_cv',n=len(y),accuracy=acc,balanced_accuracy=ba,auroc=auc,class_labels=labels,recall=recall_score(y,h,labels=list(range(len(labels))),average=None,zero_division=0).tolist(),confusion_matrix=confusion_matrix(y,h,labels=list(range(len(labels)))).tolist(),training_fold_mode_accuracy=s['mode_acc'],accuracy_gain_ci95=[s['acc_minus_mode'][0],s['acc_minus_mode'][2]],historical_locked_subset_oof_accuracy=s['locked_acc'],source=str((E/src).relative_to(R)).replace('\\','/'))
    result.append(o)
manifest=read(R/'release/final_v1/release_manifest.json')
for p,h in manifest['model_assets'].items(): assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
sources=[R/'src/nailfold_report/final_registry.py',R/'src/nailfold_report/final_inference.py',R/'scripts/eval_calibrated_router.py',R/'scripts/eval_router_loao.py',E/'calibrated_router.json',E/'calibrated_oof.csv',E/'router_loao.csv']+list(E.glob('calibrated_3band_*.csv'))
pack=dict(release_id=manifest['release_id'],inference_code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),training_exams=233,binary_head_targets=len(BINARY_TARGETS),output_rows=len(FIELDS),model_sha256=manifest['model_assets'],source_sha256={str(p.relative_to(R)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},independent_holdout_status='unavailable_original_47_in_training',external_validation_status='unavailable',evaluation_unit='exam_case_patient_identity_not_verified',uncertainty='Existing CIs cover accuracy gain over training-fold majority only; no new BA/AUC CI computed.',fields=result)
(D/'metrics.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=['# expert-router-final 交付证据','', '本包整理既有结果并从保存的折外预测复算；没有重训、没有新增测试、没有读取原始图像或重新预测原47例。','', '## 版本与入口','',f"- 推理代码提交：`{pack['inference_code_commit']}`。",f"- 发布版：`expert-router-final`；权重：`expert_router_v1`；API：`final_api.py`；推理：`final_inference.py`。",f"- 训练：233例检查；{len(BINARY_TARGETS)}个二分类目标，组合为20行输出（包含4个固定行、1个派生行）。此前说明中的“13个二分类目标”与当前注册表不一致，以本次数量为准。",'- DINOv2-B、颜色统计、分割几何、检测特征四路等权融合，Platt校准；四个核心权重已核对SHA256。','', '## 证据边界','', '- 独立锁定集：没有。原47例已进入全量训练；下述原locked子集结果属于交叉验证子集描述，不能称独立测试。','- 现有5折内部验证含折内校准；经多轮探索，指标可能乐观。分割器等上游监督是否对每个外层折隔离未在本次重新审计，不能认定为完整端到端无泄漏验证。','- 单位是检查病例；病人身份未核实，不沿用“病人级”表述。','- 现有LOAO见 `../../../artifacts/experiments/expert_routes_20260928/router_loao.csv`：ALL臂使用p>训练阳性率；发布版用校准q>0.5，因此不作为发布规则匹配的LOAO成绩。','- 固定行不使用图像；派生行无独立评估。置信等级不等于拒答，当前发布版仍输出这些行。','- 本包不证明健康建议有效性、外部泛化或可上线。','', '## 发布字段的内部交叉验证','', 'ACC、BA、二分类AUROC、召回和混淆矩阵由保存预测复算，并与原JSON核对。三分类AUROC未提供；不把单侧头AUROC冒充三分类AUROC。增益CI是相对训练折众数的accuracy差异区间，不是accuracy或BA自身区间。','', '|字段|类型|n|ACC|BA|AUROC|各类召回（类别顺序见JSON）|','|---|---|---:|---:|---:|---:|---|']
for o in result:
    if 'accuracy' not in o: lines.append(f"|{o['name']}|{o['kind']}|—|—|—|—|不适用|");continue
    auc='—' if o['auroc'] is None else f"{o['auroc']:.4f}"
    lines.append(f"|{o['name']}|{o['kind']}|{o['n']}|{o['accuracy']:.4f}|{o['balanced_accuracy']:.4f}|{auc}|"+'/'.join(f'{v:.3f}' for v in o['recall'])+'|')
lines += ['', '## 文件及复现','', '- `metrics.json`：完整字段指标、类别顺序、混淆矩阵、旧locked子集描述、证据缺口、代码版本及来源哈希。','- `python scripts/build_delivery_evidence.py`：复算并生成本包；需要本地已下载的模型权重。','- 不包含患者原图、新增逐病例信息或凭据。','- 运行测试通过仅表示软件行为满足测试，不等于预测准确率或健康建议验证。']
(D/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Verified and exported',len(result),'fields;',len(BINARY_TARGETS),'binary targets')
