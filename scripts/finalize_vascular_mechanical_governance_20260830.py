from __future__ import annotations
import csv, json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GOV=ROOT/'artifacts/audits/vascular_dataset_governance_20260830'
DER=ROOT/'artifacts/derived/vascular_dataset_governance_20260830'

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    dup=list(csv.DictReader((GOV/'duplicate_annotation_consistency.csv').open(encoding='utf-8-sig')))
    rows=[]
    for r in dup:
        members = r['member_paths'].split(';')
        rows.append({
            'duplicate_group_id':r['duplicate_group_id'],'member_a':members[0] if members else '','member_b':members[1] if len(members)>1 else '','member_paths':r['member_paths'],'xml_signature_count':r['xml_signature_count'],
            'ai_recommendation':'HOLD','decision':'HOLD','decision_reason':'exact image duplicate but conflicting XML annotations; no safe automatic A/B/merge choice',
            'reviewer':'user_or_clinical_annotator','review_time':'','decision_guide':'A=采用member_a; B=采用member_b; MERGE=逐框合并且去重; HOLD=暂不使用','evidence_path':str(GOV/'duplicate_annotation_consistency.csv')})
    write_csv(GOV/'duplicate_group_decisions.csv',rows,list(rows[0]))
    sem=json.loads((GOV/'semantic_definition_evidence.json').read_text(encoding='utf-8'))
    sem['project_label_candidates']['cross_vessel']='crossing/anastomotic vessel region detection class (AI working assumption; preserve as separate class)'
    sem['project_label_candidates']['corss_vessel']='normalize spelling to cross_vessel only; do not alter semantic class'
    sem['ai_working_decision']={'cross_vessel':'retain_separate_class','reason':'most likely an object/region class in a detection dataset; collapsing to vessel would erase positive examples and is harder to recover later','confidence':'moderate','clinical_approval_required':True}
    (GOV/'semantic_definition_evidence.json').write_text(json.dumps(sem,ensure_ascii=False,indent=2),encoding='utf-8')
    (GOV/'cross_vessel_working_decision.md').write_text('''# cross_vessel 工作决策（2026-08-30）\n\n在没有临床负责人即时裁决的情况下，采用保守且可回退的工程决策：`cross_vessel` 保留为独立检测类别，解释为交叉/吻合区域或事件的框；`corss_vessel` 只做拼写归一化到 `cross_vessel`。\n\n理由：保留独立类不会丢失信息，后续可以做三类/二类消融；如果现在合并为 `vessel`，后续无法恢复原始语义。此决策仅用于辅助训练候选，不是临床定义认可。\n''',encoding='utf-8')
    device={
      'status':'UNCALIBRATED_BATCH_CONSISTENCY_ASSUMPTION','user_statement':'同一数据集大概率使用同一设备/倍率/标尺','allowed':['pixel-domain detection','pixel-domain segmentation','relative geometry features','within-batch comparisons'],'forbidden':['absolute micron measurements','cross-device calibration claims','clinical geometry claims'],'required_evidence':['device/camera model','magnification','pixel scale or stage micrometer','resampling/crop history']}
    (GOV/'device_calibration_status.json').write_text(json.dumps(device,ensure_ascii=False,indent=2),encoding='utf-8')
    # Normalize the one observed typo in a separate XML layer only.
    src_dir = DER / 'xml_repaired_all'; norm_dir = DER / 'xml_repaired_normalized'; norm_dir.mkdir(parents=True, exist_ok=True)
    norm_rows=[]
    for src in sorted(src_dir.glob('*.xml'), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        root=ET.parse(src).getroot(); changed=0
        for obj in root.findall('object'):
            if (obj.findtext('name') or '') == 'corss_vessel': obj.find('name').text='cross_vessel'; changed+=1
        dst=norm_dir/src.name; ET.ElementTree(root).write(dst,encoding='utf-8',xml_declaration=True)
        norm_rows.append({'source_xml':str(src),'derived_xml':str(dst),'source_md5':hashlib.md5(src.read_bytes()).hexdigest(),'derived_md5':hashlib.md5(dst.read_bytes()).hexdigest(),'typo_replacements':changed,'status':'AUTO_NORMALIZED_SPELLING_ONLY','raw_overwritten':'no'})
    write_csv(DER/'xml_spelling_normalization_manifest.csv',norm_rows,list(norm_rows[0]))
    (GOV/'duplicate_group_decisions.md').write_text('''# 重复组决策入口\n\n逐组决策表：`duplicate_group_decisions.csv`。\n\n当前 AI 默认决策为 `HOLD`，因为每组图像 MD5 相同但 XML 框/类别签名不一致。请在 `decision` 列填写 `A`、`B`、`MERGE` 或保持 `HOLD`，并补充 `reviewer`、`review_time`、`decision_reason`。\n''',encoding='utf-8')
    # Update final status with explicit assumptions.
    fs=json.loads((GOV/'final_status.json').read_text(encoding='utf-8'))
    fs['ai_working_decisions']={'cross_vessel':'retain_separate_class; normalize typo corss_vessel only','duplicate_groups':'all 10 HOLD pending human conflict resolution','device_calibration':'uncalibrated batch-consistency assumption; pixel-only use'}
    fs['human_required_minimum']=['source/provenance and permission for 454 original images','final clinical acceptance of cross_vessel semantics','resolve 10 duplicate-group XML conflicts','clinical acceptance of weak-field definitions','provide calibration evidence or keep geometry pixel-only','review 829 unresolved segmentation pairs']
    (GOV/'final_status.json').write_text(json.dumps(fs,ensure_ascii=False,indent=2),encoding='utf-8')
    # Add machine-readable assumption to training README.
    readme=DER/'train_ready/TRAINING_README.md'
    readme.write_text(readme.read_text(encoding='utf-8')+'\n\n工程决策：`cross_vessel` 保留独立类别，`corss_vessel` 仅归一化拼写；设备未物理标定，训练资产只允许像素域/相对几何用途。\n',encoding='utf-8')

if __name__=='__main__':main()
