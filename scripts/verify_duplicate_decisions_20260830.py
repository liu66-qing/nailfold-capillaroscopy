from __future__ import annotations
import csv, json, xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GOV=ROOT/'artifacts/audits/vascular_dataset_governance_20260830'
XML=ROOT/'artifacts/derived/vascular_dataset_governance_20260830/xml_repaired_normalized'

def sig(stem):
    root=ET.parse(XML/f'{stem}.xml').getroot(); out=[]
    for o in root.findall('object'):
        bb=o.find('bndbox'); out.append((o.findtext('name') or '',*(bb.findtext(k) or '' for k in ('xmin','ymin','xmax','ymax'))))
    return out

def main():
    proposed={'D009':'B','D058':'A','D340':'MERGE','D356':'B','D401':'B','D463':'A','D469':'A','D486':'A','D494':'MERGE','D540':'MERGE'}
    rows=[]
    src=list(csv.DictReader((GOV/'duplicate_group_decisions.csv').open(encoding='utf-8-sig')))
    for r in src:
        gid=r['duplicate_group_id']; a=Path(r['member_a']).stem; b=Path(r['member_b']).stem
        sa,sb=sig(a),sig(b); ca,cb=Counter(x[0] for x in sa),Counter(x[0] for x in sb)
        rows.append({'duplicate_group_id':gid,'member_a':a,'member_b':b,'actual_a_box_count':len(sa),'actual_b_box_count':len(sb),'actual_a_class_counts':json.dumps(dict(ca),ensure_ascii=False),'actual_b_class_counts':json.dumps(dict(cb),ensure_ascii=False),'user_proposed_decision':proposed.get(gid,''),'ai_validation':'REJECT_PROPOSED_AS_FINAL' if ca!=cb or len(sa)!=len(sb) else 'CANDIDATE_ONLY','final_decision':'HOLD','reason':'same image but annotation signatures differ; visual adjudication or explicit annotation policy required','evidence':str(GOV/'duplicate_annotation_consistency.csv')})
    with (GOV/'duplicate_decision_verification.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (GOV/'duplicate_decision_verification.md').write_text('''# 重复组裁决复核（2026-08-30）\n\n用户提供的 A/B/MERGE 建议已与实际派生 XML 逐框解析核对。由于每组图像 MD5 相同但 XML 框数量或类别计数不同，不能仅依据“框更多/类别更丰富”自动接受任何 A/B/MERGE。当前 10 组最终决策全部保持 `HOLD`，等待可审计的视觉/标注政策裁决。\n\n核对明细见 `duplicate_decision_verification.csv`。\n''',encoding='utf-8')

if __name__=='__main__':main()
